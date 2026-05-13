import os
from dotenv import load_dotenv

load_dotenv()
from flask import Flask, render_template, redirect, url_for, request, flash, Response, jsonify, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
from twilio.rest import Client
from ultralytics import YOLO
import cv2
import numpy as np
import os
import time
from datetime import datetime

# -------------------------------------------------
# APP CONFIG
# -------------------------------------------------
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secretkey123_secure_token'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

db = SQLAlchemy(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# -------------------------------------------------
# LOAD YOLO MODEL
# -------------------------------------------------
model = YOLO("yolov8n.pt")

# -------------------------------------------------
# SYSTEM SETTINGS
# -------------------------------------------------
CONFIDENCE_THRESHOLD = 0.4
ALERT_COOLDOWN = 30  # Seconds between WhatsApp alerts in Live/Video feeds

# -------------------------------------------------
# TWILIO CONFIG
# -------------------------------------------------
TWILIO_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP = "whatsapp:+14155238886"

client = Client(TWILIO_SID, TWILIO_AUTH)

# -------------------------------------------------
# DATABASE MODELS
# -------------------------------------------------
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))

class EmergencyContact(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150))
    whatsapp = db.Column(db.String(20))
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())

class AlertLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    alert_source = db.Column(db.String(50))
    density_score = db.Column(db.Float)
    status = db.Column(db.String(100))
    head_range = db.Column(db.String(100)) 
    message_status = db.Column(db.String(50))
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# -------------------------------------------------
# CREATE DB + DEFAULT ADMIN
# -------------------------------------------------
with app.app_context():
    db.create_all()
    if not User.query.filter_by(username="admin").first():
        admin = User(username="admin", password="admin123")
        db.session.add(admin)
        db.session.commit()

# -------------------------------------------------
# GLOBAL STATE VARIABLES
# -------------------------------------------------
density_history = []
current_status = "Low Density"
current_head_range = "0"  
current_estimated_people = 0
current_detected_count = 0
current_video_path = os.path.join(app.root_path, "videos", "sample.mp4")

heatmap_accumulator = None
last_alert_time = 0

# -------------------------------------------------
# DENSITY ESTIMATION & CLASSIFICATION
# -------------------------------------------------
def classify_density(density_score, detected_yolo_count=0, is_live=False):
    xp = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]
    fp = [0, 40, 150, 350, 750, 1500]
    
    edge_estimated_count = int(np.interp(density_score, xp, fp))
    
    if is_live:
        estimated_people = detected_yolo_count
    else:
        estimated_people = max(detected_yolo_count, edge_estimated_count)
    
    approx_str = str(estimated_people) if estimated_people < 10 else f"~{estimated_people}"

    if estimated_people < 100:
        return "Low Density", approx_str, estimated_people, (0, 255, 0)
    elif estimated_people < 300:
        return "Moderate Density", approx_str, estimated_people, (0, 255, 255)
    elif estimated_people < 700:
        return "High Density", approx_str, estimated_people, (0, 165, 255)
    elif estimated_people < 1200:
        return "Very High Density", approx_str, estimated_people, (0, 100, 255)
    else:
        return "Extreme Density", approx_str, estimated_people, (0, 0, 255)

def calculate_density(frame):
    frame_resized = cv2.resize(frame, (800, 600))
    gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 80, 200)

    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    edge_count = np.sum(edges > 0)
    density_score = edge_count / (800 * 600)

    return density_score, frame_resized

def detect_people(frame):
    results = model(frame, classes=[0], conf=CONFIDENCE_THRESHOLD, verbose=False)
    boxes = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            boxes.append((x1, y1, x2, y2))
    return boxes

def send_whatsapp_alert(message):
    contacts = EmergencyContact.query.all()
    if not contacts:
        return "failed"
    try:
        for contact in contacts:
            client.messages.create(
                body=message,
                from_=TWILIO_WHATSAPP,
                to=f"whatsapp:{contact.whatsapp}"
            )
        return "sent"
    except Exception as e:
        print(f"WhatsApp Error: {e}")
        return "failed"

# -------------------------------------------------
# CENTRAL ALERT PROCESSOR (UPDATED)
# -------------------------------------------------
def process_alert(source, density_score, status, approx_str, estimated_people, force_alert=False):
    global last_alert_time
    current_time = time.time()

    # If it's a live feed or video, we apply cooldowns and checks
    if not force_alert:
        if estimated_people == 0:
            return  # Do not send an alert if the room/frame is completely empty

        if current_time - last_alert_time < ALERT_COOLDOWN:
            return  # Wait 30 seconds before sending the next Live/Video alert

    # Change message title based on how crowded it is
    if "High" in status or "Extreme" in status:
        title = "🚨 CROWD DENSITY ALERT 🚨\nImmediate attention required."
    else:
        title = "📊 CROWD ANALYSIS REPORT 📊\nStandard check complete."

    try:
        with app.app_context():
            current_clock = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            message = f"""{title}

📍 Source: {source}
⚠️ Status: {status}
👥 Estimated People: {approx_str}
⏱️ Time: {current_clock}"""

            message_status = send_whatsapp_alert(message)

            new_alert = AlertLog(
                alert_source=source,
                density_score=round(density_score, 4),
                status=status,
                head_range=approx_str,
                message_status=message_status
            )
            db.session.add(new_alert)
            db.session.commit()

        # Update the cooldown timer so it doesn't spam
        last_alert_time = current_time
    except Exception as e:
        print(f"Alert Processing Error: {e}")

# -------------------------------------------------
# VIDEO STREAM PIPELINE 
# -------------------------------------------------
def generate_video_frames():
    global density_history, current_status, current_head_range, current_estimated_people
    global current_video_path, current_detected_count, heatmap_accumulator

    cap = cv2.VideoCapture(current_video_path)
    if not cap.isOpened():
        return

    density_history.clear()
    frame_count = 0
    current_boxes = []

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        density_score, frame = calculate_density(frame)

        if heatmap_accumulator is None:
            heatmap_accumulator = np.zeros((600, 800), dtype=np.float32)
        else:
            heatmap_accumulator *= 0.95

        if frame_count % 3 == 0:
            current_boxes = detect_people(frame)
            current_detected_count = len(current_boxes)

        for (x1, y1, x2, y2) in current_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            heatmap_accumulator[y1:y2, x1:x2] += 2

        heatmap_norm = cv2.normalize(heatmap_accumulator, None, 0, 255, cv2.NORM_MINMAX)
        heatmap_color = cv2.applyColorMap(heatmap_norm.astype(np.uint8), cv2.COLORMAP_JET)
        frame = cv2.addWeighted(frame, 0.7, heatmap_color, 0.3, 0)

        status, approx_str, estimated_people, color = classify_density(density_score, current_detected_count, is_live=False)
        current_status = status
        current_head_range = approx_str
        current_estimated_people = estimated_people

        cv2.putText(frame, f'Estimated People: {approx_str}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f'Status: {status}', (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        density_history.append(round(density_score, 4))
        if len(density_history) > 50:
            density_history.pop(0)

        # Trigger Video Alert (Requires 1+ people, respects 30s cooldown)
        process_alert("Video Stream", density_score, status, approx_str, estimated_people, force_alert=False)

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()

# -------------------------------------------------
# LIVE WEBCAM PIPELINE
# -------------------------------------------------
def generate_live_frames():
    global density_history, current_status, current_head_range, current_estimated_people
    global current_detected_count

    cap = cv2.VideoCapture(0)
    frame_count = 0
    current_boxes = []

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        density_score, frame = calculate_density(frame)

        if frame_count % 3 == 0:
            current_boxes = detect_people(frame)
            current_detected_count = len(current_boxes)

        for (x1, y1, x2, y2) in current_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        status, approx_str, estimated_people, color = classify_density(density_score, current_detected_count, is_live=True)
        current_status = status
        current_head_range = approx_str
        current_estimated_people = estimated_people

        cv2.putText(frame, f'Estimated People: {approx_str}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f'Status: {status}', (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        density_history.append(round(density_score, 4))
        if len(density_history) > 50:
            density_history.pop(0)

        # Trigger Live Alert (Requires 1+ people, respects 30s cooldown)
        process_alert("Live Webcam", density_score, status, approx_str, estimated_people, force_alert=False)

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()

# -------------------------------------------------
# STATIC IMAGE ANALYSIS
# -------------------------------------------------
@app.route("/image-analysis", methods=["GET", "POST"])
@login_required
def image_analysis():
    result = None
    filename = None

    if request.method == "POST":
        file = request.files.get('image')
        if file and file.filename:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            image = cv2.imread(filepath)
            density_score, image = calculate_density(image)

            boxes = detect_people(image)
            detected_count = len(boxes)

            for (x1, y1, x2, y2) in boxes:
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.imwrite(filepath, image)

            status, approx_str, estimated_people, _ = classify_density(density_score, detected_count, is_live=False)
            
            # BYPASSES COOLDOWN: Always sends report instantly when you manually upload an image
            process_alert("Image Upload", density_score, status, approx_str, estimated_people, force_alert=True)

            result = {
                "density_score": round(density_score, 4),
                "status": status,
                "head_range": approx_str,
                "estimated_people": estimated_people,
                "detected_count": detected_count
            }

    return render_template("image-analysis.html", result=result, filename=filename)

import base64

# -------------------------------------------------
# BROWSER WEBCAM PROCESSING (NEW)
# -------------------------------------------------
@app.route("/process_webcam_frame", methods=["POST"])
@login_required
def process_webcam_frame():
    global current_status, current_head_range, current_estimated_people, current_detected_count
    
    try:
        # Get base64 image from browser
        data = request.get_json()
        image_data = data['image'].split(',')[1]  # Remove "data:image/jpeg;base64,"
        
        # Decode to OpenCV format
        img_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Process frame
        density_score, frame = calculate_density(frame)
        boxes = detect_people(frame)
        current_detected_count = len(boxes)
        
        # Draw boxes
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Classify
        status, approx_str, estimated_people, color = classify_density(
            density_score, current_detected_count, is_live=True
        )
        current_status = status
        current_head_range = approx_str
        current_estimated_people = estimated_people
        
        # Add overlay text
        cv2.putText(frame, f'Estimated People: {approx_str}', (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f'Status: {status}', (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Update history
        density_history.append(round(density_score, 4))
        if len(density_history) > 50:
            density_history.pop(0)
        
        # Trigger alert
        process_alert("Browser Webcam", density_score, status, approx_str, 
                     estimated_people, force_alert=False)
        
        # Encode processed frame back to base64
        _, buffer = cv2.imencode('.jpg', frame)
        processed_image = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            "processed_image": f"data:image/jpeg;base64,{processed_image}",
            "status": status,
            "head_range": approx_str,
            "estimated_people": estimated_people,
            "detected_count": current_detected_count
        })
    
    except Exception as e:
        print(f"Webcam Processing Error: {e}")
        return jsonify({"error": str(e)}), 500

# -------------------------------------------------
# GENERAL ROUTES
# -------------------------------------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = User.query.filter_by(
            username=request.form.get("username"),
            password=request.form.get("password")
        ).first()

        if user:
            login_user(user)
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid credentials")

    return render_template("login.html")

@app.route("/")
@login_required
def dashboard():
    return render_template("index.html")

@app.route("/video-analysis", methods=["GET", "POST"])
@login_required
def video_analysis():
    global current_video_path, heatmap_accumulator
    if request.method == "POST":
        file = request.files.get('video')
        if file and file.filename:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            current_video_path = filepath
            heatmap_accumulator = None 

    return render_template("video-analysis.html")

@app.route("/live-analysis")
@login_required
def live_analysis():
    return render_template("live-analysis.html")

@app.route('/uploads/<filename>')
@login_required
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route("/emergency-data", methods=["GET", "POST"])
@login_required
def emergency_data():
    if request.method == "POST":
        new_contact = EmergencyContact(
            name=request.form.get("name"),
            whatsapp=request.form.get("whatsapp")
        )
        db.session.add(new_contact)
        db.session.commit()
        return redirect(url_for("emergency_data"))

    contacts = EmergencyContact.query.order_by(EmergencyContact.created_at.desc()).all()
    return render_template("emergency-data.html", contacts=contacts)

@app.route("/edit-contact/<int:id>", methods=["POST"])
@login_required
def edit_contact(id):
    contact = EmergencyContact.query.get_or_404(id)
    contact.name = request.form.get("name")
    contact.whatsapp = request.form.get("whatsapp")
    db.session.commit()
    return redirect(url_for("emergency_data"))

@app.route("/delete-contact/<int:id>")
@login_required
def delete_contact(id):
    contact = EmergencyContact.query.get_or_404(id)
    db.session.delete(contact)
    db.session.commit()
    return redirect(url_for("emergency_data"))

@app.route("/alert-management")
@login_required
def alert_management():
    alerts = AlertLog.query.order_by(AlertLog.created_at.desc()).all()
    return render_template("alert-management.html", alerts=alerts)

# -------------------------------------------------
# STREAM FEEDS & DATA API
# -------------------------------------------------
@app.route("/video_feed")
@login_required
def video_feed():
    return Response(generate_video_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/live_feed")
@login_required
def live_feed():
    return Response(generate_live_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/density_data")
@login_required
def density_data():
    return jsonify({
        "history": density_history,
        "status": current_status,
        "head_range": current_head_range,          
        "estimated_people": current_estimated_people, 
        "detected_count": current_detected_count
    })

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))

# -------------------------------------------------
# RUN APP
# -------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
