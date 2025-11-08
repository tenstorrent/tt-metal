# YOLOv11 Pose Estimation Web Demo

Real-time pose estimation web demo using YOLOv11 on Tenstorrent hardware.

## Architecture

This web demo consists of:
- **Server**: FastAPI backend running YOLOv11 pose estimation inference
- **Client**: Streamlit frontend with webcam streaming and pose visualization

## Features

- Real-time pose detection and tracking
- 17-point COCO pose keypoints visualization
- Pose skeleton drawing with anatomical connections
- Webcam streaming with WebRTC
- Confidence thresholding and NMS for clean detections

## Setup and Running

### Simple Method (Recommended - Like YOLOv4)

**Server (Ubuntu with TTNN):**
```bash
cd pose_web_demo/server
./run_uvicorn.sh
```

**Client (Mac - via SSH tunnel):**
```bash
# SSH tunnel from Mac to Ubuntu
ssh -L 8000:localhost:8000 ubuntu@UBUNTU_IP

# In another terminal, run simple client
cd pose_web_demo/client
pip install -r requirements.txt
./run_simple.sh --api-url http://localhost:8000
```

**Open browser:** `http://localhost:8501`

---

### Advanced HTTPS Method

#### 1. Server Setup

```bash
cd pose/tt-metal/models/demos/yolov11/pose_web_demo/server
pip install -r requirements.txt
chmod +x run_uvicorn.sh
./run_uvicorn.sh
```

The server will start on `http://localhost:8000`

### 2. Client Setup

```bash
cd pose/tt-metal/models/demos/yolov11/pose_web_demo/client
pip install -r requirements.txt
chmod +x run_streamlit.sh
./run_streamlit.sh
```

The client will start on `http://localhost:8501`

#### Connecting to Remote Server

To connect the client to a server running on a different machine:

```bash
# Set environment variables
export SERVER_URL="http://your-server-ip:8000"
export DEVICE=0  # Camera device (optional)

# Run client
./run_streamlit.sh
```

Or pass parameters directly:
```bash
streamlit run yolov11_pose.py --server.port 8501 --server.address 0.0.0.0 -- --server-url "http://your-server-ip:8000" --device 0
```

**Note:** Ensure the server machine's firewall allows connections on port 8000, and the server is bound to `0.0.0.0` (not just localhost).

**Browser Compatibility:** Use Chrome or Firefox for best WebRTC support. Allow camera permissions when prompted.

**HTTPS Requirement:** Camera access requires HTTPS. Use `./run_streamlit_https.sh` or ngrok for secure access.

**Distributed Setup:** Server can run on TTNN machine, client on separate machine. Update `SERVER_URL` in client scripts to point to server IP.

### 3. HTTPS Setup for Camera Access

Modern browsers require HTTPS for camera access. Choose one of these options:

**Option A: Self-signed HTTPS (Recommended for local development)**
```bash
# Run with auto-generated SSL certificates
./run_streamlit_https.sh
```
- Automatically generates self-signed certificates with proper SAN (Subject Alternative Names)
- Accept the security warning in your browser (click "Advanced" → "Proceed to localhost")
- **Access at: `https://localhost:8501` or `https://127.0.0.1:8501`**
- ❌ **DO NOT use `https://0.0.0.0:8501`** (certificate won't match)

**Option B: Ngrok (Easy HTTPS tunneling)**
```bash
# Install ngrok first: https://ngrok.com/download
ngrok http 8501

# Then run normally (ngrok provides HTTPS URL)
./run_streamlit.sh
```

**Option C: Chrome Development Mode**
```bash
# Run Chrome with HTTP camera permission
google-chrome --allow-http-screen-capture --unsafely-treat-insecure-origin-as-secure=http://localhost:8501
```

### 4. Distributed Setup (Server + Client on Different Machines)

**Server Machine (Ubuntu with TTNN):**
```bash
# Get server IP
ip addr show | grep 'inet ' | grep -v 127.0.0.1

# Start server
cd pose_web_demo/server
./run_uvicorn.sh
```

**Client Machine (Mac/Windows):**
```bash
# Transfer client files from server
scp user@server-ip:/path/to/pose_web_demo_client.tar.gz .

# Extract and setup
tar -xzf pose_web_demo_client.tar.gz
cd pose_web_demo/client
pip install -r requirements.txt

# Configure server URL (edit run scripts)
# Replace YOUR_UBUNTU_IP with actual server IP
export SERVER_URL=http://YOUR_SERVER_IP:8000

# Run with HTTPS
./run_streamlit_https.sh
```

**Network Requirements:**
- Both machines on same network
- Server firewall allows port 8000
- Client can reach server IP:8000

### 5. Alternative Manual Run

**Server:**
```bash
cd /home/ubuntu/pose/tt-metal
PYTHONPATH=/home/ubuntu/pose/tt-metal:$PYTHONPATH uvicorn models.demos.yolov11.pose_web_demo.server.fast_api_yolov11_pose:app --host 0.0.0.0 --port 8000
```

**Client:**
```bash
cd /home/ubuntu/pose/tt-metal/models/demos/yolov11/pose_web_demo/client
streamlit run yolov11_pose.py --server.port 8501 --server.address 0.0.0.0
```

## API Endpoints

### POST `/pose_estimation_v2`
Upload an image to get pose detection results.

**Input**: Image file (JPEG/PNG)
**Output**: List of detections with format:
```json
[
  [x1, y1, x2, y2, confidence, class_id, kpt1_x, kpt1_y, kpt1_v, ...]
]
```

All coordinates are normalized to [0,1] relative to image dimensions.

## Pose Keypoints (COCO Format)

The model detects 17 keypoints:
1. nose
2. left_eye, right_eye
3. left_ear, right_ear
4. left_shoulder, right_shoulder
5. left_elbow, right_elbow
6. left_wrist, right_wrist
7. left_hip, right_hip
8. left_knee, right_knee
9. left_ankle, right_ankle

Each keypoint includes (x, y, visibility) values.

## Configuration

- **Confidence Threshold**: 0.6 (configurable in client)
- **NMS IoU Threshold**: 0.5
- **Max Detections**: 300 per image
- **Input Resolution**: 640x640

## Hardware Requirements

- Tenstorrent device with TTNN support
- Webcam for real-time demo
- Sufficient RAM for model loading

## Troubleshooting

1. **HTTPS/Camera issues**: Try the **Simple Method** above - it works with HTTP and SSH tunneling
2. **Python 3.13 compatibility**: Some packages may not support Python 3.13. Use Python 3.11-3.12 or conda for installation
3. **Server won't start**: Check TTNN device availability
4. **Client can't connect**: Ensure server is running on port 8000 and accessible from client machine
5. **Remote server connection fails**: Verify server is bound to `0.0.0.0` and firewall allows port 8000
6. **No detections**: Check confidence threshold and lighting conditions
7. **Performance issues**: Reduce frame processing rate in client code
8. **Network timeout**: Check network connectivity between client and server machines
9. **WebRTC/camera errors**: Try Chrome browser, refresh page, or check camera permissions
10. **"navigator.mediaDevices is undefined"**: Page not loaded securely - use HTTPS (see HTTPS Setup section)
11. **"net::ERR_CERT_AUTHORITY_INVALID"**: Use `https://localhost:8501` not `https://0.0.0.0:8501`, click "Advanced" → "Proceed to localhost"
12. **"Cannot set properties of undefined"**: Browser compatibility issue - use Chrome/Firefox

## Comparison with Object Detection Demo

| Feature | Object Detection | Pose Estimation |
|---------|------------------|-----------------|
| Model | YOLOv11 | YOLOv11-Pose |
| Output | Bounding boxes | Keypoints + boxes |
| Visualization | Boxes only | Skeleton + keypoints |
| API Endpoint | `/objdetection_v2` | `/pose_estimation_v2` |
| Use Case | Object detection | Human pose analysis |
