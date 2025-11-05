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

### 1. Server Setup

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

### 3. Alternative Manual Run

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

1. **Server won't start**: Check TTNN device availability
2. **Client can't connect**: Ensure server is running on port 8000 and accessible from client machine
3. **Remote server connection fails**: Verify server is bound to `0.0.0.0` and firewall allows port 8000
4. **No detections**: Check confidence threshold and lighting conditions
5. **Performance issues**: Reduce frame processing rate in client code
6. **Network timeout**: Check network connectivity between client and server machines
7. **WebRTC/camera errors**: Try Chrome browser, refresh page, or check camera permissions
8. **"Cannot set properties of undefined"**: Browser compatibility issue - use Chrome/Firefox

## Comparison with Object Detection Demo

| Feature | Object Detection | Pose Estimation |
|---------|------------------|-----------------|
| Model | YOLOv11 | YOLOv11-Pose |
| Output | Bounding boxes | Keypoints + boxes |
| Visualization | Boxes only | Skeleton + keypoints |
| API Endpoint | `/objdetection_v2` | `/pose_estimation_v2` |
| Use Case | Object detection | Human pose analysis |
