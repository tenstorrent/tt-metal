# YuNet Face Detection Web Demo

Real-time face detection demo using YuNet model on Tenstorrent hardware.

## Architecture

```
┌─────────────────┐     HTTP/REST     ┌─────────────────┐
│  Streamlit      │ ───────────────▶ │  FastAPI        │
│  Client         │                   │  Server         │
│  (WebRTC)       │ ◀─────────────── │  (TT Hardware)  │
└─────────────────┘    Detections     └─────────────────┘
```

- **Server (FastAPI):** Runs YuNet inference on Tenstorrent hardware
- **Client (Streamlit):** Web UI with WebRTC for live camera feed

## Prerequisites

- Tenstorrent hardware (Wormhole/Blackhole)
- tt-metal environment set up and built
- Python 3.10+

## Installation

### 1. Activate tt-metal Environment

```bash
cd /path/to/tt-metal
source python_env/bin/activate
```

### 2. Install Server Dependencies

```bash
pip3 install fastapi uvicorn python-multipart
```

### 3. Install Client Dependencies

```bash
pip3 install streamlit streamlit-webrtc opencv-python requests av
```

## Running the Demo

### Step 1: Start the Server

```bash
cd models/experimental/yunet/web_demo/server
./run_server.sh
```

You should see:
```
Starting YuNet FastAPI server...
INFO: YuNet model loaded successfully!
INFO: Uvicorn running on http://0.0.0.0:8000
```

### Step 2: Start the Client (New Terminal)

```bash
cd models/experimental/yunet/web_demo/client
./run_client.sh
```

You should see:
```
Starting YuNet Streamlit client...
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
```

### Step 3: Open in Browser

- **Local:** http://localhost:8501
- **Remote:** http://<hostname>:8501 (e.g., http://sjc-snva-tp101:8501)

Allow camera access when prompted.

## Features

- **Real-time face detection** via WebRTC
- **Two input sizes:** 320x320 (faster ~5ms) or 640x640 (more accurate)
- **5 facial keypoints:** Left eye, Right eye, Nose, Mouth corners
- **Performance logging** in terminal

## Performance

| Input Size | Inference Time | Throughput |
|------------|----------------|------------|
| 320x320    | ~4.8ms         | ~208 FPS   |
| 640x640    | ~15-20ms       | ~50-65 FPS |

*Note: Inference time is pure TTNN model execution on Tenstorrent hardware.*

## Keypoint Legend

- 🔵 Left Eye
- 🟢 Right Eye
- 🔴 Nose
- 🟡 Left Mouth Corner
- 🟣 Right Mouth Corner

## API Reference

### `GET /`

Health check endpoint.

### `POST /facedetection`

Detect faces in an uploaded image.

**Parameters:**
- `file`: Image file (JPEG/PNG)
- `input_size`: Model input size (320 or 640, default: 320)
- `conf_thresh`: Confidence threshold (0.0-1.0, default: 0.35)

**Response:**
```json
{
  "detections": [
    {
      "box": [x1, y1, x2, y2],
      "conf": 0.95,
      "keypoints": [[x, y], ...]
    }
  ],
  "inference_time_ms": 4.8,
  "num_faces": 1
}
```

## Troubleshooting

### Camera not working
- Ensure you're accessing via `localhost` or HTTPS
- Check browser permissions for camera access

### Server connection failed
- Verify FastAPI server is running on port 8000
- Check firewall settings if accessing remotely

### Module not found errors
- Ensure tt-metal python_env is activated
- Run from tt-metal root directory

### Model loading error
- Ensure YuNet weights are available
- Clone YuNet repo if needed: `git clone https://github.com/ShiqiYu/libfacedetection.train.git models/experimental/yunet/YUNet`
