# YUNet Face Detection Model

YUNet is a lightweight face detection model optimized for real-time inference. This implementation runs on Tenstorrent hardware using TTNN.

## Setup

First, clone the YUNet repository and download weights:

```bash
cd models/experimental/yunet
./setup.sh
```

This will:
1. Clone the original YUNet PyTorch repo to `YUNet/`
2. Set up the weights in `YUNet/weights/`

## Model Architecture

- **Backbone**: 5-stage feature extractor producing 3 scales (p3, p4, p5)
- **Neck**: FPN-style feature pyramid network with upsampling
- **Head**: Multi-scale detection heads outputting:
  - Classification (cls): 1 channel per scale
  - Bounding box (box): 4 channels per scale
  - Objectness (obj): 1 channel per scale
  - Keypoints (kpt): 10 channels per scale (5 landmarks × 2 coords)

## Input Sizes

The model supports two input sizes:
- **640×640** (default): Higher accuracy for smaller faces
- **320×320**: Faster inference, use `--input-size 320`

## Performance

### P150 Blackhole

| Input Size | Device FPS (kernel) | E2E FPS (trace+2CQ) |
|------------|---------------------|---------------------|
| 320×320    | ~1150 samples/s     | ~250 FPS |
| 640×640    | ~450 samples/s      | ~60 FPS |

### N150 Wormhole

| Input Size | Device FPS (kernel) | E2E FPS (trace+2CQ) |
|------------|---------------------|---------------------|
| 320×320    | ~545 samples/s      | ~171 FPS |
| 640×640    | ~185 samples/s      | ~43 FPS |

## Usage

### Running Demo

```bash
# Default 640x640 input size
python models/experimental/yunet/demo/demo.py --input <image_path>
#test_images available in yunet folder

# With 320x320 input size
python models/experimental/yunet/demo/demo.py --input <image_path> --input-size 320
```

### Running PCC Tests

```bash
# Default 640x640
pytest models/experimental/yunet/tests/pcc/test_pcc.py -v

# With 320x320 input size
pytest models/experimental/yunet/tests/pcc/test_pcc.py -v --input-size 320
```

### Running Performance Tests

```bash
# E2E performant test (default 640x640)
pytest models/experimental/yunet/tests/perf/test_e2e_performant.py -v

# E2E performant test with 320x320
pytest models/experimental/yunet/tests/perf/test_e2e_performant.py -v --input-size 320

# Device perf test (default 640x640)
pytest models/experimental/yunet/tests/perf/test_yunet_device_perf.py -v

# Device perf test with 320x320
pytest models/experimental/yunet/tests/perf/test_yunet_device_perf.py -v --input-size 320
```

### Web Demo (Real-time Face Detection)

The web demo provides a FastAPI server for inference and a Streamlit client for real-time webcam face detection.

> **Note:** For best results, run the Streamlit client locally on your laptop to avoid WebRTC NAT/firewall issues. The server runs on the Tenstorrent machine.

**1. Start the FastAPI server (on Tenstorrent machine):**

```bash
cd models/experimental/yunet/web_demo/server
pip install fastapi uvicorn python-multipart
./run_server.sh
```

Server runs at `http://0.0.0.0:8000`

**2. Run the Streamlit client:**

For best results, run the client locally on your laptop (avoids WebRTC NAT issues):

```bash
# Install dependencies
pip install streamlit streamlit-webrtc opencv-python requests av

# Copy client file to your laptop
scp user@server:~/tt-metal/models/experimental/yunet/web_demo/client/yunet_streamlit.py .

# Run client
streamlit run yunet_streamlit.py
```

In the sidebar, set **API Server URL** to `http://<server-ip>:8000`

Alternatively, run the client on the server:

```bash
cd models/experimental/yunet/web_demo/client
pip install streamlit streamlit-webrtc opencv-python requests av
./run_client.sh
```

## Directory Structure

```
models/experimental/yunet/
├── common.py                 # Common utilities and constants
├── README.md                 # This file
├── setup.sh                  # Setup script to clone YUNet repo
├── demo/
│   └── demo.py              # Demo with visualization
├── reference/
│   └── yunet_graph.svg      # Model architecture visualization
├── runner/
│   ├── performant_runner.py # Trace+2CQ runner
│   └── performant_runner_infra.py
├── test_images/             # Test images for demo
│   ├── group1.jpg
│   ├── group2.jpg
│   └── group3.jpg
├── tests/
│   ├── conftest.py          # Pytest configuration (--input-size option)
│   ├── pcc/
│   │   └── test_pcc.py     # PCC comparison tests
│   └── perf/
│       ├── test_e2e_performant.py      # E2E Trace+2CQ test
│       └── test_yunet_device_perf.py   # Device kernel perf test
├── tt/
│   ├── model_preprocessing.py  # Weight loading utilities
│   └── ttnn_yunet.py          # Main TTNN model
├── web_demo/
│   ├── client/
│   │   ├── yunet_streamlit.py # Streamlit webcam client
│   │   └── run_client.sh
│   └── server/
│       ├── fast_api_yunet.py  # FastAPI inference server
│       └── run_server.sh
└── YUNet/                     # Cloned PyTorch repo (after setup.sh)
    └── weights/best.pt
```

## Output Format

The model outputs 4 lists (one tensor per scale):
- `cls`: Classification scores [B, H, W, 1]
- `box`: Bounding box offsets [B, H, W, 4]
- `obj`: Objectness scores [B, H, W, 1]
- `kpt`: Keypoint offsets [B, H, W, 10]

Scales correspond to strides [8, 16, 32], producing feature maps of sizes:
- **320×320**: [40×40, 20×20, 10×10]
- **640×640**: [80×80, 40×40, 20×20]

## Keypoint Layout

The 5 facial landmarks detected:
1. Left Eye
2. Right Eye
3. Nose
4. Left Mouth Corner
5. Right Mouth Corner

## References

- [YUNet Paper](https://arxiv.org/abs/2107.14039)
- [YUNet GitHub](https://github.com/jahongir7174/YUNet)
