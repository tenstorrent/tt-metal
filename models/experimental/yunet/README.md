# YUNet Face Detection - TTNN Implementation

TTNN implementation of YUNet face detection model for Tenstorrent Blackhole.

## Performance

| Metric | FPS | Latency |
|--------|-----|---------|
| Device Performance | 706 FPS | 1.42 ms |
| E2E (Trace + 2CQ) | 553 FPS | 1.81 ms |

Input size: 224x224

## Setup

1. Run setup script (clones YUNet repo and creates `__init__.py` files):

```bash
cd models/experimental/yunet
./setup.sh
```

2. Place pretrained weights at:
```
models/experimental/yunet/YUNet/weights/best.pt
```

The setup script clones https://github.com/jahongir7174/YUNet into the `YUNet/` folder and makes it importable as a Python package.

## Usage

### Run Demo

```bash
# Face detection on an image
python -m models.experimental.yunet.demo.demo --input <image.jpg> --output <result.jpg>

# With custom confidence threshold
python -m models.experimental.yunet.demo.demo -i face.jpg -o result.jpg -t 0.5
```

### Run Tests

```bash
# PCC test (validates TTNN vs PyTorch, PCC > 0.99)
pytest models/experimental/yunet/tests/pcc/test_pcc.py -v

# Performance tests
pytest models/experimental/yunet/tests/perf/test_yunet_perf.py -v
```

## Architecture

YUNet is a lightweight face detection model with:
- **Backbone**: 5 stages with DPUnit blocks (depthwise separable convolutions)
- **Neck**: FPN-style feature fusion with upsampling
- **Head**: Multi-scale detection (3 scales) outputting:
  - Classification scores (face/no-face)
  - Bounding box coordinates
  - Objectness scores
  - 5 facial keypoints (eyes, nose, mouth corners)

## Files

```
models/experimental/yunet/
├── setup.sh                 # Setup script (clone repo, download weights)
├── common.py                # Constants, model loading utilities
├── demo/
│   └── demo.py              # Command-line demo with visualization
├── runner/
│   ├── performant_runner.py       # Trace + 2CQ runner
│   └── performant_runner_infra.py # Runner infrastructure
├── tests/
│   ├── pcc/
│   │   └── test_pcc.py      # PCC validation test
│   └── perf/
│       └── test_yunet_perf.py  # Performance tests (device + trace 2CQ)
├── tt/
│   └── ttnn_yunet.py        # TTNN model implementation
└── YUNet/                   # (Created by setup.sh) PyTorch reference
    ├── nets/nn.py           # PyTorch model definition
    └── weights/best.pt      # Pretrained weights
```

## References

- Original YUNet: https://github.com/jahongir7174/YUNet
