# YOLO26 Object Detection Model

YOLO26 is Ultralytics' latest object detection model, optimized for edge and low-power devices. This implementation runs on Tenstorrent hardware using TTNN.

## Features

- **End-to-End NMS-Free**: Native predictions without Non-Maximum Suppression post-processing
- **SiLU Activation**: Modern activation function with native TTNN support
- **Multi-Scale Detection**: P3, P4, P5 feature pyramid for objects of all sizes
- **Optimized for Tenstorrent**: BatchNorm folding, optimal sharding strategies

## Setup

First, install dependencies and download weights:

```bash
cd models/experimental/yolo26
./setup.sh
```

This will:
1. Install ultralytics package if not present
2. Download YOLO26 weights (starting with yolo26n - nano variant)

## Model Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         YOLO26                                   │
├─────────────────────────────────────────────────────────────────┤
│  Backbone (CSP-style)                                            │
│  ├── Stem: Conv 3x3 stride=2                                     │
│  ├── Stage 1: Conv + C2f                                         │
│  ├── Stage 2: Conv + C2f → P3 (stride 8)                        │
│  ├── Stage 3: Conv + C2f → P4 (stride 16)                       │
│  └── Stage 4: Conv + C2f + SPPF → P5 (stride 32)                │
├─────────────────────────────────────────────────────────────────┤
│  Neck (PAN - Path Aggregation Network)                           │
│  ├── Top-Down: P5 → upsample → concat P4 → C2f                  │
│  │             P4 → upsample → concat P3 → C2f → N3             │
│  └── Bottom-Up: N3 → downsample → concat → C2f → N4             │
│                 N4 → downsample → concat → C2f → N5             │
├─────────────────────────────────────────────────────────────────┤
│  Head (Detection)                                                │
│  ├── P3 Head: Conv 1x1 → [B, H/8, W/8, 84]                      │
│  ├── P4 Head: Conv 1x1 → [B, H/16, W/16, 84]                    │
│  └── P5 Head: Conv 1x1 → [B, H/32, W/32, 84]                    │
└─────────────────────────────────────────────────────────────────┘
```

## Model Variants

| Variant | Parameters | Input Size | Description |
|---------|-----------|------------|-------------|
| yolo26n | ~2.5M | 640×640 | Nano - fastest, for initial bringup |
| yolo26s | ~9M | 640×640 | Small - balanced |
| yolo26m | ~20M | 640×640 | Medium - accuracy focused |
| yolo26l | ~43M | 640×640 | Large - high accuracy |
| yolo26x | ~68M | 640×640 | Extra - maximum accuracy |

## Usage

### Running Demo

```bash
# Basic inference
python models/experimental/yolo26/demo/demo.py --input <image_path>

# With options
python models/experimental/yolo26/demo/demo.py \
    --input test.jpg \
    --output result.jpg \
    --variant yolo26n \
    --input-size 640 \
    --conf-threshold 0.25
```

### Running PCC Tests

```bash
# Full model PCC test
pytest models/experimental/yolo26/tests/pcc/test_pcc.py -v

# With specific input size
pytest models/experimental/yolo26/tests/pcc/test_pcc.py -v --input-size 320

# Backbone-only PCC test
pytest models/experimental/yolo26/tests/pcc/test_pcc.py::test_yolo26_backbone_pcc -v
```

### Running Performance Tests

```bash
# Device performance (kernel time)
pytest models/experimental/yolo26/tests/perf/test_yolo26_device_perf.py -v

# End-to-end performance
pytest models/experimental/yolo26/tests/perf/test_yolo26_device_perf.py::test_yolo26_e2e_perf -v
```

## Directory Structure

```
models/experimental/yolo26/
├── __init__.py
├── common.py                 # Common utilities (BN folding, sharding)
├── README.md                 # This file
├── setup.sh                  # Setup script for weights
├── demo/
│   └── demo.py              # Inference demo with visualization
├── reference/               # Reference PyTorch implementation (optional)
├── runner/
│   ├── __init__.py
│   ├── performant_runner.py     # Trace+2CQ runner (TODO)
│   └── performant_runner_infra.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py          # Pytest configuration
│   ├── pcc/
│   │   ├── __init__.py
│   │   └── test_pcc.py      # PCC comparison tests
│   └── perf/
│       ├── __init__.py
│       └── test_yolo26_device_perf.py  # Performance tests
└── tt/
    ├── __init__.py
    ├── model_preprocessing.py  # Weight loading utilities
    └── ttnn_yolo26.py          # Main TTNN model implementation
```

## Key Optimizations

### 1. BatchNorm Folding

BatchNorm parameters are folded into Conv weights at load time:
```python
# From common.py
folded_weight = conv_weight * (bn_weight / sqrt(bn_var + eps))
folded_bias = bn_bias - (bn_weight * bn_mean / sqrt(bn_var + eps))
```

### 2. Optimal Sharding Strategy

Sharding is selected based on tensor dimensions:
- **HEIGHT_SHARDED**: When N×H×W >> C (spatial dominant)
- **BLOCK_SHARDED**: When N×H×W ≈ C (balanced)
- **WIDTH_SHARDED**: When C >> N×H×W (channel dominant)

### 3. Memory Management

- `bfloat8_b` for weights (reduced memory, faster compute)
- Aggressive tensor deallocation
- Double buffering for activations and weights

### 4. Activation Function

YOLO26 uses SiLU (Swish) activation, natively supported:
```python
output = ttnn.silu(conv_output)
```

## Output Format

The model outputs 3 tensors (one per scale):
- P3: `[batch, input_size/8, input_size/8, num_classes + 4]`
- P4: `[batch, input_size/16, input_size/16, num_classes + 4]`
- P5: `[batch, input_size/32, input_size/32, num_classes + 4]`

Where `num_classes + 4` = 84 for COCO (80 classes + 4 bbox coordinates).

## Supported Input Sizes

- 320×320 - Fast inference
- 416×416
- 512×512
- **640×640** - Default, recommended
- 1024×1024 - Higher accuracy for small objects

## Current Status

**Work in Progress** - Initial implementation complete, needs testing on device.

### Completed
- [x] Project structure and scaffolding
- [x] Weight download from Ultralytics (yolo26n)
- [x] BatchNorm folding utilities
- [x] Backbone implementation (Conv, C2f, SPPF blocks)
- [x] Basic Neck and Head structure
- [x] PCC test framework
- [x] Demo script

### In Progress
- [ ] Verify backbone PCC against PyTorch reference
- [ ] Fix C2f channel calculations for exact match
- [ ] Implement attention mechanism in neck (model.10)

### TODO
- [ ] Add Trace+2CQ performant runner for higher throughput
- [ ] Add web demo (FastAPI + Streamlit)
- [ ] Support yolo26s, yolo26m variants
- [ ] Add segmentation head support (yolo26-seg)
- [ ] Optimize sharding for specific hardware (N150, P150)

## YOLO26n Actual Architecture (from weights)

```
Backbone:
  model.0:  Conv 3→16 (k3, stride 2)
  model.1:  Conv 16→32 (k3, stride 2)
  model.2:  C2f 32→64 (1 bottleneck)
  model.3:  Conv 64→64 (k3, stride 2)
  model.4:  C2f 64→128 (2 bottlenecks) → P3 (128ch)
  model.5:  Conv 128→128 (k3, stride 2)
  model.6:  C2f 128→128 (2 bottlenecks) → P4 (128ch)
  model.7:  Conv 128→256 (k3, stride 2)
  model.8:  C2f 256→256 (1 bottleneck)
  model.9:  SPPF 256→256 → P5 (256ch)

Neck (with attention):
  model.10: C2f+Attn 256→256
  model.13: C2f 384→128 (upsample path)
  model.16: C2f 256→64 → N3 (64ch)
  model.17: Conv 64→64 (downsample)
  model.19: C2f 192→128 → N4 (128ch)
  model.20: Conv 128→128 (downsample)
  model.22: C2f 384→256 → N5 (256ch)

Head (model.23):
  cv2: bbox predictions (4 outputs per scale)
  cv3: class predictions (80 outputs per scale)
  one2one_cv2/cv3: end-to-end predictions
```

## References

- [Ultralytics YOLO26 Documentation](https://docs.ultralytics.com/models/yolo26/)
- [YOLOv4 TTNN Tech Report](../../tech_reports/YoloV4-TTNN/yolov4.md)
- [YUNet Implementation](../yunet/) - Similar face detection model

## Performance Targets

| Hardware | Input Size | Target FPS |
|----------|-----------|------------|
| N150 Wormhole | 640×640 | >30 FPS |
| P150 Blackhole | 640×640 | >60 FPS |

Performance measurements will be updated after optimization.
