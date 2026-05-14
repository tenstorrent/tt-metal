# ViTPose-B on Tenstorrent

Human pose estimation using [ViTPose-Base](https://huggingface.co/usyd-community/vitpose-base-simple) on Tenstorrent Wormhole/Blackhole hardware via TT-NN.

## Model Architecture

| Component | Details |
|---|---|
| Backbone | ViT-Base: 12 transformer layers, 768 hidden dim, 12 attention heads |
| Decoder | SimpleDecoder: Conv2d (768 → 256) + ReLU + bilinear upsample 4x + Conv2d (256 → 17) |
| Input | 256 x 192 RGB image (single person crop) |
| Output | 17 COCO keypoint heatmaps (64 x 48 each) |
| Precision | bfloat16 weights, HiFi4 math fidelity with fp32 accumulation |
| HuggingFace model | `usyd-community/vitpose-base-simple` |

### Supported Configurations

| Parameter | Values |
|---|---|
| Batch size | 1, 2, 4 |
| Input resolution | 256 x 192 (H x W), fixed |
| Input format | NCHW float32/bfloat16 (converted to NHWC on device) |
| Device | Wormhole B0, Blackhole P150 |

Batch size 8 exceeds L1 circular buffer capacity (~1.5 MB per Tensix core) and is not supported.

## Benchmark Results

### Accuracy — COCO val2017 Keypoint Detection (200-image subset, GT bounding boxes)

| Configuration | AP | AP50 | AP75 |
|---|---|---|---|
| TT bfloat16 | 0.764 | 0.934 | 0.846 |
| HuggingFace fp32 | 0.764 | 0.934 | 0.846 |
| Published (full val2017) | 0.758 | — | — |

PCC (Pearson Correlation Coefficient) vs HuggingFace fp32: **0.9989**

### Throughput (Blackhole P150, single card)

| Batch Size | Throughput (img/s) | Speedup |
|---|---|---|
| 1 | 37.5 | 1.0x |
| 2 | 72.3 | 1.9x |
| 4 | 142.0 | 3.8x |

## Quick Start

### Prerequisites

- Tenstorrent device (Wormhole B0 or Blackhole P150)
- Docker container with tt-metal built (e.g., `metalcon:may11build`)
- HuggingFace model weights (downloaded automatically on first run)

### Running in Docker

```bash
sudo docker run --rm \
  -v /home:/home \
  -v /dev/hugepages-1G:/dev/hugepages-1G \
  --device /dev/tenstorrent/3 \
  -e HF_HOME=/home/<user>/.cache/huggingface \
  -w /path/to/tt-metal \
  metalcon:may11build \
  bash -c "export TT_METAL_HOME=\$PWD && export PYTHONPATH=\$PWD && <command>"
```

### Demo — Single Image Pose Estimation

```bash
pytest models/demos/vision/pose_estimation/vitpose/wormhole/demo/demo_vitpose.py -v --timeout=600
```

Runs inference on a sample COCO image and prints detected keypoint coordinates and confidence scores. Tests batch sizes 1, 2, and 4.

### End-to-End Accuracy Tests

```bash
pytest models/demos/vision/pose_estimation/vitpose/common/tests/test_vitpose_e2e.py -v --timeout=600
```

Compares TT output against:
- bfloat16 reference model (PCC > 0.999)
- HuggingFace fp32 model (PCC > 0.998)

Tests batch sizes 1, 2, and 4.

### COCO Keypoint Evaluation

Reproduces published AP scores on COCO val2017.

**Additional setup:**
```bash
uv pip install pycocotools
```

**Download COCO val2017 data** to a local directory:
- `annotations/person_keypoints_val2017.json`
- `val2017/` (5000 images)

**Run evaluation:**
```bash
export COCO_DATA_DIR=/path/to/coco
export VITPOSE_EVAL_MAX_IMAGES=200  # 0 = full val2017

# TT device evaluation
pytest models/demos/vision/pose_estimation/vitpose/common/tests/test_vitpose_coco_eval.py::test_vitpose_coco_eval -v -s --timeout=3600

# HuggingFace fp32 baseline (CPU only, no device needed)
pytest models/demos/vision/pose_estimation/vitpose/common/tests/test_vitpose_coco_eval.py::test_vitpose_coco_eval_hf -v -s --timeout=3600
```

## File Structure

```
models/demos/vision/pose_estimation/vitpose/
├── README.md
├── common/
│   ├── common.py                    # Model loading utility
│   ├── reference/
│   │   └── vitpose_reference.py     # Pure-PyTorch reference implementation
│   ├── tt/
│   │   ├── ttnn_vitpose.py          # Top-level VitPose class
│   │   ├── ttnn_vitpose_backbone.py # ViT-Base backbone
│   │   ├── ttnn_vitpose_encoder.py  # Transformer encoder (12 layers)
│   │   ├── ttnn_vitpose_layer.py    # Single transformer layer
│   │   ├── ttnn_vitpose_attention.py# Multi-head self-attention
│   │   ├── ttnn_vitpose_mlp.py      # FFN (fc1 → GELU → fc2)
│   │   ├── ttnn_vitpose_embeddings.py # Patch embedding + position encoding
│   │   └── ttnn_vitpose_decoder.py  # SimpleDecoder head
│   └── tests/
│       ├── test_vitpose_e2e.py      # End-to-end accuracy tests
│       └── test_vitpose_coco_eval.py# COCO benchmark evaluation
└── wormhole/
    └── demo/
        └── demo_vitpose.py          # Interactive demo
```
