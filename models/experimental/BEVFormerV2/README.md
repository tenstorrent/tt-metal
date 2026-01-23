# 🚗 BEVFormerV2: Bird's Eye View Transformer for 3D Object Detection

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Supported Device](https://img.shields.io/badge/device-Wormhole%20(n150)-blue)
![Precision](https://img.shields.io/badge/precision-BF16-green)
![Input Resolution](https://img.shields.io/badge/input-Multi-view%20(6%20cameras)-lightgrey)
![Status](https://img.shields.io/badge/status-Stable-brightgreen)

---

## 🔍 Introduction

**BEVFormerV2** is a transformer-based approach for multi-view 3D object detection that generates Bird's Eye View (BEV) representations from multi-camera inputs. This implementation brings BEVFormerV2 to Tenstorrent hardware using the Tenstorrent Neural Network (TTNN) and TT-Metalium stack.

The model processes multi-view camera images to generate a unified BEV representation, enabling accurate 3D object detection in autonomous driving scenarios. It combines temporal self-attention, spatial cross-attention, and custom deformable attention mechanisms for efficient feature aggregation.

---

## 📘 Overview

This implementation adapts **BEVFormerV2** for **Tenstorrent hardware**, optimized for throughput and low-latency inference on **Wormhole** device.

The model is validated using internal test suites under `tests/` with PCC (Pearson Correlation Coefficient) validation against PyTorch reference implementations.

### Key Capabilities

- **Multi-View Fusion**: Processes 6 camera views simultaneously
- **Temporal Modeling**: Leverages historical BEV features for improved detection
- **3D Object Detection**: Detects 10 object classes (cars, pedestrians, cyclists, etc.)
- **BEV Representation**: Generates unified Bird's Eye View features at 100×100 resolution
- **Hardware Acceleration**: Optimized for Tenstorrent Wormhole device

### Model Specifications

- **Input**: 6 camera views (RGB images)
- **BEV Resolution**: 100×100 (bev_h × bev_w)
- **Number of Queries**: 900
- **Number of Classes**: 10 (nuScenes)
- **Backbone**: ResNet-50
- **FPN Levels**: 5 (P0-P4)
- **Transformer Layers**: 6 encoder + 6 decoder layers
- **Feature Channels**: 256

---

## :heavy_check_mark: Prerequisites

- Clone the **tt-metal** repository (source code & toolchains):
  <https://github.com/tenstorrent/tt-metal>
- Install **TT-Metalium™ / TT-NN™**:
  Follow the official instructions: <https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md>
- **Python Dependencies**:
  ```bash
  pip install torch torchvision
  pip install numpy opencv-python pillow
  pip install pyquaternion  # For coordinate transformations
  ```
- (Optional, for profiling) Build with profiler enabled:
  ```bash
  ./build_metal.sh --enable-profiler
  ```

---

## 🗂️ Repository Layout

| Directory | Purpose |
|------------|----------|
| `tt/` | Core Tenstorrent native modules of **BEVFormerV2** |
| `reference/` | PyTorch reference implementation |
| `demo/` | Demo scripts and sample data |
| `tests/` | Validation (PCC) and Performance test scripts |
| `common.py` | Common utilities (weight loading, key mapping) |

The `BEVFormerV2/` directory plugs into this structure, exposing inference, profiling, and test utilities consistent with other models in the repo.

---

## 🚀 Quickstart: Run BEVFormerV2

### Run Tests

```
pytest models/experimental/BEVFormerV2/tests/pcc/test_bevformer_v2.py::test_bevformerv2 -v
```

This runs an end-to-end flow that:

  - Loads the Torch reference model,
  - Runs the Tenstorrent Neural Network graph,
  - Compares results using PCC validation (threshold: 0.97).

**Individual Component Tests:**
```bash
pytest models/experimental/BEVFormerV2/tests/pcc/test_perception_transformer.py -v
pytest models/experimental/BEVFormerV2/tests/pcc/test_bevformer_head.py -v
pytest models/experimental/BEVFormerV2/tests/pcc/test_decoder_layer.py -v
pytest models/experimental/BEVFormerV2/tests/pcc/test_ffn.py -v
```

### Run the Demo

```
python models/experimental/BEVFormerV2/demo/test.py \
  --data-root models/experimental/BEVFormerV2/demo/demo_data \
  --sample-idx 0 \
  --out models/experimental/BEVFormerV2/demo/outputs/results.json
```

**Demo Arguments:**
- `--data-root`: Path to demo data directory (default: `models/experimental/BEVFormerV2/demo/demo_data`)
- `--sample-idx`: Sample index to process (default: 0, use -1 for all samples)
- `--out`: Output JSON file path (default: `models/experimental/BEVFormerV2/demo/outputs/results.json`)
- `--eval`: Run evaluation mode (optional)

**Expected output:**
```
Demo completed.
Detection results saved in JSON format with 3D bounding boxes, class labels, and confidence scores.
```

### Custom Images

You can place your camera images under:
```
models/experimental/BEVFormerV2/demo/demo_data/samples/
├── CAM_FRONT/
├── CAM_FRONT_LEFT/
├── CAM_FRONT_RIGHT/
├── CAM_BACK/
├── CAM_BACK_LEFT/
└── CAM_BACK_RIGHT/
```

Then re-run the demo:
```
python models/experimental/BEVFormerV2/demo/test.py
```

---

## 🧪 Validation

BEVFormerV2 is verified against PyTorch reference implementations for correctness.

| Output | PCC Score |
|--------|-----------|
| bev_embed | 0.956 |
| all_cls_scores | 0.99 |
| all_bbox_preds | 0.99 |

All tests use PCC validation to compare Tenstorrent Neural Network outputs with PyTorch reference outputs (threshold: 0.97).

---

## 🧮 Profiling & Debugging

Tenstorrent profiling tools provide detailed visibility into kernel and tensor operations.

Capture a short performance trace:
```bash
tt-trace capture --model bevformerv2.ttnn --duration 5s
tt-analyze trace.json --view timeline
```

Refer to the [Profiling Guide](../../docs/profiling.md) for more usage patterns.

---

## Performance

### Single Device (BS=1):

- end-2-end perf is **0.33 FPS** (Device Kernel)

To run perf test:
```
pytest models/experimental/BEVFormerV2/tests/perf/test_bevformerv2_perf.py -v
```

To collect perf reports with the profiler, build with `--enable-profiler`

### Performance Metrics (on N150)

| Metric | Value |
|--------|-------|
| **AVG DEVICE KERNEL SAMPLES/S** | 0.333 FPS |
| **AVG DEVICE FW SAMPLES/S** | 0.326 FPS |
| **AVG DEVICE BRISC KERNEL SAMPLES/S** | 0.337 FPS |
| **Device Kernel Duration** | 3.001 seconds |
| **Device FW Duration** | 3.064 seconds |

**Configuration:** Batch size 1, bfloat16 precision, BEV resolution 100×100

---

## Configuration Notes

- **Resolution**: Multi-view camera images (6 cameras) are supported end-to-end.
- **Device**: The demo opens a Wormhole device (default id typically 0). If you need to change it, adjust the device open call in the demo.
- **Batch Size**: Demo/tests are written for BS=1. For larger BS you'll need to verify memory layouts and tile alignment.
- **Memory Layouts**: The Tenstorrent Neural Network path uses TILE_LAYOUT and ROW_MAJOR_LAYOUT as needed for different operations.
- **Weights**: The loader maps PyTorch checkpoint keys → internal module keys. Weights are automatically downloaded on first use via `common.py`.

---

## Input/Output Format

### Input

**Multi-View Camera Images:**
- **Format**: RGB images from 6 cameras
- **Cameras**: CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT
- **Preprocessing**:
  - Resize to model input size (maintaining aspect ratio)
  - Normalize with Caffe-style mean: `[103.53, 116.28, 123.675]`
  - Convert to tensor format: `(batch, 3, H, W)`

### Output

**Detection Results:**
The model outputs 3D bounding boxes in JSON format:

```json
{
  "sample_token": "...",
  "results": {
    "car": [
      {
        "translation": [x, y, z],
        "size": [w, l, h],
        "rotation": [w, x, y, z],
        "velocity": [vx, vy],
        "detection_score": 0.5
      }
    ],
    "pedestrian": [...],
    ...
  }
}
```

**Coordinate System:**
- **Input**: Camera space → Ego space → Global space
- **Output**: Global space (lidar → ego → global transformation)
- Uses `pyquaternion` for accurate quaternion-based rotations

**Object Classes:**
1. barrier, 2. bicycle, 3. bus, 4. car, 5. construction_vehicle, 6. motorcycle, 7. pedestrian, 8. traffic_cone, 9. trailer, 10. truck

---

## 🔗 References

- [BEVFormer Paper (Li et al., 2022)](https://arxiv.org/abs/2203.17270)
- [BEVFormerV2 Paper (Li et al., 2023)](https://arxiv.org/abs/2211.10439)
- [BEVFormer GitHub](https://github.com/fundamentalvision/BEVFormer)
- [Tenstorrent Developer SDK Docs](https://tenstorrent.com/developer-docs)

---

## Citation

**BEVFormerV2 Reference:**
```bibtex
@article{li2023bevformerv2,
    title={BEVFormer v2: Adapting Modern Image Backbones to Bird's-Eye-View Recognition via Perspective Supervision},
    author={Li, Zhiqi and Chen, Wenhai and Li, Hongyang and Xie, Enze and Sima, Chonghao and Lu, Tong and Qiao, Yu and Dai, Jifeng},
    journal={CVPR},
    year={2023}
}
```

**Attribution:**
This implementation is adapted from [BEVFormer](https://github.com/fundamentalvision/BEVFormer), which is licensed under the Apache License, Version 2.0. Original work Copyright (c) OpenMMLab. Modified by Zhiqi Li.

---

**Copyright © 2026 Tenstorrent AI ULC.**
