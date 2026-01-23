# BEVFormerV2: Bird's Eye View Transformer for 3D Object Detection

**Platforms:** Wormhole (n150)
**Supported Input Resolution:** Multi-view camera images (6 cameras: front, front-left, front-right, back, back-left, back-right)
**Performance:** ~0.33 FPS (Device Kernel) on Tenstorrent hardware

## Introduction

BEVFormerV2 is a transformer-based approach for multi-view 3D object detection that generates Bird's Eye View (BEV) representations from multi-camera inputs. This implementation brings BEVFormerV2 to Tenstorrent hardware using the TTNN (Tenstorrent Neural Network) and TT-Metalium stack.

This repository provides:
- A **reference PyTorch model** for correctness validation
- A **TTNN implementation** optimized for Tenstorrent hardware (Wormhole)
- **Comprehensive test suite** (PCC tests and performance benchmarks)
- **Demo application** with on-the-fly data generation
- **Model preprocessing utilities** for parameter conversion

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Repository Layout](#repository-layout)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Quickstart](#quickstart)
  - [Run Tests](#run-tests)
  - [Run the Demo](#run-the-demo)
  - [Performance Testing](#performance-testing)
- [Model Components](#model-components)
- [Input/Output Format](#inputoutput-format)
- [Performance](#performance)
- [Citation](#citation)
- [License](#license)

## Overview

BEVFormerV2 extends the original BEVFormer architecture with improved temporal modeling and spatial cross-attention mechanisms. The model processes multi-view camera images to generate a unified BEV representation, enabling accurate 3D object detection in autonomous driving scenarios.

### Key Capabilities

- **Multi-View Fusion**: Processes 6 camera views simultaneously
- **Temporal Modeling**: Leverages historical BEV features for improved detection
- **3D Object Detection**: Detects 10 object classes (cars, pedestrians, cyclists, etc.)
- **BEV Representation**: Generates unified Bird's Eye View features at 100×100 resolution
- **Hardware Acceleration**: Optimized for Tenstorrent Wormhole device

## Architecture

The BEVFormerV2 model consists of three main components:

### 1. Backbone Network (ResNet-50)
- Extracts multi-scale features from input images
- Outputs feature maps at 3 scales (C3, C4, C5)
- Supports Caffe-style normalization

### 2. Feature Pyramid Network (FPN)
- Fuses multi-scale backbone features
- Generates 5 feature levels (P0-P4) for multi-scale detection
- Output channels: 256

### 3. Perception Transformer
The perception transformer is the core of BEVFormerV2:

- **Temporal Self-Attention (TSA)**: Models temporal relationships in BEV space
  - Processes historical BEV features
  - Enables temporal consistency across frames

- **Spatial Cross-Attention (SCA)**: Aggregates multi-view image features
  - Projects image features to BEV space
  - Uses deformable attention for efficient feature sampling
  - Handles 6 camera views simultaneously

- **Encoder-Decoder Architecture**:
  - **Encoder**: 6 transformer layers with TSA and SCA
  - **Decoder**: 6 decoder layers with multi-head attention and custom deformable attention
  - **Feed-Forward Networks**: MLP layers with GELU activation

### 4. Detection Head (BEVFormerHead)
- Predicts 3D bounding boxes and class labels
- Outputs: bounding box coordinates, dimensions, rotation, and class scores
- Supports 10 object classes (nuScenes dataset)

### Model Specifications

- **Input**: 6 camera views (RGB images)
- **BEV Resolution**: 100×100 (bev_h × bev_w)
- **Number of Queries**: 900
- **Number of Classes**: 10 (nuScenes)
- **Backbone**: ResNet-50
- **FPN Levels**: 5 (P0-P4)
- **Transformer Layers**: 6 encoder + 6 decoder layers
- **Feature Channels**: 256

## Repository Layout

```
models/experimental/BEVFormerV2/
├── README.md
├── common.py
│
├── demo/
│   ├── test.py                       # Main demo script
│   ├── demo_data_loader.py
│   └── demo_data/                    # Sample camera images
│       └── samples/
│           ├── CAM_FRONT/
│           ├── CAM_FRONT_LEFT/
│           ├── CAM_FRONT_RIGHT/
│           ├── CAM_BACK/
│           ├── CAM_BACK_LEFT/
│           └── CAM_BACK_RIGHT/
│
├── reference/                         # PyTorch reference implementation
│   ├── bevformer_v2.py
│   ├── resnet.py
│   ├── fpn.py
│   ├── encoder.py
│   ├── decoder.py
│   ├── decoder.py
│   ├── perception_transformer.py
│   ├── temporal_self_attention.py
│   ├── spatial_cross_attention.py
│   ├── multihead_attention.py
│   ├── ffn.py
│   ├── head.py
│   ├── nms_free_coder.py
│   ├── modules.py
│   └── utils.py
│
├── tests/                             # Test suite
│   ├── pcc/
│   │   ├── test_bevformer_v2.py
│   │   ├── test_perception_transformer.py
│   │   ├── test_bevformer_head.py
│   │   ├── test_decoder_layer.py
│   │   ├── test_ffn.py
│   │   └── custom_preprocessors.py
│   └── perf/
│       └── test_bevformerv2_perf.py
│
└── tt/                                # TTNN implementation
    ├── ttnn_bevformer_v2.py
    ├── ttnn_backbone.py
    ├── ttnn_fpn.py
    ├── ttnn_encoder.py
    ├── ttnn_decoder.py
    ├── ttnn_decoder_layer.py
    ├── ttnn_perception_transformer.py
    ├── ttnn_temporal_self_attention.py
    ├── ttnn_spatial_cross_attention.py
    ├── ttnn_multihead_attention.py
    ├── ttnn_custom_ms_deformable_attention.py
    ├── ttnn_ffn.py
    ├── ttnn_bevformer_head.py
    ├── model_preprocessing.py        # Model parameter preprocessing
    ├── ttnn_utils.py
    └── utils.py
```

## Prerequisites

- Clone the **tt-metal** repository:
  ```bash
  git clone https://github.com/tenstorrent/tt-metal
  ```

- Install **TT-Metalium™ / TT-NN™**:
  Follow the official instructions: [INSTALLING.md](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)

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

## Setup

### Download Model Weights

The model weights are required for inference. The weights will be automatically downloaded on first use:

```bash
# Weights are downloaded automatically via common.py
# Default location: /tmp/bevformerv2_weights.pth
```

**Manual Download:**
The weights can be downloaded from Google Drive (ID: `1hC49RBbDW_qZJNHAfAjsmIezTtPKRevc`) or will be automatically fetched using `gdown` or `urllib`.

**CI Environment:**
In CI environments, weights are loaded via `model_location_generator` from the `vision-models/bevformer_v2` directory.

## Quickstart

### Run Tests

Run the PCC (Pearson Correlation Coefficient) test suite to verify model correctness:

```bash
# Full model integration test
pytest models/experimental/BEVFormerV2/tests/pcc/test_bevformer_v2.py::test_bevformerv2 -v

# Individual component tests
pytest models/experimental/BEVFormerV2/tests/pcc/test_perception_transformer.py -v
pytest models/experimental/BEVFormerV2/tests/pcc/test_bevformer_head.py -v
pytest models/experimental/BEVFormerV2/tests/pcc/test_decoder_layer.py -v
pytest models/experimental/BEVFormerV2/tests/pcc/test_ffn.py -v

# Run all PCC tests
pytest models/experimental/BEVFormerV2/tests/pcc/ -v
```

**Test Coverage:**
- `test_bevformer_v2.py`: Full model end-to-end test
- `test_perception_transformer.py`: Perception transformer component test
- `test_bevformer_head.py`: Detection head test
- `test_decoder_layer.py`: Decoder layer test
- `test_ffn.py`: Feed-forward network test

All tests use PCC validation to compare TTNN outputs with PyTorch reference outputs (threshold: 0.97).

### Run the Demo

The demo script demonstrates 3D object detection on multi-view camera images:

```bash
# Basic usage with default settings
python models/experimental/BEVFormerV2/demo/test.py

# Specify custom data root
python models/experimental/BEVFormerV2/demo/test.py \
    --data-root /path/to/demo_data

# Process specific sample
python models/experimental/BEVFormerV2/demo/test.py \
    --sample-idx 0

# Process all samples
python models/experimental/BEVFormerV2/demo/test.py \
    --sample-idx -1

# Custom output path
python models/experimental/BEVFormerV2/demo/test.py \
    --out /path/to/results.json
```

**Demo Arguments:**
- `--data-root`: Path to demo data directory (default: `models/experimental/BEVFormerV2/demo/demo_data`)
- `--sample-idx`: Sample index to process (default: 0, use -1 for all samples)
- `--out`: Output JSON file path (default: `models/experimental/BEVFormerV2/demo/outputs/results.json`)
- `--eval`: Run evaluation mode (optional)

**Demo Output:**
The demo generates a JSON file containing:
- Detected 3D bounding boxes
- Class labels and confidence scores
- Box coordinates in global space (lidar → ego → global transformation)
- Sample metadata

**Note:** The demo uses on-the-fly data generation via `demo_data_loader.py`, eliminating the need for external dataset files.

### Performance Testing

Run the performance benchmark to measure inference speed:

```bash
# Run performance test
pytest models/experimental/BEVFormerV2/tests/perf/test_bevformerv2_perf.py -v
```

**Performance Metrics:**
The test reports:
- **AVG DEVICE KERNEL SAMPLES/S**: Primary FPS metric (kernel execution only)
- **AVG DEVICE FW SAMPLES/S**: End-to-end device FPS (includes firmware overhead)
- **AVG DEVICE BRISC KERNEL SAMPLES/S**: BRISC kernel FPS

**Expected Performance:**
- **Device Kernel FPS**: ~0.33 samples/s
- **Device FW FPS**: ~0.33 samples/s
- **Device BRISC Kernel FPS**: ~0.34 samples/s

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
        "translation": [x, y, z],  // Global coordinates
        "size": [w, l, h],         // Width, length, height
        "rotation": [w, x, y, z],   // Quaternion rotation
        "velocity": [vx, vy],       // Velocity in x, y
        "detection_score": 0.5    // Confidence score
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
1. barrier
2. bicycle
3. bus
4. car
5. construction_vehicle
6. motorcycle
7. pedestrian
8. traffic_cone
9. trailer
10. truck

## Performance

### Performance Metrics (on N150)

| Metric | Value |
|--------|-------|
| **AVG DEVICE KERNEL SAMPLES/S** | 0.333 FPS |
| **AVG DEVICE FW SAMPLES/S** | 0.326 FPS |
| **AVG DEVICE BRISC KERNEL SAMPLES/S** | 0.337 FPS |
| **Device Kernel Duration** | 3.001 seconds |
| **Device FW Duration** | 3.064 seconds |
| **PCC (bev_embed)** | 0.956 |
| **PCC (all_cls_scores)** | 0.99 |
| **PCC (all_bbox_preds)** | 0.99 |

**Configuration:** Batch size 1, bfloat16 precision, BEV resolution 100×00

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
