# Faster-RCNN Object Detection using TTNN APIs

## Overview

This is an implementation of Faster-RCNN with ResNet-50-FPN backbone using TTNN APIs for Tenstorrent hardware (Wormhole/Blackhole). The model performs object detection on images, producing bounding boxes, class labels, and confidence scores for 91 COCO object categories.

### Architecture

Faster-RCNN consists of:
1. **ResNet-50 Backbone** (TTNN) - Extracts multi-scale feature maps
2. **Feature Pyramid Network (FPN)** (TTNN) - Produces 256-channel feature pyramids
3. **Region Proposal Network (RPN)** (TTNN convolutions + CPU post-processing) - Generates object proposals
4. **ROI Align + Box Head** (CPU) - Classifies proposals and refines bounding boxes

The compute-intensive backbone, FPN, and RPN convolution layers run entirely on TT hardware. Dynamic operations (NMS, ROI Align, anchor generation) run on CPU.

### Optimizations Applied

- **BatchNorm Folding**: All FrozenBatchNorm2d layers are folded into preceding Conv2d weights during preprocessing
- **ReLU Fusion**: ReLU activations are fused with Conv2d operations using `ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)`
- **LoFi Math**: `MathFidelity.LoFi` for maximum throughput
- **bfloat8_b Weights**: Reduced precision weights for faster computation
- **Memory Deallocation**: Intermediate tensors are deallocated when no longer needed

## Setup

### Prerequisites

- Tenstorrent hardware (N150 or N300)
- tt-metal and TTNN libraries installed
- Python 3.8+

### Dependencies

```bash
pip install torch torchvision
```

The pretrained weights are downloaded automatically from torchvision.

## Running the Model

### Demo (with sample images)

Place sample images in `models/demos/vision/detection/faster_rcnn/demo/images/` or run with random input:

```bash
pytest models/demos/vision/detection/faster_rcnn/demo/demo.py::test_faster_rcnn_demo_sample -sv
```

### PCC Validation Tests

Validate backbone + FPN output against PyTorch reference:

```bash
pytest models/demos/vision/detection/faster_rcnn/tests/pcc/test_faster_rcnn.py::test_faster_rcnn_backbone_pcc -sv
```

Validate full model output format:

```bash
pytest models/demos/vision/detection/faster_rcnn/tests/pcc/test_faster_rcnn.py::test_faster_rcnn_full_model -sv
```

Run all PCC tests:

```bash
pytest models/demos/vision/detection/faster_rcnn/tests/pcc/test_faster_rcnn.py -sv
```

### Performance Benchmark

```bash
pytest models/demos/vision/detection/faster_rcnn/demo/demo.py::test_faster_rcnn_perf -sv
```

## File Structure

```
models/demos/vision/detection/faster_rcnn/
├── README.md                           # This file
├── common.py                           # Constants and model loading
├── reference/
│   └── faster_rcnn.py                  # PyTorch reference model wrapper
├── tt/
│   ├── model_preprocessing.py          # Weight preprocessing and BN folding
│   ├── ttnn_resnet50_backbone.py       # ResNet-50 backbone in TTNN
│   ├── ttnn_fpn.py                     # Feature Pyramid Network in TTNN
│   └── ttnn_faster_rcnn.py             # Full model assembly
├── tests/
│   └── pcc/
│       └── test_faster_rcnn.py         # PCC validation tests
└── demo/
    ├── demo.py                         # Demo and performance tests
    └── images/                         # Sample images for demo
```

## Model Details

| Parameter | Value |
|-----------|-------|
| Backbone | ResNet-50 |
| Neck | FPN (256 channels) |
| Input Resolution | 320x320 (configurable to 640x640) |
| Number of Classes | 91 (COCO) |
| Pretrained Weights | torchvision FasterRCNN_ResNet50_FPN_Weights.DEFAULT |
| Batch Size | 1 |

## Performance

Target: >= 10 FPS on N150/N300 hardware at 320x320 resolution.

## Known Limitations

- ROI Align and NMS operations run on CPU (these are dynamic, data-dependent operations)
- Batch size > 1 requires additional memory tuning
- Input resolution changes may require adjusting L1 memory configuration

## References

- [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497)
- [TTNN Model Bring-up Tech Report](../../../../../../tech_reports/ttnn/TTNN-model-bringup.md)
- [CNN Bring-up & Optimization in TT-NN](../../../../../../tech_reports/CNNs/cnn_optimizations.md)
- [torchvision Faster R-CNN](https://docs.pytorch.org/vision/main/models/faster_rcnn.html)
