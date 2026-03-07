# Retinanet

### Platforms: Wormhole (n150)
### Supported Input Resolution:** `(512, 512)` = (Height, Width)

## Introduction
RetinaNet employs a ResNet50 backbone with an FPN to extract multi-scale features from five pyramid levels. The regression head predicts bounding box coordinates for each anchor, while the classification head outputs class probabilities across all anchors. Together, these components enable efficient multi-scale object detection.

## Prerequisites

- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)

## Directory Structure

```
models/experimental/retinanet/
├── README.md                          # This file
│
├── demo/                              # Demo application
│   ├── demo.py                       # Main demo script
│   └── processing.py                 # Image preprocessing and postprocessing utilities
│
├── resources/                         # Resources and sample images
│   ├── dog_800x800.jpg               # Sample test image
│   └── outputs/                      # Default output directory
│
├── tests/                             # Test suite
│   ├── pcc/                          # PCC (Pearson Correlation Coefficient) tests
│   │   ├── test_retinanet.py         # Main integration test
│   │   ├── test_resnet50_fpn.py      # FPN component test
│   │   ├── test_resnet50_backbone.py # Backbone component test
│   │   ├── test_resnet50_stem.py     # Stem component test
│   │   ├── test_resnet50_bottleneck.py # Bottleneck component test
│   │   ├── test_reg_head.py          # Regression head test
│   │   └── test_cls_head.py          # Classification head test
│   └── perf/                         # Performance tests
│       ├── test_perf.py              # Device performance test
│       └── test_perf_e2e.py          # End-to-end performance test
│
└── tt/                                # TTNN implementation
    ├── tt_retinanet.py               # Main TTNN RetinaNet model
    ├── tt_backbone.py                # TTNN ResNet50 backbone
    ├── tt_fpn.py                     # TTNN FPN implementation
    ├── tt_stem.py                    # TTNN ResNet stem
    ├── tt_bottleneck.py              # TTNN ResNet bottleneck
    ├── tt_reg_head.py                # TTNN Regression head
    ├── tt_cls_head.py                # TTNN Classification head
    ├── custom_preprocessor.py        # Custom weight preprocessing
    └── utils.py                      # TTNN utility functions
```

## Setup

### Download Weights (Optional)

The demo uses default pre-trained weights from Torchvision (`RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT`). These weights are automatically downloaded when you first run the model.

## Usage

### Running the Demo

The demo script demonstrates object detection on images:

```bash
# Basic usage with default settings (uses sample image)
python models/experimental/retinanet/demo/demo.py

# Specify custom output directory
python models/experimental/retinanet/demo/demo.py \
    --input path/to/image.jpg \
    --output path/to/output_dir

# Example with sample image
python models/experimental/retinanet/demo/demo.py \
    --input models/experimental/retinanet/resources/dog_800x800.jpg \
    --output models/experimental/retinanet/resources/outputs
```
### Demo Output Files

The demo generates output files for each processed image:
- `{result}.jpg`: TTNN detection results with bounding boxes and labels

**Default output directory**: `models/experimental/retinanet/resources/outputs/`

Expected output:
```
Demo completed. Output dir: models/experimental/retinanet/resources/outputs
```

Predicted classification labels overlaid on images will be saved in the output directory.

### Running Tests

Run the test suite to verify model correctness:

```bash
# Run all PCC tests
pytest models/experimental/retinanet/tests/pcc/ -v

# Run main integration test
pytest models/experimental/retinanet/tests/pcc/test_retinanet.py -v

# Run specific component tests
pytest models/experimental/retinanet/tests/pcc/test_resnet50_fpn.py -v
pytest models/experimental/retinanet/tests/pcc/test_resnet50_backbone.py -v
pytest models/experimental/retinanet/tests/pcc/test_resnet50_bottleneck.py -v
pytest models/experimental/retinanet/tests/pcc/test_resnet50_stem.py -v
pytest models/experimental/retinanet/tests/pcc/test_reg_head.py -v
pytest models/experimental/retinanet/tests/pcc/test_cls_head.py -v
```

### Performance Testing

```bash
# Test device performance
## Fps: 39.4
pytest models/experimental/retinanet/tests/perf/test_perf.py -v

# Test end-to-end performance
## Fps: 0.21 (with trace + 2CQ)
export FALLBACK_ON_GROUPNORM=0
pytest models/experimental/retinanet/tests/perf/test_perf_e2e.py -v
```

**Note**: GroupNorm in the head is set to fall back to the PyTorch implementation because the TTNN version resulted in lower PCC — around 0.96 for the regression head and 0.91 for the classification head. With the fallback enabled, the PCC improves to approximately 0.99. To disable this behavior, set:
```bash
export FALLBACK_ON_GROUPNORM=0
```

## Configuration Notes

- **Resolution**: (H, W) = (512, 512) is supported end-to-end
- **Device**: The demo opens a Wormhole device (default id typically 0). If you need to change it, adjust the device open call in the demo
- **Batch Size**: Demo/tests are written for BS=1. For larger BS you'll need to verify memory layouts and tile alignment
- **GroupNorm Fallback**: By default, GroupNorm operations fall back to PyTorch for better accuracy. Set `FALLBACK_ON_GROUPNORM=0` to use TTNN implementation
