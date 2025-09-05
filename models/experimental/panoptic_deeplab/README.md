# Panoptic DeepLab - TTNN Implementation

TTNN implementation of Panoptic-DeepLab model for panoptic segmentation.

## Setup

### 1. Download Weights

Download the ResNet-52 weights file: [https://dl.fbaipublicfiles.com/detectron2/DeepLab/R-52.pkl]

Place the downloaded `R-52.pkl` file in `models/experimental/panoptic_deeplab/weights/`

### 2. Running the Model

```bash
# From tt-metal root directory
pytest models/experimental/panoptic_deeplab/tests/pcc/test_tt_model.py::test_panoptic_deeplab
```

This runs PCC comparison between PyTorch and TTNN implementations on 512x1024 input.
