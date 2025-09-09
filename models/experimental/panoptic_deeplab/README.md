# Panoptic DeepLab - TTNN Implementation

TTNN implementation of Panoptic-DeepLab model for panoptic segmentation.

## Setup

### 1. Download Weights

Download the Panoptic-DeepLab CityScapes weights file: [https://dl.fbaipublicfiles.com/detectron2/PanopticDeepLab/Cityscapes-PanopticSegmentation/panoptic_deeplab_R_52_os16_mg124_poly_90k_bs32/model_final_bd324a.pkl]

Place the downloaded `model_final_bd324a.pkl` file in `models/experimental/panoptic_deeplab/weights/`

### 2. Running the Model

```bash
# From tt-metal root directory
pytest models/experimental/panoptic_deeplab/tests/pcc/test_tt_model.py::test_panoptic_deeplab
```

This runs PCC comparison between PyTorch and TTNN implementations on 512x1024 input.
