# Panoptic DeepLab

## Platforms:
    Made for BOS chips, tested on Blackhole with both 20-core (5x4 grid) and P150 all-core (13x10 grid - 130 cores) configurations.

## Introduction
Panoptic DeepLab is a unified model for panoptic segmentation that combines semantic segmentation and instance segmentation into a single framework. The model uses a shared ResNet backbone with separate heads for semantic segmentation and instance embedding prediction, enabling comprehensive scene understanding by simultaneously identifying both "stuff" (background regions like road, sky) and "things" (countable objects like cars, people).

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)

## Setup

### Download Weights

Download the Panoptic-DeepLab CityScapes weights file:
[https://dl.fbaipublicfiles.com/detectron2/PanopticDeepLab/Cityscapes-PanopticSegmentation/panoptic_deeplab_R_52_os16_mg124_poly_90k_bs32/model_final_bd324a.pkl]

Place the downloaded `model_final_bd324a.pkl` file in `models/experimental/panoptic_deeplab/weights/`

## How to Run

The model supports both optimized 20-core (5x4 grid) and all-core (13x10 grid) configurations:
- **20 cores (optimized)**: Set `TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE="4,3"`
- **130 cores**: Omit/Unset `TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE` for P150 device to use all cores (13x10 grid)


### Run the Full Model Test
```bash
# 20-core optimized configuration
TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE="4,3" pytest models/experimental/panoptic_deeplab/tests/pcc/test_tt_model.py

# 130-core grid configuration (13x10 on Blackhole P150)
pytest models/experimental/panoptic_deeplab/tests/pcc/test_tt_model.py
```
**Note**: All following tests can be run with or without TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE

### Run Component Tests
```bash
# Test ASPP component (works on any core configuration)
TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE="4,3" pytest models/experimental/panoptic_deeplab/tests/pcc/test_aspp.py

# Test ResNet backbone
TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE="4,3" pytest models/experimental/panoptic_deeplab/tests/pcc/test_resnet.py

# Test semantic segmentation head
TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE="4,3" pytest models/experimental/panoptic_deeplab/tests/pcc/test_semseg.py

# Test instance embedding head
TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE="4,3" pytest models/experimental/panoptic_deeplab/tests/pcc/test_insemb.py
```

### Run Device Performance Tests
```bash
# Test full model performance on 20 cores
TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE="4,3" pytest models/experimental/panoptic_deeplab/tests/test_device_perf_pdl.py::test_device_perf_pdl_20_cores
# Test full model performance on all cores
pytest models/experimental/panoptic_deeplab/tests/test_device_perf_pdl.py::test_device_perf_pdl_all_cores
```

### Run the Demo
```bash
# Single image with custom output directory
TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE="4,3" python models/experimental/panoptic_deeplab/tt/demo.py <image_path> <weights_path> <output_dir>

# Batch processing of directory
TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE="4,3" python models/experimental/panoptic_deeplab/tt/demo.py <input_dir> <weights_path> <output_dir> --batch
```

For help with demo options:
```bash
TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE="4,3" python models/experimental/panoptic_deeplab/tt/demo.py --help
```

### Demo Output Files

The demo generates several output files for each processed image in separate directories:

**TTNN Output** (`ttnn_output/`):
- `{image_name}_original.jpg`: Original input image
- `{image_name}_panoptic.jpg`: TTNN panoptic segmentation visualization (blended with original)

**PyTorch Reference** (`pytorch_output/`):
- `{image_name}_original.jpg`: Original input image
- `{image_name}_panoptic.jpg`: PyTorch reference panoptic segmentation visualization

### Image Requirements

- **Best Results**: Street scene images similar to Cityscapes dataset
- **Supported Formats**: jpg, jpeg, png, bmp, tiff
- **Auto-Resize**: All images are automatically resized to 512x1024 for inference
- **Classes**: Predicts 19 Cityscapes classes (road, car, person, etc.)

## Details

- The entry point to the TTNN Panoptic DeepLab model is `TtPanopticDeepLab` in `models/experimental/panoptic_deeplab/tt/tt_model.py`. The model uses weights from the Detectron2 implementation.

**Input Size: 512x1024**
- Input size is optimized for Cityscapes dataset format with height=512 and width=1024.

**Batch Size: 1**
- Current implementation uses batch size of 1 for optimal memory usage with L1 memory configuration.

**Memory Configuration**
- The model is currently configured to mostly run in DRAM memory, as we optimize it we will be mostly in L1.

### Model Components

1. **ResNet Backbone**: Provides hierarchical feature extraction with multiple resolution levels
2. **ASPP Module**: Atrous Spatial Pyramid Pooling for multi-scale feature aggregation
3. **Semantic Segmentation Head**: Predicts pixel-wise semantic classes (19 Cityscapes categories)
4. **Instance Embedding Head**: Generates center heatmaps and offset vectors for instance segmentation
5. **Panoptic Fusion**: Combines semantic and instance predictions into unified panoptic segmentation

### Outputs

The model produces:
- **Semantic Logits**: Per-pixel classification scores for semantic classes
- **Center Heatmaps**: Instance center point predictions
- **Offset Vectors**: Pixel-to-instance-center displacement vectors
- **Panoptic Visualization**: Combined semantic and instance segmentation results
