# MapTR - Map TRansformer for TTNN

MapTR is a Bird's-Eye-View (BEV) map detection model implemented for Tenstorrent Neural Network (TTNN) framework. This implementation supports inference-only mode and is optimized for running on Tenstorrent hardware.

## Overview

MapTR is a transformer-based model for detecting map elements (lane dividers, pedestrian crossings, and boundaries) from multi-view camera images. This implementation uses:

- **Backbone**: ResNet50
- **Encoder**: BEVFormer encoder
- **Framework**: TTNN (Tenstorrent Neural Network) for hardware acceleration
- **Dataset**: NuScenes

## Features

- ✅ **Inference-only**: All training code has been removed for a clean inference codebase
- ✅ **Automatic checkpoint download**: Checkpoints are automatically downloaded when needed
- ✅ **Comprehensive testing**: PCC (Pearson Correlation Coefficient) tests for all components
- ✅ **Demo visualization**: Full demo script with visualization capabilities
- ✅ **Modular architecture**: Clean separation of PyTorch reference and TTNN implementations

## Project Structure

```
MapTR/
├── chkpt/                          # Checkpoint directory
│   └── downloaded_weights.pth      # Auto-downloaded model weights
├── demo/                           # Demo scripts
│   ├── demo.py                     # Main inference demo
│   └── processing.py               # Data generation utilities
├── reference/                      # Reference implementations and configs
│   ├── config_maptr_tiny_r50_24e_bevformer.py  # Main config file
│   ├── maptr.py                    # MapTR detector implementation
│   ├── dependency.py               # Shared dependencies and utilities
│   ├── datasets_nuscenes.py        # NuScenes dataset implementation
│   ├── datasets_nuscenes_map.py    # NuScenes map dataset implementation
│   ├── pipelines.py                # Data processing pipelines
│   └── ...                         # Other reference modules
├── projects/                       # MMDetection3D plugin modules (legacy)
│   └── mmdet3d_plugin/            # MMDetection3D plugin modules
│       ├── bevformer/              # BEVFormer encoder modules
│       ├── maptr/                  # MapTR detector and head modules
│       └── datasets/               # Dataset loaders
├── resources/                      # Resource utilities
│   ├── download_chkpoint.py        # Checkpoint download utility
│   └── nuScenes/                   # Sample data files
├── tests/                          # Test suite
│   ├── pcc/                        # PCC tests for numerical validation
│   │   ├── test_backbone.py        # ResNet50 backbone test
│   │   ├── test_encoder.py         # BEVFormer encoder test
│   │   ├── test_decoder.py         # MapTR decoder test
│   │   ├── test_head.py            # MapTR head test
│   │   ├── test_fpn.py             # FPN test
│   │   ├── test_mha.py             # Multi-head attention test
│   │   ├── test_transformer.py     # Transformer test
│   │   └── test_maptr.py           # Full model end-to-end test
│   └── perf/                       # Performance tests
└── tt/                             # TTNN implementations
    ├── ttnn_backbone.py            # ResNet50 TTNN implementation
    ├── ttnn_encoder.py             # BEVFormer encoder TTNN implementation
    ├── ttnn_decoder.py             # MapTR decoder TTNN implementation
    ├── ttnn_head.py                # MapTR head TTNN implementation
    ├── ttnn_transformer.py         # Transformer TTNN implementation
    ├── ttnn_maptr.py               # Full MapTR TTNN model
    └── model_preprocessing.py       # Model preprocessing utilities
```

## Installation

### Prerequisites

- Python 3.8+
- PyTorch
- TTNN framework
- Required Python packages:
  - `gdown` (for checkpoint download)
  - `loguru` (for logging)
  - `mmcv` (MMDetection3D dependencies)
  - `numpy`, `opencv-python`, `matplotlib` (for visualization)

### Setup

1. Ensure you're in the TTNN environment:
```bash
cd /home/ubuntu/christyv1/tt-metal
```

2. The checkpoint will be automatically downloaded when you run the demo or tests. If you want to download it manually:

```bash
python models/experimental/MapTR/resources/download_chkpoint.py
```

## Quick Start

### Running the Demo

The demo script automatically downloads the checkpoint if it's missing:

```bash
# Using default checkpoint (auto-downloads if missing)
python_env/bin/python \
    models/experimental/MapTR/demo/demo.py \
    models/experimental/MapTR/reference/config_maptr_tiny_r50_24e_bevformer.py

# Using custom checkpoint
python_env/bin/python \
    models/experimental/MapTR/demo/demo.py \
    models/experimental/MapTR/reference/config_maptr_tiny_r50_24e_bevformer.py \
    path/to/your/checkpoint.pth

# With custom options
python_env/bin/python \
    models/experimental/MapTR/demo/demo.py \
    models/experimental/MapTR/reference/config_maptr_tiny_r50_24e_bevformer.py \
    --score-thresh 0.5 \
    --show-dir ./output \
    --device-params '{"l1_small_size": 32768}'
```

### Demo Options

- `config`: Path to the configuration file (required)
- `checkpoint`: Path to checkpoint file (optional, defaults to auto-downloaded weights)
- `--score-thresh`: Score threshold for predictions (default: 0.4)
- `--show-dir`: Directory to save visualizations (default: `./work_dirs/...`)
- `--show-cam`: Show camera images in visualization
- `--gt-format`: Ground truth visualization format (default: `fixed_num_pts`)
- `--device-params`: TTNN device parameters as JSON string

### Running Tests

All tests automatically download the checkpoint if needed:

```bash
# Run all PCC tests
pytest models/experimental/MapTR/tests/pcc/

# Run specific test
pytest models/experimental/MapTR/tests/pcc/test_tt_maptr.py

# Run with verbose output
pytest models/experimental/MapTR/tests/pcc/ -v

# Run performance tests
pytest models/experimental/MapTR/tests/perf/
```

### Test Files

- `test_backbone.py`: Tests ResNet50 backbone implementation
- `test_encoder.py`: Tests BEVFormer encoder implementation
- `test_deocder.py`: Tests MapTR decoder implementation
- `test_head.py`: Tests MapTR head implementation
- `test_fpn.py`: Tests Feature Pyramid Network implementation
- `test_mha.py`: Tests Multi-head Attention implementation
- `test_spatial_cross_attention.py`: Tests Spatial Cross Attention
- `test_temporal_self_attention.py`: Tests Temporal Self Attention
- `test_transformer.py`: Tests MapTRPerceptionTransformer
- `test_maptr.py`: End-to-end full model test

## Configuration

The main configuration file is located at:
```
models/experimental/MapTR/projects/configs/maptr/maptr_tiny_r50_24e_bevformer.py
```

Key configuration parameters:

- `bev_h_`, `bev_w_`: BEV feature map dimensions (default: 200x100)
- `point_cloud_range`: 3D point cloud range for detection
- `num_vec`: Number of map vectors to predict
- `num_pts_per_vec`: Number of points per vector
- `num_classes`: Number of map classes (divider, ped_crossing, boundary)
- `embed_dims`: Embedding dimensions

## Model Architecture

### Components

1. **Backbone (ResNet50)**: Extracts features from multi-view camera images
2. **FPN**: Feature Pyramid Network for multi-scale feature extraction
3. **BEVFormer Encoder**: Transforms image features to BEV representation
4. **MapTR Decoder**: Decodes BEV features to map elements
5. **MapTR Head**: Final prediction head for map element detection

### Input/Output

- **Input**: Multi-view camera images (6 cameras: front, front-left, front-right, back, back-left, back-right)
- **Output**: Map elements (lane dividers, pedestrian crossings, boundaries) in BEV space

## Checkpoint Management

The checkpoint is automatically managed:

- **Location**: `models/experimental/MapTR/chkpt/downloaded_weights.pth`
- **Auto-download**: Checkpoints are automatically downloaded when running demo or tests
- **Manual download**: Use `python models/experimental/MapTR/resources/download_chkpoint.py`

The checkpoint download utility:
- Checks if checkpoint exists before downloading
- Installs `gdown` automatically if needed
- Downloads from Google Drive
- Creates necessary directories

## Development

### Adding New Components

When adding new TTNN components:

1. Implement the TTNN version in `tt/` directory
2. Create a corresponding PCC test in `tests/pcc/`
3. Ensure the test uses `ensure_checkpoint_downloaded()` for checkpoint loading

### Code Style

- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Add docstrings for public functions
- Keep inference-only: no training code

## Troubleshooting

### Checkpoint Download Issues

If checkpoint download fails:

1. Check internet connection
2. Verify `gdown` is installed: `pip install gdown`
3. Manually download using the script: `python models/experimental/MapTR/resources/download_chkpoint.py`

### Test Failures

If PCC tests fail:

1. Check that checkpoint is downloaded correctly
2. Verify TTNN device is properly initialized
3. Check device parameters match hardware capabilities
4. Review test logs for specific component failures

### Demo Issues

If demo fails:

1. Verify configuration file path is correct (should be `models/experimental/MapTR/reference/config_maptr_tiny_r50_24e_bevformer.py`)
2. Check that dataset paths in config are valid
3. Ensure TTNN device parameters are appropriate for your hardware
4. Check image input format matches expected dimensions
5. Verify you're using `python_env/bin/python` or have the correct Python environment activated

## License

SPDX-License-Identifier: Apache-2.0

Copyright © 2026 Tenstorrent AI ULC

## References

- MapTR: [Original MapTR Paper](https://arxiv.org/abs/2208.14437)
- BEVFormer: [BEVFormer Paper](https://arxiv.org/abs/2203.17270)
- MMDetection3D: [MMDetection3D Framework](https://github.com/open-mmlab/mmdetection3d)

## Support

For issues and questions, please refer to the main TTNN documentation or contact the Tenstorrent support team.
