# TransFuser

TransFuser is a multi-modal transformer-based architecture for autonomous driving that fuses image and LiDAR features for perception and planning tasks. This implementation provides both PyTorch reference models and optimized Tenstorrent (TTNN) implementations.

## Overview

TransFuser combines:
- **Image Encoder**: Processes RGB camera images using architectures like ResNet or RegNet
- **LiDAR Encoder**: Processes LiDAR BEV (Bird's Eye View) representations
- **Transformer Fusion**: Multi-scale transformer blocks that fuse image and LiDAR features at different resolutions
- **Detection Head**: LidarCenterNet head for object detection in BEV space
- **Waypoint Prediction**: GRU-based waypoint prediction for planning

The model is designed for autonomous driving applications, particularly for CARLA simulation environments.

## Architecture

### TransFuserBackbone

The backbone consists of:

1. **Dual Encoders**: Separate image and LiDAR encoders that extract features at multiple scales
2. **Multi-Scale Transformer Fusion**: Four transformer blocks (GPT-based) that fuse features at different stages:
   - `transformer1`: Fuses features after layer1
   - `transformer2`: Fuses features after layer2
   - `transformer3`: Fuses features after layer3
   - `transformer4`: Fuses features after layer4
3. **FPN (Feature Pyramid Network)**: Top-down pathway for multi-scale feature extraction
4. **Feature Fusion**: Global average pooling and addition of image and LiDAR features

### Bottleneck Blocks

The RegNet bottleneck blocks include Squeeze-and-Excitation (SE) modules with `se_fc1` and `se_fc2` convolutions. These can run in either TTNN or PyTorch mode:

- **TTNN mode** (default): `se_fc1` and `se_fc2` run on Tenstorrent hardware
- **PyTorch fallback mode**: Set `use_fallback=True` to run SE modules in PyTorch on CPU

The `use_fallback` flag in the bottleneck block constructor allows switching between implementations for debugging or performance optimization.

### LidarCenterNet

Complete model that includes:
- TransFuserBackbone for feature extraction
- LidarCenterNetHead for object detection (heatmap, bounding boxes, velocity, brake prediction)
- Waypoint prediction GRU for trajectory planning

**Note**: Several components in LidarCenterNet run in PyTorch (CPU):
- `pred_bev`: BEV prediction head (torch.nn.Sequential)
- `join`: Feature projection layer (torch.nn.Sequential)
- `decoder`: GRU decoder for waypoint prediction (torch.nn.GRUCell)
- `output`: Final waypoint output layer (torch.nn.Linear)
- `forward_gru()`: Waypoint prediction method

## Directory Structure

```
transfuser/
├── reference/          # PyTorch reference implementations
│   ├── transfuser_backbone.py    # Main backbone model
│   ├── lidar_center_net.py       # Complete model with detection head
│   ├── config.py                  # Global configuration
│   ├── gpt.py                     # GPT transformer blocks
│   ├── self_attention.py          # Self-attention mechanism
│   ├── bottleneck.py              # Bottleneck blocks
│   ├── stage.py                   # ResNet/RegNet stages
│   ├── topdown.py                 # FPN top-down pathway
│   └── ...
├── tt/                  # Tenstorrent (TTNN) optimized implementations
│   ├── transfuser_backbone.py    # TTNN backbone
│   ├── lidar_center_net.py        # TTNN complete model
│   ├── custom_preprocessing.py    # Model parameter preprocessing
│   ├── gpt.py                     # TTNN transformer blocks
│   ├── self_attn.py               # TTNN self-attention
│   └── ...
├── tests/               # Test suite
│   ├── test_transfuser_backbone.py    # Backbone tests
│   ├── test_lidar_center_net.py        # Complete model tests
│   ├── test_gpt.py                     # Transformer tests
│   ├── test_self_attention.py          # Self-attention tests
│   ├── test_bottleneck.py              # Bottleneck tests
│   ├── test_stages.py                  # Stage tests
│   ├── test_head.py                    # Detection head tests
│   └── test_topdown.py                 # FPN tests
└── demo/                # Demo scripts
    └── lidar_center_net_demo.py
```

## Key Components

### 1. TransFuserBackbone

Multi-scale fusion transformer that processes image and LiDAR inputs:

```python
from models.experimental.transfuser.reference.transfuser_backbone import TransfuserBackbone
from models.experimental.transfuser.reference.config import GlobalConfig

config = GlobalConfig(setting="eval")
config.n_layer = 4  # Number of transformer layers

model = TransfuserBackbone(
    config,
    image_architecture="regnety_032",
    lidar_architecture="regnety_032",
    use_velocity=False
)

features, image_grid, fused_features = model(image, lidar, velocity)
```

### 2. LidarCenterNet

Complete model with detection and planning:

```python
from models.experimental.transfuser.reference.lidar_center_net import LidarCenterNet

model = LidarCenterNet(
    config,
    backbone="transFuser",
    image_architecture="regnety_032",
    lidar_architecture="regnety_032",
    use_velocity=False
)

fused_features, feature, pred_wp, head_results, boxes, rotated_bboxes = model.forward_ego(
    image, lidar_bev, target_point, velocity
)
```

### 3. Configuration

The `GlobalConfig` class contains all model hyperparameters:

- **Image/LiDAR**: Resolution, anchors, sequence lengths
- **Transformer**: Number of layers, embedding dimensions
- **Detection**: Confidence thresholds, bounding box parameters
- **Planning**: GRU hidden size, waypoint prediction length

See `reference/config.py` for all available options.

## Setup

### Downloading Checkpoints

Download the pre-trained TransFuser model checkpoint from the official repository:

```bash
# Download checkpoint (example: model_seed1_39.pth)
# Option 1: Direct download from TransFuser repository
wget https://github.com/autonomousvision/transfuser/releases/download/v1.0/model_seed1_39.pth

# Option 2: Clone the repository and use checkpoints from there
git clone https://github.com/autonomousvision/transfuser.git
# Checkpoints are typically in the transfuser/models_2022/transfuser/ directory
```

**Checkpoint location:**
- Place the checkpoint file (e.g., `model_seed1_39.pth`) in your working directory or specify the full path when running tests/demo.

**Alternative checkpoint paths:**
- Some checkpoints may be in `model_ckpt/models_2022/transfuser/` directory structure
- Adjust the checkpoint path in your code accordingly

### Downloading Data

For running the demo and tests, you need CARLA simulation data. Download scenario data from the TransFuser repository:

```bash
# Option 1: Download preprocessed scenario data
# Scenario data typically contains:
# - images/ (RGB camera images)
# - lidar/ (LiDAR BEV representations)
# - metadata files

# Example: Download scenario data from TransFuser releases
# Check the TransFuser repository for data download links
```

**Data structure:**
```
Scenario3_Town01_curved_route0_11_23_20_02_59/
├── images/
│   └── <frame_id>.png  (e.g., 0120.png)
├── lidar/
│   └── <frame_id>.npy  (e.g., 0120.npy)
└── metadata files
```

**For testing:**
- Tests use pre-loaded inputs: `transfuser_inputs_final.pt` (for backbone tests)
- Demo requires actual scenario folders with images and lidar data

**Note:** If you have CARLA simulation data from running the original TransFuser, you can use that directly. The `process_input()` function in `reference/lidar_center_net.py` handles loading and preprocessing the data.

## Testing

Run the test suite to verify model correctness:

```bash
# Test backbone
pytest models/experimental/transfuser/tests/test_transfuser_backbone.py

# Test complete model
pytest models/experimental/transfuser/tests/test_lidar_center_net.py

# Test individual components
pytest models/experimental/transfuser/tests/test_gpt.py
pytest models/experimental/transfuser/tests/test_self_attention.py
pytest models/experimental/transfuser/tests/test_bottleneck.py
pytest models/experimental/transfuser/tests/test_stages.py
pytest models/experimental/transfuser/tests/test_head.py
pytest models/experimental/transfuser/tests/test_topdown.py
```

### Test Coverage

- **test_transfuser_backbone.py**: Tests backbone feature extraction and fusion
- **test_lidar_center_net.py**: Tests complete model with detection and waypoint prediction
- **test_gpt.py**: Tests transformer blocks
- **test_self_attention.py**: Tests self-attention mechanism (both optimized and non-optimized)
- **test_bottleneck.py**: Tests bottleneck blocks
- **test_stages.py**: Tests ResNet/RegNet stages
- **test_head.py**: Tests LidarCenterNet detection head
- **test_topdown.py**: Tests FPN top-down pathway

All tests use PCC (Pearson Correlation Coefficient) validation to compare TTNN outputs with PyTorch reference outputs.

## Demo

Run the LidarCenterNet demo to compare TTNN and PyTorch implementations:

```bash
python models/experimental/transfuser/demo/lidar_center_net_demo.py \
    --data-root <path_to_scenario_folder> \
    --frame <frame_id> \
    --weights <path_to_model.pth> \
    --device-id 0
```

**Required arguments:**
- `--data-root`: Path to folder containing scenario data (images/lidar)
- `--frame`: Frame ID inside data_root (e.g., "0120")
- `--weights`: Path to Transfuser weight file (.pth)
- `--device-id`: TTNN device ID (default: 0)


**Example:**
```bash
python models/experimental/transfuser/demo/lidar_center_net_demo.py \
    --data-root Scenario3_Town01_curved_route0_11_23_20_02_59/ \
    --frame 0120 \
    --weights model_seed1_39.pth \
    --device-id 0
```

The demo will:
1. Load inputs from the specified data root and frame
2. Run both PyTorch reference and TTNN implementations
3. Compare outputs using PCC validation
4. Print detailed results for all components (features, waypoints, detection heads, bounding boxes)

## Input Format

### Image Input
- Shape: `(batch_size, 3, 160, 704)` (H, W)
- Format: RGB, normalized to ImageNet statistics
- Channels: RGB (3 channels)

### LiDAR Input
- Shape: `(batch_size, 3, 256, 256)` (H, W)
- Format: BEV (Bird's Eye View) representation
- Channels: 3 (can include target point image channel if `use_target_point_image=True`)

### Velocity Input
- Shape: `(batch_size, 1)`
- Format: Ego vehicle velocity scalar

### Target Point (for waypoint prediction)
- Shape: `(batch_size, 2)`
- Format: `[x, y]` coordinates in metric space

## Output Format

### Backbone Outputs
1. **Features**: Tuple of FPN features `(p2, p3, p4, p5)` at different scales
2. **Image Grid**: Features at grid resolution for auxiliary tasks
3. **Fused Features**: Global fused features for waypoint prediction

### LidarCenterNet Outputs
1. **Fused Features**: Global fused features
2. **Feature**: Single-scale feature map for detection
3. **Predicted Waypoints**: Future trajectory waypoints
4. **Head Results**: Detection outputs (heatmap, bounding boxes, velocity, brake)
5. **Boxes**: Bounding boxes in grid coordinates
6. **Rotated BBoxes**: Bounding boxes in metric coordinates


## License

SPDX-License-Identifier: Apache-2.0

Copyright © 2025 Tenstorrent Inc.
