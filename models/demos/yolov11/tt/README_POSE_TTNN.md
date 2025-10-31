# YOLO11 Pose Estimation - TTNN Implementation

## Overview

This directory contains the **TTNN (TT-Metal)** implementation of YOLO11 Pose Estimation, enabling hardware-accelerated inference on Tenstorrent devices.

## Files for Pose Estimation

### Core TTNN Implementation Files

| File | Purpose | Status |
|------|---------|--------|
| `ttnn_yolov11_pose.py` | Pose head (cv2, cv3, cv4) | ✅ **New for Pose** |
| `ttnn_yolov11_dwconv.py` | Depthwise convolution layer | ✅ **New for Pose** |
| `ttnn_yolov11_pose_model.py` | Complete YoloV11Pose model | ✅ **New for Pose** |
| `model_preprocessing_pose.py` | Parameter preprocessing | ✅ **New for Pose** |

### Shared Components (Used by Both Detection and Pose)

| File | Purpose | Used By |
|------|---------|---------|
| `common.py` | Shared utilities (TtnnConv, concat, etc.) | Both |
| `ttnn_yolov11_c3k2.py` | C3k2 block | Both |
| `ttnn_yolov11_c2psa.py` | C2PSA block | Both |
| `ttnn_yolov11_sppf.py` | SPPF block | Both |
| `ttnn_yolov11_attention.py` | Attention mechanism | Both |
| `ttnn_yolov11_bottleneck.py` | Bottleneck block | Both |
| `ttnn_yolov11_c3k.py` | C3k block | Both |
| `ttnn_yolov11_psa.py` | PSA block | Both |

### Object Detection Files (For Comparison)

| File | Purpose |
|------|---------|
| `ttnn_yolov11_detect.py` | Detection head (cv2, cv3) |
| `ttnn_yolov11.py` | Complete YoloV11 detection model |
| `model_preprocessing.py` | Detection preprocessing |

---

## Architecture Comparison

### Object Detection (ttnn_yolov11_detect.py)
```
TtnnDetect:
├─ cv2: Bbox regression (64 channels)
└─ cv3: Class prediction (80 channels)
   Output: [batch, 84, 8400]
```

### Pose Estimation (ttnn_yolov11_pose.py)
```
TtnnPoseHead:
├─ cv2: Bbox regression (64 channels)
├─ cv3: Person confidence (1 channel) - Uses DWConv!
└─ cv4: Keypoints (51 channels)
   Output: [batch, 56, 8400]
```

---

## New Components for Pose

### 1. DWConv (Depthwise Convolution)

```python
# ttnn_yolov11_dwconv.py
class TtnnDWConv:
    """
    Depthwise convolution where groups = in_channels
    More parameter-efficient than regular convolution
    Used in cv3 (confidence head)
    """
```

**Key feature:** Each input channel has its own filter (groups=in_channels)

### 2. PoseHead

```python
# ttnn_yolov11_pose.py
class TtnnPoseHead:
    """
    Pose estimation head with three prediction branches:
    - cv2: Bounding boxes (same as object detection)
    - cv3: Person confidence (uses DWConv)
    - cv4: Keypoints (17 × 3 = 51 values)
    """
```

**Output format:**
- Channels 0-3: Bounding box (x, y, w, h)
- Channel 4: Person confidence
- Channels 5-55: Keypoints (x, y, visibility for 17 keypoints)

---

## Usage Example

### Basic Usage

```python
import torch
import ttnn
from models.demos.yolov11.reference.yolov11_pose_correct import YoloV11Pose
from models.demos.yolov11.tt.ttnn_yolov11_pose_model import TtnnYoloV11Pose
from models.demos.yolov11.tt.model_preprocessing_pose import (
    create_yolov11_pose_model_parameters,
    create_yolov11_input_tensors
)

# Initialize device
device = ttnn.open_device(device_id=0)

# Load PyTorch model with pretrained weights
torch_model = YoloV11Pose()
torch_model.load_state_dict(
    torch.load('reference/yolov11_pose_pretrained_correct.pth')
)
torch_model.eval()

# Create input tensor
torch_input, ttnn_input = create_yolov11_input_tensors(
    device, batch=1, input_channels=3,
    input_height=640, input_width=640
)

# Preprocess model parameters (convert weights to TTNN format)
parameters = create_yolov11_pose_model_parameters(
    torch_model, torch_input, device
)

# Initialize TTNN model
ttnn_model = TtnnYoloV11Pose(device, parameters)

# Run inference
output = ttnn_model(ttnn_input)

# Convert output back to torch
output_torch = ttnn.to_torch(output)

# Output shape: [batch, 56, 8400]
# - 4: bbox
# - 1: confidence
# - 51: keypoints
```

---

## Model Pipeline

```
Input Image (640×640)
        ↓
┌──────────────────────────────┐
│   Backbone (Layers 0-10)     │
│   - Conv, C3k2, SPPF, C2PSA  │
│   (Shared with detection)    │
└──────────────────────────────┘
        ↓
┌──────────────────────────────┐
│    Neck (Layers 11-22)       │
│   - Upsample, Concat, C3k2   │
│   (Shared with detection)    │
└──────────────────────────────┘
        ↓
    ┌───┴───┬───────┐
    │       │       │
   x16     x19     x22  (3 scales)
    │       │       │
    └───┬───┴───┬───┘
        ↓       ↓
┌──────────────────────────────┐
│   Pose Head (Layer 23)       │
│                              │
│  cv2: Bbox → 64 channels     │
│  cv3: Conf → 1 channel       │
│  cv4: Kpts → 51 channels     │
│                              │
│  DFL decode + Concat         │
└──────────────────────────────┘
        ↓
Output: [batch, 56, 8400]
```

---

## Key Implementation Details

### 1. Bounding Box (cv2)
- **Same as object detection**
- 3 scales with Conv layers
- Outputs 64 channels for DFL
- Decoded to 4 bbox coordinates

### 2. Confidence (cv3)
- **Different from object detection!**
- Uses **DWConv** (depthwise convolution)
- 3 scales, each outputs 1 channel
- Sigmoid activation applied
- Predicts person confidence (not 80 classes)

### 3. Keypoints (cv4)
- **New for pose estimation!**
- 3 scales with Conv layers
- Each outputs 51 channels (17 keypoints × 3)
- Format: [x1, y1, v1, x2, y2, v2, ..., x17, y17, v17]
- x,y: pixel coordinates (decoded with anchor + stride)
- v: visibility score (0-1 after sigmoid)

### 4. Keypoint Decoding

The keypoint decoding formula (from reference model):
```python
# Only visibility gets sigmoid
kpt_v = sigmoid(kpt_v)

# X,Y coordinates: NO sigmoid, use raw values
kpt_x = (kpt_x * 2.0 - 0.5 + anchor_x) * stride
kpt_y = (kpt_y * 2.0 - 0.5 + anchor_y) * stride
```

**Note:** In the current TTNN implementation, keypoint coordinate decoding is simplified and may need to be done in postprocessing for optimal performance.

---

## Differences from Object Detection TTNN

| Component | Object Detection | Pose Estimation |
|-----------|------------------|-----------------|
| **Head class** | `TtnnDetect` | `TtnnPoseHead` |
| **Main model** | `TtnnYoloV11` | `TtnnYoloV11Pose` |
| **Preprocessing** | `model_preprocessing.py` | `model_preprocessing_pose.py` |
| **New components** | None | `TtnnDWConv` |
| **cv2 (bbox)** | ✓ Same | ✓ Same |
| **cv3** | 80 classes (Conv) | 1 confidence (DWConv) |
| **cv4** | N/A | 51 keypoints |
| **Output channels** | 84 | 56 |

---

## Prerequisites

### Required Files from Reference Directory

```bash
# PyTorch reference implementations
models/demos/yolov11/reference/yolov11.py                    # Shared components
models/demos/yolov11/reference/yolov11_pose_correct.py       # Pose model
models/demos/yolov11/reference/yolov11_pose_pretrained_correct.pth  # Pretrained weights
models/demos/yolov11/reference/load_weights_correct.py       # Weight loader
```

### Setup Steps

1. **Get pretrained weights:**
```bash
cd models/demos/yolov11/reference
pip install ultralytics
python3 load_weights_correct.py
```

2. **Test PyTorch model first:**
```bash
cd models/demos/yolov11/demo
python3 pose_demo.py
```

3. **Then use TTNN implementation** (once available)

---

## Testing

Test files should be created in:
```
models/demos/yolov11/tests/pcc/test_ttnn_yolov11_pose.py
models/demos/yolov11/tests/pcc/test_ttnn_yolov11_dwconv.py
```

Similar structure to existing tests like `test_ttnn_yolov11_detect.py`.

---

## Performance Considerations

### Memory Layout
- Input: ROW_MAJOR or TILE_LAYOUT
- Intermediate: TILE_LAYOUT for compute efficiency
- Output: ROW_MAJOR for postprocessing

### Sharding Strategy
- Height-sharded for most operations
- Sharded concat for efficiency
- Reshard operations when needed

### Optimizations
- Double buffering for activations and weights
- LoFi math fidelity for speed
- Math approximation mode enabled

---

## Output Format

```
Shape: [batch, 56, 8400]

Channel breakdown:
├─ 0-3:   Bounding box (x, y, w, h) - pixel coordinates
├─ 4:     Person confidence - sigmoid activated, range [0, 1]
└─ 5-55:  Keypoints (17 × 3 = 51 values)
          Each keypoint has:
          ├─ x: pixel coordinate (can be negative)
          ├─ y: pixel coordinate (can be negative)
          └─ v: visibility score [0, 1]

Number of anchors: 8400 = 80×80 + 40×40 + 20×20
                          (3 scales: stride 8, 16, 32)
```

---

## Files Summary for Commit

### Must Include
```
✅ ttnn_yolov11_dwconv.py         # DWConv implementation
✅ ttnn_yolov11_pose.py            # Pose head
✅ ttnn_yolov11_pose_model.py      # Complete model
✅ model_preprocessing_pose.py     # Preprocessing
✅ README_POSE_TTNN.md            # This file
```

### Shared Dependencies (Already exist)
```
✅ common.py
✅ ttnn_yolov11_c3k2.py
✅ ttnn_yolov11_c2psa.py
✅ ttnn_yolov11_sppf.py
✅ (and other shared components)
```

---

## Next Steps

1. ✅ Core TTNN files created
2. ⏳ Create test files (test_ttnn_yolov11_pose.py)
3. ⏳ Create runner/performant implementation
4. ⏳ Integration testing on TT hardware
5. ⏳ Performance benchmarking

---

## Related Documentation

- `../reference/OBJECT_DETECTION_VS_POSE_COMPARISON.md` - Architecture comparison
- `../reference/POSE_ESTIMATION_README.md` - PyTorch implementation details
- `../demo/pose_demo.py` - Usage example

---

## References

- [Ultralytics YOLO11 Pose](https://docs.ultralytics.com/models/yolo11/)
- [YOLO11 Pose Task](https://docs.ultralytics.com/tasks/pose/)
- TTNN Documentation: tt-metal/ttnn/README.md
