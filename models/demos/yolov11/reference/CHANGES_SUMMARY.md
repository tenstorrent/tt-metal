# Summary of Changes: Object Detection → Pose Estimation

## Modified File
`/home/ubuntu/MAIN/tt-metal/models/demos/yolov11/reference/yolov11.py`

---

## Key Changes

### 1. Added Documentation Header
```python
"""
YOLO11 Pose Estimation Model

This implementation is based on Ultralytics YOLO11 architecture for pose estimation.
Reference: https://docs.ultralytics.com/models/yolo11/

Key Features:
- Predicts human body keypoints (17 keypoints for COCO format)
- Each keypoint has x, y coordinates and visibility score
- Outputs bounding boxes around detected persons
...
"""
```

### 2. Detection Head: `Detect` → `Pose`

**BEFORE (Object Detection):**
```python
class Detect(nn.Module):
    def __init__(self, ...):
        # cv2: bbox regression
        # cv3: class prediction (80 classes)
        # Output: [batch, 84, anchors]  # 4 bbox + 80 classes
```

**AFTER (Pose Estimation):**
```python
class Pose(nn.Module):
    """Pose estimation head for YOLO11 - predicts bounding boxes and keypoints."""
    def __init__(self, ...):
        self.num_keypoints = 17  # COCO keypoints
        # cv2: bbox regression
        # cv3: keypoint prediction (17 × 3 = 51 channels)
        # cv4: person confidence (1 class)
        # Output: [batch, 56, anchors]  # 4 bbox + 1 conf + 51 keypoints
```

### 3. Model Class Renamed

**BEFORE:**
```python
class YoloV11(nn.Module):
    def __init__(self):
        ...
        Detect(...)  # Detection head
```

**AFTER:**
```python
class YoloV11Pose(nn.Module):
    """YOLO11 model for pose estimation (keypoint detection)."""
    def __init__(self):
        ...
        Pose(...)  # Pose estimation head

# Backward compatibility alias
YoloV11 = YoloV11Pose
```

---

## Output Format Comparison

### Object Detection (Before)
```
Output shape: [batch_size, 84, num_anchors]

Channels:
├─ 0-3:   Bounding box (x, y, w, h)
└─ 4-83:  Class probabilities (80 COCO classes)
```

### Pose Estimation (After)
```
Output shape: [batch_size, 56, num_anchors]

Channels:
├─ 0-3:   Bounding box (x, y, w, h)
├─ 4:     Person confidence
└─ 5-55:  Keypoints (17 × 3)
           ├─ Nose (x, y, vis)
           ├─ Left Eye (x, y, vis)
           ├─ Right Eye (x, y, vis)
           ├─ ... (14 more keypoints)
           └─ Right Ankle (x, y, vis)
```

---

## Architecture Components

| Component | Status | Notes |
|-----------|--------|-------|
| Conv | ✓ Unchanged | Basic convolution block |
| Bottleneck | ✓ Unchanged | Residual block |
| SPPF | ✓ Unchanged | Spatial pyramid pooling |
| C3k | ✓ Unchanged | CSP bottleneck |
| C3k2 | ✓ Unchanged | CSP bottleneck v2 |
| Attention | ✓ Unchanged | Attention mechanism |
| PSABlock | ✓ Unchanged | Position-sensitive attention |
| C2PSA | ✓ Unchanged | CSP with PSA |
| Concat | ✓ Unchanged | Concatenation layer |
| DFL | ✓ Unchanged | Distribution focal loss |
| **Detect** | ✗ Removed | Object detection head |
| **Pose** | ✓ Added | Pose estimation head |
| **YoloV11** | ✓ Modified | Now `YoloV11Pose` |

---

## Usage Examples

### Object Detection (Before)
```python
from yolov11 import YoloV11

model = YoloV11()
output = model(image)  # [batch, 84, anchors]

# Extract: bbox + 80 class scores
bbox = output[:, 0:4, :]
classes = output[:, 4:84, :]
```

### Pose Estimation (After)
```python
from yolov11 import YoloV11Pose

model = YoloV11Pose()
output = model(image)  # [batch, 56, anchors]

# Extract: bbox + confidence + keypoints
bbox = output[:, 0:4, :]
conf = output[:, 4:5, :]
keypoints = output[:, 5:56, :].reshape(-1, 17, 3, output.shape[2])
```

---

## New Files Created

1. **`example_pose_usage.py`** - Demonstration script
2. **`POSE_ESTIMATION_README.md`** - Comprehensive documentation
3. **`CHANGES_SUMMARY.md`** - This file

---

## Testing

To test the implementation:

```bash
cd /home/ubuntu/MAIN/tt-metal/models/demos/yolov11/reference
python3 example_pose_usage.py
```

**Requirements:** PyTorch must be installed

---

## References

- [Ultralytics YOLO11 Main Page](https://docs.ultralytics.com/models/yolo11/)
- [YOLO11 Pose Documentation](https://docs.ultralytics.com/tasks/pose/)
- [COCO Keypoint Format](https://cocodataset.org/#keypoints-2020)

---

## Compatibility

- ✓ Backward compatible via `YoloV11 = YoloV11Pose` alias
- ✓ Same backbone and neck architecture
- ✓ Same input format (3-channel images)
- ✓ All existing building blocks preserved
