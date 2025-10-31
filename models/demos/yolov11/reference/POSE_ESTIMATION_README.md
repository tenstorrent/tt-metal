# YOLO11 Pose Estimation Implementation

## Overview

This implementation modifies the YOLO11 model to perform **pose estimation** instead of object detection. The model now predicts human body keypoints (17 COCO format keypoints) along with bounding boxes around detected persons.

Based on the [Ultralytics YOLO11 documentation](https://docs.ultralytics.com/models/yolo11/).

## Changes Made

### 1. **New `Pose` Class** (replaces `Detect`)

The `Pose` class is a custom detection head that outputs:
- **Bounding boxes** (4 coordinates)
- **Person confidence** (1 value)
- **Keypoints** (17 keypoints × 3 values = 51 values for x, y, visibility)

#### Architecture:
- **cv2**: Bounding box regression head (same as object detection)
- **cv3**: Keypoint prediction head (predicts 51 channels: 17 keypoints × 3)
- **cv4**: Person confidence head (single class - person detection)

### 2. **Model Class Renamed**

- Old: `YoloV11` (for object detection)
- New: `YoloV11Pose` (for pose estimation)
- Backward compatibility: `YoloV11 = YoloV11Pose` alias added

### 3. **Output Format**

**Shape**: `[batch_size, 56, num_anchors]`

| Channels | Description |
|----------|-------------|
| 0-3      | Bounding box coordinates (x, y, w, h) |
| 4        | Person confidence score |
| 5-55     | 17 keypoints × 3 (x, y, visibility) |

### 4. **COCO Keypoint Format** (17 keypoints)

```
0:  nose
1:  left_eye
2:  right_eye
3:  left_ear
4:  right_ear
5:  left_shoulder
6:  right_shoulder
7:  left_elbow
8:  right_elbow
9:  left_wrist
10: right_wrist
11: left_hip
12: right_hip
13: left_knee
14: right_knee
15: left_ankle
16: right_ankle
```

## Usage

### Basic Usage

```python
import torch
from yolov11 import YoloV11Pose

# Initialize model
model = YoloV11Pose()
model.eval()

# Create input image (batch_size, channels, height, width)
input_image = torch.randn(1, 3, 640, 640)

# Run inference
with torch.no_grad():
    output = model(input_image)

# Output shape: [1, 56, num_anchors]
print(f"Output shape: {output.shape}")
```

### Extracting Predictions

```python
# For each detection (anchor):
for i in range(output.shape[2]):
    # Extract bounding box
    bbox = output[0, 0:4, i]  # x, y, w, h

    # Extract confidence
    conf = output[0, 4, i]

    # Extract keypoints
    keypoints = output[0, 5:56, i].reshape(17, 3)  # 17 keypoints, each with (x, y, visibility)

    # Process detection if confidence is high enough
    if conf > 0.5:
        print(f"Person detected at {bbox} with confidence {conf}")
        print(f"Nose position (keypoint 0): {keypoints[0]}")
```

## Example Script

Run the included example script to test the model:

```bash
cd /home/ubuntu/MAIN/tt-metal/models/demos/yolov11/reference
python3 example_pose_usage.py
```

**Note**: Requires PyTorch to be installed.

## Key Differences from Object Detection

| Feature | Object Detection | Pose Estimation |
|---------|-----------------|-----------------|
| Output head | `Detect` | `Pose` |
| Number of classes | 80 (COCO dataset) | 1 (person only) |
| Output channels | 84 (4 bbox + 80 classes) | 56 (4 bbox + 1 conf + 51 keypoints) |
| Task | Detect objects | Detect persons + keypoints |
| Model file | `yolo11n.pt` | `yolo11n-pose.pt` |

## Architecture Overview

The model uses the same **backbone** and **neck** architecture as YOLO11 object detection:

1. **Backbone**: Feature extraction with Conv, C3k2, SPPF, and C2PSA blocks
2. **Neck**: Feature pyramid with upsampling and concatenation
3. **Head**: Custom `Pose` head (instead of `Detect` head)

The backbone and neck remain unchanged from the original YOLO11 implementation, only the detection head is modified for pose estimation.

## References

- [Ultralytics YOLO11 Documentation](https://docs.ultralytics.com/models/yolo11/)
- [YOLO11 Pose Models](https://docs.ultralytics.com/tasks/pose/)
- [COCO Keypoint Detection](https://cocodataset.org/#keypoints-2020)

## Files Modified

- `yolov11.py` - Main implementation file
  - Added `Pose` class for pose estimation head
  - Renamed `YoloV11` to `YoloV11Pose`
  - Added detailed documentation
  - Added backward compatibility alias

## Files Added

- `example_pose_usage.py` - Example script demonstrating model usage
- `POSE_ESTIMATION_README.md` - This documentation file
