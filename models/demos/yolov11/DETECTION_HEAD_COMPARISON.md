# YOLO11: Object Detection Head vs Pose Detection Head

## Overview

This document compares the **Detect** head (object detection) with the **PoseHead** (pose estimation) in YOLO11.

---

## Architecture Comparison

### Object Detection Head: `Detect`

**File:** `models/demos/yolov11/reference/yolov11.py` (lines 444-739)

**Purpose:** Detect objects from 80 COCO classes

**Components:**
```python
class Detect(nn.Module):
    def __init__(self, ...):
        self.cv2 = nn.ModuleList([...])  # Bbox regression
        self.cv3 = nn.ModuleList([...])  # Class prediction (80 classes)
        self.dfl = DFL()                  # Distribution Focal Loss
```

**Heads:**
- **cv2**: Bounding box regression (3 scales)
- **cv3**: Class prediction (3 scales)
- **cv4**: ❌ Not present

---

### Pose Detection Head: `PoseHead`

**File:** `models/demos/yolov11/reference/yolov11_pose_correct.py` (lines 37-231)

**Purpose:** Detect people and their 17 body keypoints

**Components:**
```python
class PoseHead(nn.Module):
    def __init__(self):
        self.cv2 = nn.ModuleList([...])  # Bbox regression
        self.cv3 = nn.ModuleList([...])  # Person confidence (uses DWConv!)
        self.cv4 = nn.ModuleList([...])  # Keypoints (17 × 3)
        self.dfl = DFL()                  # Distribution Focal Loss
```

**Heads:**
- **cv2**: Bounding box regression (3 scales)
- **cv3**: Person confidence (3 scales) - **Uses DWConv**
- **cv4**: Keypoint prediction (3 scales) - **New for pose**

---

## Detailed Component Comparison

### cv2: Bounding Box Regression (IDENTICAL)

Both heads use the same architecture for bbox regression:

**Structure (3 scales):**
```python
# Scale 0: 64 channels
Conv(64, 64, k=3) → Conv(64, 64, k=3) → Conv2d(64, 64, k=1)

# Scale 1: 128 → 64
Conv(128, 64, k=3) → Conv(64, 64, k=3) → Conv2d(64, 64, k=1)

# Scale 2: 256 → 64
Conv(256, 64, k=3) → Conv(64, 64, k=3) → Conv2d(64, 64, k=1)
```

**Output:** 64 channels (4 coords × 16 bins for DFL)

**Decoding:**
```python
# Reshape to [batch, 4, 16, num_anchors]
# Apply softmax over 16 bins
# DFL conv → 4 coordinates (x, y, w, h)
# Decode with anchors and strides
```

✅ **Same in both Detection and Pose**

---

### cv3: Classes vs Confidence (DIFFERENT!)

#### Object Detection (cv3 = Classes)

**Structure:**
```python
# Regular Conv layers
nn.Sequential(
    nn.Sequential(
        Conv(64, 64, k=3),
        Conv(64, 80, k=1),  # → 80 classes
    ),
    nn.Sequential(
        Conv(80, 80, k=3),
        Conv(80, 80, k=1),
    ),
    Conv2d(80, 80, k=1)  # Final: 80 channels
)
```

**Output:** 80 channels (one per COCO class)

**Processing:**
```python
yb = sigmoid(yb)  # Class probabilities
```

#### Pose Estimation (cv3 = Confidence)

**Structure:**
```python
# Uses DWConv (Depthwise Convolution)!
nn.Sequential(
    nn.Sequential(
        DWConv(64, 64, k=3),  # ← Depthwise!
        Conv(64, 64, k=1),
    ),
    nn.Sequential(
        DWConv(64, 64, k=3),  # ← Depthwise!
        Conv(64, 64, k=1),
    ),
    Conv2d(64, 1, k=1)  # Final: 1 channel
)
```

**Output:** 1 channel (person confidence only)

**Processing:**
```python
yb = sigmoid(yb)  # Person confidence [0, 1]
```

**Key Difference:** Uses **DWConv** (groups=in_channels) for efficiency

---

### cv4: Keypoints (ONLY IN POSE)

#### Object Detection
❌ **No cv4** - Not needed for detection

#### Pose Estimation (cv4 = Keypoints)

**Structure:**
```python
# Regular Conv layers
nn.Sequential(
    Conv(64, 51, k=3),   # 64 → 51 channels
    Conv(51, 51, k=3),   # 51 → 51
    Conv2d(51, 51, k=1)  # Final: 51 channels
)

# 51 channels = 17 keypoints × 3 values
# Each keypoint: (x, y, visibility)
```

**Output:** 51 channels

**Processing:**
```python
# Reshape to [batch, 17, 3, num_anchors]
kpt_x, kpt_y, kpt_v = split keypoints

# Apply sigmoid ONLY to visibility
kpt_v = sigmoid(kpt_v)

# Decode x, y coordinates (NO sigmoid!)
kpt_x = (kpt_x * 2.0 - 0.5 + anchor_x) * stride
kpt_y = (kpt_y * 2.0 - 0.5 + anchor_y) * stride
```

---

## New Component: DWConv

### What is DWConv?

**Depthwise Convolution** - Used in cv3 (confidence head) of pose model

```python
class DWConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channel, out_channel, kernel,
            stride=stride, padding=padding,
            groups=in_channel,  # ← KEY: groups = in_channels!
            bias=False
        )
        self.bn = BatchNorm2d(out_channel)
        self.act = SiLU(inplace=True)
```

**Key Feature:** `groups=in_channel`
- Each input channel has its own filter
- More parameter-efficient than regular Conv
- Commonly used in mobile/efficient architectures

**Why use it?**
- Pose model only predicts 1 class (person)
- DWConv is more efficient for this simpler task
- Reduces parameters while maintaining accuracy

---

## Forward Pass Comparison

### Object Detection

```python
def forward(self, y1, y2, y3):
    # cv2: Bbox regression
    x1 = self.cv2[0](y1)  # Scale 0: 64 channels
    x2 = self.cv2[1](y2)  # Scale 1: 64 channels
    x3 = self.cv2[2](y3)  # Scale 2: 64 channels

    # cv3: Class prediction
    x4 = self.cv3[0](y1)  # Scale 0: 80 channels
    x5 = self.cv3[1](y2)  # Scale 1: 80 channels
    x6 = self.cv3[2](y3)  # Scale 2: 80 channels

    # Concatenate: bbox + classes
    y1 = concat(x1, x4)  # 64 + 80 = 144 channels
    y2 = concat(x2, x5)  # 64 + 80 = 144 channels
    y3 = concat(x3, x6)  # 64 + 80 = 144 channels

    # ... reshape and concat all scales ...

    # Split
    ya, yb = y.split((64, 80), dim=1)

    # Decode bbox with DFL
    ya = softmax(ya) → DFL → bbox  # 64 → 4 coords

    # Sigmoid on classes
    yb = sigmoid(yb)  # 80 class scores

    # Output
    return concat(bbox, yb)  # [4 + 80 = 84 channels]
```

---

### Pose Estimation

```python
def forward(self, y1, y2, y3):
    # cv2: Bbox regression (SAME as detection)
    x1_bbox = self.cv2[0](y1)  # Scale 0: 64 channels
    x2_bbox = self.cv2[1](y2)  # Scale 1: 64 channels
    x3_bbox = self.cv2[2](y3)  # Scale 2: 64 channels

    # cv3: Person confidence (uses DWConv)
    x1_conf = self.cv3[0](y1)  # Scale 0: 1 channel
    x2_conf = self.cv3[1](y2)  # Scale 1: 1 channel
    x3_conf = self.cv3[2](y3)  # Scale 2: 1 channel

    # cv4: Keypoints (NEW!)
    x1_kpts = self.cv4[0](y1)  # Scale 0: 51 channels
    x2_kpts = self.cv4[1](y2)  # Scale 1: 51 channels
    x3_kpts = self.cv4[2](y3)  # Scale 2: 51 channels

    # Concatenate: bbox + conf + keypoints
    y1 = concat(x1_bbox, x1_conf, x1_kpts)  # 64 + 1 + 51 = 116 channels
    y2 = concat(x2_bbox, x2_conf, x2_kpts)  # 64 + 1 + 51 = 116 channels
    y3 = concat(x3_bbox, x3_conf, x3_kpts)  # 64 + 1 + 51 = 116 channels

    # ... reshape and concat all scales ...

    # Split
    ya, yb, yc = y.split((64, 1, 51), dim=1)

    # Decode bbox with DFL (SAME as detection)
    ya = softmax(ya) → DFL → bbox  # 64 → 4 coords

    # Sigmoid on confidence
    yb = sigmoid(yb)  # 1 person score

    # Decode keypoints
    yc = reshape to [batch, 17, 3, num_anchors]
    kpt_x, kpt_y, kpt_v = split(yc)

    kpt_v = sigmoid(kpt_v)  # Visibility [0, 1]

    # Decode x, y (NO sigmoid!)
    kpt_x = (kpt_x * 2 - 0.5 + anchor_x) * stride
    kpt_y = (kpt_y * 2 - 0.5 + anchor_y) * stride

    keypoints = stack(kpt_x, kpt_y, kpt_v)

    # Output
    return concat(bbox, yb, keypoints)  # [4 + 1 + 51 = 56 channels]
```

---

## Output Format Comparison

### Object Detection Output

**Shape:** `[batch, 84, 8400]`

**Channel Breakdown:**
```
├─ 0-3:   Bounding box (x, y, w, h) - pixel coordinates
└─ 4-83:  Class probabilities (80 COCO classes)
          [person, bicycle, car, motorcycle, airplane, ...]
```

**Interpretation:**
```python
bbox = output[:, 0:4, :]
classes = output[:, 4:84, :]

# Get best class
class_id = classes.argmax(dim=1)
confidence = classes.max(dim=1)
```

---

### Pose Estimation Output

**Shape:** `[batch, 56, 8400]`

**Channel Breakdown:**
```
├─ 0-3:   Bounding box (x, y, w, h) - pixel coordinates
├─ 4:     Person confidence - sigmoid [0, 1]
└─ 5-55:  Keypoints (17 × 3 = 51 values)
          ├─ 5-7:   nose (x, y, visibility)
          ├─ 8-10:  left_eye (x, y, visibility)
          ├─ 11-13: right_eye (x, y, visibility)
          ├─ ...    (14 more keypoints)
          └─ 53-55: right_ankle (x, y, visibility)
```

**Interpretation:**
```python
bbox = output[:, 0:4, :]
conf = output[:, 4, :]
keypoints = output[:, 5:56, :].reshape(batch, 17, 3, num_anchors)

# Each keypoint
kpt_x, kpt_y, kpt_vis = keypoints[:, i, :, :]  # i = 0 to 16
```

---

## Layer-by-Layer Comparison

### cv2: Bbox Regression

| Aspect | Detection | Pose | Same? |
|--------|-----------|------|-------|
| **Architecture** | Conv→Conv→Conv2d | Conv→Conv→Conv2d | ✅ Yes |
| **Input channels** | 64, 128, 256 | 64, 128, 256 | ✅ Yes |
| **Output channels** | 64, 64, 64 | 64, 64, 64 | ✅ Yes |
| **Layer type** | Regular Conv | Regular Conv | ✅ Yes |
| **DFL used** | ✅ Yes | ✅ Yes | ✅ Yes |
| **Decoding** | (anchor+offset)*stride | (anchor+offset)*stride | ✅ Yes |

**Conclusion:** cv2 is **identical** in both heads

---

### cv3: Classes vs Confidence

| Aspect | Detection (Classes) | Pose (Confidence) | Same? |
|--------|---------------------|-------------------|-------|
| **Architecture** | Conv→Conv→Conv→Conv→Conv2d | DWConv→Conv→DWConv→Conv→Conv2d | ❌ Different |
| **Layer type** | Regular Conv | **DWConv** | ❌ Different |
| **Input channels** | 64, 128, 256 | 64, 128, 256 | ✅ Yes |
| **Output channels** | **80** | **1** | ❌ Different |
| **Purpose** | 80 class probabilities | Single person confidence | ❌ Different |
| **Activation** | Sigmoid | Sigmoid | ✅ Yes |
| **Output range** | [0, 1] per class | [0, 1] | ✅ Yes |

**Key Differences:**
1. **DWConv vs Conv**: Pose uses depthwise convolution (more efficient)
2. **80 vs 1 outputs**: Pose only predicts person class
3. **5 layers vs 5 layers**: Same depth, different layer types

---

### cv4: Keypoints (ONLY IN POSE)

| Aspect | Detection | Pose |
|--------|-----------|------|
| **Exists** | ❌ No | ✅ Yes |
| **Architecture** | N/A | Conv→Conv→Conv2d |
| **Input channels** | N/A | 64, 128, 256 |
| **Output channels** | N/A | 51, 51, 51 |
| **Layer type** | N/A | Regular Conv |
| **Purpose** | N/A | Keypoint prediction |

**Structure (3 scales):**
```python
# Scale 0: 64 → 51
Conv(64, 51, k=3) → Conv(51, 51, k=3) → Conv2d(51, 51, k=1)

# Scale 1: 128 → 51
Conv(128, 51, k=3) → Conv(51, 51, k=3) → Conv2d(51, 51, k=1)

# Scale 2: 256 → 51
Conv(256, 51, k=3) → Conv(51, 51, k=3) → Conv2d(51, 51, k=1)
```

**51 Channels =** 17 keypoints × 3 (x, y, visibility)

**Decoding:**
```python
# Visibility gets sigmoid
kpt_v = sigmoid(kpt_v)

# X, Y get special decoding (NO sigmoid!)
kpt_x = (kpt_x * 2 - 0.5 + anchor_x) * stride
kpt_y = (kpt_y * 2 - 0.5 + anchor_y) * stride
```

---

## Code Side-by-Side

### Object Detection: Concatenation
```python
# Concatenate bbox + classes for each scale
y1 = torch.cat((x1, x4), 1)  # 64 + 80 = 144
y2 = torch.cat((x2, x5), 1)  # 64 + 80 = 144
y3 = torch.cat((x3, x6), 1)  # 64 + 80 = 144

# After reshaping and concatenating scales
ya, yb = y.split((64, 80), 1)  # bbox, classes
```

### Pose Estimation: Concatenation
```python
# Concatenate bbox + conf + keypoints for each scale
y1 = torch.cat((x1_bbox, x1_conf, x1_kpts), 1)  # 64 + 1 + 51 = 116
y2 = torch.cat((x2_bbox, x2_conf, x2_kpts), 1)  # 64 + 1 + 51 = 116
y3 = torch.cat((x3_bbox, x3_conf, x3_kpts), 1)  # 64 + 1 + 51 = 116

# After reshaping and concatenating scales
ya, yb, yc = y.split((64, 1, 51), 1)  # bbox, conf, keypoints
```

---

## Parameter Comparison

### Detection Head

```python
Detect(
    in_channel=[64, 64, 64, ...],      # Complex arrays
    out_channel=[64, 64, 64, ...],     # Passed as parameters
    kernel=[3, 3, 1, ...],
    ...
)
```

**Pros:** Flexible configuration
**Cons:** Complex parameter arrays

### Pose Head

```python
PoseHead()  # No parameters!
```

**Pros:** Self-contained, simpler to read
**Cons:** Hardcoded architecture

---

## Channel Flow Diagram

### Object Detection
```
Input Features (y1, y2, y3)
        ↓
    ┌───────────┬────────────┐
    │           │            │
   cv2         cv3          │
  (bbox)     (classes)      │
    64          80           │
    │           │            │
    └───────┬───┴────────────┘
            ↓
      Concat: 144 channels
            ↓
    Split: 64 bbox + 80 classes
            ↓
      DFL decode bbox
            ↓
      Sigmoid classes
            ↓
    Output: 84 channels
```

### Pose Estimation
```
Input Features (y1, y2, y3)
        ↓
    ┌───────────┬────────────┬────────────┐
    │           │            │            │
   cv2         cv3          cv4          │
  (bbox)     (conf)      (keypoints)    │
    64          1            51          │
 (regular)  (DWConv)     (regular)      │
    │           │            │            │
    └───────┬───┴────────┬───┴────────────┘
            ↓            ↓
      Concat: 116 channels
            ↓
    Split: 64 bbox + 1 conf + 51 kpts
            ↓
      DFL decode bbox
            ↓
      Sigmoid conf
            ↓
      Decode keypoints
            ↓
    Output: 56 channels
```

---

## Summary Table

| Feature | Object Detection | Pose Estimation |
|---------|------------------|-----------------|
| **Head class** | `Detect` | `PoseHead` |
| **cv2 (bbox)** | ✅ Same | ✅ Same |
| **cv3** | 80 classes (Conv) | 1 confidence (DWConv) |
| **cv4** | ❌ None | 51 keypoints (Conv) |
| **DWConv used** | ❌ No | ✅ Yes (in cv3) |
| **Output channels** | 84 | 56 |
| **Task** | Multi-class detection | Human pose estimation |
| **Classes** | 80 COCO classes | 1 (person only) |
| **Additional output** | None | 17 keypoints × 3 |
| **Complexity** | Higher (80 classes) | Lower (1 class + keypoints) |

---

## Key Takeaways

### Similarities
- ✅ Both use same backbone and neck (layers 0-22)
- ✅ Both use DFL for bbox regression
- ✅ Both use 3-scale multi-scale detection
- ✅ Both apply sigmoid to confidence/class scores

### Differences
- ❌ Detection predicts 80 classes, Pose predicts 1 class + keypoints
- ❌ Pose uses DWConv (depthwise) for efficiency in cv3
- ❌ Pose has cv4 head for keypoint prediction
- ❌ Pose has special keypoint decoding (no sigmoid on x,y)

### Why Pose is Different
1. **Single class** (person only) → Can use simpler cv3 with DWConv
2. **Keypoints needed** → Requires new cv4 head
3. **Different output** → 17 body joint locations instead of class labels

---

## Implementation Files

### Object Detection
```
reference/yolov11.py
  - class Detect (lines 444-739)

tt/ttnn_yolov11_detect.py
  - class TtnnDetect
```

### Pose Estimation
```
reference/yolov11_pose_correct.py
  - class DWConv (lines 22-34)
  - class PoseHead (lines 37-231)

tt/ttnn_yolov11_dwconv.py
  - class TtnnDWConv

tt/ttnn_yolov11_pose.py
  - class TtnnPoseHead
```

---

## COCO Keypoint Format (17 keypoints)

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

Each keypoint has 3 values:
- **x**: horizontal pixel coordinate
- **y**: vertical pixel coordinate
- **visibility**: confidence score [0, 1]

Total: 17 × 3 = 51 values
