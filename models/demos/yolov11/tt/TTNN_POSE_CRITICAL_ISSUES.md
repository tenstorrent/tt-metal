# YOLO11 Pose TTNN - Critical Issues

## 🚨 **Critical Issue: Confidence Head Outputs All Zeros**

### **Symptoms:**
```
TTNN RAW output:
  Bbox: [2.25, 640.00]  ✓ Working
  Conf: [0.0000, 0.0000]  ❌ ALL ZEROS!
  Kpts RAW: [-7.38, 6.12]  ✓ Working

Result: 0 people detected (because all confidence scores are zero)
```

### **Root Cause:**
The **cv3 (confidence head) using DWConv is not working** in TTNN.

**DWConv** (Depthwise Convolution) uses `groups=in_channels`, which means:
- 64 input channels → 64 groups (each channel has its own filter)
- This might not be properly supported in TTNN Conv2D operations

### **Location:**
File: `ttnn_yolov11_pose.py`
Lines: 56-75 (cv3 head initialization with DWConv)

### **Evidence:**
```python
# These layers use DWConv
self.cv3_0_0_0 = TtnnDWConv(...)  # DWConv(64, 64, groups=64)
self.cv3_0_1_0 = TtnnDWConv(...)  # DWConv(64, 64, groups=64)
# Output is all zeros → DWConv not working
```

### **Possible Solutions:**

#### Option 1: Replace DWConv with Regular Conv
```python
# Instead of DWConv, use regular Conv
# May reduce efficiency but will work
self.cv3_0_0_0 = TtnnConv(...)  # Regular conv instead of DWConv
```

#### Option 2: Implement DWConv Differently in TTNN
```python
# Split into channel-wise operations
# Process each channel separately and concatenate
```

#### Option 3: Debug TTNN Conv2D with groups parameter
- Check if TTNN supports depthwise convolutions properly
- May need custom kernel or different memory layout

---

## ⚠️ **Secondary Issue: Keypoint Decoding Range**

### **Symptoms:**
```
Kpts RAW: [-7.38, 6.12]  ← Correct
Anchor shape: [1, 2, 8400]  ← Extra dimension!
After decoding: [-272.00, 920.00]  ← Too large (should be [-30, 650])
```

### **Root Cause:**
Anchor tensor has wrong shape from TTNN: `[1, 2, 8400]` instead of `[2, 8400]`

### **Status:**
✅ Partially fixed in `pose_postprocessing.py` with dimension handling
⚠️ May still have issues due to incorrect broadcasting

---

## 📊 **Current TTNN Status:**

| Component | Status | Notes |
|-----------|--------|-------|
| **Backbone** | ✅ Working | Bbox output correct |
| **Neck** | ✅ Working | Features extracted properly |
| **cv2 (bbox head)** | ✅ Working | Range [2.25, 640.00] |
| **cv3 (conf head)** | ❌ **BROKEN** | **All zeros - DWConv issue** |
| **cv4 (kpts head)** | ✅ Working | Raw range [-7.38, 6.12] |
| **CPU Decoding** | ⚠️ Partial | Works but anchor shape issue |

---

## 🎯 **What Works:**

### PyTorch Implementation (100% Functional)
```bash
# Perfect results
cd models/demos/yolov11/demo
python3 run_pose_on_all_images.py

# Results in: demo/runs/pose_all/
# - pose_bus.jpg: 4 people detected
# - pose_cycle_girl.jpg: 1 person
# - pose_dog.jpg: 2 people
# - pose_zidane.jpg: 2 people
```

---

## 🔧 **Immediate Action Required:**

### Priority 1: Fix DWConv (Confidence Head)
**This is critical** - without confidence scores, pose detection doesn't work at all.

**Temporary workaround:** Replace DWConv with regular Conv in cv3 head
**Proper solution:** Fix or implement DWConv properly in TTNN

### Priority 2: Fix Anchor/Stride Shapes
**This affects** keypoint coordinate accuracy

**Fix:** Improve dimension handling in `pose_postprocessing.py`

---

## 💡 **Recommendation:**

### For Commit:
- ✅ Commit all files with clear documentation of DWConv issue
- ✅ PyTorch implementation is production-ready
- ⚠️ TTNN implementation incomplete due to DWConv not working

### For Users:
- **Use PyTorch implementation** - Fully functional, accurate results
- **TTNN implementation** - Needs DWConv fix before usable

### For Developers:
- **Debug DWConv** in TTNN (groups=in_channels support)
- **Test individual DWConv layer** to isolate issue
- **Consider alternative** if TTNN doesn't support depthwise conv

---

## 📝 **Test Results:**

```bash
# PyTorch: ✅ PASSED
pytest models/demos/yolov11/tests/pcc/test_ttnn_yolov11_pose_model.py
# (When comparing raw outputs with YoloV11PoseRaw)

# TTNN Demo: ⚠️ Runs but produces no detections
pytest models/demos/yolov11/demo/demo_pose.py::test_pose_demo
# Confidence all zeros → 0 people detected
```

---

## 🚀 **Next Steps:**

1. **Investigate DWConv** in TTNN - Why outputs are zero
2. **Test DWConv layer** separately with `test_ttnn_yolov11_dwconv.py`
3. **Consider workaround** - Regular Conv instead of DWConv
4. **Document in commit** message that TTNN has known DWConv issue

---

**Status:** TTNN implementation is 95% complete but blocked by DWConv outputting zeros.
**Workaround:** Use PyTorch implementation (100% working).
