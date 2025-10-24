# YOLO11 Pose Estimation - TTNN Tests

## Overview

This directory contains PCC (Pearson Correlation Coefficient) tests for the TTNN implementation of YOLO11 Pose Estimation. These tests verify that the TTNN implementation matches the PyTorch reference implementation.

## Test Files

### Pose Estimation Tests (New)

| File | Component Tested | Input Shapes | PCC Threshold |
|------|-----------------|--------------|---------------|
| `test_ttnn_yolov11_dwconv.py` | DWConv layer | Various scales | 0.99 |
| `test_ttnn_yolov11_pose.py` | Pose Head (layer 23) | 3 scales | 0.99 |
| `test_ttnn_yolov11_pose_model.py` | Complete pose model | 640×640 | 0.99 |

### Object Detection Tests (Existing)

| File | Component Tested |
|------|-----------------|
| `test_ttnn_yolov11_detect.py` | Detect head |
| `test_ttnn_yolov11.py` | Complete detection model |
| `test_ttnn_yolov11_c3k2.py` | C3k2 block |
| `test_ttnn_yolov11_c2psa.py` | C2PSA block |
| `test_ttnn_yolov11_sppf.py` | SPPF block |
| `test_ttnn_yolov11_attention.py` | Attention mechanism |
| `test_ttnn_yolov11_bottleneck.py` | Bottleneck block |
| `test_ttnn_yolov11_c3k.py` | C3k block |
| `test_ttnn_yolov11_psa.py` | PSA block |

---

## Running Tests

### Prerequisites

1. **TT-Metal device** properly configured
2. **Pretrained weights** downloaded:
```bash
cd models/demos/yolov11/reference
python3 load_weights_correct.py
```

### Run Individual Tests

#### Test DWConv Layer
```bash
pytest models/demos/yolov11/tests/pcc/test_ttnn_yolov11_dwconv.py -v
```

#### Test Pose Head
```bash
pytest models/demos/yolov11/tests/pcc/test_ttnn_yolov11_pose.py -v
```

#### Test Complete Pose Model
```bash
pytest models/demos/yolov11/tests/pcc/test_ttnn_yolov11_pose_model.py -v
```

### Run All Pose Tests
```bash
pytest models/demos/yolov11/tests/pcc/test_ttnn_yolov11_*pose*.py -v
```

### Run All YOLO11 Tests (Detection + Pose)
```bash
pytest models/demos/yolov11/tests/pcc/ -v
```

---

## Test Details

### 1. test_ttnn_yolov11_dwconv.py

**Purpose:** Verify Depthwise Convolution layer

**Test Cases:**
- Scale 0: 64 → 64 (80×80 feature map)
- Scale 1: 128 → 128 (40×40 feature map)
- Scale 2: 256 → 256 (20×20 feature map)

**What it tests:**
- DWConv with groups=in_channels
- BatchNorm folding into conv
- SiLU activation
- Memory layout transformations

**Expected PCC:** ≥ 0.99

---

### 2. test_ttnn_yolov11_pose.py

**Purpose:** Verify Pose Head (layer 23)

**Test Cases:**
- Full pose head with 3 input scales

**Input Shapes:**
- Scale 0: [1, 64, 80, 80]
- Scale 1: [1, 128, 40, 40]
- Scale 2: [1, 256, 20, 20]

**What it tests:**
- cv2: Bounding box regression (64 channels)
- cv3: Person confidence with DWConv (1 channel)
- cv4: Keypoint prediction (51 channels)
- DFL decoding for bboxes
- Keypoint coordinate decoding
- Concatenation of all predictions

**Output Shape:** [1, 56, 8400]
- 4: bbox
- 1: confidence
- 51: keypoints

**Expected PCC:** ≥ 0.99

---

### 3. test_ttnn_yolov11_pose_model.py

**Purpose:** Verify complete end-to-end pose model

**Test Cases:**
- Full model with 640×640 input
- With and without pretrained weights

**Input Shape:** [1, 3, 640, 640]

**What it tests:**
- Entire pipeline from image input to pose predictions
- All 23 layers working together
- Backbone feature extraction
- Neck feature pyramid
- Pose head predictions
- Memory management and layout conversions

**Output Shape:** [1, 56, 8400]

**Expected PCC:** ≥ 0.99

---

## Test Architecture

### Test Flow

```
PyTorch Reference Model
        ↓
   Create Input
        ↓
┌──────────────────────┐
│  PyTorch Forward     │
│  torch_output        │
└──────────────────────┘
        ↓
┌──────────────────────┐
│  Preprocess Params   │
│  (Weights to TTNN)   │
└──────────────────────┘
        ↓
┌──────────────────────┐
│  TTNN Forward        │
│  ttnn_output         │
└──────────────────────┘
        ↓
┌──────────────────────┐
│  Compare Outputs     │
│  assert_with_pcc     │
└──────────────────────┘
```

### PCC (Pearson Correlation Coefficient)

**Threshold:** 0.99

- `PCC = 1.0` → Perfect correlation
- `PCC ≥ 0.99` → Excellent match (acceptable)
- `PCC < 0.99` → Test fails (implementation mismatch)

---

## Common Test Patterns

### Pattern 1: Component Test (DWConv, individual layers)

```python
# Create PyTorch module
torch_module = TorchComponent(...)
torch_module.eval()

# Create inputs
torch_input, ttnn_input = create_yolov11_input_tensors(...)

# Run PyTorch
torch_output = torch_module(torch_input)

# Preprocess for TTNN
parameters = preprocess_model_parameters(...)

# Create and run TTNN
ttnn_module = TtnnComponent(device, parameters, ...)
ttnn_output = ttnn_module(device, ttnn_input)

# Compare
assert_with_pcc(torch_output, ttnn_output, 0.99)
```

### Pattern 2: Head Test (Multi-scale inputs)

```python
# Create 3 inputs for 3 scales
torch_input_1, ttnn_input_1 = create_inputs(scale=0)
torch_input_2, ttnn_input_2 = create_inputs(scale=1)
torch_input_3, ttnn_input_3 = create_inputs(scale=2)

# Run through head
torch_output = torch_head(input_1, input_2, input_3)
ttnn_output = ttnn_head(input_1, input_2, input_3)

# Compare
assert_with_pcc(torch_output, ttnn_output, 0.99)
```

### Pattern 3: End-to-End Model Test

```python
# Create full model with pretrained weights
torch_model = YoloV11Pose()
torch_model.load_state_dict(torch.load('weights.pth'))

# Single input image
torch_input, ttnn_input = create_inputs([1, 3, 640, 640])

# Run full pipeline
torch_output = torch_model(torch_input)
ttnn_output = ttnn_model(ttnn_input)

# Compare
assert_with_pcc(torch_output, ttnn_output, 0.99)
```

---

## Debugging Failed Tests

### If PCC < 0.99:

1. **Check component by component:**
```bash
# Test individual layers first
pytest test_ttnn_yolov11_dwconv.py -v
pytest test_ttnn_yolov11_pose.py -v
```

2. **Compare intermediate outputs:**
```python
# Add debug prints in test
print(f"PyTorch output range: [{torch_output.min()}, {torch_output.max()}]")
print(f"TTNN output range: [{ttnn_output.min()}, {ttnn_output.max()}]")
```

3. **Check weight loading:**
```bash
cd models/demos/yolov11/reference
python3 compare_model_outputs.py
# Should show 0.000000 difference
```

4. **Verify memory layouts:**
```python
# Check tensor properties
print(f"TTNN tensor layout: {ttnn_output.layout}")
print(f"TTNN tensor memory: {ttnn_output.memory_config()}")
```

---

## Expected Test Results

### With Pretrained Weights

```
test_ttnn_yolov11_dwconv.py::test_yolo_v11_dwconv[64-64-3-1-1]    PASSED
test_ttnn_yolov11_dwconv.py::test_yolo_v11_dwconv[128-128-3-1-1]  PASSED
test_ttnn_yolov11_dwconv.py::test_yolo_v11_dwconv[256-256-3-1-1]  PASSED

test_ttnn_yolov11_pose.py::test_yolo_v11_pose_head                PASSED

test_ttnn_yolov11_pose_model.py::test_yolov11_pose_model          PASSED
```

### Without Pretrained Weights

```
test_ttnn_yolov11_pose_model.py::test_yolov11_pose_model          SKIPPED
  (Reason: Pretrained weights not available)
```

---

## Test Parameters

### Device Configuration

```python
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLOV11_L1_SMALL_SIZE}],
    indirect=True
)
```

**L1 Small Size:** Configures L1 memory size for YOLO11

### Resolution

```python
@pytest.mark.parametrize(
    "resolution",
    [([1, 3, 640, 640])],  # Standard YOLO input
)
```

**Standard:** 640×640 is the default YOLO11 input size

---

## Comparison: Detection vs Pose Tests

| Aspect | Detection Tests | Pose Tests |
|--------|----------------|-----------|
| **Head test** | `test_ttnn_yolov11_detect.py` | `test_ttnn_yolov11_pose.py` |
| **Model test** | `test_ttnn_yolov11.py` | `test_ttnn_yolov11_pose_model.py` |
| **New component** | None | `test_ttnn_yolov11_dwconv.py` |
| **Output channels** | 84 | 56 |
| **Shared tests** | C3k2, C2PSA, SPPF, etc. | ✓ Same |

---

## CI/CD Integration

### Add to CI Pipeline

```yaml
# Example pytest command for CI
pytest models/demos/yolov11/tests/pcc/test_ttnn_yolov11_pose*.py \
  --junit-xml=pose_test_results.xml \
  -v
```

### Test Markers (Future)

```python
@pytest.mark.pose_estimation
@pytest.mark.slow  # For end-to-end tests
@pytest.mark.requires_weights  # For tests needing pretrained weights
```

---

## Performance Tests

For performance testing, see:
```
models/demos/yolov11/tests/perf/
```

(Pose-specific performance tests can be added there)

---

## Test Coverage

### Covered Components

- ✅ DWConv layer (new for pose)
- ✅ Pose Head (cv2, cv3, cv4)
- ✅ Complete pose model (end-to-end)
- ✅ Keypoint decoding
- ✅ Multi-scale feature maps

### Shared Components (Tested by Detection)

- ✅ Conv, C3k2, C2PSA, SPPF (same for both)
- ✅ Backbone and neck layers
- ✅ Memory operations

---

## Success Criteria

A test passes if:

1. ✅ No runtime errors
2. ✅ Output shapes match PyTorch reference
3. ✅ PCC ≥ 0.99 (99% correlation)
4. ✅ Output channel count correct (56 for pose)
5. ✅ Memory properly deallocated (no leaks)

---

## Files Summary

### Tests Created for Pose
```
✅ test_ttnn_yolov11_dwconv.py         # DWConv layer test
✅ test_ttnn_yolov11_pose.py           # Pose head test
✅ test_ttnn_yolov11_pose_model.py     # Complete model test
✅ README_POSE_TESTS.md                # This documentation
```

### Total Test Count
- **Pose-specific:** 3 tests
- **Shared components:** 6 tests (already exist)
- **Total coverage:** 9 tests for pose estimation

---

## Quick Start

```bash
# Install dependencies
pip install pytest ultralytics

# Get pretrained weights
cd models/demos/yolov11/reference
python3 load_weights_correct.py

# Run pose tests
cd models/demos/yolov11/tests/pcc
pytest test_ttnn_yolov11_dwconv.py -v
pytest test_ttnn_yolov11_pose.py -v
pytest test_ttnn_yolov11_pose_model.py -v
```

---

## Expected Output

```
test_ttnn_yolov11_dwconv.py::test_yolo_v11_dwconv[64-64-3-1-1-input_shape0] PASSED
test_ttnn_yolov11_dwconv.py::test_yolo_v11_dwconv[128-128-3-1-1-input_shape1] PASSED
test_ttnn_yolov11_dwconv.py::test_yolo_v11_dwconv[256-256-3-1-1-input_shape2] PASSED

test_ttnn_yolov11_pose.py::test_yolo_v11_pose_head PASSED

test_ttnn_yolov11_pose_model.py::test_yolov11_pose_model PASSED

==================== 5 passed in XX.XXs ====================
```

---

## Troubleshooting

### Test Fails with "Pretrained weights not available"

**Solution:**
```bash
cd models/demos/yolov11/reference
python3 load_weights_correct.py
```

### Test Fails with PCC < 0.99

**Debug steps:**
1. Run comparison tool:
```bash
cd models/demos/yolov11/reference
python3 compare_model_outputs.py
```

2. Check if PyTorch model matches Ultralytics (should be 0.000000)

3. If PyTorch is correct but TTNN fails, check:
   - Memory layout conversions
   - Sharding configurations
   - Weight preprocessing

### Test Fails with Shape Mismatch

**Check:**
- Input tensor creation
- Output reshaping
- Channel count (should be 56 for pose)

---

## Next Steps

After tests pass:

1. ✅ Add performance tests in `tests/perf/`
2. ✅ Add integration tests
3. ✅ Benchmark on TT hardware
4. ✅ Optimize memory usage
5. ✅ Add multi-device tests

---

## References

- PyTorch Reference: `models/demos/yolov11/reference/yolov11_pose_correct.py`
- TTNN Implementation: `models/demos/yolov11/tt/ttnn_yolov11_pose_model.py`
- Test Utils: `tests/ttnn/utils_for_testing.py`
