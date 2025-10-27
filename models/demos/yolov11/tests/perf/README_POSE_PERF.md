# YOLO11 Pose Estimation - Performance Tests

## Overview

Performance tests for YOLO11 Pose Estimation on TT-Metal hardware.

## Test Files

| File | Type | Purpose |
|------|------|---------|
| `test_perf_pose.py` | Device Performance | Measures device kernel execution time |
| `test_e2e_performant_pose.py` | End-to-End Performance | Measures full inference pipeline throughput |

---

## Running Performance Tests

### 1. Device Performance Test

**Measures:** Device kernel execution time, samples per second

```bash
cd /home/ubuntu/MAIN/tt-metal
pytest models/demos/yolov11/tests/perf/test_perf_pose.py -v
```

**Output:**
- Device firmware time
- Device kernel time
- BRISC kernel time
- Average samples/second

---

### 2. End-to-End Performance Test

**Measures:** Full inference time including preprocessing and postprocessing

```bash
# Single device
pytest models/demos/yolov11/tests/perf/test_e2e_performant_pose.py::test_e2e_performant_pose -v

# Multi-device (data parallel)
pytest models/demos/yolov11/tests/perf/test_e2e_performant_pose.py::test_e2e_performant_pose_dp -v
```

**Output:**
```
YOLO11 Pose Performance Results:
  Model: ttnn_yolov11_pose
  Batch size: 1
  Resolution: (640, 640)
  Total time (100 iters): X.XX sec
  Avg inference time: X.XXXXXX sec
  FPS (frames per second): XX.XX
  Throughput: XX.XX images/sec
```

---

## Performance Comparison

### Expected Performance

| Model | Batch Size | FPS (est.) | Notes |
|-------|-----------|-----------|-------|
| YOLO11 Detection | 1 | ~276 | 80 classes |
| YOLO11 Pose | 1 | ~250-280 | 1 class + 17 keypoints |

**Note:** Pose estimation may be slightly faster than detection due to:
- Only 1 class prediction (vs 80)
- cv3 uses efficient DWConv
- But adds cv4 (keypoint head)

---

## Performance Metrics Explained

### Device Kernel Time
- Time spent executing kernels on TT device
- Does NOT include host-device communication
- Pure compute time

### End-to-End Time
- Full pipeline: input prep → TTNN inference → output conversion
- Includes memory transfers
- Real-world performance metric

### FPS (Frames Per Second)
- How many images processed per second
- Key metric for real-time applications
- Higher is better

### Throughput
- Total images/second across all devices
- Important for batch processing
- Scales with number of devices

---

## Optimization Opportunities

### Current Implementation
- ✅ TTNN architecture verified correct (PCC test passed)
- ✅ DWConv implemented
- ✅ Multi-scale pose head
- ⚠️ No performant runner yet (unlike object detection)

### Future Optimizations
1. **Performant Runner**: Create `YOLOv11PosePerformantRunner` (like detection)
2. **Memory optimization**: Tune sharding strategies
3. **Kernel fusion**: Fuse consecutive operations
4. **Double buffering**: Pipeline input/output transfers
5. **Multi-device**: Scale across multiple TT chips

---

## Comparison with Object Detection Perf Tests

| Aspect | Detection | Pose |
|--------|-----------|------|
| **Model** | YOLOv11 (80 classes) | YOLOv11 Pose (keypoints) |
| **Test files** | `test_perf.py`, `test_e2e_performant.py` | `test_perf_pose.py`, `test_e2e_performant_pose.py` |
| **Output channels** | 84 | 56 |
| **Computation** | cv2 + cv3 (80 classes) | cv2 + cv3 (DWConv, 1 class) + cv4 (keypoints) |
| **Performant runner** | ✅ Implemented | ❌ Not yet (uses direct model call) |
| **Expected FPS** | ~276 | ~250-280 (estimated) |

---

## Running All Performance Tests

```bash
# Run all YOLO11 perf tests (detection + pose)
pytest models/demos/yolov11/tests/perf/ -v -k "yolov11"

# Run only pose perf tests
pytest models/demos/yolov11/tests/perf/ -v -k "pose"

# Device performance only
pytest models/demos/yolov11/tests/perf/test_perf_pose.py -v

# E2E performance only
pytest models/demos/yolov11/tests/perf/test_e2e_performant_pose.py -v
```

---

## Markers

Tests use pytest markers for categorization:

```python
@pytest.mark.models_device_performance_bare_metal  # Device-level perf
@pytest.mark.models_performance_bare_metal         # E2E on bare metal
@pytest.mark.models_performance_virtual_machine    # E2E on VM
@run_for_wormhole_b0()                              # Wormhole B0 only
```

---

## Prerequisites

1. **TT-Metal hardware** (Wormhole B0)
2. **Pretrained weights**:
```bash
cd models/demos/yolov11/reference
python3 load_weights_correct.py
```

3. **Performance profiling enabled**:
```bash
# Set environment variable if needed
export TTNN_CONFIG_OVERRIDES='{"enable_logging": true}'
```

---

## Interpreting Results

### Good Performance
```
✓ FPS > 250 (batch_size=1, 640x640)
✓ Avg inference time < 4ms
✓ Device kernel samples/s > 270
```

### Performance Issues
```
⚠ FPS < 200
⚠ Avg inference time > 5ms
⚠ Large variance between runs
```

**If performance is lower than expected:**
1. Check device utilization
2. Review memory layout (sharding strategy)
3. Profile individual layers
4. Consider implementing performant runner

---

## Next Steps

1. ✅ Baseline performance established
2. ⏳ Implement `YOLOv11PosePerformantRunner` for optimization
3. ⏳ Profile individual components (DWConv, pose head)
4. ⏳ Optimize memory layouts
5. ⏳ Benchmark against PyTorch CPU/GPU

---

## Files Created

```
✅ tests/perf/test_perf_pose.py           # Device performance
✅ tests/perf/test_e2e_performant_pose.py # E2E performance
✅ tests/perf/README_POSE_PERF.md         # This documentation
```

**Total:** 3 new performance test files

---

## Related Documentation

- `../pcc/README_POSE_TESTS.md` - PCC/accuracy tests
- `../../tt/README_POSE_TTNN.md` - TTNN implementation guide
- `../../reference/POSE_ESTIMATION_README.md` - PyTorch reference
