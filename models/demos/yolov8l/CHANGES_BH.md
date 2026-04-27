# YOLOv8L Blackhole (BH) Optimization Changes

## Background

Blackhole (BH) has **12 rows × 10 cols = 120 Tensix cores** per device with **1.5 MB L1/core**, vs Wormhole B0 (WH) with **8×8 = 64 cores** and **1 MB L1/core**.
Due to typical 1-row harvesting, effective BH usable cores ≈ 110, but safe rectangular grids are 80 (8×10) or 120 (12×10).

---

## Changes Made

### 1. `models/demos/yolov8l/runner/performant_runner_infra.py`

**Bug fixed**: `_setup_l1_sharded_input` was computing a BH-aware `core_grid` variable but then ignoring it and always passing a hardcoded `CoreGrid(x=8, y=8)` (64 WH cores) to `create_sharded_memory_config`.

**Fix**: Use device-appropriate core grid:
```python
if is_wormhole_b0():
    core_grid = ttnn.CoreGrid(y=8, x=8)
else:  # BH: 8x10=80 cores divides both 640 and 1280 evenly
    core_grid = ttnn.CoreGrid(y=8, x=10)
```

**Why 80 and not 120**: For input tensor `[1, 16, 640, 640]`, shard height = `1 × 16 × 640 = 10240`. The constraint is `10240 % num_cores == 0`. Divisors ≤ 120: 80 is the largest valid value. `10240 % 120 ≠ 0`, so 120 cannot be used.

---

### 2. `models/demos/yolov8l/tt/ttnn_yolov8l.py`

#### 2a. SPPF concat uses 80 BH cores instead of 64 WH cores

Added device-aware globals initialized in `TtDetectionModel.__init__`:
```python
global _SPPF_CORE_GRID, _SPPF_NUM_CORES
if is_wormhole_b0():
    _SPPF_CORE_GRID = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))})
    _SPPF_NUM_CORES = 64
else:  # BH: 8x10=80 cores
    _SPPF_CORE_GRID = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(9, 7))})
    _SPPF_NUM_CORES = 80
```

`sharded_concat_sppf` and `sharded_concat` both use `_SPPF_CORE_GRID` and `_SPPF_NUM_CORES` instead of hardcoded WH values.

#### 2b. `packer_l1_acc=True` on BH

In `TtConv._initialize_compute_config()`:
```python
packer_l1_acc=not is_wormhole_b0(),  # BH has 1.5MB L1 vs WH 1MB — safe to accumulate in L1
```

This keeps matmul partial sums in L1 instead of bouncing through DRAM, reducing memory bandwidth pressure on BH.

---

### 3. `models/experimental/yolo_common/yolo_utils.py`

#### 3a. Device-aware grid helper

Added `_device_grid_params()`:
```python
def _device_grid_params():
    if is_wormhole_b0():
        return 8, 8, 64
    return 12, 10, 120  # Blackhole
```

#### 3b. `determine_num_cores` uses BH max_cores=120

Changed signature to `def determine_num_cores(nhw, width, max_cores=None)` with:
```python
if max_cores is None:
    _, _, max_cores = _device_grid_params()
```

#### 3c. Bug fix in `get_core_grid_from_num_cores`

Original code had two bugs (only correct for square WH 8×8 grid):
- Used `grid_rows - 1` as x-coordinate → should be `grid_cols - 1`
- Used `% grid_rows` for partial-row remainder → should be `% grid_cols`

Fixed to correctly handle non-square BH 12×10 grid.

Updated signature: `def get_core_grid_from_num_cores(num_cores, grid_rows=None, grid_cols=None)` with defaults from `_device_grid_params()`.

---

### 4. `models/demos/yolov8l/tests/perf/test_e2e_performant.py`

Removed `@run_for_wormhole_b0()` decorator from `test_run_yolov8l_trace_2cqs_inference` so it runs on BH.

The data-parallel test `test_run_yolov8l_trace_2cqs_dp_inference` retains `@run_for_wormhole_b0()` as it targets T3K multi-device which is WH-only.

---

## Performance Results

| Chip | Resolution | FPS (before) | FPS (after) |
|------|-----------|--------------|-------------|
| BH single device | 640×640 | ~100 (WH code path) | **~130** |

Primary gain: `packer_l1_acc=True` + correct 80-core input sharding (vs 64-core WH grid).

---

## What Was Tried But Reverted

| Attempt | Reason reverted |
|---------|----------------|
| `CoreGrid(y=12, x=10)` = 120 cores for input sharding | `10240 % 120 ≠ 0` → `RuntimeError: Invalid sharding core_grid` |
| `reshard_if_not_optimal=True` on BH convs | Downstream fixed shard layouts incompatible → runtime crash at `program.cpp:1136` |
| `act_block_h=64` on conv_0/conv_1 | Activation block too large for kernel → crash at `program.cpp:1145` |
| Scale `l1_small_size` by 1.5× for BH in `common.py` | SPPF conv crash at `program.cpp:1145` — likely reservation conflict |
