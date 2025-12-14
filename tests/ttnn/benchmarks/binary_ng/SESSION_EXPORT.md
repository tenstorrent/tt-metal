# Session Export - Binary NG Benchmark Testing

**Date**: November 13, 2025
**Session Summary**: Fixed block sharding grid validation and compute cores tracking in benchmark tests

---

## Overview

This session focused on fixing errors in the `example_single_test.py` benchmark test, specifically:
1. Block sharding grid validation errors
2. Compute core grid tracking from C++ implementation
3. Printing tensor sharding information

---

## Key Changes Made

### 1. Block Sharding Grid Validation (MAIN FIX)

**Problem**: Test generated many "ValueError: BLOCK needs 16+ cores" errors because block sharding requires 2D grids that are compatible with tensor shapes.

**Solution**: Added intelligent grid computation based on tensor shape constraints.

#### New Function: `compute_valid_block_grid(shape, cores)`

Location: `tests/ttnn/benchmarks/binary_ng/example_single_test.py` (lines 142-184)

```python
def compute_valid_block_grid(shape, cores):
    """
    Compute a valid block sharding grid for the given shape and cores.

    For block sharding with shape (H, W) and N cores, the grid (GH, GW) must satisfy:
    - GH * GW = N
    - H (padded to tile size) must be divisible by GH
    - W (padded to tile size) must be divisible by GW

    Returns:
        Tuple (grid_h, grid_w) or None if no valid grid exists
    """
```

**Key Insight**: Block sharding is 2D, unlike height/width sharding which is 1D.

**Examples**:
- Tensor `(1, 1024)` with 8 cores → grid `(2, 4)` ✅ (not `(8, 1)` ❌)
- Tensor `(1024, 1)` with 8 cores → grid `(2, 4)` ✅ (can work vertically)
- Tensor `(1024, 1024)` with 8 cores → many valid options, selects most square

#### Updated: `create_sharded_tensor()` for Block Sharding

Location: `tests/ttnn/benchmarks/binary_ng/example_single_test.py` (lines 227-242)

Changed from hardcoded grids to shape-aware computation:
```python
elif sharding == "block":
    # Compute a valid block grid based on tensor shape
    grid_shape = compute_valid_block_grid(shape, cores)
    if grid_shape is None:
        raise ValueError(f"Cannot create valid block sharding grid...")

    grid_h, grid_w = grid_shape
    print(f"  [Block Grid] shape={shape}, cores={cores} → grid=({grid_h}, {grid_w})")

    grid = ttnn.CoreRangeSet({ttnn.CoreRange(
        ttnn.CoreCoord(0, 0),
        ttnn.CoreCoord(grid_w - 1, grid_h - 1)
    )})
```

#### Added Early Filtering in Test Configuration

Location: Lines 799-805 and 836-842

```python
# Filter: block sharding needs a valid grid based on tensor shape
if a_sharding == "block" and a_cores is not None:
    if compute_valid_block_grid(shape_a, a_cores) is None:
        continue  # Skip: no valid block grid for this shape/cores combination
if b_sharding == "block" and b_cores is not None:
    if compute_valid_block_grid(shape_b, b_cores) is None:
        continue  # Skip: no valid block grid for this shape/cores combination
```

Also enabled filtering for height sharding with height=1.

---

### 2. Compute Cores Tracking from C++

**Problem**: `compute_cores` column in CSV was showing 0 when output was interleaved, because it was derived from output tensor's cores instead of the compute grid used.

**Solution**: Capture and parse the `WORKER_GRID` log from C++ implementation.

#### Implementation

Location: Lines 835-869 (in operation execution)

```python
# Run operation and capture stderr to get WORKER_GRID log from C++
# C++ fprintf(stderr, ...) writes directly to FD 2, so we need FD-level redirection
stderr_fd = sys.stderr.fileno()
saved_stderr = os.dup(stderr_fd)

with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.txt') as tmp_file:
    tmp_name = tmp_file.name
    os.dup2(tmp_file.fileno(), stderr_fd)
    sys.stderr = tmp_file

    result = op_func(tensor_a, tensor_b)
    ttnn.synchronize_device(device_with_profiling)

    os.dup2(saved_stderr, stderr_fd)
    sys.stderr = sys.__stderr__

# Parse WORKER_GRID log
with open(tmp_name, 'r') as f:
    stderr_output = f.read()

# Extract compute_cores from "WORKER_GRID: strategy=... cores=..."
worker_grid_match = re.search(r'WORKER_GRID:\\s*strategy=(\\S+)\\s+cores=(\\d+)', stderr_output)
if worker_grid_match:
    compute_cores = int(worker_grid_match.group(2))
```

**Key Point**: `compute_cores` is now independent of output tensor's sharding and reflects the actual compute grid used by the operation.

---

### 3. Printing Tensor Sharding Information

**Problem**: User wanted to print tensor spec details including sharding strategy and core grid.

**Solution**: Added code to access and print memory config details.

#### Implementation

Location: `tests/ttnn/unit_tests/operations/eltwise/test_binary_bcast.py` (line 4030)

```python
mem_config = out_tt_sharded.memory_config()
print(f"Memory Layout (Sharding Strategy): {mem_config.memory_layout}")
if mem_config.shard_spec is not None:
    shard_spec = mem_config.shard_spec
    print(f"  Shard Spec - Grid: {shard_spec.grid}, Shape: {shard_spec.shape}, Orientation: {shard_spec.orientation}")
if mem_config.nd_shard_spec is not None:
    nd_shard_spec = mem_config.nd_shard_spec
    print(f"  ND Shard Spec - Grid: {nd_shard_spec.grid}, Shard Shape: {nd_shard_spec.shard_shape}, Orientation: {nd_shard_spec.orientation}, Distribution Strategy: {nd_shard_spec.shard_distribution_strategy}")
```

**Key API Details**:
- `tensor.memory_config()` - method call (with parentheses)
- `mem_config.memory_layout` - property (no parentheses)
- `mem_config.shard_spec` - property (no parentheses)
- `mem_config.nd_shard_spec` - property (no parentheses)

---

## Performance Analysis from CSV Data

### Comparison: max_ab vs max_abc Strategies

Analyzed two CSV files with 108 test configurations each:
- `example_multiple_ops_max_ab_20251112_060626.csv`
- `example_multiple_ops_max_abc_20251112_060000.csv`

#### Key Findings:

1. **Overall Performance**: Nearly identical
   - Mean difference: -0.41 μs (-0.59%) — max_ab slightly faster
   - Median difference: -0.25 μs (-0.42%)
   - 54.6% cases: max_ab faster
   - 42.6% cases: max_abc faster

2. **Pattern by C Sharding Type**:
   - **Height sharding**: max_ab is 3.24% faster on average
   - **Width sharding**: Nearly identical (0.09% difference)
   - **Interleaved**: max_ab is 0.41% faster on average

3. **Key Insight**: When C's core count differs between strategies, max_ab can be up to 16.4% faster, suggesting that compute grid selection matters more than output sharding strategy for performance.

---

## Counter-Intuitive Finding

**Observation**: When `a=interleaved`, `b=sharded`, `c=interleaved`, the kernel is fastest despite tensors being in DRAM.

**Explanation**:
1. **Compute grid selection**: Uses `b`'s sharded grid (e.g., 32 cores) → maximum parallelism
2. **Efficient data movement**:
   - Each core reads its portion of interleaved `a` directly from DRAM
   - Each core already has its shard of `b` in L1
   - Result written to DRAM interleaved (simple, no complex sharding overhead)
3. **No resharding overhead**: Avoids expensive tensor layout conversions
4. **Memory bandwidth**: DRAM reads are distributed across all cores simultaneously

This is faster than:
- Both sharded → requires careful shard alignment, potential resharding
- Output sharded → adds complexity of sharding the output correctly

---

## Files Modified

1. **`tests/ttnn/benchmarks/binary_ng/example_single_test.py`**
   - Added `compute_valid_block_grid()` function
   - Updated `create_sharded_tensor()` for block sharding
   - Added early filtering for invalid block configs
   - Implemented stderr capture for WORKER_GRID logs
   - Removed unused test functions (simplified)
   - Added operation type parameterization

2. **`tests/ttnn/unit_tests/operations/eltwise/test_binary_bcast.py`**
   - Fixed: `tensor_spec` → `spec`
   - Added: Detailed sharding info printing

3. **New Documentation Files**:
   - `tests/ttnn/benchmarks/binary_ng/BLOCK_SHARDING_FIX.md`
   - `tests/ttnn/benchmarks/binary_ng/SESSION_EXPORT.md` (this file)

---

## Current Test Status

### Latest CSV Results

File: `tests/ttnn/benchmarks/binary_ng/results/example_multiple_ops_max_ab_20251113_011040.csv`

- **Total configurations**: 142
- **All tests passing**: ✅ (no errors in error column)
- **Block sharding**: Working correctly with shape-aware grids
- **Compute cores**: Correctly tracked from C++ WORKER_GRID logs

### Sample Results:

| a_shape | a_sharding | a_cores | b_shape | b_sharding | b_cores | compute_cores | kernel_time_us |
|---------|------------|---------|---------|------------|---------|---------------|----------------|
| 1024×1024 | height | 8 | 1×1024 | width | 8 | 8 | 71.426 |
| 1024×1024 | height | 8 | 1×1024 | block | 8 | 8 | 73.373 |
| 1024×1024 | interleaved | 0 | 1×1024 | block | 32 | 32 | 56.433 |
| 1×1024 | block | 16 | 1024×1024 | block | 16 | 16 | 63.664 |

All configurations now run successfully with correct compute_cores values.

---

## How to Run Tests

### Run the benchmark test:
```bash
TT_METAL_DEVICE_PROFILER=1 pytest tests/ttnn/benchmarks/binary_ng/example_single_test.py::test_multiple_operations_with_timing -v -s
```

### Run the unit test with sharding info:
```bash
pytest tests/ttnn/unit_tests/operations/eltwise/test_binary_bcast.py::test_binary_sharded_output_spec -xvs
```

### Test block grid computation:
```python
from example_single_test import compute_valid_block_grid

# Examples
compute_valid_block_grid((1, 1024), 8)   # Returns: (2, 4)
compute_valid_block_grid((1024, 1), 16)  # Returns: (4, 4)
compute_valid_block_grid((1024, 1024), 32) # Returns: (4, 8)
```

---

## Key Technical Details

### Block Sharding Constraints

For tensor shape `(H, W)` with `N` cores:

1. **Grid must use all cores**: `grid_h × grid_w = N`
2. **Height divisibility**: `H_padded % grid_h = 0` (H padded to 32)
3. **Width divisibility**: `W_padded % grid_w = 0` (W padded to 32)
4. **Grid preference**: Most square (minimizes `|grid_h - grid_w|`)

### Grid Selection Strategies (C++ Implementation)

Located: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_device_operation.cpp`

- **max_abc**: Uses max_ab logic (ignores C)
- **max_ab**: Uses `max(A_cores, B_cores)`
- **current**: Prefers C, then A, then B, then full device grid
- **a_first**: Prefers A, then B, then C, then full device grid
- **b_first**: Prefers B, then A, then C, then full device grid
- **full_grid**: Always uses full device grid

### Python API for Tensor Sharding Info

```python
# Memory config (method)
mem_config = tensor.memory_config()

# Properties (no parentheses)
mem_config.memory_layout        # TensorMemoryLayout enum
mem_config.shard_spec          # ShardSpec or None
mem_config.nd_shard_spec       # NdShardSpec or None

# ShardSpec properties
shard_spec.grid                # CoreRangeSet
shard_spec.shape              # Shard shape tuple
shard_spec.orientation        # ShardOrientation enum

# NdShardSpec properties
nd_shard_spec.grid                      # CoreRangeSet
nd_shard_spec.shard_shape              # Shard shape
nd_shard_spec.orientation              # ShardOrientation enum
nd_shard_spec.shard_distribution_strategy  # Distribution strategy enum
```

---

## Remaining Work / Next Steps

1. **Test other operations**: Currently only ADD is tested, can add POWER, LOGEXP, etc.
2. **Test more shape combinations**: Current test uses (1024, 1024) and (1, 1024)
3. **Analyze performance patterns**: Use the CSV data to identify optimal sharding strategies
4. **Document best practices**: Create guidelines for when to use which sharding strategy
5. **Validate C++ grid selection**: Compare actual compute grids with expected strategies

---

## Important Code Locations

### Benchmark Test
- **Main test**: `tests/ttnn/benchmarks/binary_ng/example_single_test.py::test_multiple_operations_with_timing`
- **Block grid computation**: Lines 142-184
- **Create sharded tensor**: Lines 187-265
- **Test config generation**: Lines 770-852
- **Operation execution**: Lines 858-928

### C++ Implementation
- **Grid selection**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_device_operation.cpp`
- **get_worker_grid()**: Lines 67-90 (approximate)
- **WORKER_GRID log**: Written to stderr with fprintf

### Python Bindings
- **Tensor properties**: `ttnn/cpp/ttnn-pybind/pytensor.cpp`
- **Memory config**: `ttnn/cpp/ttnn-pybind/tensor.cpp`

---

## Debug Tips

### Enable C++ logging:
```bash
export TT_METAL_LOGGER_LEVEL=Debug
```

### Enable device profiling:
```bash
export TT_METAL_DEVICE_PROFILER=1
```

### View WORKER_GRID logs:
The logs appear in stderr during operation execution and show:
```
WORKER_GRID: strategy=max_ab cores=32
```

### View profiler results:
```bash
cat /workspace/generated/profiler/reports/ops_perf_results.csv
```

---

## Summary

This session successfully:
1. ✅ Fixed block sharding grid validation with shape-aware computation
2. ✅ Implemented compute cores tracking from C++ WORKER_GRID logs
3. ✅ Added comprehensive sharding info printing capabilities
4. ✅ Analyzed performance differences between grid strategies
5. ✅ Cleaned up benchmark test structure
6. ✅ Generated clean CSV results without errors

The benchmark test now correctly handles all sharding configurations and provides accurate performance data for analysis.

---

## Contact / Continuation

To continue this work:
1. Read this export document
2. Review the key changes in `example_single_test.py`
3. Check latest CSV: `results/example_multiple_ops_max_ab_20251113_011040.csv`
4. Run tests to verify everything still works
5. Proceed with next steps listed above

All code changes are committed and working as of November 13, 2025.
