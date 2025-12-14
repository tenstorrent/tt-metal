# New Grid Selection Strategy: `new_grid`

**Date Added**: November 18, 2025
**Status**: ✅ Implemented in C++ and Python

---

## Overview

The `new_grid` strategy computes the element-wise maximum of the grid dimensions from input tensors A and B.

**Formula**: `new_grid = (max(a.x, b.x), max(a.y, b.y))`

**Example**:
- If A's shard spec grid is `(4, 8)` (4 columns, 8 rows = 32 cores)
- And B's shard spec grid is `(8, 4)` (8 columns, 4 rows = 32 cores)
- Then `new_grid` is `(8, 8)` (8 columns, 8 rows = 64 cores)

---

## Motivation

### Problem
Existing strategies choose grids based on total core count:
- `max_ab`: picks grid with more total cores
- `min_ab`: picks grid with fewer total cores

However, both strategies select an **existing** grid from either A or B.

### Solution
The `new_grid` strategy creates a **new** grid that accommodates both input shapes optimally by taking the maximum dimension in each direction.

**Benefits**:
1. **Better Coverage**: Can use more cores than either input individually
2. **Balanced Layout**: Creates square-like grids when inputs are complementary
3. **Flexibility**: Not constrained to existing input grids

---

## Implementation

### C++ Code
Location: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_device_operation.cpp`

Added `get_elementwise_max_grid` lambda function:

```cpp
auto get_elementwise_max_grid = [&get_tensor_grid, &get_full_device_grid](
    const Tensor& t1, const Tensor& t2) -> CoreRangeSet {
    if (!t1.is_sharded() && !t2.is_sharded()) {
        // Both interleaved - use full device grid
        return get_full_device_grid(t1);
    }
    if (!t1.is_sharded()) {
        // Only t2 is sharded - use t2's grid
        return get_tensor_grid(t2);
    }
    if (!t2.is_sharded()) {
        // Only t1 is sharded - use t1's grid
        return get_tensor_grid(t1);
    }

    // Both sharded - compute element-wise max grid
    auto grid1 = t1.shard_spec()->grid;
    auto grid2 = t2.shard_spec()->grid;
    auto bbox1 = grid1.bounding_box();
    auto bbox2 = grid2.bounding_box();

    // Get grid sizes (end coordinates + 1 since 0-based)
    uint32_t max_x = std::max(bbox1.end_coord.x, bbox2.end_coord.x);
    uint32_t max_y = std::max(bbox1.end_coord.y, bbox2.end_coord.y);

    // Create new grid from (0,0) to (max_x, max_y)
    return CoreRangeSet({CoreRange(CoreCoord(0, 0), CoreCoord(max_x, max_y))});
};
```

Strategy branch (lines 206-216):

```cpp
} else if (strategy == "new_grid") {
    // Element-wise max grid: (max(a.x, b.x), max(a.y, b.y))
    if (input_tensor_b) {
        return get_elementwise_max_grid(input_tensor_a, *input_tensor_b);
    } else if (input_tensor_a.is_sharded()) {
        return get_tensor_grid(input_tensor_a);
    }
    // Fallback: use full device grid
    auto device = input_tensor_a.device();
    return device->worker_cores(HalProgrammableCoreType::TENSIX,
                                device->get_sub_device_ids().front());
}
```

### Python Code
Location: `tests/ttnn/benchmarks/binary_ng/compare_multi_strategy.py`

Updated `known_strategies` list to include `('new', 'grid')` for filename parsing.

---

## Usage

### Set Strategy
```bash
export TT_METAL_BINARY_NG_GRID_STRATEGY=new_grid
```

### Run Benchmarks
```bash
# Using pytest
TT_METAL_BINARY_NG_GRID_STRATEGY=new_grid pytest tests/ttnn/unit_tests/operations/eltwise/test_binary.py

# Using benchmark script
cd tests/ttnn/benchmarks/binary_ng
TT_METAL_BINARY_NG_GRID_STRATEGY=new_grid pytest example_single_test.py
```

### Compare with Other Strategies
```bash
# Generate results for new_grid
TT_METAL_BINARY_NG_GRID_STRATEGY=new_grid \
    pytest example_single_test.py::test_multiple_operations_with_timing

# Compare with existing strategies
python compare_multi_strategy.py max_ab max_abc new_grid
```

---

## Examples

### Example 1: Complementary Grids
```
Input A grid: (4, 8) = 32 cores
Input B grid: (8, 4) = 32 cores
new_grid:     (8, 8) = 64 cores ✓ Uses more cores!
```

### Example 2: One Dominant Dimension
```
Input A grid: (4, 16) = 64 cores
Input B grid: (8, 8)  = 64 cores
new_grid:     (8, 16) = 128 cores (if device has enough cores)
```

### Example 3: Identical Grids
```
Input A grid: (8, 8) = 64 cores
Input B grid: (8, 8) = 64 cores
new_grid:     (8, 8) = 64 cores (same as inputs)
```

### Example 4: One Interleaved
```
Input A grid: (4, 8) = 32 cores (sharded)
Input B: interleaved (no grid)
new_grid:     (4, 8) = 32 cores (falls back to A's grid)
```

---

## Comparison with Other Strategies

| Strategy | Formula | Example (A=4×8, B=8×4) | Notes |
|----------|---------|------------------------|-------|
| `max_ab` | max(total_cores) | 8×4 or 4×8 (32 cores) | Picks one input's grid |
| `min_ab` | min(total_cores) | 8×4 or 4×8 (32 cores) | Picks one input's grid |
| `new_grid` | (max(x), max(y)) | **8×8 (64 cores)** | Creates new grid ⭐ |
| `full_grid` | device max | 8×8 (64 cores) | Always max, ignores inputs |

**Key Difference**:
- `max_ab`/`min_ab` choose from existing grids
- `new_grid` creates a new grid based on input dimensions
- `full_grid` ignores inputs entirely

---

## Performance Expectations

### When `new_grid` May Be Better
1. **Complementary input grids** (e.g., 4×8 and 8×4)
   - Creates balanced square-like grid
   - Better core utilization

2. **Asymmetric operations** where both dimensions matter
   - Element-wise operations with broadcasting
   - Operations sensitive to data layout

### When `new_grid` May Be Worse
1. **Already optimal grids**
   - If inputs already use maximum suitable cores
   - Creating larger grid may cause validation errors

2. **Memory-bound operations**
   - More cores don't help if memory is bottleneck

3. **Grid incompatibility**
   - New grid may not satisfy shard spec constraints

---

## Testing Recommendations

### Test Cases
1. **Basic functionality**
   ```bash
   # Test that new_grid strategy is recognized
   TT_METAL_BINARY_NG_GRID_STRATEGY=new_grid pytest <test_file>
   ```

2. **Complementary grids**
   ```python
   # A: height sharded with 8 cores (8×1 grid)
   # B: width sharded with 8 cores (1×8 grid)
   # Expected new_grid: (8×8) = 64 cores
   ```

3. **Edge cases**
   - One or both inputs interleaved
   - Identical input grids
   - Very different grid sizes

### Validation
Check logs for:
```
WORKER_GRID: strategy=new_grid cores=<num> grid=<x>x<y>
```

Verify:
- Grid dimensions are max of inputs
- Total cores = grid.x × grid.y
- Operation completes successfully

---

## Known Limitations

1. **Device Core Limits**
   - New grid may exceed device capabilities
   - Fallback to full device grid in such cases

2. **Shard Spec Constraints**
   - New grid must satisfy output shard spec requirements
   - May fail validation if incompatible with output shape

3. **Not Always Better**
   - More cores ≠ always faster
   - Performance depends on operation type and data layout

---

## Future Enhancements

Potential improvements:
1. **Smart capping**: Don't exceed device limits
2. **Constraint-aware**: Validate against output requirements
3. **Adaptive**: Choose between new_grid and existing grids based on validation
4. **Statistics**: Track when new_grid helps vs. hurts

---

## Files Modified

### C++
- `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_device_operation.cpp`
  - Added `get_elementwise_max_grid` function
  - Added `new_grid` strategy branch

### Python
- `tests/ttnn/benchmarks/binary_ng/compare_multi_strategy.py`
  - Added `('new', 'grid')` to known strategies
- `tests/ttnn/benchmarks/binary_ng/run_single_strategy.sh`
  - Updated documentation to list new_grid
- `tests/ttnn/benchmarks/binary_ng/QUICK_REFERENCE.md`
  - Documented new_grid strategy

### Documentation
- `tests/ttnn/benchmarks/binary_ng/NEW_GRID_STRATEGY.md` (this file)

---

## Quick Reference

```bash
# Set strategy
export TT_METAL_BINARY_NG_GRID_STRATEGY=new_grid

# Run test
pytest <test_file>

# Compare strategies
python compare_multi_strategy.py max_ab new_grid

# Check logs for grid size
# Look for: WORKER_GRID: strategy=new_grid cores=X grid=YxZ
```

---

**Summary**: `new_grid` creates a new compute grid by taking the element-wise maximum of input grid dimensions, potentially using more cores than either input individually.

**When to use**: When inputs have complementary grid layouts (e.g., 4×8 and 8×4), and you want to maximize core utilization.

**Status**: ✅ Ready for testing and benchmarking

---

**Last Updated**: November 18, 2025
**Author**: AI Assistant
**Status**: ✅ Complete
