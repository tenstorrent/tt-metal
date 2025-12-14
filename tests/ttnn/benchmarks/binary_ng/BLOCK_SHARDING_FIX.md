# Block Sharding Grid Validation Fix

## Problem

The benchmark test was generating many errors for block-sharded tensors because block sharding requires a **2D grid** that is compatible with the tensor shape. The previous implementation used hardcoded grids that didn't consider the tensor shape constraints.

## Block Sharding Constraints

For block sharding with shape `(H, W)` and `N` cores, the grid `(grid_h, grid_w)` must satisfy:

1. **Grid must use all cores**: `grid_h * grid_w = N`
2. **Height divisibility**: `H_padded % grid_h = 0` (where H_padded is padded to tile size 32)
3. **Width divisibility**: `W_padded % grid_w = 0` (where W_padded is padded to tile size 32)

### Examples

#### Tensor (1, 1024) - height=1, width=1024
- After padding: (32, 1024)
- **8 cores**: Valid grids: (1,8), (2,4), (4,2)
  - Selected: **(2,4)** (most square)
  - Invalid: (8,1) because 32 % 8 ≠ 0
- **16 cores**: Valid grids: (1,16), (2,8), (4,4), (16,1)
  - Selected: **(4,4)** (most square)
- **32 cores**: Valid grids: (1,32), (2,16), (4,8), (8,4), (16,2), (32,1)
  - Selected: **(4,8)** (most square)

#### Tensor (1024, 1) - height=1024, width=1
- After padding: (1024, 32)
- **8 cores**: Valid grids: (1,8), (2,4), (4,2), (8,1)
  - Selected: **(2,4)** (most square)
- Same pattern for 16 and 32 cores

#### Tensor (1024, 1024) - square tensor
- After padding: (1024, 1024)
- All grid configurations work (many more valid options)
- Always selects the most square grid for better load balancing

## Solution

### 1. Added `compute_valid_block_grid()` function

```python
def compute_valid_block_grid(shape, cores):
    """
    Compute a valid block sharding grid for the given shape and cores.

    Returns:
        Tuple (grid_h, grid_w) or None if no valid grid exists
    """
```

This function:
- Finds all divisor pairs of cores
- Filters pairs that satisfy the divisibility constraints
- Prefers grids closer to square (minimizes `|grid_h - grid_w|`)

### 2. Updated `create_sharded_tensor()` for block sharding

Changed from hardcoded grids:
```python
# OLD - hardcoded
if cores == 8:
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))})
```

To shape-aware grid computation:
```python
# NEW - shape-aware
grid_shape = compute_valid_block_grid(shape, cores)
if grid_shape is None:
    raise ValueError(f"Cannot create valid block sharding grid for shape {shape} with {cores} cores")

grid_h, grid_w = grid_shape
grid = ttnn.CoreRangeSet({ttnn.CoreRange(
    ttnn.CoreCoord(0, 0),
    ttnn.CoreCoord(grid_w - 1, grid_h - 1)
)})
```

### 3. Added early filtering in test configuration generation

Filter out invalid block sharding configurations **before** creating tensors:

```python
# Filter: block sharding needs a valid grid based on tensor shape
if a_sharding == "block" and a_cores is not None:
    if compute_valid_block_grid(shape_a, a_cores) is None:
        continue  # Skip: no valid block grid for this shape/cores combination
if b_sharding == "block" and b_cores is not None:
    if compute_valid_block_grid(shape_b, b_cores) is None:
        continue  # Skip: no valid block grid for this shape/cores combination
```

### 4. Fixed height sharding validation

Also enabled the filter for height sharding with height=1:

```python
# Skip invalid combinations: height sharding when height=1
if b_sharding == "height" and shape_b[0] == 1:
    continue  # Skip: can't height-shard a tensor with height=1
```

## Results

- **Before**: Many "ValueError: BLOCK needs 16+ cores" errors and tensor creation failures
- **After**: Only valid block sharding configurations are tested
- **Benefit**: Cleaner CSV results, faster test execution, no spurious errors

## Key Insight

Block sharding is fundamentally different from height/width sharding:
- **Height/Width sharding**: 1D distribution (fixed grid shape per core count)
- **Block sharding**: 2D distribution (grid must be computed based on tensor shape)

For tensors with one dimension equal to 1:
- The grid is **highly constrained** in one direction
- Example: (1, 1024) → grid must be mostly horizontal like (2, 4) not vertical like (8, 1)
- Example: (1024, 1) → grid must be mostly vertical

## Testing

Run the benchmark test to verify:
```bash
TT_METAL_DEVICE_PROFILER=1 pytest tests/ttnn/benchmarks/binary_ng/example_single_test.py::test_multiple_operations_with_grid_strategy -v -s
```

The test will now:
1. Skip invalid block sharding configurations upfront
2. Print the selected grid for each block-sharded tensor
3. Generate CSV without block sharding errors
