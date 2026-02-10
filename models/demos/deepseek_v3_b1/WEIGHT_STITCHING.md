# Weight Tensor Stitching for CB Optimization

## Overview

This document describes the `stitch_weight_tensors()` function that allows combining multiple weight tensors with different core grids into a single unified tensor, saving precious CB (Circular Buffer) indices in fused kernels.

## Problem Statement

When fusing multiple operations (e.g., multiple matmuls in pre-SDPA), each operation's weights typically require a separate CB. With a hard limit of 64 CBs per kernel, this becomes a bottleneck for complex fused operations.

**Challenge**: Different matmul operations run on different core grids, meaning their weight tensors have different shard specifications.

## Solution: 2D Spatial Concatenation

The `stitch_weight_tensors()` function concatenates multiple weight tensors along the width dimension, creating a single unified tensor where different column ranges correspond to different operations.

### Key Features

1. **Column-based layout**: Tensors are concatenated along width (N dimension)
2. **Handles overlapping grids**: Cores can participate in multiple matmuls
3. **Tile-aligned**: Ensures all dimensions are properly aligned to 32-element tiles
4. **Height padding**: Pads shorter tensors to match the maximum height
5. **Metadata generation**: Provides compile-time args for kernel access

## Function Signature

```python
def stitch_weight_tensors(
    weight_specs: list[dict],
    device: ttnn.Device,
    unified_grid: ttnn.CoreRangeSet,
    memory_layout: ttnn.TensorMemoryLayout = ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    dtype: ttnn.DataType = ttnn.bfloat8_b,
    tile: ttnn.Tile = None,
) -> tuple[ttnn.Tensor, dict]
```

### Parameters

- `weight_specs`: List of dicts, each containing:
  - `'tensor'`: torch.Tensor [K, N] on host
  - `'grid'`: ttnn.CoreRangeSet for this weight
  - `'name'`: Identifier string

- `device`: ttnn.Device instance
- `unified_grid`: Union of all participating core grids
- `memory_layout`: Sharding strategy (WIDTH_SHARDED recommended)
- `dtype`: Data type (bfloat8_b for quantized weights)
- `tile`: Tile descriptor (defaults to [1, 32])

### Returns

- `stitched_tensor`: ttnn.Tensor with shape [max_K, sum_of_Ns]
- `metadata`: Dict with column offsets and shapes for each weight

## Usage Example

```python
# Step 1: Define weight specifications
matmul1_grid = ttnn.CoreRangeSet({
    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 7))
})  # 6x8 = 48 cores

matmul2_grid = ttnn.CoreRangeSet({
    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 7))
})  # 12x8 = 96 cores

weight_specs = [
    {
        'tensor': torch_matmul1_weights,  # [7168, 1536]
        'grid': matmul1_grid,
        'name': 'matmul1',
    },
    {
        'tensor': torch_matmul2_weights,  # [1536, 12288]
        'grid': matmul2_grid,
        'name': 'matmul2',
    },
]

# Step 2: Create unified grid
unified_grid = matmul1_grid.merge(matmul2_grid)

# Step 3: Stitch tensors
stitched_weights, metadata = stitch_weight_tensors(
    weight_specs,
    device=device,
    unified_grid=unified_grid,
    memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    dtype=ttnn.bfloat8_b,
)

# Result: [8192, 13824] tensor
#   columns [0:1536]     -> matmul1 weights (tiles 0:48)
#   columns [1536:13824] -> matmul2 weights (tiles 48:432)
```

## Metadata Structure

```python
metadata = {
    'unified_shape': (8192, 14336),
    'total_width_tiles': 448,
    'shard_shape': (8192, 1195),  # Per-core shard
    'weights': {
        'matmul1': {
            'col_start': 0,
            'col_end': 1536,
            'col_start_tiles': 0,
            'width_tiles': 48,
            'original_shape': (7168, 1536),
            'padded_shape': (8192, 1536),
            'grid': matmul1_grid,
        },
        'matmul2': { ... },
    }
}
```

## Kernel Integration

### CB Descriptor Creation

```python
stitched_weights_cb_id = 3
stitched_weights_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
    stitched_weights_cb_id,
    stitched_weights
)
```

### Compile-Time Args

```python
# Generate args automatically
weight_compile_args = generate_stitched_weight_compile_args(metadata)

# Or manually:
compile_args = [
    ('stitched_weights_cb', stitched_weights_cb_id),
    ('matmul1_col_start_tiles', 0),
    ('matmul1_width_tiles', 48),
    ('matmul2_col_start_tiles', 48),
    ('matmul2_width_tiles', 384),
]
```

### C++ Kernel Access

```cpp
// Compile-time args
constexpr uint32_t stitched_weights_cb = get_compile_time_arg_val(X);
constexpr uint32_t matmul1_col_start_tiles = get_compile_time_arg_val(Y);
constexpr uint32_t matmul1_width_tiles = get_compile_time_arg_val(Z);

// In NCRISC reader
#if defined(is_matmul_core)
    // Read matmul1 weights from columns 0-1535
    for (uint32_t tile_x = matmul1_col_start_tiles;
         tile_x < matmul1_col_start_tiles + matmul1_width_tiles;
         ++tile_x) {

        cb_reserve_back(stitched_weights_cb, 1);

        uint32_t l1_write_addr = get_write_ptr(stitched_weights_cb);
        noc_async_read_tile(
            stitched_weights_cb_addr + tile_x * tile_size,
            l1_write_addr,
            tile_size
        );
        noc_async_read_barrier();

        cb_push_back(stitched_weights_cb, 1);
    }
#endif

#if defined(is_matmul2_core)
    // Read matmul2 weights from columns 1536-13823
    for (uint32_t tile_x = matmul2_col_start_tiles;
         tile_x < matmul2_col_start_tiles + matmul2_width_tiles;
         ++tile_x) {
        // Similar read logic
    }
#endif
```

## Design Rationale

### Why 2D Instead of 3D?

While a 3D tensor `[num_tensors, K, N]` seems natural, it has drawbacks:
- CB infrastructure may not fully support 3D tensors
- More complex addressing in kernels
- Unclear memory layout benefits

The 2D approach:
- ✓ Works with standard CB operations
- ✓ Simple column-based indexing
- ✓ Natural width-sharding semantics
- ✓ Handles overlapping grids seamlessly

### Why Width Concatenation?

For matmul weights `[K, N]`:
- **K dimension** (height): Usually large (e.g., 7168)
- **N dimension** (width): Varies per matmul (e.g., 1536, 12288)

WIDTH_SHARDED distributes columns across cores, which aligns naturally with:
- Each core computing a subset of output columns
- Weight access patterns in matmul (each core reads its N-range)

### Overlapping Grids

Cores can be in multiple grids (e.g., matmul1_grid ⊆ matmul2_grid):
- Core (0,0) participates in both matmul1 and matmul2
- The stitched tensor is visible to all cores in unified_grid
- Compile-time flags (`is_matmul_core`, `is_matmul2_core`) control which columns each core reads

## Benefits

1. **CB savings**: Use 1 CB instead of N CBs (saves N-1 indices)
2. **Overlapping grid support**: Cores can participate in multiple operations
3. **Simple addressing**: Column-based access via compile-time offsets
4. **Standard infrastructure**: Works with existing CB and sharding mechanisms
5. **Tile-aligned**: Ensures proper alignment for hardware requirements

## Limitations

1. **Memory overhead**: Max height padding means shorter tensors waste space
2. **Same data type required**: All weights must use same dtype (e.g., bfp8)
3. **Same tile size required**: All weights must use compatible tiles
4. **Width-only concatenation**: Only concatenates along one dimension

## Future Enhancements

- Support for block-based layouts to reduce padding waste
- Compression for padded regions
- Automatic dtype conversion if needed
- Support for 3D weight tensors (e.g., batched matmuls)

## Testing

Validated with test case:
- matmul1: [7168, 1536] on 48 cores
- matmul2: [1536, 12288] on 96 cores
- matmul3: [8192, 512] on 64 cores

Result: [8192, 14336] stitched tensor with all data integrity preserved.
