# LEFT_SHIFT (binary_ng) Implementation Analysis

## Overview

The LEFT_SHIFT operation performs element-wise bitwise left shift (`a << b`) on two integer tensors using the SFPU (Special Function Processing Unit). It is implemented through the `binary_ng` program factory, which is a generalized framework for binary element-wise operations in TTNN. LEFT_SHIFT is an SFPU-only operation -- it has no FPU path and will throw an error if the FPU path is attempted.

**Program factory path**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_program_factory.cpp`

## Path Selection: FPU vs SFPU

The `binary_ng` program factory supports both FPU and SFPU execution paths, selected via the `is_sfpu` flag in `operation_attributes_t`. This flag is determined by the `utils::is_binary_sfpu_op()` function in `binary_ng_device_operation.cpp` (line 15), which examines the `BinaryOpType` and input data types.

For `LEFT_SHIFT`, the SFPU path is **always** selected when the input data types satisfy:
- `a` is INT32, UINT32, or UINT16, **and**
- `b` is INT32, UINT32, or UINT16

If these conditions are not met, the operation will not be classified as SFPU and the FPU path in `OpConfig` will throw `TT_THROW("Unsupported binary op for FPU")` (line 263 in `binary_ng_utils.cpp`). Thus, LEFT_SHIFT is exclusively an SFPU operation.

Inside `ProgramFactory::create`, the `is_sfpu` flag controls:
1. Which compute kernel file is selected via `get_kernel_file_path()` (SFPU variants like `eltwise_binary_sfpu_no_bcast.cpp` vs FPU variants like `eltwise_binary_no_bcast.cpp`)
2. The `OpConfig` construction: `OpConfig(op_type, std::in_place_type<OpConfig::SfpuBinaryOp>, a_dtype)` for SFPU
3. The `UnpackToDestMode` settings: SFPU ops (except POWER) use `UnpackToDestFp32` for all source CBs
4. The `b_dtype` when scalar: SFPU uses `a_dtype` for the scalar's inferred dtype, whereas FPU uses `DataType::BFLOAT16`

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Tile (32x32 elements) |
| **Unit size** | 1 tile |
| **Total units** | `c.physical_volume() / (tile_height * tile_width)` -- total output tiles |
| **Loop structure** | Each core processes its assigned chunk of output tiles in a flat loop. The compute kernel processes 1 tile per cycle (`num_tiles_per_cycle = 1`). |

## Tensor Format and Layout

### Input Tensors

| Property | Input Tensor A | Input Tensor B |
|----------|---------------|---------------|
| **Logical shape** | Arbitrary (up to rank 6+) | Arbitrary (broadcastable to A) |
| **Dimension convention** | [..., D, N, C, H, W] (last 5 dims explicit) | [..., D, N, C, H, W] |
| **Tensor layout** | TILE_LAYOUT | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED or SHARDED | INTERLEAVED or SHARDED |
| **Buffer type** | DRAM or L1 | DRAM or L1 |
| **Data type** | INT32, UINT32, or UINT16 | INT32, UINT32, or UINT16 |

### Output Tensor

| Property | Output Tensor C |
|----------|----------------|
| **Logical shape** | Broadcast of A and B shapes |
| **Dimension convention** | [..., D, N, C, H, W] |
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED or SHARDED |
| **Buffer type** | DRAM or L1 |
| **Data type** | Same as input (or user-specified) |

### Layout Transformations

No tilize/untilize conversions occur within the operation. Both inputs must already be in TILE_LAYOUT. The operation operates directly on tiles in their native tiled format.

## Data Flow Pattern

The data flow varies depending on whether operand B is a tensor or a scalar. Two main paths exist:

### Two-Tensor Path (B is a tensor)

1. **Reader** reads tiles of both A and B from DRAM/L1 into CB c_0 and CB c_1 respectively, one tile at a time per input. For sharded inputs, the reader simply does `cb_reserve_back` / `cb_push_back` to expose the pre-existing L1 data.
2. **Compute** waits for one tile each in CB c_0 (LHS) and CB c_1 (RHS), copies both to DEST registers, executes `binary_left_shift_tile`, packs the result into CB c_2.
3. **Writer** waits for one tile in CB c_2, writes it to DRAM/L1 output buffer. For sharded output, no write is needed (data already in L1).

### Scalar Path (B is a scalar)

1. **Writer** fills a single tile in CB c_1 with the packed scalar value (once, before the main loop), then writes output tiles from CB c_2 to DRAM.
2. **Reader** reads tiles of A from DRAM/L1 into CB c_0.
3. **Compute** processes each A tile against the single scalar tile in CB c_1.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_src_a | Input A staging | 2 tiles (interleaved) or shard volume (sharded) | 1 tile | Double (interleaved) / Single (sharded) | Reader | Compute | Program |
| c_1 | cb_src_b | Input B staging | 2 tiles (interleaved, tensor B) or 1 tile (scalar B) or shard volume (sharded) | 1 tile | Double (interleaved tensor) / Single (scalar or sharded) | Reader or Writer (scalar) | Compute | Program |
| c_2 | cb_out | Output staging | 2 tiles (interleaved) or shard volume (sharded) | 1 tile | Double (interleaved) / Single (sharded) | Compute | Writer | Program |
| c_3 | cb_lhs_intermediate | LHS activation intermediate | 1 tile | 1 tile | Single | Compute | Compute | Block |
| c_4 | cb_rhs_intermediate | RHS activation intermediate | 1 tile | 1 tile | Single | Compute | Compute | Block |

**Notes**: CB c_3 and c_4 are only created if LHS or RHS activations are defined (i.e., `PROCESS_LHS_ACTIVATIONS(i)` or `PROCESS_RHS_ACTIVATIONS(i)` are non-empty). For plain LEFT_SHIFT without pre/post activations, only c_0, c_1, and c_2 are used. CB c_5 and c_6 are only created for ROW_A/ROW_B broadcast types.

## Pipeline Pattern Summary

- **Interleaved mode**: CB c_0 and c_1 have capacity 2 tiles with block size 1, enabling **double-buffered** overlap between reader and compute. CB c_2 similarly allows double-buffered overlap between compute and writer.
- **Sharded mode**: CBs are sized to the full shard volume, operating as **single-buffered** since all data is pre-loaded.
- **Scalar mode**: CB c_1 holds 1 tile (the scalar), loaded once. Effectively single-buffered for this input.

## Index Calculations

The reader kernel computes per-tile offsets using a stride-based decomposition of the 5D+ tensor shape:

```
tile_offset = start_nd * nD_stride + start_d * d_stride + start_n * n_stride + start_c * c_stride + start_th * Wt
```

Where strides for input A are computed as:
- `nD_stride = aHt * aWt * aC * aN * aD * (aND > 1)` -- stride across collapsed dims >5 (zero if dim is 1, enabling broadcast)
- `d_stride = aHt * aWt * aC * aN * (aD > 1)`
- `n_stride = aHt * aWt * aC * (aN > 1)`
- `c_stride = aHt * aWt * (aC > 1)`

The `(dim > 1)` multiplier implements broadcasting: when a dimension is 1, the stride becomes 0, causing the same data to be re-read for each iteration along that dimension.

The `start_tile_id` for each core is computed from the output tile space. The reader decomposes this into ND/D/N/C/H/W coordinates to compute the correct input tile offset with broadcasting.

The TensorAccessor utility is used for compile-time args to configure the DRAM bank mapping.

## Memory Access Patterns

### Read Pattern
- **Interleaved**: Tiles are read one at a time via `noc_async_read_page()` with an immediate barrier. The access pattern follows the output tile ordering (row-major within each tile-row, progressing through H, C, N, D, ND dimensions). Broadcasting causes repeated reads of the same source tile when a dimension stride is 0.
- **Sharded**: No reads -- data is already in L1. The reader kernel simply exposes the shard via `cb_reserve_back` / `cb_push_back`.

### Write Pattern
- **Interleaved**: Tiles are written one at a time via `noc_async_write_page()` with an immediate barrier. The ordering follows the same dimensional decomposition as the reader.
- **Sharded**: No writes -- output data is written directly to the sharded CB backed by the output buffer.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 2D (when zero_start_grid) or arbitrary CoreRangeSet |
| **Grid dimensions** | Device compute grid (e.g., 8x8) or shard grid for sharded |
| **Total cores** | `compute_with_storage_grid.x * compute_with_storage_grid.y` (interleaved) or shard grid core count (sharded) |
| **Work per core** | `num_tiles_per_core_group_1` or `num_tiles_per_core_group_2` (interleaved); shard volume (sharded) |
| **Load balancing** | Two-group split: group 1 gets `ceil(total_tiles / num_cores)` tiles, group 2 gets `floor(...)`. Cores outside both groups receive zero-args (no-op). |

The `split_work_to_cores()` utility divides the total output tiles across available cores. For sharded tensors, each core processes exactly its shard's tiles. Cores not in any core group are given zero runtime args and skip execution.

## Arguments

### Compile-Time Arguments

**Compute kernel**:

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | num_tiles_per_cycle | uint32_t | Always 1 -- tiles produced per read-compute-write cycle |

**Reader kernel** (via TensorAccessorArgs):

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0..N | TensorAccessor A args | uint32_t[] | Bank mapping for tensor A |
| N+1..M | TensorAccessor B args | uint32_t[] | Bank mapping for tensor B |
| M+1 | has_sharding | uint32_t | 1 if native L1 sharding is active |

**Writer kernel** (via TensorAccessorArgs):

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0..N | TensorAccessor C args | uint32_t[] | Bank mapping for output tensor C |
| N+1 | has_sharding | uint32_t | 1 if native L1 sharding is active |

### Runtime Arguments

**Reader kernel** (two-tensor path, `ReaderNoBcastNg`):

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Base address of tensor A buffer |
| 1 | c_start_id | uint32_t | Starting output tile ID for this core |
| 2 | a_num_tiles | uint32_t | Number of A shard tiles (sharded) or 0 |
| 3 | c_num_tiles | uint32_t | Number of output tiles for this core |
| 4 | c_current_shard_width | uint32_t | Shard width in tiles (sharded) or 0 |
| 5 | nD_stride | uint32_t | A stride for collapsed dims >5 |
| 6 | d_stride | uint32_t | A stride for D dimension |
| 7 | n_stride | uint32_t | A stride for N dimension |
| 8 | c_stride | uint32_t | A stride for C dimension |
| 9 | cD | uint32_t | Output D dimension |
| 10 | cN | uint32_t | Output N dimension |
| 11 | cC | uint32_t | Output C dimension |
| 12 | cHt | uint32_t | Output H in tiles |
| 13 | cWt | uint32_t | Output W in tiles |
| 14 | cND | uint32_t | Output collapsed ND dimension |
| 15 | src_addr_b | uint32_t | Base address of tensor B buffer |
| 16 | nD_stride_b | uint32_t | B stride for collapsed dims >5 |
| 17 | d_stride_b | uint32_t | B stride for D dimension |
| 18 | n_stride_b | uint32_t | B stride for N dimension |
| 19 | c_stride_b | uint32_t | B stride for C dimension |
| 20 | b_num_tiles | uint32_t | Number of B shard tiles (sharded) or 0 |

**Writer kernel** (two-tensor path, `WriterNoBcastNg`):

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Base address of output buffer |
| 1 | start_tile_id | uint32_t | Starting output tile ID |
| 2 | dst_num_tiles | uint32_t | Number of tiles to write |
| 3 | dst_shard_width | uint32_t | Shard width in tiles (sharded) or 0 |
| 4 | cD | uint32_t | Output D dimension |
| 5 | cN | uint32_t | Output N dimension |
| 6 | cC | uint32_t | Output C dimension |
| 7 | cHt | uint32_t | Output H in tiles |
| 8 | cWt | uint32_t | Output W in tiles |
| 9 | cND | uint32_t | Output collapsed ND dimension |
| 10 | (unused) | uint32_t | Padding (always 0) |

**Compute kernel**:

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | num_tiles | uint32_t | Total tiles for this core to process |
| 1 | freq | uint32_t | Broadcast frequency (1 for NONE, Wt for COL, Ht*Wt for SCALAR) |
| 2 | counter | uint32_t | Starting counter within broadcast cycle |
| 3 | compute_scalar_value | uint32_t | 0 for non-quant ops |

## Kernel Implementations

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| Reader (`reader_interleaved_no_bcast.cpp` -- `kernels_ng/`) | BRISC (RISCV_0) | NOC0 | DRAM/L1 (A, B) | CB c_0, CB c_1 | Read A and B tiles with stride-based broadcast |
| Compute (`eltwise_binary_sfpu_no_bcast.cpp`) | Compute (RISCV_2) | N/A | CB c_0, CB c_1 | CB c_2 | SFPU `binary_left_shift_tile` |
| Writer (`writer_interleaved_no_bcast.cpp` -- `kernels_ng/`) | TRISC (RISCV_1) | NOC1 | CB c_2 | DRAM/L1 | Write output tiles |

### Reader Kernel (`kernels_ng/dataflow/reader_interleaved_no_bcast.cpp`)

| Property | Value |
|----------|-------|
| File | `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/dataflow/reader_interleaved_no_bcast.cpp` |
| Assigned cores | All worker cores in the grid |

**Key Logic**:
- Reads both input A (into CB c_0) and input B (into CB c_1) in a single kernel -- this is the "merged reader" pattern used by `binary_ng`.
- For **sharded** inputs, the kernel simply calls `cb_reserve_back(cb_id, num_tiles)` followed by `cb_push_back(cb_id, num_tiles)` to expose the pre-loaded shard data. No NoC reads occur.
- For **interleaved** inputs, uses a 6-level nested loop (ND, D, N, C, Ht, Wt) iterating over output tile coordinates. Input tile offsets are computed using stride-based addressing with separate strides for A and B, enabling broadcasting (stride = 0 when dimension is 1).
- Uses `TensorAccessor` for DRAM bank resolution and `noc_async_read_page()` for tile reads.
- Issues `noc_async_read_barrier()` after each pair of reads (A and B for the same output tile).
- **Synchronization**: Produces tiles into CB c_0 and CB c_1 via `cb_reserve_back` / `cb_push_back`. Does not consume from any CB.

### Compute Kernel (`kernels/compute/eltwise_binary_sfpu_no_bcast.cpp`)

| Property | Value |
|----------|-------|
| File | `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_sfpu_no_bcast.cpp` |
| Assigned cores | All worker cores in the grid |

**Key Logic**:
- Main loop iterates `num_tiles` times (runtime arg 0), processing 1 tile per iteration (`num_tiles_per_cycle = 1`).
- Supports optional pre-processing of LHS/RHS via `PREPROCESS(LHS/RHS, ...)` macros. For plain LEFT_SHIFT, these are no-ops.
- Each iteration:
  1. Waits for LHS tile in `cb_post_lhs` (c_0) and RHS tile in `cb_post_rhs` (c_1)
  2. Reserves output space in `cb_out` (c_2)
  3. Calls `BINARY_SFPU_INIT` which expands to `binary_shift_tile_init();`
  4. Acquires tile registers, copies LHS tile to DEST slot 0, copies RHS tile to DEST slot 1
  5. Executes `BINARY_SFPU_OP(0, 1, 0)` which expands to `binary_left_shift_tile<DataFormat::Int32>(0, 1, 0)` (or `UInt32`/`UInt16` depending on dtype)
  6. Optionally applies `PROCESS_POST_ACTIVATIONS` (no-op for plain LEFT_SHIFT)
  7. Commits registers, waits, packs result from DEST slot 0 into `cb_out`
  8. Releases registers, pushes to `cb_out`, pops from both input CBs
- **Synchronization**: Consumes from CB c_0 (`cb_wait_front` / `cb_pop_front`) and CB c_1 (`cb_wait_front` / `cb_pop_front`). Produces to CB c_2 (`cb_reserve_back` / `cb_push_back`).

### Writer Kernel (`kernels_ng/dataflow/writer_interleaved_no_bcast.cpp`)

| Property | Value |
|----------|-------|
| File | `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/dataflow/writer_interleaved_no_bcast.cpp` |
| Assigned cores | All worker cores in the grid |

**Key Logic**:
- For **sharded** output (`DST_SHARDED` define), the kernel is a no-op -- output is already in L1 via the sharded CB.
- For **interleaved** output, uses the same 6-level nested loop as the reader to iterate output tile coordinates.
- Each iteration: waits for one tile in CB c_2, gets the read pointer, writes via `noc_async_write_page()`, barriers, then pops.
- Handles sharding-aware tile offset adjustment: when `has_sharding` is true, adjusts `dst_tile_offset` by `(Wt - dst_shard_width)` per row to account for partial shard widths.
- **Synchronization**: Consumes from CB c_2 via `cb_wait_front` / `cb_pop_front`. Does not produce to any CB.

## Implementation Notes

- **Program factory variants**: There is a single `ProgramFactory` for all `binary_ng` operations. The factory internally selects between different reader/compute/writer kernel files based on `SubtileBroadcastType`, `is_sfpu`, and `is_where_op` flags.
- **Type-based operation variants**: LEFT_SHIFT supports INT32, UINT32, and UINT16 inputs. The SFPU function is templated on `DataFormat` (`binary_left_shift_tile<DataFormat::Int32>`, `<DataFormat::UInt32>`, or the default `Int32`). The specific format is determined at kernel compile time via the `BINARY_SFPU_OP` define.
- **UnpackToDestFP32 mode**: Enabled for all SFPU binary ops except POWER. For LEFT_SHIFT, all four source CB indices (c_0, c_1, c_3, c_4) use `UnpackToDestMode::UnpackToDestFp32`.
- **Broadcast type selection**: All `SubtileBroadcastType` variants are supported (NONE, SCALAR_A/B, ROW_A/B, COL_A/B, ROW_A_COL_B, ROW_B_COL_A). The broadcast type determines which reader, compute kernel variant, and stride configuration are used. Stride-based broadcasting is used for dimension-level broadcasts (dims > tile dimensions), while subtile broadcasting handles within-tile replication.
- **Sharding support and constraints**: Height, width, and block sharding are supported for native L1 sharding. Constraints include: (1) both inputs must have the same shape and memory config for native sharding, (2) no DRAM buffers, (3) no uneven shards on the output, (4) all shard grids must match. When constraints are not met, the operation falls back to interleaved (tensor accessor) mode.
- **FP32 dest accumulation**: Enabled when output format is UInt32, Int32, or Float32, or when both input formats are Float32, Int32, or UInt32. For LEFT_SHIFT with INT32 inputs and INT32 output, `fp32_dest_acc_en` is always true.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the binary_ng operation work in TTNN? What are its program factory variants, kernel types (SFPU vs FPU), and how does it select between them? What is the SubtileBroadcastType enum and how does it affect kernel selection?"
   **Reason**: Needed architectural overview of the binary_ng framework before diving into source code.
   **Key Findings**: Confirmed single ProgramFactory with SFPU/FPU selection via `is_sfpu` flag. `SubtileBroadcastType` drives reader/compute kernel selection. `BinaryNgKernelConfig` maps broadcast types to kernel names.

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_utils.cpp`
   **Reason**: Needed to understand `OpConfig` construction, SFPU init/op function mapping, and kernel file path resolution.
   **Key Information**: LEFT_SHIFT maps to `binary_shift_tile_init()` and `binary_left_shift_tile<DataFormat::Int32>`. The `get_kernel_file_path()` function selects SFPU kernel variants when `is_sfpu` is true.

2. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_device_operation.cpp`
   **Reason**: Needed to understand how `is_sfpu` flag is determined and what conditions enable the SFPU path for LEFT_SHIFT.
   **Key Information**: `is_binary_sfpu_op()` returns true for LEFT_SHIFT when both inputs are INT32, UINT32, or UINT16. The function also sets `SubtileBroadcastType` based on input tensor shapes.

3. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils_common.hpp`
   **Reason**: Needed to understand the macro framework used by compute kernels (PREPROCESS, HAS_ACTIVATIONS, BCAST_INPUT).
   **Key Information**: The macro system conditionally enables pre/post-processing of operands. `HAS_ACTIVATIONS` checks if a `PROCESS_*_ACTIVATIONS` macro expands to non-empty text.
