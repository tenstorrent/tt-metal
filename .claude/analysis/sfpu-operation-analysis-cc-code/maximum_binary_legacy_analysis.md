# MAXIMUM (binary_legacy) Implementation Analysis

## Overview

The MAXIMUM operation computes the element-wise maximum of two input tensors: `y = max(a, b)`. It is implemented through the binary legacy SFPU program factory, which dispatches tile-level max comparisons on the SFPU (vector unit). Three data-type variants exist: floating-point (`binary_max_tile`), INT32 (`binary_max_int32_tile`), and UINT32 (`binary_max_uint32_tile`).

**Program factory path**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/element_wise_multi_core_sfpu_pgm_factory.cpp`

## Path Selection: FPU vs SFPU

The `BinaryDeviceOperation::select_program_factory` function (in `binary_device_operation.cpp`) decides between the FPU path (`ElementWiseMultiCore`) and the SFPU path (`ElementWiseMultiCoreSfpu`). When both input tensors have the same height and width (no broadcasting required), the function calls `utils::is_binary_sfpu_op(op, dtype1, dtype2)`. For `BinaryOpType::MAXIMUM`, this function unconditionally returns `true` regardless of the input data types — meaning MAXIMUM **always** routes to the SFPU program factory when tensor shapes are element-wise compatible.

If broadcasting is needed (height_b==1 or width_b==1), the operation is dispatched to a broadcast-specific program factory instead of either element-wise factory.

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Tile (32x32 elements) |
| **Unit size** | `block_size` tiles (1 tile when interleaved, up to `max_block_size` when sharded) |
| **Total units** | `physical_volume / TILE_HW` tiles across all cores |
| **Loop structure** | Outer loop over `per_core_block_cnt` blocks, inner loop over `per_core_block_size` tiles per block |

## Tensor Format and Layout

### Input Tensors

| Property | Input Tensor A | Input Tensor B |
|----------|---------------|---------------|
| **Logical shape** | [N, C, H, W] | [N, C, H, W] (same H, W as A) |
| **Dimension convention** | NCHW | NCHW |
| **Tensor layout** | TILE_LAYOUT | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED or SHARDED | INTERLEAVED or SHARDED |
| **Buffer type** | DRAM or L1 | DRAM or L1 |
| **Data type** | BFLOAT16, FLOAT32, INT32, UINT32 | BFLOAT16, FLOAT32, INT32, UINT32 |

### Output Tensor

| Property | Output Tensor |
|----------|---------------|
| **Logical shape** | Same as input |
| **Dimension convention** | NCHW |
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED or SHARDED |
| **Buffer type** | DRAM or L1 |
| **Data type** | Same as determined by output config |

### Layout Transformations

No tilize/untilize operations. All tensors must already be in TILE_LAYOUT. No format conversion is performed within the operation.

## Data Flow Pattern

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader | DRAM/L1 (src0_buffer, src1_buffer) | CB c_0, CB c_1 | reserve_back, push_back (per tile) |
| 2 | Compute | CB c_0 (as cb_inp0), CB c_1 (as cb_inp1) | CB c_2 | wait_front, copy_tile to DST, SFPU max, pack_tile, pop_front, push_back |
| 3 | Writer | CB c_2 | DRAM/L1 (dst_buffer) | wait_front, pop_front (per tile) |

For MAXIMUM, there is no pre-scaling step (no `SFPU_OP_INIT_PRE_IN0_0` or `SFPU_OP_INIT_PRE_IN1_0` defines are set), so CBs c_3 and c_4 are not created. The compute kernel reads directly from c_0 and c_1.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_src0 | Input A staging | 2 tiles (interleaved) or `num_tiles_per_shard` (sharded) | 1 tile (interleaved) or `max_block_size` (sharded) | Double (interleaved) / Single (sharded) | Reader | Compute | Program |
| c_1 | cb_src1 | Input B staging | 2 tiles (interleaved) or `num_tiles_per_shard` (sharded) | 1 tile (interleaved) or `max_block_size` (sharded) | Double (interleaved) / Single (sharded) | Reader | Compute | Program |
| c_2 | cb_output | Output staging | 2 tiles (interleaved) or `num_tiles_per_shard` (sharded) | 1 tile (interleaved) or `max_block_size` (sharded) | Double (interleaved) / Single (sharded) | Compute | Writer | Program |

Note: For MAXIMUM, CBs c_3 and c_4 (interim buffers for pre-scaling) are **not allocated** because no `SFPU_OP_INIT_PRE_IN0_0` or `SFPU_OP_INIT_PRE_IN1_0` defines are emitted.

Capacity formula (interleaved): `2 * max_block_size * tile_size`. With `max_block_size = 1`, this yields 2 tiles.

## Pipeline Pattern Summary

- **Interleaved path**: CB c_0, c_1, and c_2 each have capacity = 2 tiles with block_size = 1 tile, yielding **double-buffered** operation. The reader can fill the next tile while compute processes the current one.
- **Sharded path**: CB capacity equals `num_tiles_per_shard` and the CB is globally allocated to the tensor buffer. The entire shard is available at once (single-buffered bulk transfer).

## Index Calculations

The reader kernel uses `TensorAccessor` for DRAM-interleaved reads. Tile IDs are assigned sequentially per core via `start_id` (the first tile ID for that core) and `num_tiles` (count). For block/width sharded layouts, tile access follows a 2D pattern: `row_start_tile_id = start_id`, incrementing by `num_cores_y * block_width` per row, and tile_id increments by 1 within each row of width `block_width`.

The writer uses the same `TensorAccessor` pattern with sequential tile IDs for interleaved output, or simply waits on the output CB for sharded output (no write needed since the CB is backed by the output buffer).

## Memory Access Patterns

### Read Pattern

- **Interleaved**: Sequential tile reads via `noc_async_read_tile`. One tile at a time with a barrier after each tile (both inputs read before barrier). Tiles are read in order from `start_id` to `start_id + num_tiles - 1`.
- **Block/width sharded**: 2D access pattern iterating height then width within the shard, with tile ID stride of `num_cores_y * block_width` between rows.
- **Sharded input**: No DRAM reads; the CB is directly backed by the L1 shard buffer. The reader just does `cb_reserve_back` / `cb_push_back` to make the data available.

### Write Pattern

- **Interleaved**: Sequential single-tile writes via `noc_async_write_page` with `noc_async_writes_flushed` after each tile.
- **Sharded output**: No writes needed; the output CB is backed by the output tensor's L1 shard buffer. The writer kernel is `writer_unary_interleaved_start_id.cpp` with `OUT_SHARDED` defined, which just calls `cb_wait_front`.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (row-major) for interleaved; matches shard grid for sharded |
| **Grid dimensions** | Determined by `operation_attributes.worker_grid` |
| **Total cores** | `all_device_cores.num_cores()` or grid area |
| **Work per core** | `num_tiles_per_core_group_1` for group 1 cores, `num_tiles_per_core_group_2` for group 2 (remainder cores) |
| **Load balancing** | Two-group split: group 1 gets `ceil(num_tiles / num_cores)` tiles, group 2 gets one fewer tile. For sharded, all cores get the same number of tiles. |

The runtime args function (`set_eltwise_binary_runtime_args`) supports a `zero_start_grid` optimization when the core range is a single rectangle starting at (0,0), enabling faster work-splitting via `split_work_to_cores` with a `CoreCoord` rather than a `CoreRangeSet`.

## Arguments

### Compile-Time Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | block_or_width_sharded | uint32_t | 1 if block or width sharded, 0 otherwise |
| 1+ | TensorAccessorArgs(src0) | varies | Tensor accessor args for input A (omitted if IN0_SHARDED) |
| N+ | TensorAccessorArgs(src1) | varies | Tensor accessor args for input B (omitted if IN1_SHARDED) |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | output_cb_index | uint32_t | CB index for output (c_2) |
| 1+ | TensorAccessorArgs(dst) | varies | Tensor accessor args for output buffer |

#### Compute Kernel

No explicit compile-time args. Configuration is via `#define` macros:

| Define | Value (for MAXIMUM) | Description |
|--------|-------------------|-------------|
| `BINOP_INIT` | `binary_max_tile_init();` (float) / `binary_max_int32_tile_init();` (INT32) / `binary_max_uint32_tile_init();` (UINT32) | Initializes the SFPU max operation |
| `BINARY_SFPU_OP` | `binary_max_tile(i*2, i*2+1, i*2);` (float) / `binary_max_int32_tile(...)` / `binary_max_uint32_tile(...)` | Executes the max comparison on two DST tiles |

### Runtime Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src0_addr | uint32_t | Address of input tensor A buffer |
| 1 | src1_addr | uint32_t | Address of input tensor B buffer |
| 2 | num_tiles | uint32_t | Number of tiles this core processes |
| 3 | start_id | uint32_t | Starting tile ID for this core |
| 4 | block_height | uint32_t | Shard block height in tiles (0 if not sharded) |
| 5 | block_width | uint32_t | Shard block width in tiles (0 if not sharded) |
| 6 | num_cores_y | uint32_t | Number of shards per width (used for stride calculation) |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_cnt | uint32_t | Number of blocks to process (outer loop count) |
| 1 | per_core_block_size | uint32_t | Tiles per block (inner loop count) |

#### Writer Kernel (interleaved output)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Address of output tensor buffer |
| 1 | num_pages | uint32_t | Number of tiles to write |
| 2 | start_id | uint32_t | Starting tile ID for output writes |

## Kernel Implementations

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| Reader | BRISC (RISCV_0) | NOC0 | DRAM/L1 src0, src1 | CB c_0, CB c_1 | Read input tiles via TensorAccessor |
| Compute | TRISC (RISCV_2) | N/A | CB c_0 (DST even slots), CB c_1 (DST odd slots) | CB c_2 | copy_tile to DST, binary_max_tile SFPU op, pack_tile |
| Writer | BRISC (RISCV_1) | NOC1 | CB c_2 | DRAM/L1 dst | Write output tiles via TensorAccessor |

### Reader Kernel

| Property | Value |
|----------|-------|
| **File** | `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/dataflow/reader_binary_interleaved_start_id.cpp` |
| **Assigned cores** | All worker cores (`all_device_cores`) |

**Key Logic**:
- If `IN0_SHARDED` is defined, the reader skips DRAM reads for input A and simply makes the sharded L1 data available via `cb_reserve_back` / `cb_push_back` for the full `num_tiles`.
- If `IN1_SHARDED` is defined, same logic applies to input B.
- For interleaved (non-sharded) inputs in the default path (not `block_or_width_sharded`): iterates tile-by-tile from `start_id` to `start_id + num_tiles - 1`, reading one tile at a time from each input buffer via `noc_async_read_tile`, calling `noc_async_read_barrier()` after each pair, then pushing both tiles.
- For `block_or_width_sharded` path: uses a 2D loop over `block_height` x `block_width`, with row stride of `num_cores_y * block_width` to compute tile IDs.
- **Synchronization**: Produces into CB c_0 and CB c_1 via `cb_reserve_back(cb, 1)` -> write -> `cb_push_back(cb, 1)`.

### Compute Kernel

| Property | Value |
|----------|-------|
| **File** | `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/eltwise_binary_sfpu_kernel.cpp` |
| **Assigned cores** | All worker cores (`all_device_cores`) |

**Key Logic**:
- Outer loop: `per_core_block_cnt` iterations (blocks).
- For MAXIMUM, no pre-scaling step is active (no `SFPU_OP_INIT_PRE_IN0_0` / `SFPU_OP_INIT_PRE_IN1_0`), so `cb_inp0 = cb_in0 = c_0` and `cb_inp1 = cb_in1 = c_1`.
- Waits on both input CBs for `per_core_block_size` tiles, reserves output CB.
- Acquires DST registers via `tile_regs_acquire()` and `tile_regs_wait()`.
- Copies input A tiles to even DST slots (`i*2`) and input B tiles to odd DST slots (`i*2+1`) using `copy_tile` with proper data type initialization via `copy_tile_to_dst_init_short_with_dt`.
- Executes `BINOP_INIT` (e.g., `binary_max_tile_init()`) then `BINARY_SFPU_OP` (e.g., `binary_max_tile(i*2, i*2+1, i*2)`) per tile. The result overwrites the even DST slot (idst0).
- Packs result from DST slot `i*2` to the output CB via `pack_tile(i*2, cb_out0)`.
- Commits and releases DST registers, pops input CBs, pushes output CB.
- **Synchronization**: `cb_wait_front(cb_inp0, block_size)` and `cb_wait_front(cb_inp1, block_size)` to consume from reader. `cb_reserve_back(cb_out0, block_size)` then `cb_push_back(cb_out0, block_size)` to produce for writer. `cb_pop_front` on both inputs after processing.

### Writer Kernel

| Property | Value |
|----------|-------|
| **File** | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` (interleaved) or `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded_blocks_interleaved_start_id.cpp` (block/width sharded to interleaved) |
| **Assigned cores** | All worker cores (`all_device_cores`) |

**Key Logic**:
- For sharded output (`OUT_SHARDED` defined): simply calls `cb_wait_front(cb_id_out, num_pages)` to ensure compute is complete. No DRAM write needed since the output CB is backed by the output tensor's L1 buffer.
- For interleaved output: iterates from `start_id` to `start_id + num_pages`, writing one tile at a time via `noc_async_write_page`, flushing after each write, and a final `noc_async_write_barrier`.
- **Synchronization**: Consumes from CB c_2 via `cb_wait_front(cb_out, 1)` -> read -> `cb_pop_front(cb_out, 1)` per tile.

## Implementation Notes

- **Program factory variants**: The `ElementWiseMultiCoreSfpu::create` factory is selected when `is_binary_sfpu_op` returns true and tensor shapes are element-wise compatible (same H, W). For broadcast cases, separate factories handle height/width broadcasting.
- **Type-based operation variants**: Three SFPU function variants based on data types: `binary_max_tile` for floating-point types (BFLOAT16, FLOAT32), `binary_max_int32_tile` for INT32, and `binary_max_uint32_tile` for UINT32. The variant is selected at define-generation time in `get_defines_fp32`.
- **UnpackToDestFP32 mode**: Enabled for all CBs (c_0, c_1, c_3, c_4) since the op type is not POWER. This forces FP32 unpacking to DEST regardless of the input data format.
- **Broadcast type selection**: No broadcasting in this factory. MAXIMUM with different-shaped inputs is routed to broadcast-specific program factories.
- **Sharding support and constraints**: Supports HEIGHT_SHARDED, WIDTH_SHARDED, and BLOCK_SHARDED. Any of the three tensors (input A, input B, output) can be independently sharded or interleaved. The writer kernel selection changes based on whether block/width sharded output writes to interleaved DRAM.
- **FP32 dest accumulation**: Enabled when the output data format is Float32, Int32, or UInt32 (`fp32_dest_acc_en` flag in `ComputeConfig`).

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the binary eltwise SFPU program factory work in ttnn? What kernels does it use and how does it handle broadcast modes?"
   **Reason**: Initial reconnaissance to understand the factory structure, kernel assignments, and broadcast handling.
   **Key Findings**: Confirmed three kernels (reader, compute, writer), that broadcast is handled by separate factories, and that `ElementWiseMultiCoreSfpu` is selected via `is_binary_sfpu_op`.

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/binary_device_operation.cpp` (lines 21-65)
   **Reason**: Needed to understand SFPU path selection logic.
   **Key Information**: `is_binary_sfpu_op` returns `true` unconditionally for MAXIMUM/MINIMUM (line 57-61).

2. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary/common/binary_op_utils.cpp` (lines 184-552)
   **Reason**: Needed to understand how MAXIMUM maps to SFPU defines.
   **Key Information**: MAXIMUM sets `BINOP_INIT` and the `BINARY_SFPU_OP` define with `binary_max_tile(i*2, i*2+1, i*2)` (or int32/uint32 variants). No pre-scaling defines are emitted.

3. **Source**: `tt_metal/hw/inc/api/compute/binary_max_min.h`
   **Reason**: Needed to understand the SFPU compute API for max/min operations.
   **Key Information**: Three variants exist (float, int32, uint32). Each takes (idst0, idst1, odst) and computes element-wise max, writing result to odst in DST register buffer.
