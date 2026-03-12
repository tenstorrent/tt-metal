# ADD (element_wise_sfpu) Implementation Analysis

## Overview

The ADD operation via the SFPU path performs element-wise addition of two input tensors using the Special Function Processing Unit (SFPU) instead of the FPU matrix engine. This path is selected when both input tensors share the same data type and that type is one of FLOAT32, INT32, UINT32, or UINT16. The SFPU path copies both input tiles into the DEST register and executes `add_binary_tile` (for float types) or `add_int_tile` (for integer types) as an SFPU function, rather than using the FPU's `add_tiles` instruction.

**Program factory path**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/element_wise_multi_core_sfpu_pgm_factory.cpp`

## Path Selection: FPU vs SFPU

The factory selection is performed in `BinaryDeviceOperation::select_program_factory()` (in `binary_device_operation.cpp`). When both tensors have the same shape (no broadcasting needed), the function calls `utils::is_binary_sfpu_op(op, dtype1, dtype2)`. For `BinaryOpType::ADD`, this function returns `true` when `dtype1 == dtype2` and the common type is one of `FLOAT32`, `INT32`, `UINT32`, or `UINT16`. When the check returns true, `ElementWiseMultiCoreSfpu{}` is selected; otherwise `ElementWiseMultiCore{}` (the FPU path) is selected. The FPU path is therefore used for BF16 and other non-32-bit / non-UINT16 types, or when broadcasting is required.

## Work Unit Definition

One work unit is one **tile** (32x32 elements). The compute kernel processes tiles in blocks of `per_core_block_size` tiles, iterating over `per_core_block_cnt` blocks. For non-sharded tensors, `block_size` is 1 and `block_cnt` equals the number of tiles assigned to the core. For sharded tensors, `block_size` is the largest power-of-2 factor of `num_tiles_per_shard`, and `block_cnt = num_tiles_per_shard / block_size`.

## Tensor Format and Layout

### Input Tensors

| Property | Input A (src0) | Input B (src1) |
|---|---|---|
| **Dim Convention** | NHWC (logical) | NHWC (logical) |
| **Tensor Layout** | TILE (32x32) | TILE (32x32) |
| **Memory Layout** | Interleaved or Sharded | Interleaved or Sharded |
| **Buffer Type** | DRAM or L1 | DRAM or L1 |
| **Data Type** | FLOAT32, INT32, UINT32, or UINT16 (must match B) | FLOAT32, INT32, UINT32, or UINT16 (must match A) |

### Output Tensor

| Property | Output |
|---|---|
| **Dim Convention** | NHWC (logical) |
| **Tensor Layout** | TILE (32x32) |
| **Memory Layout** | Interleaved or Sharded |
| **Buffer Type** | DRAM or L1 |
| **Data Type** | Same as input (FLOAT32/INT32/UINT32/UINT16) |

### Layout Transformations

No tilize/untilize operations are performed. Data remains in tile layout throughout the pipeline. No format conversions occur for the basic ADD case without fused activations.

## Data Flow Pattern

1. **Reader kernel** reads tiles for both input A and input B from DRAM/L1 (or signals sharded data availability) into CB c_0 and CB c_1 respectively, one tile at a time (interleaved) or all tiles at once (sharded).
2. **Compute kernel** waits for tiles in CB c_0 and CB c_1 (or c_3/c_4 if pre-scaling is active).
3. Both input tiles are copied into DEST registers: input A goes to DEST[i*2], input B goes to DEST[i*2+1].
4. The SFPU binary operation (`add_binary_tile` or `add_int_tile`) is invoked, reading from DEST[i*2] and DEST[i*2+1], writing the result back to DEST[i*2].
5. The result tile at DEST[i*2] is packed into CB c_2.
6. **Writer kernel** reads tiles from CB c_2 and writes them to the output buffer in DRAM/L1 (or leaves them in place for sharded output).

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity (tiles) | Block Size (tiles) | Buffering | Producer | Consumer | Lifetime |
|---|---|---|---|---|---|---|---|---|
| c_0 | cb_src0 | Input A tiles | Sharded: `num_tiles_per_shard`; Interleaved: `2 * max_block_size` | 1 (interleaved read) or `per_core_block_size` (compute) | Sharded: Single; Interleaved: Double | Reader | Compute | Full pipeline |
| c_1 | cb_src1 | Input B tiles | Sharded: `num_tiles_per_shard`; Interleaved: `2 * max_block_size` | 1 (interleaved read) or `per_core_block_size` (compute) | Sharded: Single; Interleaved: Double | Reader | Compute | Full pipeline |
| c_3 | cb_interim0 | Pre-scaled input A (conditional) | `max_block_size` | `per_core_block_size` | Single | Compute (pre-scale) | Compute (main) | Conditional: only when `SFPU_OP_INIT_PRE_IN0_0` defined |
| c_4 | cb_interim1 | Pre-scaled input B (conditional) | `max_block_size` | `per_core_block_size` | Single | Compute (pre-scale) | Compute (main) | Conditional: only when `SFPU_OP_INIT_PRE_IN1_0` defined |
| c_2 | cb_output | Output tiles | Sharded/block-width: `num_tiles_per_shard`; Interleaved: `2 * max_block_size` | `per_core_block_size` (compute push) / 1 (writer pop, interleaved) | Sharded: Single; Interleaved: Double | Compute | Writer | Full pipeline |

**Note for ADD specifically**: For basic ADD (FLOAT32 or integer types) without `input_tensor_a_activation`, the pre-scale defines `SFPU_OP_INIT_PRE_IN0_0` and `SFPU_OP_INIT_PRE_IN1_0` are NOT defined. Therefore CB c_3 and CB c_4 are NOT created. The compute kernel reads directly from c_0 and c_1.

## Pipeline Pattern Summary

- **Interleaved path**: CB c_0, c_1, and c_2 each have capacity `2 * max_block_size` tiles. The reader pushes 1 tile at a time, compute consumes `per_core_block_size` (which equals `max_block_size` = 1 for interleaved), and writer pops 1 tile at a time. With capacity of 2 tiles, this provides **double-buffering**, allowing reader and compute to overlap.
- **Sharded path**: CB c_0 and c_1 are globally allocated (backed by the input tensor's L1 buffer). All shard tiles are signaled at once. CB c_2 is similarly backed by the output buffer. This is effectively **single-buffered** -- all tiles are produced then consumed in one shot.

## Index Calculations

The reader kernel uses `TensorAccessor` for DRAM address resolution. For interleaved tensors, `noc_async_read_tile(tile_id, accessor, l1_addr)` is called with a linear tile ID. The `TensorAccessor` maps this to the correct bank and offset.

For the block/width sharded path, tile IDs are computed as:
```
start_id = (core_index / num_shards_per_width) * (block_height * block_width * num_shards_per_width)
         + (core_index % num_shards_per_width) * block_width
```
Within each shard, tiles are read row-by-row: `row_start_tile_id += num_cores_y * block_width` per row.

## Memory Access Patterns

### Read Pattern
- **Interleaved**: Sequential tile-by-tile reads. Each tile is read via `noc_async_read_tile` with an `noc_async_read_barrier` after each pair of reads (one from each input). This is a linear scan through the tile ID space.
- **Block/Width sharded reader**: Two-level nested loop -- outer over block height, inner over block width. Row stride = `num_cores_y * block_width` tiles.
- **Sharded (both inputs)**: No reads; the reader just signals `cb_reserve_back` + `cb_push_back` on the full shard.

### Write Pattern
- **Interleaved**: Sequential single-tile writes via `noc_async_write_page` with `noc_async_writes_flushed` after each tile, followed by a final `noc_async_write_barrier`.
- **Sharded output**: No writes; the writer just calls `cb_wait_front` on all output tiles (data is already in place).
- **Block/Width sharded to interleaved**: The special writer kernel reads the full block from CB, then writes unpadded tiles row-by-row to DRAM, skipping padding tiles.

## Core Distribution Strategy

| Property | Value |
|---|---|
| **Grid Topology** | Rectangular grid from `operation_attributes.worker_grid` |
| **Work Splitting** | `split_work_to_cores()` for interleaved; shard grid for sharded |
| **Row Major** | Yes (interleaved default); shard orientation for sharded |
| **Core Group 1** | Cores with `ceil(num_tiles / num_cores)` tiles |
| **Core Group 2** | Remaining cores with `floor(num_tiles / num_cores)` tiles (interleaved only) |
| **Remainder Handling** | Two core groups with different tile counts; excess cores receive 0 tiles |
| **Zero-start optimization** | When the grid is a single rectangle starting at (0,0) and any shard grid also starts at (0,0), a faster `grid_to_cores` path is used instead of the generic `corerange_to_cores` |

## Arguments

### Compile-Time Arguments

**Reader Kernel:**

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | `block_or_width_sharded` | uint32_t | 1 if block or width sharded, 0 otherwise |
| 1+ | `TensorAccessorArgs(src0)` | variable | Accessor args for input A (only if not IN0_SHARDED) |
| N+ | `TensorAccessorArgs(src1)` | variable | Accessor args for input B (only if not IN1_SHARDED) |

**Writer Kernel:**

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | `output_cb_index` | uint32_t | CB index for output (c_2) |
| 1+ | `TensorAccessorArgs(dst)` | variable | Accessor args for output buffer |

**Compute Kernel:**

No indexed compile-time args. Configuration is via `ComputeConfig`:
- `fp32_dest_acc_en`: true if output is Float32/Int32/UInt32
- `unpack_to_dest_mode`: `UnpackToDestFp32` for all CBs (since op is not POWER)
- `defines`: The operation-specific defines from `get_defines_fp32()`

### Runtime Arguments

**Reader Kernel:**

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | `src0_addr` | uint32_t | Base address of input A buffer |
| 1 | `src1_addr` | uint32_t | Base address of input B buffer |
| 2 | `num_tiles` | uint32_t | Total tiles this core processes |
| 3 | `start_id` | uint32_t | Starting tile ID for this core |
| 4 | `block_height` | uint32_t | Shard block height in tiles (0 if not sharded) |
| 5 | `block_width` | uint32_t | Shard block width in tiles (0 if not sharded) |
| 6 | `num_cores_y` | uint32_t | Number of shards per width dimension |

**Compute Kernel:**

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | `per_core_block_cnt` | uint32_t | Number of blocks to process |
| 1 | `per_core_block_size` | uint32_t | Tiles per block |

**Writer Kernel (interleaved, non-block-sharded):**

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | `dst_addr` | uint32_t | Base address of output buffer |
| 1 | `num_pages` | uint32_t | Number of tiles to write |
| 2 | `start_id` | uint32_t | Starting tile ID for writes |

**Writer Kernel (block/width sharded to interleaved):**

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | `dst_addr` | uint32_t | Base address of output buffer |
| 1 | `block_height_tiles` | uint32_t | Block height in tiles |
| 2 | `block_width_tiles` | uint32_t | Block width in tiles |
| 3 | `unpadded_block_height_tiles` | uint32_t | Unpadded block height |
| 4 | `unpadded_block_width_tiles` | uint32_t | Unpadded block width |
| 5 | `output_width_tiles` | uint32_t | Full output width in tiles |
| 6 | `block_num_tiles` | uint32_t | Total tiles in block |
| 7 | `start_id_offset` | uint32_t | Starting tile offset |
| 8 | `start_id_base` | uint32_t | Base tile ID (0) |

## Kernel Implementations

| Kernel | File | Type | Assigned Cores |
|---|---|---|---|
| Reader | `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/dataflow/reader_binary_interleaved_start_id.cpp` | DataMovement (Reader) | All worker cores |
| Compute | `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/eltwise_binary_sfpu_kernel.cpp` | Compute | All worker cores |
| Writer (interleaved) | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | DataMovement (Writer) | All worker cores |
| Writer (block-sharded to interleaved) | `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded_blocks_interleaved_start_id.cpp` | DataMovement (Writer) | All worker cores |

### Reader Kernel

| Property | Value |
|---|---|
| **File** | `reader_binary_interleaved_start_id.cpp` |
| **Assigned Cores** | All worker cores in `all_device_cores` |

**Key Logic:**
- If `IN0_SHARDED` is defined, immediately calls `cb_reserve_back(cb_id_in0, num_tiles)` and `cb_push_back(cb_id_in0, num_tiles)` to signal that data is already in place. Same for `IN1_SHARDED`.
- For non-sharded inputs with `block_or_width_sharded` mode: uses a nested loop (height x width) to read tiles. Row stride is `num_cores_y * block_width`, which maps the 2D shard layout to the linear tile space.
- For non-sharded inputs without block/width sharding: simple sequential loop from `start_id` to `start_id + num_tiles`, reading one tile at a time.
- Each non-sharded tile read uses `noc_async_read_tile` via `TensorAccessor`, followed by `noc_async_read_barrier` before pushing to the CB. This serializes reads with a barrier after each tile pair.
- **Synchronization**: Pushes tiles to CB c_0 and CB c_1. The compute kernel consumes from these CBs via `cb_wait_front`.

### Compute Kernel

| Property | Value |
|---|---|
| **File** | `eltwise_binary_sfpu_kernel.cpp` |
| **Assigned Cores** | All worker cores in `all_device_cores` |

**Key Logic:**
- Initializes with `unary_op_init_common(cb_in0, cb_out0)`.
- For ADD with FLOAT32: defines `BINOP_INIT` = `add_binary_tile_init();` and `BINARY_SFPU_OP` = `add_binary_tile(i*2, i*2+1, i*2);`. No pre-scaling defines are set.
- For ADD with INT32: defines `ADD_INT_INIT` = `add_int_tile_init();` and `BINARY_SFPU_OP` = `add_int_tile<DataFormat::Int32>(i*2, i*2+1, i*2);`. Similar patterns for UINT32 and UINT16.
- Main loop iterates `per_core_block_cnt` times. Each iteration:
  1. Waits for `per_core_block_size` tiles in `cb_inp0` (c_0) and `cb_inp1` (c_1).
  2. Reserves `per_core_block_size` tiles in `cb_out0` (c_2).
  3. Acquires tile registers (`tile_regs_acquire` + `tile_regs_wait`).
  4. Copies all input A tiles to DEST[i*2] via `copy_tile(cb_inp0, i, i*2)`.
  5. Switches data type context with `copy_tile_to_dst_init_short_with_dt`.
  6. For each input B tile: copies to DEST[i*2+1], runs the init macro, then runs the SFPU op macro, then packs result from DEST[i*2] to `cb_out0`.
  7. Commits and releases tile registers.
  8. Pops tiles from input CBs, pushes to output CB.
- **Synchronization**: Waits on CB c_0 and c_1 via `cb_wait_front`, pushes to CB c_2 via `cb_push_back`. Uses `tile_regs_acquire/commit/wait/release` for DEST register synchronization between unpack and pack pipelines.

### Writer Kernel (Interleaved)

| Property | Value |
|---|---|
| **File** | `writer_unary_interleaved_start_id.cpp` |
| **Assigned Cores** | All worker cores in `all_device_cores` |

**Key Logic:**
- If `OUT_SHARDED` is defined, just calls `cb_wait_front(cb_id_out, num_pages)` -- data is already in the output buffer via the globally-allocated CB.
- For non-sharded output: sequential loop writing one tile at a time via `noc_async_write_page` using `TensorAccessor`. Each write is followed by `noc_async_writes_flushed`, then `cb_pop_front(cb_id_out, 1)`. Final `noc_async_write_barrier` ensures all writes complete.
- **Synchronization**: Waits on CB c_2 via `cb_wait_front`, pops via `cb_pop_front`.

### Writer Kernel (Block-Sharded to Interleaved)

| Property | Value |
|---|---|
| **File** | `writer_unary_sharded_blocks_interleaved_start_id.cpp` |
| **Assigned Cores** | All worker cores (only used when `block_or_width_sharded && !out_sharded`) |

**Key Logic:**
- Waits for the entire block (`cb_wait_front(cb_id_out, block_num_tiles)`), then writes unpadded tiles to DRAM using a nested height x width loop. Skips padding tiles by advancing the L1 read pointer past padded width.
- Row stride in output is `output_width_tiles` tiles.
- **Synchronization**: Single `cb_wait_front` for all block tiles, single `cb_pop_front` after all writes complete.

## Implementation Notes

- **Program factory variants**: Two program factories exist for non-broadcast binary operations: `ElementWiseMultiCore` (FPU path) and `ElementWiseMultiCoreSfpu` (SFPU path). Selection is based on `is_binary_sfpu_op()` which checks the operation type and data types. Additional broadcast factories exist (`BroadcastWidthMultiCore`, `BroadcastHeightMultiCore`, `BroadcastHeightAndWidthMultiCore`) but these do not have SFPU variants.
- **Type-based operation variants**: For ADD, the SFPU path supports FLOAT32 (using `add_binary_tile`), INT32 (using `add_int_tile<DataFormat::Int32>`), UINT32 (using `add_int_tile<DataFormat::UInt32>`), and UINT16 (using `add_int_tile<DataFormat::UInt16>`). BF16 ADD uses the FPU path instead.
- **UnpackToDestFP32 mode**: For all ops except POWER, `UnpackToDestMode::UnpackToDestFp32` is set on all input and interim CBs (c_0, c_1, c_3, c_4). This ensures data is unpacked to full FP32 precision in the DEST register regardless of the source format.
- **Broadcast type selection**: N/A for this SFPU path. Broadcasting requires equal tensor shapes; when shapes differ, the FPU-based broadcast factories are selected instead.
- **Sharding support and constraints**: All three sharding modes (height, width, block) are supported for inputs and output independently. The program factory detects sharding from any of the three tensors (src0, src1, output) and configures globally-allocated circular buffers accordingly. A special writer kernel (`writer_unary_sharded_blocks_interleaved_start_id`) handles the case where inputs are block/width-sharded but output is interleaved.
- **FP32 dest accumulation**: Enabled when the output data format is Float32, Int32, or UInt32 (via `fp32_dest_acc_en` in `ComputeConfig`). This keeps the DEST accumulator in full 32-bit precision.

## External Knowledge Sources

### DeepWiki Queries
1. **Query**: "How does the binary eltwise SFPU program factory work? What kernels does it use for reader, compute, and writer? How does it differ from the FPU binary path?"
   **Reason**: Initial reconnaissance to understand the overall architecture of the SFPU binary path before reading source code.
   **Key Findings**: Confirmed the three kernels used (reader_binary_interleaved_start_id, eltwise_binary_sfpu_kernel, writer_unary_interleaved_start_id), the factory selection mechanism via `is_binary_sfpu_op`, and the fundamental difference: SFPU copies tiles to DEST then invokes SFPU functions, whereas FPU operates directly on tiles via matrix engine instructions.

### Documentation References
1. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/binary_device_operation.cpp` (lines 22-66)
   **Reason**: Needed to understand the exact conditions under which the SFPU path is selected for ADD.
   **Key Information**: ADD uses SFPU path when `a == b && (a == FLOAT32 || a == INT32 || a == UINT32 || a == UINT16)`.

2. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary/common/binary_op_utils.cpp` (lines 184-552)
   **Reason**: Needed to understand what preprocessor defines are generated for ADD in the SFPU path.
   **Key Information**: For FLOAT32 ADD, defines `BINOP_INIT` = `add_binary_tile_init()` and `BINARY_SFPU_OP` = `add_binary_tile(i*2, i*2+1, i*2)`. For INT32 ADD, defines `ADD_INT_INIT` = `add_int_tile_init()` and uses `add_int_tile<DataFormat::Int32>`.

3. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/eltwise_multi_core_program_factory_common.hpp`
   **Reason**: Needed to understand the runtime argument setup and core distribution logic shared between FPU and SFPU paths.
   **Key Information**: The `set_eltwise_binary_runtime_args` template function handles both creation and override cases, with optimized zero-start-grid paths and two-group core splitting for load balancing.
