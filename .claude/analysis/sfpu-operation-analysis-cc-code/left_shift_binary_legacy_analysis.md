# LEFT_SHIFT (Binary Legacy) Implementation Analysis

## Overview
The LEFT_SHIFT operation performs element-wise bitwise left shift on two integer tensors (`A << B`), where each element of tensor A is shifted left by the corresponding element of tensor B. It is implemented through the binary legacy SFPU program factory, which routes integer shift operations through the SFPU (vector unit) rather than the FPU (matrix unit).

**Program factory path**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/element_wise_multi_core_sfpu_pgm_factory.cpp`

## Path Selection: FPU vs SFPU

The binary device operation uses `BinaryDeviceOperation::select_program_factory()` (in `binary_device_operation.cpp`) to choose between program factories. When both input tensors have the same shape (no broadcasting needed), the function calls `utils::is_binary_sfpu_op(op, dtype1, dtype2)` to decide between `ElementWiseMultiCore` (FPU) and `ElementWiseMultiCoreSfpu` (SFPU).

For `BinaryOpType::LEFT_SHIFT`, `is_binary_sfpu_op` returns `true` when:
- Both inputs are `INT32`, or
- Both inputs are `UINT32`

(Note: `UINT16` is also listed in the switch case alongside `GCD`, `LCM`, `RIGHT_SHIFT`, and `LOGICAL_RIGHT_SHIFT`.)

If the shapes differ (broadcasting needed), the operation is routed to broadcast-specific factories (`BroadcastHeightMultiCore`, `BroadcastWidthMultiCore`, etc.) instead. The SFPU path is only selected for element-wise (same-shape) cases.

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Tile (32x32 elements) |
| **Unit size** | 1 tile per inner loop iteration; `per_core_block_size` tiles per block |
| **Total units** | `total_tiles = physical_volume / TILE_HW` |
| **Loop structure** | Outer loop: `per_core_block_cnt` blocks; inner loop: `per_core_block_size` tiles per block |

In the non-sharded case, `per_core_block_size = 1` and `per_core_block_cnt = num_tiles_per_core`. In the sharded case, `per_core_block_size = find_max_block_size(num_tiles_per_shard)` and `per_core_block_cnt = num_tiles_per_shard / per_core_block_size`.

## Tensor Format and Layout

### Input Tensors

| Property | Input Tensor A | Input Tensor B |
|----------|---------------|---------------|
| **Logical shape** | Arbitrary (must match B) | Arbitrary (must match A) |
| **Dimension convention** | NHWC/any | NHWC/any |
| **Tensor layout** | TILE_LAYOUT | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED or SHARDED | INTERLEAVED or SHARDED |
| **Buffer type** | DRAM or L1 | DRAM or L1 |
| **Data type** | INT32 or UINT32 | INT32 or UINT32 |

### Output Tensor

| Property | Output Tensor |
|----------|---------------|
| **Logical shape** | Same as inputs |
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED or SHARDED |
| **Buffer type** | DRAM or L1 |
| **Data type** | INT32 or UINT32 |

### Layout Transformations
No tilize/untilize or format conversions are performed within the operation. Both inputs and the output must already be in TILE_LAYOUT.

## Data Flow Pattern

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader | DRAM/L1 (src0_buffer) | CB c_0 | reserve_back, push_back |
| 1 | Reader | DRAM/L1 (src1_buffer) | CB c_1 | reserve_back, push_back |
| 2 | Compute | CB c_0, CB c_1 | CB c_2 | wait_front, pop_front, reserve_back, push_back |
| 3 | Writer | CB c_2 | DRAM/L1 (dst_buffer) | wait_front, pop_front |

**Interleaved path**: The reader reads one tile at a time from each input buffer using `noc_async_read_tile` via TensorAccessor, pushing tiles into CB c_0 and CB c_1. The compute kernel waits for `per_core_block_size` tiles in both input CBs, copies both inputs into DST registers (A into even slots, B into odd slots), executes the SFPU shift operation, packs the result to CB c_2, and pops both input CBs. The writer reads one tile at a time from CB c_2 and writes it to the output buffer via `noc_async_write_page`.

**Sharded path**: When an input is sharded, the reader simply does `cb_reserve_back / cb_push_back` for the full shard (the CB is backed by the sharded buffer directly). When the output is sharded, the writer just does `cb_wait_front` and the data is already in place.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_src0 | Input A staging | 2 * max_block_size tiles (interleaved) or num_tiles_per_shard (sharded) | 1 tile (interleaved) or max_block_size tiles (sharded) | Double (interleaved) / Single (sharded) | Reader | Compute | Program |
| c_1 | cb_src1 | Input B staging | 2 * max_block_size tiles (interleaved) or num_tiles_per_shard (sharded) | 1 tile (interleaved) or max_block_size tiles (sharded) | Double (interleaved) / Single (sharded) | Reader | Compute | Program |
| c_2 | cb_output | Output staging | 2 * max_block_size tiles (interleaved) or num_tiles_per_shard (sharded) | 1 tile (interleaved) or max_block_size tiles (sharded) | Double (interleaved) / Single (sharded) | Compute | Writer | Program |

**Note**: CB c_3 and c_4 (interim buffers for pre-scale operations) are NOT created for LEFT_SHIFT because the operation does not define `SFPU_OP_INIT_PRE_IN0_0` or `SFPU_OP_INIT_PRE_IN1_0`.

## Pipeline Pattern Summary

- **Interleaved**: Double-buffered on all three CBs (capacity = 2 * block_size). This allows the reader and compute (or compute and writer) to overlap -- while compute processes one block, the reader can fill the next.
- **Sharded**: Single-buffered (capacity = num_tiles_per_shard = total shard). The sharded CB is globally allocated on the tensor's buffer, so there is no streaming overlap -- the entire shard is available at once.

## Index Calculations

The reader kernel uses `TensorAccessor` to map linear tile IDs to physical memory addresses. For the non-sharded interleaved path, tile IDs are sequential starting from `start_id`. Each core processes a contiguous range of `num_tiles_per_core` tiles.

For the block/width-sharded path, tile traversal follows a 2D pattern: outer loop over `block_height` rows, inner loop over `block_width` columns. The `start_id` for each core is computed as:
```
start_id = (core_index / num_shards_per_width) * (block_height * block_width * num_shards_per_width)
         + (core_index % num_shards_per_width) * block_width
```
Row advancement uses stride: `row_start_tile_id += num_cores_y * block_width`.

## Memory Access Patterns

### Read Pattern
- **Interleaved**: Sequential tile reads. Each tile is read individually via `noc_async_read_tile` with a barrier after each tile pair (one from each input). This is a tile-at-a-time pattern.
- **Sharded**: No reads needed; the CB is backed by the sharded L1 buffer directly.

### Write Pattern
- **Interleaved**: Sequential tile writes via `noc_async_write_page`, one tile at a time, with `noc_async_writes_flushed()` after each tile and `noc_async_write_barrier()` at the end.
- **Sharded**: No writes needed; the output CB is backed by the output sharded buffer.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (interleaved) or 2D (sharded, matching shard grid) |
| **Grid dimensions** | Determined by `operation_attributes.worker_grid` |
| **Total cores** | `num_cores_total` from worker grid |
| **Work per core** | `num_tiles_per_core_group_1` (main group) or `num_tiles_per_core_group_2` (remainder group) |
| **Load balancing** | Two-group split: group 1 gets `ceil(total_tiles/num_cores)` tiles, group 2 gets `floor(total_tiles/num_cores)` tiles. For sharded: all cores get equal `num_tiles_per_shard`. |

Work splitting uses `tt::tt_metal::split_work_to_cores()` for the interleaved case. The function divides total tiles across available cores, creating two groups to handle remainder. Non-working cores (beyond `num_cores`) receive zero-tile arguments.

An optimization is applied when the grid is a single rectangle starting at (0,0) (`zero_start_grid` flag), using faster core enumeration algorithms.

## Arguments

### Compile-Time Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | block_or_width_sharded | uint32_t | 1 if block or width sharded, 0 otherwise |
| 1+ | TensorAccessorArgs(src0) | varies | Accessor args for input A (omitted if IN0_SHARDED) |
| N+ | TensorAccessorArgs(src1) | varies | Accessor args for input B (omitted if IN1_SHARDED) |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | output_cb_index | uint32_t | CB index for output (c_2) |
| 1+ | TensorAccessorArgs(dst) | varies | Accessor args for output buffer |

#### Compute Kernel
No compile-time arguments. All configuration is via preprocessor defines:
- `SHIFT_INIT` = `binary_shift_tile_init();`
- `BINARY_SFPU_OP` = `binary_left_shift_tile<DataFormat::Int32>(i*2, i*2+1, i*2);` (or `UInt32`/`UInt16` variant)

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
| 6 | num_cores_y | uint32_t | Number of shards per width dimension (0 if not sharded) |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_cnt | uint32_t | Number of blocks to process |
| 1 | per_core_block_size | uint32_t | Number of tiles per block |

#### Writer Kernel (Interleaved, non-block-sharded output)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Address of output buffer |
| 1 | num_pages | uint32_t | Number of tiles to write |
| 2 | start_id | uint32_t | Starting tile ID for output writes |

## Kernel Implementations

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| Reader | BRISC (RISCV_0) | NOC0 | DRAM/L1 src0, src1 | CB c_0, CB c_1 | Read tiles from both input buffers |
| Compute | Tensix compute (RISCV_2) | N/A | CB c_0, CB c_1 | CB c_2 | Copy tiles to DST, execute SFPU left shift, pack result |
| Writer | NCRISC (RISCV_1) | NOC1 | CB c_2 | DRAM/L1 dst | Write output tiles to destination buffer |

### Reader Kernel

| Property | Value |
|----------|-------|
| File | `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/dataflow/reader_binary_interleaved_start_id.cpp` |
| Assigned cores | All worker cores (`all_device_cores`) |

**Key Logic**:
- When inputs are sharded (`IN0_SHARDED` / `IN1_SHARDED` defined), the reader simply calls `cb_reserve_back(cb_id, num_tiles)` then `cb_push_back(cb_id, num_tiles)` to make the globally-allocated sharded buffer available to compute. No actual data movement occurs.
- For interleaved inputs, a `TensorAccessor` is constructed from compile-time args for each non-sharded input.
- In the `block_or_width_sharded` path (sharded input with interleaved other), tile traversal uses 2D indexing: outer loop over `block_height`, inner loop over `block_width`, with row stride = `num_cores_y * block_width`.
- In the standard interleaved path, tiles are read sequentially from `start_id` to `start_id + num_tiles`.
- Each tile read uses `noc_async_read_tile` followed by `noc_async_read_barrier` before pushing to the CB.
- **Synchronization**: Produces to CB c_0 and CB c_1 via `cb_reserve_back` / `cb_push_back` (one tile at a time for interleaved).

### Compute Kernel

| Property | Value |
|----------|-------|
| File | `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/eltwise_binary_sfpu_kernel.cpp` |
| Assigned cores | All worker cores (`all_device_cores`) |

**Key Logic**:
- Outer loop iterates `per_core_block_cnt` times (one iteration per block).
- No pre-scale path is active for LEFT_SHIFT (no `SFPU_OP_INIT_PRE_IN0_0` or `SFPU_OP_INIT_PRE_IN1_0` defines), so `cb_inp0 = cb_in0 = c_0` and `cb_inp1 = cb_in1 = c_1`.
- Waits for `per_core_block_size` tiles in both input CBs and reserves space in the output CB.
- Acquires tile registers (`tile_regs_acquire` + `tile_regs_wait`).
- Copies input A tiles to even DST slots (`i*2`) using `copy_tile_to_dst_init_short_with_dt(cb_inp1, cb_inp0)` then `copy_tile(cb_inp0, i, i*2)`.
- Copies input B tiles to odd DST slots (`i*2+1`) using `copy_tile_to_dst_init_short_with_dt(cb_inp0, cb_inp1)` then `copy_tile(cb_inp1, i, i*2+1)`.
- Executes `SHIFT_INIT` -> `binary_shift_tile_init()` once per inner tile.
- Executes `BINARY_SFPU_OP` -> `binary_left_shift_tile<DataFormat::Int32>(i*2, i*2+1, i*2)` which reads operands from DST[i*2] and DST[i*2+1], writes the result back to DST[i*2].
- Packs result from DST[i*2] into output CB c_2 via `pack_tile(i*2, cb_out0)`.
- After inner loop: `tile_regs_commit` / `tile_regs_release`, then pops both input CBs and pushes the output CB.
- **Synchronization**: Consumes CB c_0 and c_1 via `cb_wait_front` / `cb_pop_front`. Produces to CB c_2 via `cb_reserve_back` / `cb_push_back`.

### Writer Kernel

| Property | Value |
|----------|-------|
| File | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` (standard) or `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded_blocks_interleaved_start_id.cpp` (block/width sharded input, interleaved output) |
| Assigned cores | All worker cores (`all_device_cores`) |

**Key Logic**:
- For sharded output (`OUT_SHARDED` defined): simply calls `cb_wait_front(cb_id_out, num_pages)` -- the data is already in the output buffer since the CB is globally allocated on it.
- For interleaved output: iterates from `start_id` to `start_id + num_pages`, writing one tile per iteration. Each iteration: `cb_wait_front` for 1 tile, gets L1 read address, calls `noc_async_write_page`, flushes, then `cb_pop_front`.
- Final `noc_async_write_barrier()` ensures all writes complete.
- **Synchronization**: Consumes CB c_2 via `cb_wait_front` / `cb_pop_front`.

## Implementation Notes

- **Program factory variants**: The `BinaryDeviceOperation` has multiple program factories: `ElementWiseMultiCore` (FPU), `ElementWiseMultiCoreSfpu` (SFPU), and several broadcast variants. LEFT_SHIFT always uses `ElementWiseMultiCoreSfpu` since `is_binary_sfpu_op` returns true for all its supported dtype combinations. The factory is selected in `select_program_factory()` only when input shapes match (no broadcasting).
- **Type-based operation variants**: Supports INT32 x INT32, UINT32 x UINT32, and UINT16 x UINT16. The data format string (`"Int32"`, `"UInt32"`, or `"UInt16"`) is selected in `get_defines_fp32()` and templated into the SFPU call as `binary_left_shift_tile<DataFormat::XXX>`.
- **UnpackToDestFP32 mode**: Enabled for all CBs (c_0, c_1, c_3, c_4) because `op_type != BinaryOpType::POWER` -- all non-POWER SFPU binary ops use `UnpackToDestMode::UnpackToDestFp32`.
- **Broadcast type selection**: N/A. LEFT_SHIFT through this program factory requires both inputs to have identical shapes. No broadcasting is supported in this path.
- **Sharding support and constraints**: Supports height-sharded, width-sharded, and block-sharded memory layouts. Either or both inputs and/or the output can be sharded. When sharded, CBs are globally allocated on the tensor buffer (no data movement). A special writer kernel variant handles the case of block/width-sharded input with interleaved output.
- **FP32 dest accumulation**: Enabled when the output data format is Float32, Int32, or UInt32 (`fp32_dest_acc_en`). Since LEFT_SHIFT operates on INT32 or UINT32, this is always enabled for LEFT_SHIFT.

## External Knowledge Sources

### DeepWiki Queries
1. **Query**: "How does the binary element-wise SFPU program factory work in ttnn? What kernels does it use and how does it handle different binary operations like left_shift?"
   **Reason**: Needed to understand the overall SFPU binary factory architecture and kernel selection before reading source code.
   **Key Findings**: Confirmed the three-kernel structure (reader, compute, writer), the use of preprocessor defines to parameterize the compute kernel, and that LEFT_SHIFT uses `binary_shift_tile_init()` and `binary_left_shift_tile<DataFormat::Int32>`. Also confirmed that `is_binary_sfpu_op` controls factory selection.

### Documentation References
1. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/binary_device_operation.cpp`
   **Reason**: Needed to understand `select_program_factory` logic and `is_binary_sfpu_op` conditions.
   **Key Information**: LEFT_SHIFT routes to SFPU when both inputs are INT32 or UINT32; same-shape requirement for element-wise path.

2. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary/common/binary_op_utils.cpp`
   **Reason**: Needed to understand which preprocessor defines are generated for LEFT_SHIFT.
   **Key Information**: Defines `SHIFT_INIT` = `binary_shift_tile_init();` and `BINARY_SFPU_OP` = `binary_left_shift_tile<DataFormat::XXX>(i*2, i*2+1, i*2);` with format selected based on input dtypes.

3. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/eltwise_multi_core_program_factory_common.hpp`
   **Reason**: Needed to understand runtime argument setup and core distribution logic.
   **Key Information**: Two-group work splitting for interleaved; shard-based distribution for sharded; `zero_start_grid` optimization for rectangular grids starting at (0,0).
