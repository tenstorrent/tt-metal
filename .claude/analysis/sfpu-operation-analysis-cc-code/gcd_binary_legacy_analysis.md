# GCD (Binary Legacy) Implementation Analysis

## Overview

The GCD (Greatest Common Divisor) operation computes the element-wise greatest common divisor of two integer tensors: `output = gcd(input_a, input_b)`. It uses the binary GCD algorithm implemented as an SFPU kernel, operating on INT32 or UINT32 tile data in the DST register file.

**Program factory path**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/element_wise_multi_core_sfpu_pgm_factory.cpp`

## Path Selection: FPU vs SFPU

The `BinaryDeviceOperation::select_program_factory` method (in `binary_device_operation.cpp`) selects between `ElementWiseMultiCore` (FPU) and `ElementWiseMultiCoreSfpu` (SFPU) factories. When both input tensors have the same height and width (element-wise, no broadcasting), it calls `utils::is_binary_sfpu_op(op, dtype1, dtype2)`. For `BinaryOpType::GCD`, this function returns `true` when both inputs are either `INT32/INT32` or `UINT32/UINT32`. Since GCD is exclusively an integer operation with no FPU path, the SFPU factory is always selected. If broadcasting is needed (height_b==1 or width_b==1), the broadcast program factories are used instead, but those are out of scope for this analysis.

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Block of tiles |
| **Unit size** | `block_size` tiles (1 for interleaved; `find_max_block_size(num_tiles_per_shard)` for sharded) |
| **Total units** | `per_core_block_cnt` blocks per core |
| **Loop structure** | Outer loop over `per_core_block_cnt` blocks; inner loop over `per_core_block_size` tiles within each block |

## Tensor Format and Layout

| Property | Input Tensor A | Input Tensor B | Output Tensor |
|----------|---------------|---------------|---------------|
| **Dimension convention** | NHWC (arbitrary rank, flattened to tiles) | Same shape as A | Same shape as A |
| **Tensor layout** | TILE_LAYOUT | TILE_LAYOUT | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED or SHARDED | INTERLEAVED or SHARDED | INTERLEAVED or SHARDED |
| **Buffer type** | DRAM or L1 | DRAM or L1 | DRAM or L1 |
| **Data type** | INT32 or UINT32 | INT32 or UINT32 | INT32 or UINT32 |

### Layout Transformations

No tilize/untilize or format conversions are performed. Both inputs and output must already be in TILE_LAYOUT. The data format for all circular buffers matches the corresponding tensor's data type directly.

## Data Flow Pattern

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader | DRAM/L1 (src0, src1) | CB c_0 (src0), CB c_1 (src1) | reserve_back, push_back (per tile) |
| 2 | Compute | CB c_0 (input A), CB c_1 (input B) | CB c_2 (output) | wait_front, copy_tile to DST, gcd_tile SFPU op, pack_tile, pop_front, push_back |
| 3 | Writer | CB c_2 (output) | DRAM/L1 (dst) | wait_front, pop_front (per tile) |

For GCD, there is no pre-scaling stage (no `SFPU_OP_INIT_PRE_IN0_0` or `SFPU_OP_INIT_PRE_IN1_0` defines), so data flows directly from CB c_0/c_1 through compute to CB c_2.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity (interleaved) | Capacity (sharded) | Block Size | Buffering (interleaved) | Producer | Consumer | Lifetime |
|-------|------|---------|----------------------|-------------------|------------|----------------------|----------|----------|----------|
| c_0 | cb_src0 | Input A staging | 2 tiles | num_tiles_per_shard tiles | 1 tile | Double | Reader | Compute | Program |
| c_1 | cb_src1 | Input B staging | 2 tiles | num_tiles_per_shard tiles | 1 tile | Double | Reader | Compute | Program |
| c_2 | cb_output | Output staging | 2 tiles | num_tiles_per_shard tiles | 1 tile | Double | Compute | Writer | Program |

**Notes**:
- CB c_3 and c_4 (interim buffers for pre-scaling) are NOT created for GCD since it has no `SFPU_OP_INIT_PRE_IN0_0` / `SFPU_OP_INIT_PRE_IN1_0` defines.
- For interleaved mode, capacity = `2 * max_block_size` tiles. With `max_block_size = 1` (interleaved default), this is 2 tiles, enabling double-buffering.
- For sharded inputs, the CB is backed by the globally-allocated tensor buffer and holds the entire shard.

## Pipeline Pattern Summary

- **Interleaved mode**: All three CBs (c_0, c_1, c_2) are double-buffered (capacity = 2 * block_size), allowing overlap between reader/compute and compute/writer stages.
- **Sharded mode**: Input CBs hold the entire shard at once (single logical pass); output CB similarly holds the full shard. No streaming overlap is needed since data is already in L1.

## Index Calculations

- **Interleaved path**: The reader iterates tiles linearly from `start_id` to `start_id + num_tiles`. Tile IDs are passed to `TensorAccessor` which maps logical tile indices to physical DRAM bank addresses using the interleaved page mapping.
- **Block/width-sharded path**: The reader uses a 2D loop over `block_height` x `block_width` tiles. The start tile ID for each core is computed as: `(core_index / num_shards_per_width) * (block_height * block_width * num_shards_per_width) + (core_index % num_shards_per_width) * block_width`. Row strides use `num_cores_y * block_width`.
- **Compute kernel**: Within each block, tiles are copied to DST at positions `i*2` (input A) and `i*2+1` (input B). The GCD result overwrites position `i*2`, which is then packed to the output CB.

## Memory Access Patterns

### Read Pattern
- **Interleaved**: Sequential tile reads from DRAM via NoC0, one tile at a time with `noc_async_read_tile`. Each tile read is followed by `noc_async_read_barrier` before pushing to the CB.
- **Sharded**: No DRAM reads; the CB is directly backed by the L1 shard buffer. Reader simply does `cb_reserve_back` / `cb_push_back` to make tiles available.

### Write Pattern
- **Interleaved**: Sequential tile writes to DRAM via NoC1, one tile at a time with `noc_async_write_page`. Uses `noc_async_writes_flushed` between tiles and a final `noc_async_write_barrier`.
- **Sharded**: No DRAM writes; the output CB is backed by L1. Writer simply does `cb_wait_front` on all output tiles.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (row-major linearization of available cores) or shard-grid for sharded |
| **Grid dimensions** | Determined by `operation_attributes.worker_grid` |
| **Total cores** | `num_cores_total` from worker grid |
| **Work per core** | `num_tiles_per_core_group_1` or `num_tiles_per_core_group_2` tiles |
| **Load balancing** | Two-group split: group 1 gets `ceil(total_tiles / num_cores)` tiles, group 2 gets `floor(total_tiles / num_cores)` tiles. Uses `split_work_to_cores`. |

For sharded tensors, each core processes exactly its shard's tiles (`shard_shape[0] * shard_shape[1] / TILE_HW`), and the core grid comes from the shard spec.

## Arguments

### Compile-Time Arguments

**Reader kernel:**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | block_or_width_sharded | uint32_t | 1 if block/width sharded, 0 otherwise |
| 1+ | TensorAccessorArgs(src0) | varies | Accessor params for input A (omitted if IN0_SHARDED) |
| N+ | TensorAccessorArgs(src1) | varies | Accessor params for input B (omitted if IN1_SHARDED) |

**Writer kernel:**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | output_cb_index | uint32_t | CB index for output (c_2) |
| 1+ | TensorAccessorArgs(dst) | varies | Accessor params for output buffer |

**Compute kernel**: No compile-time args. Behavior is controlled entirely through preprocessor defines:
- `BINOP_INIT` = `gcd_tile_init();`
- `BINARY_SFPU_OP` = `gcd_tile(i*2, i*2+1, i*2);`

### Runtime Arguments

**Reader kernel:**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src0_addr | uint32_t | Input A buffer address |
| 1 | src1_addr | uint32_t | Input B buffer address |
| 2 | num_tiles | uint32_t | Total tiles for this core |
| 3 | start_id | uint32_t | Starting tile ID |
| 4 | block_height | uint32_t | Shard block height in tiles (0 if interleaved) |
| 5 | block_width | uint32_t | Shard block width in tiles (0 if interleaved) |
| 6 | num_cores_y | uint32_t | Number of shards per width dimension |

**Compute kernel:**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_cnt | uint32_t | Number of blocks to process |
| 1 | per_core_block_size | uint32_t | Tiles per block |

**Writer kernel (interleaved/height-sharded):**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Output buffer address |
| 1 | num_pages | uint32_t | Total tiles to write |
| 2 | start_id | uint32_t | Starting tile ID for output |

## Kernel Implementations

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| Reader | BRISC (RISCV_0) | NOC0 | DRAM/L1 src0, src1 | CB c_0, CB c_1 | Read tiles from both inputs |
| Compute | TRISC (math + pack) | N/A | CB c_0, CB c_1 | CB c_2 | copy_tile to DST, gcd_tile SFPU, pack_tile |
| Writer | NCRISC (RISCV_1) | NOC1 | CB c_2 | DRAM/L1 dst | Write output tiles |

### Reader Kernel

| Property | Value |
|----------|-------|
| **File** | `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/dataflow/reader_binary_interleaved_start_id.cpp` |
| **Assigned cores** | All worker cores in `all_device_cores` |

**Key Logic**:
- For sharded inputs (`IN0_SHARDED` / `IN1_SHARDED`): simply calls `cb_reserve_back` + `cb_push_back` to expose the pre-loaded L1 shard data to compute.
- For interleaved inputs with `block_or_width_sharded` mode: uses a 2D loop (`block_height` x `block_width`) reading tiles with strided row access pattern (`row_start_tile_id += num_cores_y * block_width`).
- For fully interleaved inputs: simple linear loop from `start_id` to `start_id + num_tiles`, reading one tile at a time.
- Each tile read uses `noc_async_read_tile` with a `TensorAccessor` for address translation, followed by `noc_async_read_barrier`.
- **Synchronization**: Produces tiles into CB c_0 and CB c_1 via `cb_reserve_back(1)` / `cb_push_back(1)` per tile.

### Compute Kernel

| Property | Value |
|----------|-------|
| **File** | `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/eltwise_binary_sfpu_kernel.cpp` |
| **Assigned cores** | All worker cores in `all_device_cores` |

**Key Logic**:
- Outer loop iterates `per_core_block_cnt` blocks.
- No pre-scaling stage for GCD (no `SFPU_OP_INIT_PRE_IN0_0` / `SFPU_OP_INIT_PRE_IN1_0`).
- Waits for `per_core_block_size` tiles on both CB c_0 and CB c_1, reserves space on CB c_2.
- Acquires tile registers, then copies input A tiles to DST[i*2] and input B tiles to DST[i*2+1] using `copy_tile` with proper data type initialization via `copy_tile_to_dst_init_short_with_dt`.
- For each tile pair, executes `BINOP_INIT` (`gcd_tile_init()`) followed by `BINARY_SFPU_OP` (`gcd_tile(i*2, i*2+1, i*2)`).
- `gcd_tile_init()` records a 28-instruction replay buffer implementing 4 iterations of the binary GCD inner loop.
- `gcd_tile()` loads tile rows from DST into SFPU LREGs, runs `calculate_sfpu_gcd_body<31>()` which performs 30 iterations of the binary GCD algorithm using TTI_REPLAY to replay the recorded loop body, then stores the result back to DST.
- The binary GCD algorithm: takes absolute values, ensures b is odd by removing trailing zeros (using `SFPLZ` + `SFPSHFT2`), then iteratively computes `a = b - a` keeping a even and swapping to maintain `b < a`, converging to `gcd` in register LREG1.
- Packs result from DST[i*2] to CB c_2 via `pack_tile`.
- **Synchronization**: Consumes from CB c_0 and CB c_1 via `cb_wait_front` / `cb_pop_front`. Produces to CB c_2 via `cb_reserve_back` / `cb_push_back`.

### Writer Kernel

| Property | Value |
|----------|-------|
| **File** | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` (standard interleaved case) |
| **Assigned cores** | All worker cores in `all_device_cores` |

**Key Logic**:
- For sharded output (`OUT_SHARDED`): simply calls `cb_wait_front(cb_id_out, num_pages)` -- output data is already in L1 via the globally-allocated CB buffer.
- For interleaved output: iterates from `start_id` to `start_id + num_pages`, writing one tile at a time using `noc_async_write_page` with `TensorAccessor` for address mapping.
- Uses `noc_async_writes_flushed` after each tile write for flow control, and `noc_async_write_barrier` at the end.
- **Synchronization**: Consumes from CB c_2 via `cb_wait_front(1)` / `cb_pop_front(1)` per tile.

## Implementation Notes

- **Program factory variants**: Only the `ElementWiseMultiCoreSfpu` factory is used for GCD. The `ElementWiseMultiCore` (FPU) factory is never selected since `is_binary_sfpu_op` always returns `true` for GCD with valid dtypes. A separate writer kernel (`writer_unary_sharded_blocks_interleaved_start_id.cpp`) is selected when the input is block/width-sharded but the output is interleaved.
- **Type-based operation variants**: GCD supports only INT32/INT32 and UINT32/UINT32 input pairs. No floating-point path exists. The defines are identical for both integer types (`gcd_tile_init` / `gcd_tile`).
- **UnpackToDestFP32 mode**: Enabled for all input CBs (c_0, c_1, c_3, c_4) since `op_type != BinaryOpType::POWER`. This ensures 32-bit integer data is unpacked to DEST at full precision.
- **Broadcast type selection**: N/A. GCD via this factory requires both inputs to have identical height and width. Broadcasting would route through different program factories.
- **Sharding support and constraints**: Supports height-sharded, width-sharded, and block-sharded memory layouts. Any combination of sharded/interleaved inputs and output is supported. For block/width-sharded input with interleaved output, a specialized writer kernel handles the reshuffling.
- **FP32 dest accumulation**: Enabled when output dtype is Float32, Int32, or UInt32. For GCD (always integer), this is always enabled, ensuring the 32-bit integer DST accumulator retains full precision.

## External Knowledge Sources

### DeepWiki Queries
1. **Query**: "How does the binary element-wise SFPU program factory work in ttnn? What kernels does it use and how does it handle broadcasting?"
   **Reason**: Needed architectural context for the legacy binary SFPU path before reading source code.
   **Key Findings**: Confirmed the three kernels used (reader_binary_interleaved_start_id, eltwise_binary_sfpu_kernel, writer_unary_interleaved_start_id), the factory selection logic via `is_binary_sfpu_op`, and that broadcasting routes to different factories.

### Documentation References
1. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary/common/binary_op_utils.cpp` (lines 47-52, 358-361, 535)
   **Reason**: To verify how GCD defines are generated and what preprocessor macros control the compute kernel.
   **Key Information**: GCD sets `BINOP_INIT = gcd_tile_init()` and `BINARY_SFPU_OP = gcd_tile(i*2, i*2+1, i*2)`. No pre-scaling defines are set.

2. **Source**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_gcd.h`
   **Reason**: To understand the SFPU-level GCD algorithm implementation.
   **Key Information**: Implements the binary GCD algorithm using SFPU instructions (SFPABS, SFPAND, SFPLZ, SFPSHFT2, SFPSWAP, SFPIADD). Uses TTI_REPLAY for loop unrolling -- init records 4 iterations into replay buffer, then body replays 7*4 + 7*2-1 = 41 instruction groups for 30 total iterations. Handles 31-bit signed integers. Loads/stores tile rows via SFPLOAD/SFPSTORE with INT32 data format (mode 4, format 3).
