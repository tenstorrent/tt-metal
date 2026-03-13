# GCD (Binary Legacy) Implementation Analysis

## Overview

The GCD (Greatest Common Divisor) operation computes the element-wise greatest common divisor of two integer tensors: `output[i] = gcd(a[i], b[i])`. It operates on INT32 or UINT32 data types and uses the binary GCD algorithm implemented entirely on the SFPU.

**Program factory path**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/element_wise_multi_core_sfpu_pgm_factory.cpp`

## Path Selection: FPU vs SFPU

The `select_program_factory` method in `binary_device_operation.cpp` determines the path:

1. If both tensors have equal height and width, `is_binary_sfpu_op()` is consulted.
2. For `BinaryOpType::GCD`, `is_binary_sfpu_op` returns `true` when both inputs are `INT32/INT32` or `UINT32/UINT32` (line 47-52 of `binary_device_operation.cpp`).
3. When true, `ElementWiseMultiCoreSfpu` is selected (the SFPU path).
4. GCD has **no FPU path** -- it is exclusively an SFPU operation. The FPU path (`ElementWiseMultiCore`) would only be reached if the dtype check failed, which would cause a fatal error.

All remaining sections cover only the SFPU path.

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | tile (32x32 elements) |
| **Unit size** | 1 tile (interleaved) or `max_block_size` tiles (sharded) |
| **Total units** | `physical_volume / TILE_HW` tiles total |
| **Loop structure** | Outer loop over `per_core_block_cnt` blocks; inner loop over `per_core_block_size` tiles per block |

For interleaved tensors: `block_cnt = num_tiles_per_core`, `block_size = 1`.
For sharded tensors: `block_cnt = num_tiles_per_shard / max_block_size`, `block_size = max_block_size`.

## Tensor Format and Layout

### Input Tensors

| Property | Input Tensor A | Input Tensor B |
|----------|---------------|---------------|
| **Logical shape** | [N, C, H, W] | [N, C, H, W] (same H, W as A) |
| **Dimension convention** | NCHW | NCHW |
| **Tensor layout** | TILE_LAYOUT | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED or SHARDED | INTERLEAVED or SHARDED |
| **Buffer type** | DRAM or L1 | DRAM or L1 |
| **Data type** | INT32 or UINT32 | INT32 or UINT32 (must match A) |

### Output Tensor

| Property | Output Tensor |
|----------|---------------|
| **Logical shape** | Same as input A |
| **Dimension convention** | NCHW |
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED or SHARDED |
| **Buffer type** | DRAM or L1 |
| **Data type** | INT32 or UINT32 (matches input) |

### Layout Transformations

None. All tensors must already be in TILE_LAYOUT. No tilize/untilize is performed within this operation.

## Data Flow Pattern

1. **Reader** reads tile pairs (one from each input tensor) from DRAM/L1 into CB c_0 and CB c_1.
2. **Compute** waits on both input CBs, copies tiles into DST registers (input A at even indices `i*2`, input B at odd indices `i*2+1`), executes `gcd_tile(i*2, i*2+1, i*2)` via the SFPU, packs result from DST[i*2] to CB c_2.
3. **Writer** reads completed tiles from CB c_2 and writes them to DRAM/L1.

For GCD specifically, there are **no pre-scaling stages** -- `SFPU_OP_INIT_PRE_IN0_0` and `SFPU_OP_INIT_PRE_IN1_0` are NOT defined, so CBs c_3 and c_4 are not created. The `cb_inp0` and `cb_inp1` aliases point directly to `cb_in0` (c_0) and `cb_in1` (c_1).

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_src0 | Input A staging | 2 tiles (interleaved) or `num_tiles_per_shard` (sharded) | 1 tile (interleaved) or `max_block_size` (sharded) | Double (interleaved) / Single (sharded) | Reader | Compute | Block |
| c_1 | cb_src1 | Input B staging | 2 tiles (interleaved) or `num_tiles_per_shard` (sharded) | 1 tile (interleaved) or `max_block_size` (sharded) | Double (interleaved) / Single (sharded) | Reader | Compute | Block |
| c_2 | cb_output | Output staging | 2 tiles (interleaved) or `num_tiles_per_shard` (sharded) | 1 tile (interleaved) or `max_block_size` (sharded) | Double (interleaved) / Single (sharded) | Compute | Writer | Block |

**Notes**:
- Interleaved capacity formula: `2 * max_block_size` where `max_block_size = 1` for interleaved, yielding 2 tiles (double-buffered).
- Sharded capacity: `num_tiles_per_shard` tiles, with the CB backed by the tensor's L1 buffer directly (globally allocated address).
- CBs c_3 and c_4 are **not created** for GCD because no pre-scaling defines are emitted.

## Pipeline Pattern Summary

**Interleaved path**: All three CBs (c_0, c_1, c_2) are double-buffered (capacity = 2 * block_size = 2 tiles). This allows overlap between reader and compute, and between compute and writer.

**Sharded path**: CBs are single-buffered with capacity equal to the full shard size. The globally-allocated-address pattern means the CB directly aliases the sharded tensor buffer in L1.

## Index Calculations

Tile indices are computed linearly. For the interleaved (non-sharded) path:
- `start_id = num_tiles_read` (accumulated across cores)
- Reader iterates `tile_id` from `start_id` to `start_id + num_tiles`
- `TensorAccessor` maps tile IDs to physical DRAM bank addresses

For block/width-sharded paths:
- `start_id = (core_index / num_shards_per_width) * (block_height * block_width * num_shards_per_width) + (core_index % num_shards_per_width) * block_width`
- Reader iterates in a 2D pattern: outer loop over `block_height` rows, inner loop over `block_width` columns, with row stride of `num_cores_y * block_width`

In the compute kernel, DST register indexing uses interleaved pairs:
- Input A tile `i` is placed at DST index `i * 2`
- Input B tile `i` is placed at DST index `i * 2 + 1`
- Output is written from DST index `i * 2` (overwrites input A location)

## Memory Access Patterns

### Read Pattern
- **Interleaved**: Sequential tile reads. Each tile read is a single `noc_async_read_tile` followed by `noc_async_read_barrier`. One tile at a time per input.
- **Sharded**: No reads needed; the CB is backed by the L1 shard buffer directly. Reader just does `cb_reserve_back` / `cb_push_back` to signal data availability.
- **Block/width-sharded with non-sharded inputs**: 2D strided pattern with row-major tile traversal within each shard block.

### Write Pattern
- **Interleaved**: Sequential single-tile writes via `noc_async_write_page`, one page at a time with `noc_async_writes_flushed` between tiles.
- **Sharded output**: No writes needed; compute writes directly into the output shard's L1 buffer via the CB.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D or 2D (depends on worker_grid) |
| **Grid dimensions** | Determined by `operation_attributes.worker_grid` |
| **Total cores** | `num_cores_total` from grid |
| **Work per core** | `num_tiles_per_core_group_1` or `num_tiles_per_core_group_2` tiles |
| **Load balancing** | Two-group: group 1 gets `ceil(total/cores)` tiles, group 2 gets `floor(total/cores)` tiles |

For interleaved tensors, `split_work_to_cores` divides total tiles across available cores, creating two core groups for remainder handling. For sharded tensors, the shard grid defines core assignment directly.

## Arguments

### Compile-Time Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | block_or_width_sharded | uint32_t | 1 if block/width-sharded, 0 otherwise |
| 1+ | TensorAccessorArgs(src0) | varies | Accessor params for input A (omitted if IN0_SHARDED) |
| N+ | TensorAccessorArgs(src1) | varies | Accessor params for input B (omitted if IN1_SHARDED) |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | output_cb_index | uint32_t | CB index for output (c_2) |
| 1+ | TensorAccessorArgs(dst) | varies | Accessor params for output buffer |

#### Compute Kernel

No compile-time arguments. Operation behavior is controlled entirely via defines:
- `BINOP_INIT` = `gcd_tile_init();`
- `BINARY_SFPU_OP` = `gcd_tile(i*2, i*2+1, i*2);`

### Runtime Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src0_addr | uint32_t | Input A buffer address |
| 1 | src1_addr | uint32_t | Input B buffer address |
| 2 | num_tiles | uint32_t | Total tiles for this core |
| 3 | start_id | uint32_t | Starting tile ID |
| 4 | block_height | uint32_t | Block height in tiles (sharded) |
| 5 | block_width | uint32_t | Block width in tiles (sharded) |
| 6 | num_cores_y | uint32_t | Shards per width dimension |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_cnt | uint32_t | Number of blocks to process |
| 1 | per_core_block_size | uint32_t | Tiles per block |

#### Writer Kernel (interleaved output)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Output buffer address |
| 1 | num_pages | uint32_t | Tiles to write |
| 2 | start_id | uint32_t | Starting tile ID for output |

## Kernel Implementations

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| Reader | BRISC (RISCV_0) | NOC0 | DRAM/L1 | CB c_0, CB c_1 | Read input tile pairs |
| Compute | TRISC (RISCV_2) | N/A | CB c_0, CB c_1 | CB c_2 | copy_tile to DST, gcd_tile SFPU op, pack_tile |
| Writer | NCRISC (RISCV_1) | NOC1 | CB c_2 | DRAM/L1 | Write output tiles |

### Reader Kernel

| Property | Value |
|----------|-------|
| **File** | `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/dataflow/reader_binary_interleaved_start_id.cpp` |
| **Assigned cores** | All worker cores |

**Key Logic**:
- For **sharded** inputs: immediately signals availability via `cb_reserve_back(cb, num_tiles)` and `cb_push_back(cb, num_tiles)` -- no actual data movement since CB aliases the L1 shard buffer.
- For **interleaved** inputs: creates a `TensorAccessor` from compile-time args and reads one tile at a time using `noc_async_read_tile`, with a barrier after each pair.
- For **block/width-sharded** read pattern: uses a 2D loop (height x width) with strided row offsets (`row_start_tile_id += num_cores_y * block_width`).
- **Synchronization**: Produces into CB c_0 and CB c_1 via `cb_reserve_back` / `cb_push_back`.

### Compute Kernel

| Property | Value |
|----------|-------|
| **File** | `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/eltwise_binary_sfpu_kernel.cpp` |
| **Assigned cores** | All worker cores |

**Key Logic**:
- Outer loop iterates `per_core_block_cnt` times; inner loop iterates `per_core_block_size` tiles.
- For GCD, the pre-scaling stages (`SFPU_OP_INIT_PRE_IN0_0`, `SFPU_OP_INIT_PRE_IN1_0`) are **not active** -- `cb_inp0 = cb_in0 = c_0` and `cb_inp1 = cb_in1 = c_1`.
- Waits on both input CBs, acquires tile registers, then:
  - Copies input A tiles to DST at even indices (`i*2`) using `copy_tile_to_dst_init_short_with_dt` + `copy_tile`.
  - Copies input B tiles to DST at odd indices (`i*2+1`).
  - For each tile: executes `BINOP_INIT` (`gcd_tile_init()`) then `BINARY_SFPU_OP` (`gcd_tile(i*2, i*2+1, i*2)`).
  - Packs result from DST[i*2] to CB c_2.
- **SFPU GCD algorithm** (in `ckernel_sfpu_gcd.h`): Implements the binary GCD algorithm using SFPU instructions. Loads operands from DST to SFPU LREGs, uses SFPU integer arithmetic (ABS, AND, OR, IADD with 2's complement, LZ for count-leading-zeros, SHFT2 for shifts, SWAP for conditional exchange). The init function programs a replay buffer with 4 iterations of the core loop body (7 instructions each), then the main function replays this buffer 7 times (28 iterations) plus 2 more partial iterations for a total of 30 iterations, sufficient for 31-bit inputs.
- **Synchronization**: Waits on CB c_0 and c_1 (`cb_wait_front`), pops them after processing (`cb_pop_front`), pushes results to CB c_2 (`cb_reserve_back` / `cb_push_back`).

### Writer Kernel

| Property | Value |
|----------|-------|
| **File** | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` |
| **Assigned cores** | All worker cores |

**Key Logic**:
- For **sharded** output: simply `cb_wait_front(cb_id_out, num_pages)` -- data is already in the correct L1 location.
- For **interleaved** output: iterates from `start_id` to `start_id + num_pages`, writing one tile at a time via `noc_async_write_page` with flush between tiles and a final barrier.
- Uses `TensorAccessor` for address computation from compile-time args.
- **Synchronization**: Consumes from CB c_2 via `cb_wait_front` / `cb_pop_front`.

## Implementation Notes

- **Program factory variants**: GCD uses only `ElementWiseMultiCoreSfpu`. When inputs are block/width-sharded but output is interleaved, the writer switches to `writer_unary_sharded_blocks_interleaved_start_id.cpp`. The factory is selected when `is_binary_sfpu_op` returns true for `BinaryOpType::GCD` with matching INT32/UINT32 dtypes.
- **Type-based operation variants**: Supports `INT32/INT32` and `UINT32/UINT32` only. Both map to the same `gcd_tile` SFPU function. No float path exists.
- **UnpackToDestFP32 mode**: Enabled for all non-POWER ops (GCD included). All input CBs (c_0, c_1, c_3, c_4) use `UnpackToDestMode::UnpackToDestFp32`. This ensures 32-bit integer fidelity when loading into the DST register.
- **Broadcast type selection**: N/A. GCD requires both inputs to have the same H and W dimensions. No broadcast modes are supported in this factory.
- **Sharding support and constraints**: Supports HEIGHT_SHARDED, WIDTH_SHARDED, and BLOCK_SHARDED. Either input or the output (or combinations) can be sharded. The CB is globally allocated to the shard buffer when sharded.
- **FP32 dest accumulation**: Enabled when output dtype is Float32, Int32, or UInt32. Since GCD only supports Int32/UInt32, `fp32_dest_acc_en` is always `true` for this operation.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the binary element-wise SFPU program factory work? What kernels does it use and how does it handle different broadcast modes?"
   **Reason**: Needed to understand the overall architecture of the SFPU binary path and how factory selection works.
   **Key Findings**: Confirmed that `ElementWiseMultiCoreSfpu` is selected when inputs have matching H/W and `is_binary_sfpu_op` returns true. Identified the three kernel files (reader, compute, writer) and the broadcast factory variants.

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/binary_device_operation.cpp` (lines 21-52)
   **Reason**: Needed to verify exactly which data types route GCD to the SFPU path.
   **Key Information**: `BinaryOpType::GCD` falls through with `LCM`, `LEFT_SHIFT`, `RIGHT_SHIFT`, `LOGICAL_RIGHT_SHIFT` -- all requiring `INT32/INT32` or `UINT32/UINT32`.

2. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary/common/binary_op_utils.cpp` (lines 358-361, 535)
   **Reason**: Needed to determine the exact defines emitted for GCD.
   **Key Information**: GCD emits `BINOP_INIT = "gcd_tile_init();"` and `BINARY_SFPU_OP = "gcd_tile(i*2, i*2+1, i*2);"`. No pre-scaling defines are emitted.

3. **Source**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_gcd.h`
   **Reason**: Needed to understand the actual SFPU algorithm implementation.
   **Key Information**: Implements the binary GCD algorithm using SFPU instructions with a replay buffer for loop optimization. Uses 30 iterations of the core loop (sufficient for 31-bit inputs). The init function sets up the replay buffer, and the main function calls it with dst register operands. Key SFPU instructions used: SFPABS, SFPAND, SFPOR, SFPIADD (2's complement negation/addition), SFPLZ (count leading zeros), SFPSHFT2 (shift), SFPSWAP (conditional min/max swap), SFPSETCC/SFPENCC (conditional execution).

4. **Source**: `tt_metal/hw/inc/api/compute/gcd.h`
   **Reason**: Needed to understand the compute API surface for GCD.
   **Key Information**: `gcd_tile(idst0, idst1, odst)` takes two DST tile indices and an output DST index. `gcd_tile_init()` initializes the SFPU state. Both inputs must be int32.
