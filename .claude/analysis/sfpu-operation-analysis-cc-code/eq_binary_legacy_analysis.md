# EQ (Binary Legacy SFPU) Implementation Analysis

## Overview
The EQ binary operation compares two tensors element-wise for equality, producing 1.0 where elements are equal and 0.0 where they differ. It is implemented as a two-step SFPU operation: first subtract the two inputs (`A - B`), then apply the unary `eqz` (equal-to-zero) comparison on the result. The SFPU path is selected when both input tensors have matching data types of FLOAT32, INT32, UINT32, or UINT16.

**Program factory path**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/element_wise_multi_core_sfpu_pgm_factory.cpp`

## Path Selection: FPU vs SFPU

The factory selection occurs in `binary_device_operation.cpp` (`select_program_factory`). When the two input tensors have the same shape (height_a == height_b, width_a == width_b), `is_binary_sfpu_op()` is consulted. For `BinaryOpType::EQ`, it returns true when both inputs share the same dtype and that dtype is one of: FLOAT32, INT32, UINT32, UINT16. If this check fails (e.g., mixed dtypes or BF16 inputs), the FPU path `ElementWiseMultiCore` is used instead. The FPU path is not analyzed further here.

## Work Unit Definition
One work unit is one **tile** (32x32 elements). Work is distributed across cores in blocks of tiles, where `block_size` is the largest power-of-2 divisor of the per-core tile count (capped at 8). For the interleaved (non-sharded) case, `block_size` defaults to 1, so each work unit is a single tile.

## Tensor Format and Layout

### Input Tensor(s)

| Property | Input A (src0) | Input B (src1) |
|---|---|---|
| Rank | N-dimensional (any) | Same shape as A |
| Dimension convention | [..., H, W] | [..., H, W] |
| Tensor layout | TILE (32x32) | TILE (32x32) |
| Memory layout | Interleaved or Sharded | Interleaved or Sharded |
| Buffer type | DRAM or L1 | DRAM or L1 |
| Data type | FLOAT32, INT32, UINT32, or UINT16 (must match B) | Same as A |

### Output Tensor(s)

| Property | Output |
|---|---|
| Rank | Same as inputs |
| Dimension convention | [..., H, W] |
| Tensor layout | TILE (32x32) |
| Memory layout | Interleaved or Sharded |
| Buffer type | DRAM or L1 |
| Data type | Same as input dtype (output contains 1.0/0.0 for float, 1/0 for int) |

### Layout Transformations
No tilize/untilize is performed. Both inputs and output must already be in tiled layout.

## Data Flow Pattern

1. **Reader kernel** reads tiles of input A into CB c_0 and tiles of input B into CB c_1 from DRAM/L1 (or marks them as available if sharded).
2. **Compute kernel** waits for tiles in CB c_0 and CB c_1.
3. Copies input A tiles to even DST registers (i*2) and input B tiles to odd DST registers (i*2+1).
4. Executes `BINARY_SFPU_OP` which performs subtraction: `sub_binary_tile(i*2, i*2+1, i*2)` for float, or `sub_int_tile<DataFormat::Int32>(i*2, i*2+1, i*2)` for INT32.
5. Executes `SFPU_OP_INIT_0` / `SFPU_OP_FUNC_0` which applies `eqz_tile(i*2)` (or the int/uint variant), comparing the subtraction result to zero.
6. Packs result from DST register i*2 to output CB c_2.
7. **Writer kernel** reads completed tiles from CB c_2 and writes them to DRAM/L1 (or does nothing if output is sharded).

## Circular Buffer Configuration

| CB ID | Name | Purpose | Data Format | Capacity (tiles) | Block Size (tiles) | Buffering | Producer | Consumer |
|---|---|---|---|---|---|---|---|---|
| c_0 | cb_src0 | Input A tiles | src0 dtype format | 2*max_block_size (interleaved) or num_tiles_per_shard (sharded) | 1 (interleaved) / max_block_size (sharded) | Double (interleaved) / Single (sharded) | Reader | Compute |
| c_1 | cb_src1 | Input B tiles | src1 dtype format | 2*max_block_size (interleaved) or num_tiles_per_shard (sharded) | 1 (interleaved) / max_block_size (sharded) | Double (interleaved) / Single (sharded) | Reader | Compute |
| c_2 | cb_out0 | Output tiles | output dtype format | 2*max_block_size (interleaved) or num_tiles_per_shard (sharded) | 1 (interleaved) / max_block_size (sharded) | Double (interleaved) / Single (sharded) | Compute | Writer |

Note: CB c_3 and c_4 (interim buffers for pre-scaling) are **not created** for EQ because EQ does not define `SFPU_OP_INIT_PRE_IN0_0` or `SFPU_OP_INIT_PRE_IN1_0`.

## Pipeline Pattern Summary

- **Interleaved path**: CB capacity = 2 * max_block_size, block_size defaults to 1, so CBs hold 2 tiles. This provides **double buffering** -- the reader can fill the next tile while compute processes the current one.
- **Sharded path**: CB capacity = num_tiles_per_shard (all tiles for the shard). This is **single-buffered** since all shard data is present in L1 from the start via globally allocated address.

## Index Calculations

- **Interleaved reader**: Tiles are read sequentially from `start_id` to `start_id + num_tiles`. `TensorAccessor` maps linear tile IDs to physical DRAM bank addresses.
- **Block/width-sharded reader**: Uses a 2D loop over `block_height` x `block_width`, with `row_start_tile_id` advancing by `num_cores_y * block_width` per row.
- **Compute kernel**: Input A is copied to DST[i*2], Input B to DST[i*2+1]. The subtraction writes result to DST[i*2], and eqz operates in-place on DST[i*2]. Output is packed from DST[i*2].
- **Writer**: Sequential tile-by-tile write from `start_id` using `TensorAccessor` for address mapping.

## Memory Access Patterns

### Read Pattern
- **Interleaved**: Sequential tile reads, one tile at a time per input, with `noc_async_read_barrier()` after each pair.
- **Sharded**: No reads needed; data is already in L1. The reader kernel simply calls `cb_reserve_back` / `cb_push_back` to signal availability.

### Write Pattern
- **Interleaved**: Sequential single-tile writes with `noc_async_writes_flushed()` after each tile, and a final `noc_async_write_barrier()`.
- **Sharded**: No writes needed; output CB is backed by the output tensor's L1 buffer. Writer simply calls `cb_wait_front`.

## Core Distribution Strategy

| Property | Interleaved | Sharded |
|---|---|---|
| Grid topology | Rectangular grid from worker_grid | Shard spec grid |
| Work splitting | `split_work_to_cores()` divides total tiles across cores | Each core processes its shard (num_tiles_per_shard) |
| Core group 1 | Cores with `ceil(num_tiles / num_cores)` tiles | All shard cores |
| Core group 2 | Remaining cores with one fewer tile (or 0) | Empty |
| Remainder handling | Last core(s) may have fewer tiles; excess cores get 0 tiles | N/A -- each shard is fixed size |
| Traversal order | Row-major (interleaved) or shard orientation (sharded) | Per shard_spec orientation |

The factory supports a `zero_start_grid` optimization: when the worker grid is a single rectangle starting at (0,0) and shards also start at (0,0), faster grid-to-core mapping is used.

## Arguments

### Compile-Time Arguments

**Reader Kernel:**

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | block_or_width_sharded | uint32_t | 1 if block or width sharded, 0 otherwise |
| 1+ | TensorAccessorArgs(src0) | varies | Accessor args for input A (omitted if IN0_SHARDED) |
| N+ | TensorAccessorArgs(src1) | varies | Accessor args for input B (omitted if IN1_SHARDED) |

**Writer Kernel:**

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | output_cb_index | uint32_t | CB index for output (c_2) |
| 1+ | TensorAccessorArgs(dst) | varies | Accessor args for output buffer |

**Compute Kernel:**
No compile-time args. Operation is parameterized entirely through `#define` macros passed via `ComputeConfig::defines`.

### Runtime Arguments

**Reader Kernel:**

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | src0_addr | uint32_t | Address of input A buffer |
| 1 | src1_addr | uint32_t | Address of input B buffer |
| 2 | num_tiles | uint32_t | Total tiles to read on this core |
| 3 | start_id | uint32_t | Starting tile ID for this core |
| 4 | block_height | uint32_t | Shard block height in tiles (0 if interleaved) |
| 5 | block_width | uint32_t | Shard block width in tiles (0 if interleaved) |
| 6 | num_cores_y | uint32_t | Number of shards per width dimension (0 if interleaved) |

**Compute Kernel:**

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | per_core_block_cnt | uint32_t | Number of blocks to process on this core |
| 1 | per_core_block_size | uint32_t | Number of tiles per block |

**Writer Kernel (interleaved/height-sharded):**

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | dst_addr | uint32_t | Address of output buffer |
| 1 | num_pages | uint32_t | Number of tiles to write |
| 2 | start_id | uint32_t | Starting tile ID |

**Writer Kernel (block/width-sharded to interleaved):**

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | dst_addr | uint32_t | Address of output buffer |
| 1 | block_height | uint32_t | Shard block height in tiles |
| 2 | block_width | uint32_t | Shard block width in tiles |
| 3 | unpadded_block_height | uint32_t | Unpadded block height |
| 4 | unpadded_block_width | uint32_t | Unpadded block width |
| 5 | output_width | uint32_t | Total output width in tiles |
| 6 | block_size | uint32_t | block_width * block_height |
| 7 | start_id | uint32_t | Starting tile ID |

## Kernel Implementations

| Kernel | File | Type | Assigned Cores |
|---|---|---|---|
| Reader | `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/dataflow/reader_binary_interleaved_start_id.cpp` | DataMovement (Reader) | all_device_cores |
| Compute | `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/eltwise_binary_sfpu_kernel.cpp` | Compute | all_device_cores |
| Writer | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | DataMovement (Writer) | all_device_cores |

### Reader Kernel

| Property | Value |
|---|---|
| File | `reader_binary_interleaved_start_id.cpp` |
| Assigned cores | all_device_cores |

**Key Logic:**
- If `IN0_SHARDED` is defined, immediately calls `cb_reserve_back(c_0, num_tiles)` and `cb_push_back(c_0, num_tiles)` to mark sharded data as available. Same for `IN1_SHARDED` with c_1.
- For interleaved inputs, constructs `TensorAccessor` objects from compile-time args and base addresses.
- **Block/width-sharded path**: Uses 2D tile iteration over `block_height` x `block_width`. Row start advances by `num_cores_y * block_width` to account for multi-core stride in the tiled layout.
- **Interleaved path**: Simple sequential loop from `start_id` to `start_id + num_tiles`, reading one tile at a time per input.
- Each tile read uses `noc_async_read_tile()` followed by `noc_async_read_barrier()` before pushing to CB.
- **Synchronization**: Produces to CB c_0 and CB c_1 via `cb_reserve_back` / `cb_push_back`.

### Compute Kernel

| Property | Value |
|---|---|
| File | `eltwise_binary_sfpu_kernel.cpp` |
| Assigned cores | all_device_cores |

**Key Logic:**
- Receives `per_core_block_cnt` (number of blocks) and `per_core_block_size` (tiles per block) as runtime args.
- For EQ, no pre-scaling is needed (no `SFPU_OP_INIT_PRE_IN0_0` or `SFPU_OP_INIT_PRE_IN1_0` defined), so `cb_inp0 = cb_in0 = c_0` and `cb_inp1 = cb_in1 = c_1`.
- Main loop iterates `per_core_block_cnt` times:
  1. `cb_wait_front(c_0, block_size)` and `cb_wait_front(c_1, block_size)` -- waits for reader to provide data.
  2. `cb_reserve_back(c_2, block_size)` -- ensures space in output CB.
  3. `tile_regs_acquire()` and `tile_regs_wait()` -- acquires DST registers.
  4. Copies input A tiles to DST even slots: `copy_tile(c_0, i, i*2)`.
  5. Switches unpack config, copies input B tiles to DST odd slots: `copy_tile(c_1, i, i*2+1)`.
  6. For each tile, applies the binary SFPU op (subtraction) and then the unary comparison:
     - **FLOAT32**: `sub_binary_tile(i*2, i*2+1, i*2)` then `eqz_tile_init(); eqz_tile(i*2);`
     - **INT32**: `sub_int_tile<DataFormat::Int32>(i*2, i*2+1, i*2)` then `eqz_tile_init(); eqz_tile_int32(i*2);`
     - **UINT32**: `sub_int_tile<DataFormat::UInt32>(i*2, i*2+1, i*2)` then `eqz_tile_init(); eqz_tile_uint32(i*2);`
     - **UINT16**: `sub_int_tile<DataFormat::UInt16>(i*2, i*2+1, i*2)` then `eqz_tile_init(); eqz_tile_uint16(i*2);`
  7. `pack_tile(i*2, c_2)` -- packs result from even DST slot to output CB.
  8. `tile_regs_commit()` / `tile_regs_release()` -- releases DST registers.
  9. `cb_pop_front(c_0, block_size)` and `cb_pop_front(c_1, block_size)` -- frees input CBs.
  10. `cb_push_back(c_2, block_size)` -- signals output ready for writer.

### Writer Kernel

| Property | Value |
|---|---|
| File | `writer_unary_interleaved_start_id.cpp` |
| Assigned cores | all_device_cores |

**Key Logic:**
- If `OUT_SHARDED` is defined, simply calls `cb_wait_front(c_2, num_pages)` and returns (data is already in the output buffer's L1 location).
- For interleaved output, constructs `TensorAccessor` from compile-time args and base address.
- Iterates sequentially from `start_id` to `start_id + num_pages`, writing one tile at a time.
- **Synchronization**: Consumes from CB c_2 via `cb_wait_front(c_2, 1)` / `cb_pop_front(c_2, 1)`.
- Calls `noc_async_writes_flushed()` after each tile for ordered writes, and `noc_async_write_barrier()` at the end.

## Implementation Notes

- **Program factory variants**: Two program factories exist for binary operations: `ElementWiseMultiCore` (FPU path) and `ElementWiseMultiCoreSfpu` (SFPU path). Selection is based on `is_binary_sfpu_op()` which checks op type and dtype compatibility. Additional factories handle broadcast cases (`BroadcastHeightAndWidthMultiCore`, `BroadcastHeightMultiCore`, `BroadcastWidthMultiCore`).
- **Type-based operation variants**: EQ supports FLOAT32, INT32, UINT32, and UINT16. Each dtype uses a different subtraction kernel (`sub_binary_tile` for float, `sub_int_tile<DataFormat::X>` for integer types) and a different eqz variant (`eqz_tile`, `eqz_tile_int32`, `eqz_tile_uint32`, `eqz_tile_uint16`).
- **UnpackToDestFP32 mode**: Enabled for all CBs (c_0, c_1, c_3, c_4) since the op type is not POWER. This means all inputs are unpacked to FP32 in the DEST register regardless of source dtype.
- **Broadcast type selection**: N/A for this SFPU path. Broadcasting is handled by separate program factories before reaching `ElementWiseMultiCoreSfpu`.
- **Sharding support and constraints**: Supports height-sharded, width-sharded, and block-sharded inputs/outputs. Any combination of sharded/interleaved inputs and output is allowed. When block/width sharded input feeds an interleaved output, a specialized writer kernel (`writer_unary_sharded_blocks_interleaved_start_id.cpp`) is used.
- **FP32 dest accumulation**: Enabled when the output dtype is Float32, Int32, or UInt32. Controlled by `fp32_dest_acc_en` in `ComputeConfig`.

## External Knowledge Sources

### DeepWiki Queries
1. **Query**: "How does the binary element-wise SFPU program factory work in ttnn? What kernels does it use and how does it handle different binary operations like add, sub, mul, eq?"
   **Reason**: Initial orientation to understand the SFPU binary operation architecture and identify kernel files.
   **Key Findings**: Confirmed the three-kernel structure (reader, compute, writer), identified that EQ uses `sub_binary_tile` followed by `eqz_tile`, and learned that operation-specific behavior is controlled through macro defines.

### Documentation References
1. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary/common/binary_op_utils.cpp` (lines 490-501)
   **Reason**: Needed to trace exactly which defines are generated for the EQ operation.
   **Key Information**: EQ uses `sub_binary_tile` (float) or `sub_int_tile<DataFormat::X>` (integer) as the binary op, followed by `get_defines(UnaryOpType::EQZ, ...)` for the unary comparison step.

2. **Source**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp` (lines 667-679)
   **Reason**: Needed to understand what `SFPU_OP_INIT_0` and `SFPU_OP_FUNC_0` expand to for EQZ.
   **Key Information**: EQZ expands to `eqz_tile_init()` and `eqz_tile(idst)` for float, with `_int32`, `_uint16`, `_uint32` variants for integer types.

3. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/binary_device_operation.cpp` (lines 22-66, 69-94)
   **Reason**: Needed to understand the SFPU path selection criteria.
   **Key Information**: `is_binary_sfpu_op(EQ, ...)` returns true when both inputs share the same dtype from {FLOAT32, INT32, UINT32, UINT16}. The SFPU factory is only selected when input shapes match exactly.

4. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/eltwise_multi_core_program_factory_common.hpp`
   **Reason**: Needed to understand the runtime argument setup and core distribution logic.
   **Key Information**: The `set_eltwise_binary_runtime_args` template handles both initial argument setup and cached override, manages sharded vs interleaved paths, and computes per-core tile assignments with two core groups for remainder handling.
