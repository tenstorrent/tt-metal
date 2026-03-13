# EQ (Binary Legacy) Implementation Analysis

## Overview

The EQ (equal) binary legacy operation performs element-wise equality comparison between two tensors, producing an output where each element is 1.0 if the corresponding input elements are equal, and 0.0 otherwise. It is implemented via the SFPU path of the binary legacy program factory.

**Program factory path**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/element_wise_multi_core_sfpu_pgm_factory.cpp`

The operation is a two-step SFPU computation: first subtract the two inputs element-wise (`sub_binary_tile`), then apply a unary equal-to-zero check (`eqz_tile`) on the result. This "subtract then compare to zero" pattern avoids a direct two-operand comparison instruction on the SFPU.

## Path Selection: FPU vs SFPU

The program factory is selected in `BinaryDeviceOperation::select_program_factory()` (file: `binary_device_operation.cpp`). The logic is:

1. If `operation_attributes.scalar.has_value()` (i.e., scalar broadcast), the `BroadcastHeightAndWidthMultiCore` factory is selected -- neither FPU nor SFPU element-wise path.
2. If both tensors have the same height and width (`height_a == height_b and width_a == width_b`), the function calls `utils::is_binary_sfpu_op(op, dtype1, dtype2)`. `BinaryOpType::EQ` is unconditionally listed in the SFPU switch-case (lines 26 of `binary_device_operation.cpp`), so for equal-shaped tensors, **the SFPU path (`ElementWiseMultiCoreSfpu`) is always selected for EQ regardless of data type**.
3. If shapes differ (broadcasting needed), the FPU broadcast factories are used instead.

The FPU path (`ElementWiseMultiCore`) uses `eltwise_binary_kernel.cpp` with FPU-based subtract + unary SFPU comparison. The SFPU path performs the entire computation (subtract + eqz) on the SFPU via `eltwise_binary_sfpu_kernel.cpp`.

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | tile (32x32 elements) |
| **Unit size** | `per_core_block_size` tiles (max block size that evenly divides tiles-per-shard; defaults to 1 for interleaved) |
| **Total units** | `per_core_block_cnt` blocks per core, where `block_cnt = num_tiles_per_core / block_size` |
| **Loop structure** | Outer loop over `per_core_block_cnt` blocks; inner loop over `per_core_block_size` tiles within each block |

## Tensor Format and Layout

### Input Tensors

| Property | Input Tensor A | Input Tensor B |
|----------|---------------|---------------|
| **Logical shape** | Arbitrary (same as B) | Arbitrary (same as A) |
| **Dimension convention** | NHWC / any | NHWC / any |
| **Tensor layout** | TILE_LAYOUT | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED or SHARDED | INTERLEAVED or SHARDED |
| **Buffer type** | DRAM or L1 | DRAM or L1 |
| **Data type** | BFLOAT16 / FLOAT32 / INT32 / UINT32 / UINT16 | BFLOAT16 / FLOAT32 / INT32 / UINT32 / UINT16 |

### Output Tensor

| Property | Output Tensor |
|----------|---------------|
| **Logical shape** | Same as input |
| **Dimension convention** | Same as input |
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED or SHARDED |
| **Buffer type** | DRAM or L1 |
| **Data type** | Same as output config (typically matches input A) |

### Layout Transformations

No tilize/untilize or resharding is performed within the operation. Both inputs and outputs must already be in TILE_LAYOUT. If inputs are sharded, the shard grid must be compatible across inputs and output.

## Data Flow Pattern

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader | DRAM/L1 (src0, src1) | CB c_0, CB c_1 | reserve_back, push_back (per tile or bulk for sharded) |
| 2 | Compute | CB c_0, CB c_1 | CB c_2 | wait_front(c_0), wait_front(c_1), reserve_back(c_2); copy tiles to DST, sub_binary_tile, eqz_tile, pack_tile; pop_front(c_0), pop_front(c_1), push_back(c_2) |
| 3 | Writer | CB c_2 | DRAM/L1 (output) | wait_front, pop_front (per tile; or just wait_front for sharded output) |

For the EQ operation specifically, no pre-scaling is used (no `SFPU_OP_INIT_PRE_IN0_0` or `SFPU_OP_INIT_PRE_IN1_0` defines are emitted), so interim CBs c_3 and c_4 are not created.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_src0 | Input A staging | 2 * max_block_size tiles (interleaved) or num_tiles_per_shard tiles (sharded) | 1 tile (reader pushes 1 at a time for interleaved) | Double (interleaved) / Single (sharded) | Reader | Compute | Program |
| c_1 | cb_src1 | Input B staging | 2 * max_block_size tiles (interleaved) or num_tiles_per_shard tiles (sharded) | 1 tile (reader pushes 1 at a time for interleaved) | Double (interleaved) / Single (sharded) | Reader | Compute | Program |
| c_2 | cb_output | Output staging | 2 * max_block_size tiles (interleaved) or num_tiles_per_shard tiles (sharded/block-width) | per_core_block_size tiles | Double (interleaved) / Single (sharded) | Compute | Writer | Program |

Notes:
- For EQ, interim CBs c_3 and c_4 are **not created** since no pre-scaling defines are emitted.
- When sharded, CB capacity equals `num_tiles_per_shard` and the CB is backed by the globally-allocated tensor buffer in L1.
- `max_block_size` for interleaved is 1, so interleaved capacity is 2 tiles per input CB (double-buffered).

## Pipeline Pattern Summary

**Interleaved path**: CBs c_0 and c_1 have capacity = 2 * block_size, enabling double-buffering. The reader can fill the next tile while compute processes the current one. CB c_2 similarly has capacity = 2 * block_size for writer overlap.

**Sharded path**: All data is pre-loaded; CBs are single-buffered with capacity = num_tiles_per_shard. No reader/writer overlap is needed since data is already in L1.

## Index Calculations

- **Interleaved (non-sharded)**: Tile indices are sequential starting from `start_id`. Each core processes a contiguous range of `num_tiles_per_core` tiles from `[start_id, start_id + num_tiles_per_core)`.
- **Height-sharded**: `start_id` = cumulative tile count; tiles are contiguous per shard.
- **Block/Width-sharded**: `start_id = (core_index / num_shards_per_width) * (block_height * block_width * num_shards_per_width) + (core_index % num_shards_per_width) * block_width`. Reader iterates in a 2D pattern: outer loop over `block_height` rows, inner loop over `block_width` columns, with row stride = `num_cores_y * block_width`.

In the compute kernel, tiles from input A are copied to DST register slots `i*2` and tiles from input B to DST register slots `i*2+1`. The SFPU subtract operates on `(i*2, i*2+1) -> i*2`, then eqz operates on `i*2`.

## Memory Access Patterns

### Read Pattern

- **Interleaved**: Sequential tile reads via `noc_async_read_tile`. One tile at a time per input, with a `noc_async_read_barrier` after each pair of reads. TensorAccessor handles bank-interleaved addressing.
- **Block/Width-sharded (non-sharded input)**: 2D tile reads: row-by-row within a block, row stride = `num_cores_y * block_width` tiles.
- **Sharded input**: No DRAM reads; CB is directly backed by L1 shard buffer. Reader just does `cb_reserve_back` + `cb_push_back` to make tiles visible.

### Write Pattern

- **Interleaved**: Sequential single-tile writes via `noc_async_write_page`. One tile at a time with `noc_async_writes_flushed` between tiles, final `noc_async_write_barrier`.
- **Block/Width-sharded to interleaved**: 2D write pattern respecting unpadded block dimensions (skips padding tiles). Writes all tiles in one batch after `cb_wait_front`.
- **Sharded output**: No DRAM writes; CB is backed by the output L1 buffer. Writer just does `cb_wait_front`.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (interleaved) or shard-grid-defined (sharded) |
| **Grid dimensions** | Determined by `operation_attributes.worker_grid` |
| **Total cores** | `all_device_cores.num_cores()` (or `grid_x * grid_y` for zero-start grids) |
| **Work per core** | `num_tiles_per_core_group_1` or `num_tiles_per_core_group_2` tiles |
| **Load balancing** | Two core groups: group 1 gets `ceil(num_tiles / num_cores)` tiles, group 2 gets one fewer tile. For sharded: all cores get `num_tiles_per_shard` tiles (equal). |

Work splitting uses `tt::tt_metal::split_work_to_cores` for interleaved tensors. For sharded tensors, the shard grid defines the core assignment directly. Cores beyond the active count receive zero-length work (no-op).

## Arguments

### Compile-Time Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | block_or_width_sharded | uint32_t | 1 if block or width sharded, 0 otherwise |
| 1+ | src0 TensorAccessorArgs | uint32_t[] | TensorAccessor parameters for input A (omitted if IN0_SHARDED) |
| N+ | src1 TensorAccessorArgs | uint32_t[] | TensorAccessor parameters for input B (omitted if IN1_SHARDED) |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | output_cb_index | uint32_t | CB index for output (c_2) |
| 1+ | dst TensorAccessorArgs | uint32_t[] | TensorAccessor parameters for output buffer |

#### Compute Kernel

No explicit compile-time args. Behavior is controlled entirely through preprocessor defines:
- `BINARY_SFPU_OP`: `sub_binary_tile(i*2, i*2+1, i*2);` (float) or `sub_int_tile<DataFormat::Int32>(i*2, i*2+1, i*2);` (INT32)
- `SFPU_OP_INIT_0`: `eqz_tile_init();`
- `SFPU_OP_FUNC_0`: `eqz_tile(i*2);` (float), `eqz_tile_int32(i*2);` (INT32), `eqz_tile_uint16(i*2);` (UINT16), or `eqz_tile_uint32(i*2);` (UINT32)

### Runtime Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src0_addr | uint32_t | Source buffer A base address |
| 1 | src1_addr | uint32_t | Source buffer B base address |
| 2 | num_tiles | uint32_t | Total tiles to process on this core |
| 3 | start_id | uint32_t | Starting tile index for this core |
| 4 | block_height | uint32_t | Block height in tiles (sharded) or 0 (interleaved) |
| 5 | block_width | uint32_t | Block width in tiles (sharded) or 0 (interleaved) |
| 6 | num_cores_y | uint32_t | Number of shards per width dimension (sharded) or 0 |

#### Writer Kernel (interleaved / height-sharded output)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Output buffer base address |
| 1 | num_pages | uint32_t | Number of tiles to write |
| 2 | start_id | uint32_t | Starting tile index for writes |

#### Writer Kernel (block/width-sharded to interleaved)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Output buffer base address |
| 1 | block_height_tiles | uint32_t | Block height in tiles |
| 2 | block_width_tiles | uint32_t | Block width in tiles |
| 3 | unpadded_block_height_tiles | uint32_t | Actual (non-padding) rows in block |
| 4 | unpadded_block_width_tiles | uint32_t | Actual (non-padding) columns in block |
| 5 | output_width_tiles | uint32_t | Full output width in tiles |
| 6 | block_num_tiles | uint32_t | Total tiles in block (height * width) |
| 7 | start_id_offset | uint32_t | Starting tile offset for this core |
| 8 | start_id_base | uint32_t | Base starting tile id (always 0) |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_cnt | uint32_t | Number of blocks to process |
| 1 | per_core_block_size | uint32_t | Number of tiles per block |

## Kernel Implementations

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| Reader | BRISC (RISCV_0) | NOC0 | DRAM/L1 src0, src1 | CB c_0, CB c_1 | Read tiles from both inputs |
| Compute | TRISC (compute) | N/A | CB c_0, CB c_1 | CB c_2 | copy_tile to DST, sub_binary_tile, eqz_tile, pack_tile |
| Writer | NCRISC (RISCV_1) | NOC1 | CB c_2 | DRAM/L1 output | Write output tiles |

### Reader Kernel

| Property | Value |
|----------|-------|
| File | `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/dataflow/reader_binary_interleaved_start_id.cpp` |
| Assigned cores | All worker cores in `all_device_cores` |

**Key Logic**:
- If `IN0_SHARDED` or `IN1_SHARDED` is defined, the corresponding input is already in L1; the reader merely calls `cb_reserve_back(cb_id, num_tiles)` followed by `cb_push_back(cb_id, num_tiles)` to make the full shard visible to compute.
- For non-sharded inputs in the `block_or_width_sharded` path, tiles are read in a 2D loop: outer `block_height`, inner `block_width`, with `row_start_tile_id += num_cores_y * block_width` stride between rows.
- For non-sharded inputs in the standard (height-sharded or interleaved) path, tiles are read sequentially in a simple `for` loop from `start_id` to `start_id + num_tiles`.
- Each non-sharded tile read uses `noc_async_read_tile` via TensorAccessor, followed by `noc_async_read_barrier` before pushing to the CB.
- **Synchronization**: Produces to CB c_0 and CB c_1. For each tile: `cb_reserve_back -> noc_async_read_tile -> noc_async_read_barrier -> cb_push_back`.

### Compute Kernel

| Property | Value |
|----------|-------|
| File | `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/eltwise_binary_sfpu_kernel.cpp` |
| Assigned cores | All worker cores in `all_device_cores` |

**Key Logic**:
- Outer loop iterates `per_core_block_cnt` times (one iteration per block).
- For EQ, no pre-scaling path is active (no `SFPU_OP_INIT_PRE_IN0_0` or `SFPU_OP_INIT_PRE_IN1_0`), so `cb_inp0 = cb_in0 = c_0` and `cb_inp1 = cb_in1 = c_1`.
- Waits on `per_core_block_size` tiles from both input CBs, then reserves output CB space.
- Acquires tile registers (`tile_regs_acquire` + `tile_regs_wait`).
- Copies input A tiles to DST slots `i*2` via `copy_tile(cb_inp0, i, i*2)`.
- Switches copy data type and copies input B tiles to DST slots `i*2+1` via `copy_tile(cb_inp1, i, i*2+1)`.
- For each tile in the inner loop:
  - Executes `BINARY_SFPU_OP` which expands to `sub_binary_tile(i*2, i*2+1, i*2);` -- subtracts B from A, result in DST[i*2].
  - Executes `SFPU_OP_INIT_0` which expands to `eqz_tile_init();` -- initializes the equal-to-zero comparison.
  - Executes `SFPU_OP_FUNC_0` which expands to `eqz_tile(i*2);` -- applies EQZ on the subtraction result, producing 1.0 or 0.0.
  - Packs result from DST[i*2] to output CB via `pack_tile(i*2, cb_out0)`.
- After inner loop: `tile_regs_commit` + `tile_regs_release`, then `cb_pop_front` on both inputs, `cb_push_back` on output.
- **Synchronization**: Consumes from CB c_0 and CB c_1 (`cb_wait_front` / `cb_pop_front`); produces to CB c_2 (`cb_reserve_back` / `cb_push_back`).

### Writer Kernel

| Property | Value |
|----------|-------|
| File | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` (standard) or `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded_blocks_interleaved_start_id.cpp` (block/width-sharded to interleaved) |
| Assigned cores | All worker cores in `all_device_cores` |

**Key Logic (standard writer)**:
- If `OUT_SHARDED` is defined, simply calls `cb_wait_front(cb_id_out, num_pages)` and returns -- data is already in the output L1 buffer.
- Otherwise, iterates sequentially from `start_id` to `start_id + num_pages`, writing one tile at a time: `cb_wait_front(1) -> get_read_ptr -> noc_async_write_page -> noc_async_writes_flushed -> cb_pop_front(1)`.
- Final `noc_async_write_barrier` ensures all writes complete.
- **Synchronization**: Consumes from CB c_2 (`cb_wait_front` / `cb_pop_front`).

**Key Logic (block/width-sharded to interleaved writer)**:
- Waits for all `block_num_tiles` tiles at once via `cb_wait_front`.
- Writes in a 2D pattern: iterates over `unpadded_block_height_tiles` rows and `unpadded_block_width_tiles` columns, skipping padding tiles.
- Row stride in output tile space is `output_width_tiles`.
- Single `noc_async_write_barrier` at end, then `cb_pop_front`.

## Implementation Notes

- **Program factory variants**: `ElementWiseMultiCoreSfpu` is the only factory for EQ when input shapes match (no broadcasting). When broadcasting is needed, `BroadcastWidthMultiCore`, `BroadcastHeightMultiCore`, or `BroadcastHeightAndWidthMultiCore` factories are selected instead, which use the FPU path with a different compute kernel.

- **Type-based operation variants**: The SFPU operation adapts to data types: for INT32 inputs, `sub_int_tile<DataFormat::Int32>` and `eqz_tile_int32` are used; for UINT32, `sub_int_tile<DataFormat::UInt32>` and `eqz_tile_uint32`; for UINT16, `sub_int_tile<DataFormat::UInt16>` and `eqz_tile_uint16`; for float types (BFLOAT16/FLOAT32), `sub_binary_tile` and `eqz_tile`.

- **UnpackToDestFP32 mode**: Since `BinaryOpType::EQ` is not `POWER`, all input CBs (c_0, c_1, c_3, c_4) are set to `UnpackToDestMode::UnpackToDestFp32` unconditionally. This ensures all data is unpacked to FP32 precision in the DST register before SFPU operations.

- **Broadcast type selection**: N/A for the SFPU path. The SFPU factory is only selected when both inputs have identical height and width -- no broadcasting occurs. Broadcasting is handled by separate program factories.

- **Sharding support and constraints**: All sharding modes are supported (height, width, block). The shard spec can come from input A, input B, or the output. Mixed configurations are allowed (e.g., sharded input with interleaved output). When block/width-sharded input feeds into interleaved output, a specialized block writer kernel handles the 2D-to-linear tile mapping.

- **FP32 dest accumulation**: Enabled when the output data format is Float32, Int32, or UInt32 (`fp32_dest_acc_en` flag). This ensures the DEST register uses full FP32 precision for accumulation, which is important for integer subtraction accuracy and for float32 output fidelity.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the binary element-wise SFPU program factory work in ttnn? What is the structure of element_wise_multi_core_sfpu_pgm_factory.cpp and how does it differ from the FPU path?"
   **Reason**: To understand the overall architecture of the SFPU binary path, its kernel structure, and how it differs from the FPU path.
   **Key Findings**: Confirmed the three-kernel structure (reader, compute, writer), the use of `eltwise_binary_sfpu_kernel.cpp` for compute, the role of `UnpackToDestFp32` mode, and the interim CB mechanism for pre-scaling operations. The SFPU path explicitly loads data into DST registers before SFPU function calls, while the FPU path can work more directly on CB data.

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/binary_device_operation.cpp`
   **Reason**: To understand program factory selection logic and when the SFPU path is chosen.
   **Key Information**: `is_binary_sfpu_op` returns true for EQ unconditionally. SFPU path selected when input shapes match (no broadcasting).

2. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary/common/binary_op_utils.cpp` (function `get_defines_fp32`)
   **Reason**: To determine exact SFPU defines generated for EQ operation.
   **Key Information**: EQ generates `BINARY_SFPU_OP` = `sub_binary_tile(i*2, i*2+1, i*2)` (or integer variant), plus `SFPU_OP_INIT_0` / `SFPU_OP_FUNC_0` for `eqz_tile`. No pre-scaling defines are emitted.

3. **Source**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: To trace how `get_defines(UnaryOpType::EQZ, ...)` expands into SFPU init/func macros.
   **Key Information**: `EQZ` expands to `eqz_tile_init()` / `eqz_tile(idst)` for float, with integer variants for INT32, UINT16, UINT32.

4. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/eltwise_multi_core_program_factory_common.hpp`
   **Reason**: To understand runtime argument setup, core distribution, and work splitting logic.
   **Key Information**: Uses `split_work_to_cores` for interleaved; shard grid for sharded. Two core groups handle remainder tiles. Block/width-sharded paths have specialized start_id calculations and a different writer kernel.
