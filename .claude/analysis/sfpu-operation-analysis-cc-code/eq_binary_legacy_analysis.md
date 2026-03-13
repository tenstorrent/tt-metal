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

## SFPU Kernel Implementation

This section provides a dedicated deep dive into the underlying SFPU kernel functions that the compute kernel dispatches to. The EQ binary legacy operation uses two sequential SFPU operations: (1) a binary subtract (`sub_binary_tile`) and (2) a unary equal-to-zero comparison (`eqz_tile`).

### SFPU Abstraction Layers

The EQ operation involves two distinct SFPU call chains -- one for the binary subtract and one for the unary eqz comparison. Both are documented below.

**Binary subtract (`sub_binary_tile`):**

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_binary_sfpu.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_binop.h` |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_binary.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_binary_sfpu_params.h` |

**Unary eqz (`eqz_tile`):**

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/comp.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h` (macro `SFPU_ZERO_KERNEL`) |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_comp.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |

### Call Chain

**Binary subtract path:**
1. The compute kernel calls `sub_binary_tile(i*2, i*2+1, i*2)` (defined in `eltwise_binary_sfpu.h`).
2. This calls `llk_math_eltwise_binary_sfpu_binop<APPROX, BinaryOp::SUB>(idst0, idst1, odst)` (in `llk_math_eltwise_binary_sfpu_binop.h`).
3. That dispatches to `_llk_math_eltwise_binary_sfpu_params_<APPROX>(calculate_sfpu_binary<APPROX, BinaryOp::SUB, 8, is_fp32_dest_acc_en>, idst0, idst1, odst, VectorMode::RC)` (in `llk_math_eltwise_binary_sfpu_params.h`), which iterates over 4 faces and calls the SFPU function per face.
4. `calculate_sfpu_binary` (in the WH-specific `ckernel_sfpu_binary.h`) delegates to `_calculate_sfpu_binary_<APPROX, BinaryOp::SUB, 8>()` (in the common `ckernel_sfpu_binary.h`), which performs `result = in0 - in1` using SFPI vector subtraction.

**Unary eqz path:**
1. The compute kernel calls `eqz_tile_init()` then `eqz_tile(i*2)` (defined in `comp.h`).
2. `eqz_tile_init()` expands to `llk_math_eltwise_unary_sfpu_init<SfpuType::equal_zero, APPROX>()`, which configures ADDR_MOD_7 and resets counters.
3. `eqz_tile(idst)` expands via `SFPU_ZERO_KERNEL(equal_zero, RC, APPROX, idst)` to `_llk_math_eltwise_unary_sfpu_params_<APPROX>(calculate_comp<APPROX, SfpuType::equal_zero>, idst, VectorMode::RC, 8)`.
4. The params dispatch iterates over 4 faces, calling `calculate_comp<APPROX, SfpuType::equal_zero>(8)` per face. Each call processes 8 rows of the face (one `dst_reg++` per row).
5. `calculate_comp` (in the WH-specific `ckernel_sfpu_comp.h`) checks `_sfpu_is_fp16_zero_(v, exponent_size_8)` to determine if each element equals zero, writing 1.0f or 0.0f accordingly.

### Parameters Dispatch Summary

**Binary subtract (`_llk_math_eltwise_binary_sfpu_params_`):**

- **Vector mode**: `VectorMode::RC` -- all 4 faces of the tile are processed.
- **Operation invocation**: A loop over 4 faces calls `calculate_sfpu_binary(dst_index_in0, dst_index_in1, dst_index_out)` once per face. Each invocation internally loops 8 times (ITERATIONS=8), processing 8 rows per face (32 rows total = one full 32x32 tile).
- **DEST address progression**: After each face, `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` is called twice to advance the DEST read/write pointer by 16 rows (2 x 8 rows). Within the SFPU function, `dst_reg++` auto-increments the pointer by 1 row per iteration (using ADDR_MOD_7 with dest.incr=0, relying on SFPI's built-in `dst_reg++` which uses SFP_DESTREG_STRIDE).

**Unary eqz (`_llk_math_eltwise_unary_sfpu_params_`):**

- **Vector mode**: `VectorMode::RC` -- all 4 faces of the tile are processed.
- **Operation invocation**: A loop over 4 faces calls `calculate_comp<APPROX, SfpuType::equal_zero>(exponent_size_8)` once per face. Each invocation internally loops 8 times (ITERATIONS=8), processing 8 rows per face.
- **DEST address progression**: After each face, `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` is called twice to advance by 16 rows. Within `calculate_comp`, `dst_reg++` advances by 1 row per iteration. Before dispatching, `math::set_dst_write_addr` sets the initial DEST address based on `dst_index`, and `math::set_addr_mod_base()` switches to ADDR_MOD bank 4..7 (where ADDR_MOD_7 is configured with all-zero increments so the base SFPI `dst_reg++` stride is the only progression).

### Annotated SFPU Kernel Source

**Step 1: Binary Subtract (`_calculate_sfpu_binary_` with `BINOP=BinaryOp::SUB`)**

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_binary.h

template <bool APPROXIMATION_MODE, BinaryOp BINOP, int ITERATIONS = 8>
inline void _calculate_sfpu_binary_(const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out)
{ // For EQ: BINOP=BinaryOp::SUB, ITERATIONS=8
    static constexpr float nan = std::numeric_limits<float>::quiet_NaN();
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++)
    {
        // size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
        constexpr std::uint32_t dst_tile_size_sfpi = 32; // offset between tiles in DEST when using SFPI indexing
        sfpi::vFloat in0                           = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi]; // load row from tile A
        sfpi::vFloat in1                           = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi]; // load row from tile B
        sfpi::vFloat result                        = 0.0f;

        // Only the SUB branch is active for EQ; other branches compiled away
        if constexpr (BINOP == BinaryOp::SUB)
        {
            result = in0 - in1; // SFPIADD with negated in1 (SFPI subtraction)
        }
        // ... other BINOP branches omitted (ADD, MUL, DIV, RSUB, POW, XLOGY) ...

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result; // store result back to tile A's DEST slot
        sfpi::dst_reg++; // advance to next row within the face
    }
}
```

**Step 2: Equal-to-Zero Comparison (`calculate_comp` with `COMP_MODE=SfpuType::equal_zero`)**

```cpp
// File: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_comp.h

template <bool APPROXIMATION_MODE, SfpuType COMP_MODE, int ITERATIONS = 8>
inline void calculate_comp(uint exponent_size_8) {
    // For EQ: COMP_MODE=SfpuType::equal_zero, ITERATIONS=8
    const vFloat zero = 0.0f;
    const vFloat one = 1.0f;
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[0]; // load current row from DEST

        // a[i] == 0: only this branch is active for equal_zero
        if constexpr (COMP_MODE == SfpuType::equal_zero) {
            v_if(_sfpu_is_fp16_zero_(v, exponent_size_8)) { v = one; }  // element is zero -> output 1.0
            v_else { v = zero; }                                         // element is non-zero -> output 0.0
            v_endif;
        }
        // ... other COMP_MODE branches omitted (not_equal_zero, less_than_zero, etc.) ...

        dst_reg[0] = v; // write result back to DEST
        dst_reg++;       // advance to next row
    }
}
```

**Helper: `_sfpu_is_fp16_zero_`**

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_is_fp16_zero.h

sfpi_inline sfpi::vInt _sfpu_is_fp16_zero_(const sfpi::vFloat& v, std::uint32_t exponent_size_8)
{
    if (exponent_size_8)
    {
        // fp16b / bfloat16 / fp32: 8-bit exponent, direct comparison works
        return v == 0.0F;
    }
    else
    {
        // fp16a: 5-bit exponent was extended to 8-bit by adding bias of 112
        // A true zero in fp16a becomes 0x3800 after bias addition
        // So add 0x3800 and check if the sum is zero (overflow wraps to 0 for true zero)
        sfpi::vInt tmp = 0x3800; // loads {0, 8'd112, 10'b0}
        tmp += sfpi::reinterpret<sfpi::vInt>(v);
        return tmp == 0;
    }
}
```

### SFPU Instructions Used

The EQ binary legacy operation's SFPU kernels use SFPI abstractions that compile down to the following underlying SFPU instructions:

| Instruction / Intrinsic | Description |
|------------------------|-------------|
| `SFPLOAD` (via `dst_reg[offset]` read) | Loads a vector row from the DEST register file into an SFPU local register (LREG). The offset indexes into DEST relative to the current base address. |
| `SFPSTORE` (via `dst_reg[offset] = val` write) | Stores a vector row from an SFPU local register back to the DEST register file at the specified offset. |
| `SFPIADD` (via `in0 - in1`, integer add/compare) | Used for subtraction (negate + add) and for integer zero-comparison. The `v == 0.0F` comparison in `_sfpu_is_fp16_zero_` compiles to SFPIADD-based equality check that sets condition codes. |
| `SFPMAD` (via `in0 - in1` float subtract) | Multiply-add instruction used to implement floating-point subtraction: `in0 + (-1.0) * in1`. The SFPI compiler may use this for the `in0 - in1` operation. |
| `SFPLOADI` (via literal loads like `0x3800`, `0.0f`, `1.0f`) | Loads an immediate constant into an SFPU local register. Used for loading comparison constants (0.0, 1.0) and the fp16a bias value (0x3800). |
| `SFPSETCC` (via `v_if` conditions) | Sets the condition code register based on a comparison result, enabling conditional execution of subsequent instructions within the `v_if`/`v_else`/`v_endif` blocks. |
| `SFPENCC` (via `v_endif`) | Ends conditional execution, restoring the condition code to its prior state. |
| `SFPMOV` (via register assignments) | Moves data between SFPU local registers. Used when assigning constant values (0.0f, 1.0f) to output under conditional paths. |
| `SFPNOP` (implicit) | No-operation padding inserted by the compiler between dependent SFPU instructions to satisfy pipeline latency requirements. |
| `SETRWC` (via `dst_reg++` and inter-face advancement) | Sets the read/write counter for DEST addressing. `dst_reg++` increments by `SFP_DESTREG_STRIDE` (=2) rows. Between faces, explicit `TTI_SETRWC` calls advance the pointer by 8 rows each (called twice for 16-row face advancement). |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **DEST register file** | Primary storage. Input tiles occupy DEST[i*2] (tile A) and DEST[i*2+1] (tile B). After subtraction, the result overwrites DEST[i*2]. After eqz, the final 0.0/1.0 values overwrite DEST[i*2]. Each "row" in SFPI terms is 16 elements wide (half a tile face width). |
| **LREG0** | Working register for loading values from DEST (`dst_reg[0]` reads into LREG0 by default). Also used for intermediate computation. |
| **LREG1-LREG3** | Additional local registers used by the SFPI compiler for holding constants (0.0f, 1.0f, 0x3800 bias), intermediate comparison results, and the condition code state during `v_if`/`v_else` blocks. |
| **LCONST registers** | `vConst0` (LCONST_0 = 0.0f) and `vConst1` (LCONST_1 = 1.0f in fp16b format) are hardware-provided constants. The kernel loads explicit constants via SFPLOADI rather than relying on these for precision reasons. |
| **Condition Code (CC)** | The SFPI `v_if`/`v_else`/`v_endif` constructs manipulate the CC register per-lane. For the eqz comparison, `_sfpu_is_fp16_zero_` produces a per-lane boolean via `v == 0.0F` (or the fp16a bias trick), which sets CC. The `v_if` then uses CC to conditionally write 1.0 or 0.0 to each lane. |

### Address Mode Configuration

The address mode configuration for the EQ operation is set during `eqz_tile_init()` (via `_llk_math_eltwise_unary_sfpu_init_<SfpuType::equal_zero>()`) and applies to both the binary subtract and unary eqz phases (since the binary phase is called first without its own init overriding addr_mod for the high bank).

**ADDR_MOD_7** (configured by `eltwise_unary_sfpu_configure_addrmod<SfpuType::equal_zero>`):

| Field | Value | Description |
|-------|-------|-------------|
| `srca.incr` | 0 | No auto-increment for source A addressing (not used by SFPU) |
| `srcb.incr` | 0 | No auto-increment for source B addressing (not used by SFPU) |
| `dest.incr` | 0 | No auto-increment for DEST addressing via addr_mod; DEST progression is handled entirely by SFPI's `dst_reg++` (which uses `SFP_DESTREG_STRIDE` = 2 hardware rows per increment) and explicit `TTI_SETRWC` calls between faces |

The `equal_zero` SfpuType does not match any of the special-case addr_mod configurations (topk, typecast, max/min), so only ADDR_MOD_7 is set with all-zero increments.

During SFPU execution, `math::set_addr_mod_base()` is called, which sets the addr_mod base register to 1 via `TTI_SETC16(ADDR_MOD_SET_Base_ADDR32, 1)`. This means SFPU instructions reference ADDR_MOD indices 4-7 (base offset of 4). ADDR_MOD_3 (used by SFPLOAD/SFPSTORE in `dst_reg[]` accesses) maps to hardware ADDR_MOD_7 under this base, which has `dest.incr=0`.

This configuration is the same for Wormhole B0 and Blackhole -- the `eltwise_unary_sfpu_configure_addrmod` function has identical logic in both architectures for the `equal_zero` case.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the binary element-wise SFPU program factory work in ttnn? What is the structure of element_wise_multi_core_sfpu_pgm_factory.cpp and how does it differ from the FPU path?"
   **Reason**: To understand the overall architecture of the SFPU binary path, its kernel structure, and how it differs from the FPU path.
   **Key Findings**: Confirmed the three-kernel structure (reader, compute, writer), the use of `eltwise_binary_sfpu_kernel.cpp` for compute, the role of `UnpackToDestFp32` mode, and the interim CB mechanism for pre-scaling operations. The SFPU path explicitly loads data into DST registers before SFPU function calls, while the FPU path can work more directly on CB data.

2. [SFPU] **Query**: "How is eqz_tile implemented in the LLK? What is the call chain from eqz_tile through llk_math to the ckernel SFPU implementation? What file contains the core SFPU eqz calculation function?"
   **Reason**: To identify the full call chain from the `eqz_tile` API down to the core SFPU implementation, and to locate the relevant source files.
   **Key Findings**: Confirmed the call chain: `eqz_tile` -> `SFPU_ZERO_KERNEL` macro -> `_llk_math_eltwise_unary_sfpu_params_` -> `calculate_comp<APPROX, SfpuType::equal_zero>`. The core implementation is in `ckernel_sfpu_comp.h` (WH-specific variant at `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_comp.h`). DeepWiki correctly identified the pattern but did not have the exact file for eqz -- source code verification was required.

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

### Confluence References

No Confluence pages were consulted for this analysis. The SFPI-based kernel uses high-level abstractions (`v_if`, `dst_reg`, `vFloat`) that are well-documented via DeepWiki and source code, so the low-level SFPU ISA page was not needed.

### Glean References

No Glean searches were performed for this analysis. The SFPU kernel implementations were fully traceable through source code and DeepWiki.
