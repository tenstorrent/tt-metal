# MUL (Element-Wise SFPU) Implementation Analysis

## Overview

The MUL operation performs element-wise multiplication of two input tensors using the SFPU (Scalar Floating Point Unit) compute path. It supports both floating-point types (BF16, FP32) via `mul_binary_tile` and integer types (INT32, UINT32, UINT16) via `mul_int_tile`. The operation requires both input tensors to have matching height and width dimensions (no broadcasting).

**Program factory path**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/element_wise_multi_core_sfpu_pgm_factory.cpp`

## Path Selection: FPU vs SFPU

The binary operation framework supports two element-wise program factories: `ElementWiseMultiCore` (FPU path) and `ElementWiseMultiCoreSfpu` (SFPU path). Path selection occurs in `BinaryDeviceOperation::select_program_factory()` (file: `binary_device_operation.cpp`, line ~68).

When both input tensors have matching height and width (`height_a == height_b and width_a == width_b`), the function calls `utils::is_binary_sfpu_op(op, dtype1, dtype2)` to check if the operation type is SFPU-eligible. For `BinaryOpType::MUL`, this function returns `true` (line ~25), so the SFPU path (`ElementWiseMultiCoreSfpu`) is always selected for MUL when shapes match. If `is_binary_sfpu_op` returned `false`, the FPU path (`ElementWiseMultiCore`) would be used instead. Broadcasting cases (where one dimension is 1) are routed to entirely different factories (`BroadcastHeightMultiCore`, `BroadcastWidthMultiCore`, etc.).

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | tile (32x32 elements) |
| **Unit size** | `max_block_size` tiles (1 for interleaved, largest divisor of `num_tiles_per_shard` for sharded) |
| **Total units** | `num_tiles = a.physical_volume() / TILE_HW` |
| **Loop structure** | Outer loop: `per_core_block_cnt` blocks; inner loop: `per_core_block_size` tiles per block |

## Tensor Format and Layout

### Input Tensors

| Property | Input Tensor A (src0) | Input Tensor B (src1) |
|----------|----------------------|----------------------|
| **Logical shape** | [N, C, H, W] | [N, C, H, W] (must match A) |
| **Dimension convention** | NCHW | NCHW |
| **Tensor layout** | TILE_LAYOUT | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED or SHARDED | INTERLEAVED or SHARDED |
| **Buffer type** | DRAM or L1 | DRAM or L1 |
| **Data type** | BFLOAT16, FLOAT32, INT32, UINT32, or UINT16 | BFLOAT16, FLOAT32, INT32, UINT32, or UINT16 |

### Output Tensor

| Property | Output Tensor |
|----------|---------------|
| **Logical shape** | Same as input |
| **Dimension convention** | NCHW |
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED or SHARDED |
| **Buffer type** | DRAM or L1 |
| **Data type** | Same as configured output dtype |

### Layout Transformations

No tilize/untilize or format conversions are performed. Both inputs and the output must already be in TILE_LAYOUT. For integer types (INT32, UINT32, UINT16), the operation dispatches to `mul_int_tile<DataFormat>`, while for floating-point types, it uses `mul_binary_tile`.

## Data Flow Pattern

**Interleaved (non-sharded) path:**

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader | DRAM (src0) | CB c_0 | `cb_reserve_back(c_0, 1)`, `noc_async_read_tile`, `cb_push_back(c_0, 1)` |
| 1 | Reader | DRAM (src1) | CB c_1 | `cb_reserve_back(c_1, 1)`, `noc_async_read_tile`, `cb_push_back(c_1, 1)` |
| 2 | Compute | CB c_0, c_1 | CB c_2 | `cb_wait_front(c_0/c_1)`, `copy_tile` to DST, SFPU op, `pack_tile`, `cb_pop_front`, `cb_push_back(c_2)` |
| 3 | Writer | CB c_2 | DRAM (dst) | `cb_wait_front(c_2, 1)`, `noc_async_write_page`, `cb_pop_front(c_2, 1)` |

**Sharded path:** For sharded inputs, the reader simply does `cb_reserve_back` + `cb_push_back` for the entire shard (data is already in L1). For sharded output, the writer just does `cb_wait_front` (data stays in L1).

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_src0 | Input A staging | 2 * max_block_size tiles (interleaved) or num_tiles_per_shard (sharded) | 1 tile (interleaved) or max_block_size tiles (sharded) | Double (interleaved) / Single (sharded) | Reader | Compute | Program |
| c_1 | cb_src1 | Input B staging | 2 * max_block_size tiles (interleaved) or num_tiles_per_shard (sharded) | 1 tile (interleaved) or max_block_size tiles (sharded) | Double (interleaved) / Single (sharded) | Reader | Compute | Program |
| c_2 | cb_output | Output staging | 2 * max_block_size tiles (interleaved) or num_tiles_per_shard (sharded) | 1 tile (interleaved) or max_block_size tiles (sharded) | Double (interleaved) / Single (sharded) | Compute | Writer | Program |
| c_3 | cb_interim0 | Pre-processing scratch for input A | max_block_size tiles | max_block_size tiles | Single | Compute | Compute | Block |
| c_4 | cb_interim1 | Pre-processing scratch for input B | max_block_size tiles | max_block_size tiles | Single | Compute | Compute | Block |

**Note on c_3 and c_4**: These intermediate CBs are only created if `SFPU_OP_INIT_PRE_IN0_0` or `SFPU_OP_INIT_PRE_IN1_0` defines are present. For plain MUL (floating-point or integer), no pre-processing defines are set, so **c_3 and c_4 are NOT allocated**. They would be allocated for composite operations that reuse this factory, such as LDEXP (which applies `exp2` to input B before multiplying).

## Pipeline Pattern Summary

- **Interleaved path**: CB c_0, c_1, and c_2 each have capacity = 2 * max_block_size with block_size = 1, providing **double-buffering**. This allows the reader to fill one slot while compute processes the other.
- **Sharded path**: CBs map directly to the shard buffer in L1 (globally allocated address). Capacity equals num_tiles_per_shard, effectively **single-buffered** since all data is pre-loaded.

## Index Calculations

The reader kernel uses `TensorAccessor` for mapping tile IDs to physical memory addresses. For interleaved tensors, tiles are accessed sequentially by tile ID starting from `start_id`. The `TensorAccessor` handles bank interleaving internally.

For block/width-sharded tensors, the reader uses a 2D tile traversal pattern:
- `row_start_tile_id = start_id`, incremented by `num_cores_y * block_width` per row
- Inner loop iterates `tile_id` from `row_start_tile_id` to `row_start_tile_id + block_width`

The `start_id` for each core is computed in the host as:
```
start_id = (core_index / num_shards_per_width) * (block_height * block_width * num_shards_per_width)
         + (core_index % num_shards_per_width) * block_width
```

## Memory Access Patterns

### Read Pattern
- **Interleaved**: Sequential tile-by-tile reads from DRAM via `noc_async_read_tile`. Each tile is read individually with a barrier after each pair (src0, src1).
- **Sharded**: No reads -- data is already in L1. The reader simply marks the CB as full.
- **Block/width-sharded (mixed)**: 2D traversal reading tiles row-by-row within the shard's logical block.

### Write Pattern
- **Interleaved**: Sequential tile-by-tile writes via `noc_async_write_page` with `noc_async_writes_flushed` after each tile.
- **Sharded output**: No writes -- data remains in L1 after compute. Writer just calls `cb_wait_front`.
- **Block/width-sharded to interleaved**: Writer uses `writer_unary_sharded_blocks_interleaved_start_id.cpp`, which performs a 2D write pattern, writing only unpadded tiles (skipping padding) and advancing by `output_width_tiles` per row.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (interleaved) or 2D (sharded, follows shard grid) |
| **Grid dimensions** | Determined by `operation_attributes.worker_grid` |
| **Total cores** | `num_cores_total` from worker grid |
| **Work per core** | `num_tiles_per_core_group_1` (majority) or `num_tiles_per_core_group_2` (remainder cores) |
| **Load balancing** | `split_work_to_cores` for interleaved (two groups with at most 1 tile difference); uniform for sharded |

For interleaved tensors, `split_work_to_cores` divides total tiles into two groups: group 1 gets `ceil(num_tiles / num_cores)` tiles, group 2 gets one fewer tile. Excess cores beyond `num_cores` receive zero work.

For sharded tensors, each core processes exactly its shard (`num_tiles_per_shard` tiles), and the core grid matches the shard grid.

## Arguments

### Compile-Time Arguments

**Reader kernel:**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | block_or_width_sharded | uint32_t | 1 if block or width sharded, 0 otherwise |
| 1+ | src0_tensor_accessor_args | TensorAccessorArgs | Memory access parameters for src0 (omitted if IN0_SHARDED) |
| N+ | src1_tensor_accessor_args | TensorAccessorArgs | Memory access parameters for src1 (omitted if IN1_SHARDED) |

**Writer kernel:**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | output_cb_index | uint32_t | CB index for output (c_2) |
| 1+ | dst_tensor_accessor_args | TensorAccessorArgs | Memory access parameters for output buffer |

**Compute kernel:** No compile-time args. Behavior is controlled entirely through preprocessor defines.

### Runtime Arguments

**Reader kernel:**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src0_addr | uint32_t | Source buffer A address |
| 1 | src1_addr | uint32_t | Source buffer B address |
| 2 | num_tiles | uint32_t | Total tiles for this core |
| 3 | start_id | uint32_t | Starting tile ID |
| 4 | block_height | uint32_t | Shard block height in tiles (0 for interleaved) |
| 5 | block_width | uint32_t | Shard block width in tiles (0 for interleaved) |
| 6 | num_cores_y | uint32_t | Number of shards per width dimension |

**Compute kernel:**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_cnt | uint32_t | Number of blocks to process |
| 1 | per_core_block_size | uint32_t | Tiles per block |

**Writer kernel (interleaved output):**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Destination buffer address |
| 1 | num_pages | uint32_t | Total tiles to write |
| 2 | start_id | uint32_t | Starting tile ID for writes |

**Writer kernel (block/width-sharded to interleaved):**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Destination buffer address |
| 1 | block_height_tiles | uint32_t | Block height in tiles |
| 2 | block_width_tiles | uint32_t | Block width in tiles |
| 3 | unpadded_block_height_tiles | uint32_t | Actual (unpadded) block height |
| 4 | unpadded_block_width_tiles | uint32_t | Actual (unpadded) block width |
| 5 | output_width_tiles | uint32_t | Full output width in tiles |
| 6 | block_num_tiles | uint32_t | Total tiles in block (height * width) |
| 7 | start_id_offset | uint32_t | Start tile offset for this core |
| 8 | start_id_base | uint32_t | Base start tile ID (always 0) |

## Kernel Implementations

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| Reader | BRISC (RISCV_0) | NOC0 | DRAM/L1 | CB c_0, c_1 | Read src0 and src1 tiles |
| Compute | TRISC (Unpack+Math+Pack) | N/A | CB c_0, c_1 | CB c_2 | Copy tiles to DST, mul_binary_tile or mul_int_tile, pack |
| Writer | NCRISC (RISCV_1) | NOC1 | CB c_2 | DRAM/L1 | Write output tiles |

### Reader Kernel

| Property | Value |
|----------|-------|
| **File** | `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/dataflow/reader_binary_interleaved_start_id.cpp` |
| **Assigned cores** | All worker cores (`all_device_cores`) |

**Key Logic:**
- When `IN0_SHARDED` is defined, the reader skips DRAM reads for src0 and simply does `cb_reserve_back(c_0, num_tiles)` + `cb_push_back(c_0, num_tiles)` to signal that data is already present in L1. Same for `IN1_SHARDED` and src1.
- For interleaved inputs with `block_or_width_sharded == false`, the reader iterates sequentially from `start_id` to `start_id + num_tiles`, reading one tile at a time from each source into their respective CBs.
- For `block_or_width_sharded == true`, the reader uses a 2D loop: outer over `block_height`, inner over `block_width`, advancing the row start by `num_cores_y * block_width` to skip over tiles owned by other cores in the same column.
- Each tile read is followed by `noc_async_read_barrier()` before pushing to the CB, ensuring the read completes before the tile is consumed.
- **Synchronization**: Produces to CB c_0 and c_1 via `cb_reserve_back` / `cb_push_back`. Compute waits on these with `cb_wait_front`.

### Compute Kernel

| Property | Value |
|----------|-------|
| **File** | `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/eltwise_binary_sfpu_kernel.cpp` |
| **Assigned cores** | All worker cores (`all_device_cores`) |

**Key Logic:**
- Outer loop iterates `per_core_block_cnt` times; inner loop iterates `per_core_block_size` times.
- For plain MUL, no pre-processing stage is active (no `SFPU_OP_INIT_PRE_IN0_0` or `SFPU_OP_INIT_PRE_IN1_0`), so `cb_inp0 = cb_in0 = c_0` and `cb_inp1 = cb_in1 = c_1`.
- Waits for both inputs: `cb_wait_front(c_0, per_core_block_size)` and `cb_wait_front(c_1, per_core_block_size)`.
- Reserves output space: `cb_reserve_back(c_2, per_core_block_size)`.
- Acquires tile registers via `tile_regs_acquire()` and `tile_regs_wait()`.
- Copies input A tiles into even DST slots: `copy_tile(c_0, i, i * 2)`.
- Copies input B tiles into odd DST slots: `copy_tile(c_1, i, i * 2 + 1)`.
- For floating-point MUL: executes `BINOP_INIT` (`mul_binary_tile_init()`) then `BINARY_SFPU_OP` (`mul_binary_tile(i*2, i*2+1, i*2)`), where idst0=i*2 (input A), idst1=i*2+1 (input B), odst=i*2 (result overwrites A's DST slot).
- For integer MUL: executes `MUL_INT_INIT` (`mul_int_tile_init<DataFormat>()`) then `BINARY_SFPU_OP` (`mul_int_tile<DataFormat>(i*2, i*2+1, i*2)`).
- Packs result from DST slot `i * 2` to output CB: `pack_tile(i * 2, cb_out0)`.
- After inner loop: `tile_regs_commit()`, `tile_regs_release()`.
- **Synchronization**: Consumes from CB c_0 and c_1 via `cb_wait_front` / `cb_pop_front`. Produces to CB c_2 via `cb_reserve_back` / `cb_push_back`.

### Writer Kernel

| Property | Value |
|----------|-------|
| **File** | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` (interleaved output) or `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded_blocks_interleaved_start_id.cpp` (block/width-sharded to interleaved) |
| **Assigned cores** | All worker cores (`all_device_cores`) |

**Key Logic (interleaved output):**
- When `OUT_SHARDED` is defined, the writer simply does `cb_wait_front(c_2, num_pages)` and returns (data stays in L1).
- For interleaved output, iterates from `start_id` to `start_id + num_pages`, writing one tile at a time: `cb_wait_front(c_2, 1)`, `noc_async_write_page`, `noc_async_writes_flushed`, `cb_pop_front(c_2, 1)`.
- Final `noc_async_write_barrier()` ensures all writes complete.

**Key Logic (block/width-sharded to interleaved output):**
- Waits for entire block: `cb_wait_front(c_2, block_num_tiles)`.
- Traverses only unpadded tiles in a 2D pattern (rows x cols), skipping padding columns by advancing the L1 read pointer by `padded_width_diff`.
- Row stride in output is `output_width_tiles`.
- **Synchronization**: Consumes from CB c_2 via `cb_wait_front` / `cb_pop_front`.

## Implementation Notes

- **Program factory variants**: Two binary element-wise factories exist: `ElementWiseMultiCore` (FPU) and `ElementWiseMultiCoreSfpu` (SFPU). MUL always selects the SFPU factory when shapes match. Broadcasting cases use separate factories (`BroadcastHeightMultiCore`, `BroadcastWidthMultiCore`, `BroadcastHeightAndWidthMultiCore`).
- **Type-based operation variants**: Floating-point types (BF16, FP32) use `mul_binary_tile` with `BINOP_INIT`. Integer types use `mul_int_tile<DataFormat>` with `MUL_INT_INIT`. Supported integer formats: INT32, UINT32, UINT16.
- **UnpackToDestFP32 mode**: For non-POWER operations (including MUL), `UnpackToDestMode::UnpackToDestFp32` is set on all input CBs (c_0, c_1, c_3, c_4) unconditionally. This ensures intermediate values in DST registers use FP32 precision regardless of input format.
- **Broadcast type selection**: N/A -- the SFPU factory does not support broadcasting. Both inputs must have matching H and W dimensions.
- **Sharding support and constraints**: Supports height-sharded, width-sharded, and block-sharded memory layouts. Any combination of sharded/interleaved inputs and output is permitted. When block/width-sharded input feeds interleaved output, a specialized writer kernel handles the 2D-to-1D mapping with padding removal.
- **FP32 dest accumulation**: Enabled when output dtype is Float32, Int32, or UInt32 (`fp32_dest_acc_en = true`). This configures the pack unit to read full FP32 values from DEST.

## SFPU Kernel Implementation

This section provides a dedicated deep dive into the underlying SFPU kernel functions that the compute kernel dispatches to. MUL has two distinct SFPU paths: floating-point multiplication (`mul_binary_tile`) and integer multiplication (`mul_int_tile`). Each is analyzed separately, and the integer path further splits by data format and hardware generation.

### SFPU Abstraction Layers

#### Floating-Point MUL (`mul_binary_tile`)

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_binary_sfpu.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_binop.h` |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/ckernel_sfpu_binary.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{arch}/llk_lib/llk_math_eltwise_binary_sfpu_params.h` |

#### Integer MUL (`mul_int_tile`) -- INT32 / UINT32

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/mul_int_sfpu.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_mul_int.h` |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/ckernel_sfpu_mul_int32.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{arch}/llk_lib/llk_math_eltwise_binary_sfpu_params.h` |

#### Integer MUL (`mul_int_tile`) -- UINT16

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/mul_int_sfpu.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_mul_int.h` |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_{arch}/common/inc/sfpu/ckernel_sfpu_mul_int.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{arch}/llk_lib/llk_math_eltwise_binary_sfpu_params.h` |

### Call Chain

#### Floating-Point MUL

1. Compute kernel calls `mul_binary_tile_init()` which expands to `MATH((llk_math_eltwise_binary_sfpu_binop_init<APPROX, BinaryOp::MUL>()))`. This calls `_llk_math_eltwise_binary_sfpu_init_<SfpuType::unused>()` (configures SFPU config register, sets ADDR_MOD_7, resets counters) followed by `sfpu_binary_init<APPROX, BinaryOp::MUL>()` which calls `_sfpu_binary_init_` -- a no-op for MUL (only initializes for DIV/POW/XLOGY).

2. Compute kernel calls `mul_binary_tile(i*2, i*2+1, i*2)` which expands to `MATH((llk_math_eltwise_binary_sfpu_binop_mul<APPROX, BinaryOp::MUL, DST_ACCUM_MODE>(i*2, i*2+1, i*2)))`.

3. `llk_math_eltwise_binary_sfpu_binop_mul` calls `_llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>(calculate_sfpu_binary_mul<APPROX, BinaryOp::MUL, 8, DST_ACCUM_MODE>, ...)` which sets the DEST write address, stalls for SFPU readiness, then iterates over 4 tile faces (VectorMode::RC), calling `calculate_sfpu_binary_mul` once per face. Between faces, `TTI_SETRWC` advances the DEST pointer by 16 rows (2 increments of 8).

4. `calculate_sfpu_binary_mul` executes 8 iterations per face, each processing one row of 32 elements: loads `in0` and `in1` from DEST, computes `in0 * in1`, optionally applies BF16 rounding, and writes the result back to DEST.

#### Integer MUL (INT32/UINT32)

1. Compute kernel calls `mul_int_tile_init<DataFormat>()` which expands to `MATH((llk_math_eltwise_binary_sfpu_mul_int_init<APPROX, DataFormat>()))`. This calls `_llk_math_eltwise_binary_sfpu_init_<SfpuType::mul_int32>()` (configures ADDR_MOD_7 with dest.incr=0 AND ADDR_MOD_6 with dest.incr=2) followed by `mul_int32_init` which sets up SFPLOADMACRO instruction templates and macro configurations for pipelined 32-bit integer multiplication.

2. Compute kernel calls `mul_int_tile<DataFormat>(i*2, i*2+1, i*2)` which dispatches to `_llk_math_eltwise_binary_sfpu_params_` with `sfpu::mul_int32` as the callable. The params dispatch iterates over 4 faces (VectorMode::RC) calling `mul_int32` per face.

3. `mul_int32` uses SFPLOADMACRO-based pipelining: it splits each 32-bit integer into 23-bit mantissa and 9-bit exponent-shifted portions, performs mul24_lo/mul24_hi operations, accumulates cross-products, and stores the 32-bit result back. Throughput: 8 cycles per row.

#### Integer MUL (UINT16)

1. Same init path but selects `SfpuType::mul_uint16` and calls `_init_mul_int_` which configures a simpler SFPLOADMACRO macro set with `SFPMUL24` instruction templates.

2. `_mul_int_` is called per face. It uses SFPLOADMACRO with optimized paths depending on whether input/output DST indices overlap (1, 2, or 3 cycles per row).

### Parameters Dispatch Summary

The parameters dispatch function `_llk_math_eltwise_binary_sfpu_params_` is identical for all MUL variants (floating-point and integer) and identical across Wormhole and Blackhole.

- **Vector mode**: `VectorMode::RC` (default, always used for MUL). All 4 faces of the 32x32 tile are processed (Face 0, Face 1, Face 2, Face 3).
- **Operation invocation**: The core SFPU function is called once per face (4 calls total). Each call internally iterates `ITERATIONS=8` times, processing 8 rows of 32 elements per face, covering the full 16x32 face (a 32x32 tile has 4 faces of 16x16 elements, but in SFPU row-based addressing, each face is 8 rows of 32 SFPU datums).
- **DEST address progression**: The `_llk_math_eltwise_binary_sfpu_start_` function sets the initial DEST write address to index 0 (not the tile index -- the tile offset is baked into the SFPU function's `dst_index_in0/in1/out` arguments which are multiplied by `dst_tile_size_sfpi=32`). After each face, `TTI_SETRWC` advances the DEST row counter by 16 (two increments of 8 via `CR_D, 8`). Within each face, `dst_reg++` in the SFPU loop auto-increments the DEST row pointer by 1 per iteration. On Wormhole, `_llk_math_eltwise_binary_sfpu_done_` additionally stalls for SFPU completion and clears the addr_mod base; on Blackhole it only clears the DEST register address.

### Annotated SFPU Kernel Source

This operation has three distinct SFPU kernel implementations depending on data type and hardware generation:

#### 1. Floating-Point MUL (`calculate_sfpu_binary_mul`) -- Blackhole and Wormhole

The floating-point kernel is identical across Blackhole and Wormhole.

```cpp
// File: tt_metal/hw/ckernels/{blackhole,wormhole_b0}/metal/llk_api/llk_sfpu/ckernel_sfpu_binary.h

// Convert float32 to bfloat16 using IEEE 754 Round-to-Nearest-Even (RNE)
// This implements the "add 0x7fff + LSB" algorithm for correct tie-breaking
sfpi_inline sfpi::vFloat float32_to_bf16_rne(sfpi::vFloat in) {
    sfpi::vUInt bits = sfpi::reinterpret<sfpi::vUInt>(in);
    sfpi::vUInt lsb = (bits >> 16) & 1; // Extract bit 16 (bf16 mantissa LSB) for tie-breaking
    // Add 0x7fff + lsb: ties round to even, non-ties round to nearest
    bits = bits + 0x7fffU + lsb;
    bits = bits & 0xFFFF0000U; // Truncate lower 16 bits to produce bf16-in-fp32
    return sfpi::reinterpret<sfpi::vFloat>(bits);
}

template <bool APPROXIMATION_MODE, BinaryOp BINOP, int ITERATIONS, bool is_fp32_dest_acc_en>
inline void calculate_sfpu_binary_mul(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    // APPROXIMATION_MODE=true (from APPROX), BINOP=BinaryOp::MUL, ITERATIONS=8, is_fp32_dest_acc_en=DST_ACCUM_MODE
    constexpr uint dst_tile_size_sfpi = 32; // 64/SFP_DESTREG_STRIDE = 32 rows per tile in SFPI addressing
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi]; // Load row from input A tile in DEST
        sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi]; // Load row from input B tile in DEST

        sfpi::vFloat result = in0 * in1; // SFPMUL: element-wise fp32 multiply

        if constexpr (!is_fp32_dest_acc_en) {
            // When not in FP32 accumulation mode, round result to bf16 for FPU-matching behavior
            result = float32_to_bf16_rne(result);

            // Match FPU semantics: 0 * x = 0 and x * 0 = 0 (handles NaN * 0 case)
            v_if(in0 == 0 || in1 == 0) { result = 0.0f; }
            v_endif;
        }

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result; // Store result row to output tile in DEST
        sfpi::dst_reg++; // Advance DEST row pointer by 1
    }
}
```

#### 2. Integer MUL INT32/UINT32 (`mul_int32`) -- Blackhole

The Blackhole INT32 kernel uses SFPLOADMACRO for pipelined 32-bit integer multiply with 8-cycle-per-row throughput.

```cpp
// File: tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_mul_int32.h

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void mul_int32(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {

    constexpr uint dst_tile_size = 64;

    uint offset_in0 = dst_index_in0 * dst_tile_size;
    uint offset_in1 = dst_index_in1 * dst_tile_size;
    uint offset_out = dst_index_out * dst_tile_size;

    // Implementation notes, see the original file for more details
    //
    // In pseudocode:
    //   a1 = a >> 23
    //   b1 = b >> 23
    //   cross0 = mul24_lo(a1, b)
    //   cross1 = mul24_lo(a, b1)
    //   lo = mul24_lo(a, b)
    //   hi = mul24_hi(a, b)
    //   result = ((hi + cross0 + cross1) << 23) + lo

    constexpr uint a0 = p_sfpu::LREG0;
    constexpr uint b0 = p_sfpu::LREG0; // Shares LREG0 with a0 (reused after a0 consumed)
    constexpr uint a1 = p_sfpu::LREG1;
    constexpr uint b1 = p_sfpu::LREG2;
    constexpr uint b2 = p_sfpu::LREG3;
    constexpr uint c = p_sfpu::LREG4;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        // Load b0 from DEST
        TT_SFPLOAD(b0, INT32, ADDR_MOD_7, offset_in1);
        // Macro 0: loads a1, schedules SHFT2(a1, -23) and MUL24_LO(b0, a1)
        TT_SFPLOADMACRO((0 << 2) | (a1 & 3), INT32, ADDR_MOD_7, offset_in0 | (a1 >> 2));
        // Macro 1: loads b1, schedules SHFT2(b1, -23), MUL24_LO(a0, b1), and IADD(b1+a1)
        TT_SFPLOADMACRO((1 << 2) | (b1 & 3), INT32, ADDR_MOD_7, offset_in1 | (b1 >> 2));
        // Load a0 from DEST
        TT_SFPLOAD(a0, INT32, ADDR_MOD_7, offset_in0);
        // Macro 2: loads b2 (=b again for fresh copy), schedules MUL24_HI(a0,b2), MUL24_LO(a0,b2)
        TT_SFPLOADMACRO((2 << 2) | (b2 & 3), INT32, ADDR_MOD_7, offset_in1 | (b2 >> 2));
        // c = mul24_hi(a0, b2): upper 24 bits of 48-bit product
        TTI_SFPMUL24(a0, b2, p_sfpu::LCONST_0, c, sfpi::SFPMUL24_MOD1_UPPER);
        // b1 = b1 + a1: accumulate cross-products (hi portion)
        TTI_SFPIADD(0, a1, b1, sfpi::SFPIADD_MOD1_CC_NONE); // CC_NONE: do not update condition codes
        // Macro 3: loads c, schedules SHFT(b1,23), IADD(b2+c), SFPSTORE(c L16)
        TT_SFPLOADMACRO((3 << 2) | (c & 3), INT32, ADDR_MOD_6, offset_out | (c >> 2));
    }
    TTI_SFPNOP; // Pipeline drain: 3 NOPs needed for SFPLOADMACRO pipeline to flush
    TTI_SFPNOP;
    TTI_SFPNOP;
}
```

#### 3. Integer MUL INT32/UINT32 (`mul_int32`) -- Wormhole

The Wormhole INT32 kernel uses a different algorithm because Wormhole lacks `SFPMUL24`. It splits each 32-bit integer into three 11-bit chunks, casts them to fp32, performs six `SFPMAD` (fused multiply-add) operations, extracts mantissa bits via `SFPEXMAN`, and reconstructs the 32-bit result with shifts and integer adds.

```cpp
// File: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_mul_int32.h

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void mul_int32(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        constexpr uint dst_tile_size = 64;

        // Implementation notes, see the original file for more details
        //
        // Split: a = (a2 << 22) | (a1 << 11) | a0; b = (b2 << 22) | (b1 << 11) | b0
        // Result: a*b = (top << 22) + (mid << 11) + low
        //   where top = a0*b2 + a1*b1 + a2*b0, mid = a0*b1 + a1*b0, low = a0*b0
        // Uses fp32_to_u23(x) = mantissa_bits(x + 2**23) trick for exact integer extraction

        // Load a, split into 11-bit chunks
        TT_SFPLOAD(p_sfpu::LREG0, INT32, ADDR_MOD_3, dst_index_in0 * dst_tile_size);
        TTI_SFPSHFT2(p_sfpu::LREG0, p_sfpu::LREG13, p_sfpu::LREG2, 5); // a1 = a >> 11 (via LREG13=-11)
        TTI_SFPSHFT2(p_sfpu::LREG2, p_sfpu::LREG13, p_sfpu::LREG4, 5); // a2 = a1 >> 11

        TTI_SFPAND(0, p_sfpu::LREG12, p_sfpu::LREG2, 0); // a1 &= 0x7ff (LREG12=0x7ff)
        TTI_SFPCAST(p_sfpu::LREG2, p_sfpu::LREG2, 0);     // a1 = int_to_fp32(a1)
        TTI_SFPCAST(p_sfpu::LREG4, p_sfpu::LREG4, 0);     // a2 = int_to_fp32(a2)
        TTI_SFPAND(0, p_sfpu::LREG12, p_sfpu::LREG0, 0); // a0 &= 0x7ff
        TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG0, 0);     // a0 = int_to_fp32(a0)

        // Load b, split into 11-bit chunks
        TT_SFPLOAD(p_sfpu::LREG1, INT32, ADDR_MOD_3, dst_index_in1 * dst_tile_size);
        TTI_SFPSHFT2(p_sfpu::LREG1, p_sfpu::LREG13, p_sfpu::LREG3, 5); // b1
        TTI_SFPSHFT2(p_sfpu::LREG3, p_sfpu::LREG13, p_sfpu::LREG5, 5); // b2
        TTI_SFPCAST(p_sfpu::LREG5, p_sfpu::LREG5, 0);     // b2 = int_to_fp32(b2)

        // Accumulate cross-products using FMA: top = a0*b2 + 2**23
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG5, p_sfpu::LREG14, p_sfpu::LREG5, 0); // LREG14=2**23

        TTI_SFPAND(0, p_sfpu::LREG12, p_sfpu::LREG3, 0); // b1 &= 0x7ff
        TTI_SFPCAST(p_sfpu::LREG3, p_sfpu::LREG3, 0);     // b1 = int_to_fp32(b1)

        TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG3, p_sfpu::LREG5, p_sfpu::LREG5, 0); // top += a1*b1

        TTI_SFPAND(0, p_sfpu::LREG12, p_sfpu::LREG1, 0); // b0 &= 0x7ff
        TTI_SFPCAST(p_sfpu::LREG1, p_sfpu::LREG1, 0);     // b0 = int_to_fp32(b0)

        TTI_SFPMAD(p_sfpu::LREG4, p_sfpu::LREG1, p_sfpu::LREG5, p_sfpu::LREG5, 0); // top += a2*b0
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG3, p_sfpu::LREG14, p_sfpu::LREG6, 0); // mid = a0*b1 + 2**23
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG14, p_sfpu::LREG0, 0); // low = a0*b0 + 2**23
        TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG1, p_sfpu::LREG6, p_sfpu::LREG6, 0); // mid += a1*b0

        // Extract mantissa bits to recover exact integers from fp32 encoding
        TTI_SFPEXMAN(0, p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPEXMAN_MOD1_PAD9); // low = mantissa(low)
        TTI_SFPEXMAN(0, p_sfpu::LREG6, p_sfpu::LREG6, sfpi::SFPEXMAN_MOD1_PAD9); // mid = mantissa(mid)
        TTI_SFPEXMAN(0, p_sfpu::LREG5, p_sfpu::LREG5, sfpi::SFPEXMAN_MOD1_PAD9); // top = mantissa(top)

        TTI_SFPSHFT(22, 0, p_sfpu::LREG5, 1); // top <<= 22
        TTI_SFPSHFT(11, 0, p_sfpu::LREG6, 1); // mid <<= 11

        TTI_SFPIADD(0, p_sfpu::LREG6, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_CC_NONE); // result = low + mid
        TTI_SFPIADD(0, p_sfpu::LREG5, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_CC_NONE); // result += top

        TT_SFPSTORE(p_sfpu::LREG0, INT32, ADDR_MOD_2, dst_index_out * dst_tile_size);
    }
}
```

#### 4. Integer MUL UINT16 (`_mul_int_`) -- Blackhole and Wormhole

The UINT16 kernel is shared across both architectures (located in tt_llk common). It uses SFPLOADMACRO for efficient 16-bit integer multiply with optimized paths based on operand aliasing.

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_{arch}/common/inc/sfpu/ckernel_sfpu_mul_int.h

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _mul_int_(const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out)
{
    // Implementation notes, see the original file for more details
    //
    // Throughput depends on operand aliasing:
    // - dst_index_in0 == dst_index_in1 == dst_index_out: 1 cycle per row
    // - One input == output: 2 cycles per row
    // - All different: 3 cycles per row

    int offset0 = (dst_index_in0 * 32) << 1;
    int offset1 = (dst_index_in1 * 32) << 1;

    constexpr int a = p_sfpu::LREG0;
    constexpr int b = p_sfpu::LREG1;

    if (dst_index_out == dst_index_in0 && dst_index_out == dst_index_in1)
    {
#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++)
        {
            // All same: 1 cycle -- single macro load+mul24_lo+store
            TT_SFPLOADMACRO((0 << 2) | (a & 3), LO16, ADDR_MOD_6, offset1 | (a >> 2));
        }
    }
    else if (dst_index_out == dst_index_in0)
    {
#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++)
        {
            // Output aliases in0: load in1, then macro on in0
            TT_SFPLOAD(a, LO16, ADDR_MOD_7, offset1);
            TT_SFPLOADMACRO((0 << 2) | (b & 3), LO16, ADDR_MOD_6, offset0 | (b >> 2));
        }
    }
    else if (dst_index_out == dst_index_in1)
    {
#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++)
        {
            // Output aliases in1: load in0, then macro on in1
            TT_SFPLOAD(a, LO16, ADDR_MOD_7, offset0);
            TT_SFPLOADMACRO((0 << 2) | (b & 3), LO16, ADDR_MOD_6, offset1 | (b >> 2));
        }
    }
    else
    {
        int offset2 = (dst_index_out * 32) << 1;

#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++)
        {
            // All different: load in0, macro on out (which schedules store), load in1
            TT_SFPLOAD(a, LO16, ADDR_MOD_7, offset0);
            TT_SFPLOADMACRO((1 << 2) | (b & 3), LO16, ADDR_MOD_7, offset2 | (b >> 2));
            TT_SFPLOAD(b, LO16, ADDR_MOD_6, offset1);
        }
    }
    TTI_SFPNOP; // Pipeline drain
    TTI_SFPNOP;
    TTI_SFPNOP;
}
```

### SFPU Instructions Used

#### Floating-Point MUL

| Instruction / Intrinsic | Description |
|------------------------|-------------|
| `sfpi::dst_reg[offset]` (read) | **SFPLOAD**: Loads a 32-element vector from a DEST register row into an SFPU local register |
| `in0 * in1` | **SFPMUL**: Element-wise floating-point multiply of two SFPU vector registers |
| `sfpi::reinterpret<vUInt>(...)` | **SFPMOV**: Reinterprets float bits as unsigned integer (bitwise move, no conversion) |
| `bits >> 16`, `bits & 1` | **SFPSHFT** / **SFPAND**: Bitwise right shift and mask for extracting bf16 LSB |
| `bits + 0x7fffU + lsb` | **SFPIADD**: Integer addition for RNE rounding bias |
| `bits & 0xFFFF0000U` | **SFPAND**: Mask to truncate lower 16 bits |
| `sfpi::reinterpret<vFloat>(...)` | **SFPMOV**: Reinterpret integer bits back as float |
| `sfpi::dst_reg[offset] = result` | **SFPSTORE**: Stores a 32-element vector from an SFPU local register back to a DEST register row |
| `sfpi::dst_reg++` | **SFPINCRWC**: Increments the DEST register write counter by 1 row |
| `v_if(cond)` / `v_endif` | **SFPSETCC** / **SFPENCC**: Sets/clears condition codes for predicated execution |

#### Integer MUL -- Blackhole (INT32/UINT32)

| Instruction | Description |
|-------------|-------------|
| `TT_SFPLOAD` | Loads a 32-element vector from DEST in INT32 format |
| `TT_SFPLOADMACRO` | Loads from DEST and triggers a pre-programmed macro sequence (shift, multiply, accumulate, store) |
| `TTI_SFPMUL24` | 24-bit integer multiply (lower or upper half of 48-bit product); `SFPMUL24_MOD1_UPPER` selects upper bits |
| `TTI_SFPIADD` | Integer addition; `SFPIADD_MOD1_CC_NONE` disables condition code update |
| `TTI_SFPNOP` | No-operation; used to drain the SFPLOADMACRO pipeline |

#### Integer MUL -- Wormhole (INT32/UINT32)

| Instruction | Description |
|-------------|-------------|
| `TT_SFPLOAD` | Loads a 32-element vector from DEST in INT32 format |
| `TTI_SFPSHFT2` | Arithmetic right shift by immediate (shift amount from LREG13=-11); splits 32-bit value into 11-bit chunks |
| `TTI_SFPAND` | Bitwise AND with constant mask (LREG12=0x7ff) to isolate 11-bit chunks |
| `TTI_SFPCAST` | Converts integer to fp32 representation |
| `TTI_SFPMAD` | Fused multiply-add: `VD = VA * VB + VC`; used for cross-product accumulation with 2**23 bias |
| `TTI_SFPEXMAN` | Extracts mantissa bits from fp32; `SFPEXMAN_MOD1_PAD9` pads with 9 zero bits for 23-bit integer recovery |
| `TTI_SFPSHFT` | Integer left shift by immediate; shifts top/mid partial products to their bit positions |
| `TTI_SFPIADD` | Integer addition with `CC_NONE`; accumulates partial products into final result |
| `TT_SFPSTORE` | Stores 32-element vector to DEST in INT32 format |

#### Integer MUL -- UINT16

| Instruction | Description |
|-------------|-------------|
| `TT_SFPLOAD` | Loads a 32-element vector from DEST in LO16 (lower 16-bit) format |
| `TT_SFPLOADMACRO` | Loads from DEST and triggers macro that performs MUL24_LO and SFPSTORE in a single scheduled pipeline |
| `TTI_SFPAND` | Bitwise AND with mask (LREG12=0xff) to isolate 8-bit chunks |
| `TTI_SFPMAD` | Fused multiply-add for cross-product accumulation with 2**23 bias (Wormhole/Blackhole UINT16 path) |
| `TTI_SFPEXMAN` | Extracts mantissa bits for integer recovery |
| `TTI_SFPNOP` | Pipeline drain |

### SFPU Register Usage

#### Floating-Point MUL

| Register | Usage |
|----------|-------|
| `dst_reg[dst_index_in0 * 32]` | DEST row for input A tile (read only) |
| `dst_reg[dst_index_in1 * 32]` | DEST row for input B tile (read only) |
| `dst_reg[dst_index_out * 32]` | DEST row for output tile (write); in practice `dst_index_out == dst_index_in0`, so output overwrites input A |
| SFPU LREG (implicit via `vFloat`) | Temporary vector registers used by SFPI compiler for `in0`, `in1`, `result`, `bits`, `lsb` |

#### Integer MUL -- Blackhole (INT32/UINT32)

| Register | Usage |
|----------|-------|
| `LREG0` (a0/b0) | Input A value, then reused for input B (shared due to register pressure) |
| `LREG1` (a1) | Upper 9 bits of input A (a >> 23) |
| `LREG2` (b1) | Upper 9 bits of input B (b >> 23), then cross-product accumulator |
| `LREG3` (b2) | Fresh copy of input B for lo/hi multiply |
| `LREG4` (c) | Accumulator for high cross-products, then final result |
| DEST rows | Addressed via `offset_in0`, `offset_in1`, `offset_out` (each = dst_index * 64) |

#### Integer MUL -- Wormhole (INT32/UINT32)

| Register | Usage |
|----------|-------|
| `LREG0` | Input A lower 11 bits (a0), then reused for `low` partial product and final result |
| `LREG1` | Input B raw value, then b0 (lower 11 bits as fp32) |
| `LREG2` | a1 (middle 11 bits of A as fp32) |
| `LREG3` | b1 (middle 11 bits of B as fp32) |
| `LREG4` | a2 (upper 10 bits of A as fp32) |
| `LREG5` | b2 (upper bits of B as fp32), then `top` accumulator |
| `LREG6` | `mid` accumulator |
| `LREG12` (vConstIntPrgm0) | Constant 0x7ff (11-bit mask) |
| `LREG13` (vConstIntPrgm1) | Constant -11 (shift amount for 11-bit chunk extraction) |
| `LREG14` (vConstFloatPrgm2) | Constant 8388608.0 (2**23, magic number for exact fp32-to-int conversion) |

#### Integer MUL -- UINT16

| Register | Usage |
|----------|-------|
| `LREG0` (a/a0) | Input A or lower 8 bits of A |
| `LREG1` (b/b0) | Input B or lower 8 bits of B |
| `LREG2` (a1) | Upper 8 bits of A (as fp32, Wormhole path) |
| `LREG3` (b1) | Upper 8 bits of B (as fp32, Wormhole path) |
| `LREG4` (out) | Output accumulator |
| `LREG5` (tmp) | Temporary for mantissa extraction |
| `LREG12` (vConstIntPrgm0) | Constant 0xff (8-bit mask, Wormhole) |
| `LREG13` (vConstFloatPrgm1) | Constant 8388608.0 (2**23, Wormhole) |

### Address Mode Configuration

#### Floating-Point MUL

The init function `_llk_math_eltwise_binary_sfpu_init_<SfpuType::unused>()` configures:

- **ADDR_MOD_7**: `srca.incr=0, srcb.incr=0, dest.incr=0` -- No auto-increment; the SFPU kernel manages DEST addressing explicitly via `dst_reg++` and the params dispatch uses `TTI_SETRWC` to advance between faces.

This is the same configuration on both Wormhole and Blackhole. The `SfpuType::unused` template parameter means ADDR_MOD_6 is NOT configured (the ADDR_MOD_6 branch only activates for `mul_int32`, `mul_uint16`, `max`, `min`, and their integer variants).

#### Integer MUL (INT32/UINT32 and UINT16)

The init function `_llk_math_eltwise_binary_sfpu_init_<SfpuType::mul_int32>()` (or `SfpuType::mul_uint16`) configures:

- **ADDR_MOD_7**: `srca.incr=0, srcb.incr=0, dest.incr=0` -- Used by `TT_SFPLOAD` instructions that should not auto-increment the DEST pointer.
- **ADDR_MOD_6**: `srca.incr=0, srcb.incr=0, dest.incr=2` -- Used by the final `TT_SFPLOADMACRO` (or `TT_SFPLOAD`) in each iteration that needs to advance the DEST pointer by 2 rows (since INT32 data occupies 2 DEST row slots per logical row in the SFPU addressing scheme).

This is the same configuration on both Wormhole and Blackhole.

For Wormhole INT32 specifically, the kernel uses `ADDR_MOD_3` (for loads) and `ADDR_MOD_2` (for stores) instead, which are configured separately by the A2D (Accumulate-to-DEST) copy_tile infrastructure that runs before the SFPU operation. The `ADDR_MOD_3` provides dest.incr=2 for load auto-increment, and `ADDR_MOD_2` provides dest.incr=2 for store auto-increment.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the binary element-wise SFPU program factory work in ttnn? What kernels does it use and how does it handle broadcasting?"
   **Reason**: Needed initial architectural context before reading source code.
   **Key Findings**: Confirmed the SFPU factory uses three kernels (reader, compute, writer), does not support broadcasting (requires exact shape match), and the compute kernel processes tiles through DST registers using copy_tile + SFPU operation pattern.

2. [SFPU] **Query**: "How does mul_binary_tile work in the SFPU binary compute path? What is the call chain from mul_binary_tile through the LLK layers to the core SFPU implementation?"
   **Reason**: Needed to identify file locations and call chain for the floating-point MUL SFPU kernel before reading source code.
   **Key Findings**: Confirmed that `mul_binary_tile` dispatches through `llk_math_eltwise_binary_sfpu_binop_mul` to `calculate_sfpu_binary_mul` in `ckernel_sfpu_binary.h`. Identified separate files for Blackhole and Wormhole architectures. Learned that the kernel includes BF16 RNE rounding when FP32 dest accumulation is disabled.

3. [SFPU] **Query**: "How does mul_binary_tile work? Trace the call chain from the compute API through llk_math_eltwise_binary_sfpu to ckernel_sfpu_mul_binary. What SFPU instructions does it use? What is the VectorMode and iteration pattern?"
   **Reason**: Needed LLK-level details about the params dispatch, VectorMode configuration, and SFPU instruction usage for the multiplication kernel.
   **Key Findings**: Confirmed VectorMode::RC processes all 4 faces with 8 iterations each. Identified the `_llk_math_eltwise_binary_sfpu_params_` function as the face-level dispatch loop. Learned that the integer path (`mul_int32`) uses a significantly different instruction set including `SFPLOADMACRO`, `SFPMUL24`, `SFPSHFT2`, `SFPCAST`, `SFPAND`, `SFPMAD`, `SFPEXMAN` on Wormhole.

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/binary_device_operation.cpp`
   **Reason**: Needed to understand path selection between FPU and SFPU factories.
   **Key Information**: `select_program_factory` checks `is_binary_sfpu_op()` which returns true for MUL. SFPU path is selected when shapes match.

2. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary/common/binary_op_utils.cpp`
   **Reason**: Needed to understand what preprocessor defines are generated for MUL.
   **Key Information**: For floating-point MUL: `BINOP_INIT = mul_binary_tile_init()`, `BINARY_SFPU_OP = mul_binary_tile(...)`. For integer MUL: `MUL_INT_INIT = mul_int_tile_init<DataFormat>()`, `BINARY_SFPU_OP = mul_int_tile<DataFormat>(...)`. Confirmed argument order: `BINARY_SFPU_OP` = `op_name(idst1, idst2, idst1)` where `idst1="i*2"` and `idst2="i*2+1"`, so result overwrites input A's DST slot.

3. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/eltwise_multi_core_program_factory_common.hpp`
   **Reason**: Needed to understand runtime argument setup and core distribution logic.
   **Key Information**: Contains the `set_eltwise_binary_runtime_args` template function shared by both FPU and SFPU factories. Handles work splitting via `split_work_to_cores`, shard-aware start_id calculation, and two-group core distribution for handling remainder tiles.
