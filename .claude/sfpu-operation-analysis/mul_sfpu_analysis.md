# MUL (SFPU) Implementation Analysis

## Overview

The MUL (SFPU) operation performs element-wise multiplication of two input tensors using the SFPU (Special Function Processing Unit) path on Tenstorrent hardware. Unlike the FPU-based `mul_tiles` path, this SFPU variant is selected when both input tensors share the same data type and that type is one of: FLOAT32, INT32, UINT32, or UINT16. The operation loads both input tiles into DST registers via `copy_tile`, executes the SFPU multiply function, and packs the result to the output circular buffer.

**Program factory path**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/element_wise_multi_core_sfpu_pgm_factory.cpp`

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | tile (32x32 elements) |
| **Unit size** | `per_core_block_size` tiles (1 for interleaved, up to `max_block_size` for sharded) |
| **Total units** | `per_core_block_cnt` blocks per core; total = `physical_volume / TILE_HW` tiles across all cores |
| **Loop structure** | Outer loop over `per_core_block_cnt` blocks; inner loop over `per_core_block_size` tiles within each block |

The compute kernel iterates `per_core_block_cnt` times. Each iteration processes `per_core_block_size` tiles. For interleaved tensors, `block_size = 1` and `block_cnt = num_tiles_per_core`. For sharded tensors, `block_size = find_max_block_size(num_tiles_per_shard)` (largest power-of-2 dividing the shard tile count, capped at 8) and `block_cnt = num_tiles_per_shard / block_size`.

## Tensor Format and Layout

### Input Tensors

| Property | Input Tensor A (cb_src0) | Input Tensor B (cb_src1) |
|----------|--------------------------|--------------------------|
| **Logical shape** | Arbitrary N-D (must match B) | Arbitrary N-D (must match A) |
| **Dimension convention** | Row-major logical ordering | Row-major logical ordering |
| **Tensor layout** | TILE_LAYOUT (32x32) | TILE_LAYOUT (32x32) |
| **Memory layout** | INTERLEAVED or SHARDED | INTERLEAVED or SHARDED |
| **Buffer type** | DRAM or L1 | DRAM or L1 |
| **Data type** | FLOAT32, INT32, UINT32, or UINT16 | Must match A's dtype |

### Output Tensor

| Property | Output Tensor |
|----------|---------------|
| **Logical shape** | Same as input tensors |
| **Dimension convention** | Row-major logical ordering |
| **Tensor layout** | TILE_LAYOUT (32x32) |
| **Memory layout** | INTERLEAVED or SHARDED |
| **Buffer type** | DRAM or L1 |
| **Data type** | Same as input dtype (or as configured by operation attributes) |

### Layout Transformations

No tilize/untilize or format conversions occur within this operation. Both inputs and the output must already be in TILE_LAYOUT. The SFPU path routes MUL to different underlying functions depending on data type:

- **FLOAT32**: `mul_binary_tile` with `BINOP_INIT` = `mul_binary_tile_init()`
- **INT32**: `mul_int_tile<DataFormat::Int32>` with `MUL_INT_INIT` = `mul_int_tile_init<DataFormat::Int32>()`
- **UINT32**: `mul_int_tile<DataFormat::UInt32>` with `MUL_INT_INIT` = `mul_int_tile_init<DataFormat::UInt32>()`
- **UINT16**: `mul_int_tile<DataFormat::UInt16>` with `MUL_INT_INIT` = `mul_int_tile_init<DataFormat::UInt16>()`

## Data Flow Pattern

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader | DRAM/L1 (src0_buffer) | CB c_0 | `cb_reserve_back`, `noc_async_read_tile`, `cb_push_back` |
| 1 | Reader | DRAM/L1 (src1_buffer) | CB c_1 | `cb_reserve_back`, `noc_async_read_tile`, `cb_push_back` |
| 2 | Compute | CB c_0 (inp0) | DST regs (even slots) | `cb_wait_front`, `copy_tile(cb_inp0, i, i*2)` |
| 2 | Compute | CB c_1 (inp1) | DST regs (odd slots) | `copy_tile(cb_inp1, i, i*2+1)` |
| 2 | Compute | DST regs | DST regs | `BINARY_SFPU_OP` = `mul_binary_tile(i*2+1, i*2, i*2+1)` or `mul_int_tile(...)` |
| 2 | Compute | DST regs | CB c_2 | `pack_tile(i*2, cb_out0)`, `cb_push_back` |
| 3 | Writer | CB c_2 | DRAM/L1 (dst_buffer) | `cb_wait_front`, `noc_async_write_page`, `cb_pop_front` |

**Key detail**: The compute kernel interleaves input A tiles into even DST register slots (`i*2`) and input B tiles into odd slots (`i*2+1`). The SFPU op macro is invoked as `mul_binary_tile(idst1=i*2+1, idst2=i*2, idst_out=i*2+1)` where `idst1` is the B operand index, `idst2` is the A operand index, and the result is written to `idst1` (the even slot is then packed via `pack_tile(i*2, cb_out0)`). Note: the define format from `binary_op_utils.cpp` line 535 is `op_name(idst1, idst2, idst1)` where `idst1 = "i * 2 + 1"` and `idst2 = "i * 2"`, so the actual call is `mul_binary_tile(i*2+1, i*2, i*2+1)`. The pack reads from slot `i*2`, which is the A-slot -- examining the SFPU op semantics, the third argument is the destination register index. The result is packed from `i*2`.

**Sharded path**: When inputs are sharded, the reader simply does `cb_reserve_back(cb_id, num_tiles)` + `cb_push_back(cb_id, num_tiles)` since the CB is backed by the tensor's L1 buffer directly (globally allocated address). Similarly for sharded output, the writer just does `cb_wait_front(cb_id, num_pages)` -- no NoC writes needed.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity (interleaved) | Capacity (sharded) | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------------------|-------------------|------------|-----------|----------|----------|----------|
| c_0 | cb_src0 | Input A staging | 2 tiles (`2 * max_block_size`) | `num_tiles_per_shard` tiles | 1 tile (interleaved) / `max_block_size` (sharded) | Double (interleaved) / Single (sharded) | Reader | Compute | Block |
| c_1 | cb_src1 | Input B staging | 2 tiles (`2 * max_block_size`) | `num_tiles_per_shard` tiles | 1 tile (interleaved) / `max_block_size` (sharded) | Double (interleaved) / Single (sharded) | Reader | Compute | Block |
| c_2 | cb_output | Output staging | 2 tiles (`2 * max_block_size`) | `num_tiles_per_shard` tiles | 1 tile (interleaved) / `max_block_size` (sharded) | Double (interleaved) / Single (sharded) | Compute | Writer | Block |
| c_3 | cb_src0interim | Pre-scaling intermediate for A | `max_block_size` tiles | `max_block_size` tiles | `max_block_size` tiles | Single | Compute | Compute | Block |
| c_4 | cb_src1interim | Pre-scaling intermediate for B | `max_block_size` tiles | `max_block_size` tiles | `max_block_size` tiles | Single | Compute | Compute | Block |

**Notes on c_3 and c_4**: These interim CBs are only created when the defines `SFPU_OP_INIT_PRE_IN0_0` or `SFPU_OP_INIT_PRE_IN1_0` are present. For a plain MUL operation (no fused activations requiring pre-scaling), these CBs are NOT created. They would be used for operations like LOGADDEXP that apply `exp()` to inputs before the binary op.

For MUL specifically, `cb_inp0 = cb_in0` (c_0) and `cb_inp1 = cb_in1` (c_1) since no pre-scaling is needed.

## Pipeline Pattern Summary

**Interleaved path**: CBs c_0, c_1, and c_2 are all allocated with `2 * max_block_size` tiles capacity while block_size is `max_block_size` (which is 1 for interleaved). This gives capacity = 2, block_size = 1, so **double-buffered**. Reader can fill one slot while compute processes the other.

**Sharded path**: CBs are allocated for the entire shard (`num_tiles_per_shard`), which equals the full workload. This is effectively **single-buffered** -- all data is present in L1 at once via the globally allocated buffer.

## Index Calculations

**Interleaved path**: Tiles are accessed sequentially using `TensorAccessor` for both reads and writes. The reader uses a linear tile ID starting from `start_id` and iterates `num_tiles` times. The writer similarly uses a linear tile ID.

**Block/width sharded path**: The reader uses a 2D iteration pattern:
- Outer loop: `h` from 0 to `block_height`
- Inner loop: `w` from 0 to `block_width`
- Tile ID: `row_start_tile_id + w`, where `row_start_tile_id` advances by `num_cores_y * block_width` per row

The `start_id` for each core is computed as:
```
start_id = (core_index / num_shards_per_width) * (block_height * block_width * num_shards_per_width)
         + (core_index % num_shards_per_width) * block_width
```

**DST register mapping**: In the compute kernel, input A tiles go to even DST slots (`i * 2`) and input B tiles go to odd DST slots (`i * 2 + 1`). The result is packed from slot `i * 2`.

## Memory Access Patterns

### Read Pattern
- **Interleaved**: Sequential tile reads, one tile at a time via `noc_async_read_tile`, with `noc_async_read_barrier()` after each pair (one from each input). Both inputs read the same tile IDs since shapes must match.
- **Sharded**: No NoC reads -- data is already in L1 via globally allocated CB.
- **Block/width sharded (non-sharded input)**: 2D tiled access pattern with row strides of `num_cores_y * block_width`.

### Write Pattern
- **Interleaved**: Sequential tile writes, one tile at a time via `noc_async_write_page`, with `noc_async_writes_flushed()` after each tile and a final `noc_async_write_barrier()`.
- **Sharded**: No NoC writes -- output is already in L1 via globally allocated CB.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (rectangular grid linearized in row-major order) |
| **Grid dimensions** | Determined by `operation_attributes.worker_grid` |
| **Total cores** | `compute_with_storage_grid_size.x * compute_with_storage_grid_size.y` (zero-start grid) or `all_device_cores.num_cores()` |
| **Work per core** | `num_tiles / num_cores` tiles (group 1 gets ceil, group 2 gets floor) |
| **Load balancing** | Two-group split: `core_group_1` gets `num_tiles_per_core_group_1` tiles, `core_group_2` gets one fewer tile. Sharded: all cores get equal `num_tiles_per_shard` tiles. |

Work splitting uses `tt::tt_metal::split_work_to_cores()` which divides total tiles evenly, with remainder tiles distributed to the first N cores (core_group_1). Cores beyond the working set receive zero tiles and skip processing.

## Arguments

### Compile-Time Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | block_or_width_sharded | uint32_t | 1 if block/width sharded, 0 otherwise |
| 1+ | TensorAccessorArgs(src0) | varies | Tensor accessor parameters for input A (only if not IN0_SHARDED) |
| N+ | TensorAccessorArgs(src1) | varies | Tensor accessor parameters for input B (only if not IN1_SHARDED) |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | output_cb_index | uint32_t | CB index for output (c_2) |
| 1+ | TensorAccessorArgs(dst) | varies | Tensor accessor parameters for output buffer |

#### Compute Kernel

No explicit compile-time argument indices. Configuration is via `ComputeConfig`:
- `fp32_dest_acc_en`: true if output is Float32, Int32, or UInt32
- `unpack_to_dest_mode`: `UnpackToDestFp32` for all CBs (non-POWER ops)
- `defines`: Map containing `BINOP_INIT`/`MUL_INT_INIT` and `BINARY_SFPU_OP` macros

### Runtime Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src0_addr | uint32_t | DRAM/L1 address of input tensor A |
| 1 | src1_addr | uint32_t | DRAM/L1 address of input tensor B |
| 2 | num_tiles | uint32_t | Total tiles this core must process |
| 3 | start_id | uint32_t | Starting tile ID for this core |
| 4 | block_height | uint32_t | Shard block height in tiles (0 if not sharded) |
| 5 | block_width | uint32_t | Shard block width in tiles (0 if not sharded) |
| 6 | num_cores_y | uint32_t | Number of shards per width dimension |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_cnt | uint32_t | Number of blocks to process |
| 1 | per_core_block_size | uint32_t | Number of tiles per block |

#### Writer Kernel (interleaved output)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | DRAM/L1 address of output buffer |
| 1 | num_pages | uint32_t | Number of tiles to write |
| 2 | start_id | uint32_t | Starting tile ID for writes |

#### Writer Kernel (block/width sharded to interleaved output)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Output buffer address |
| 1 | block_height | uint32_t | Block height in tiles |
| 2 | block_width | uint32_t | Block width in tiles |
| 3 | unpadded_block_height | uint32_t | Actual data height (may be less at edges) |
| 4 | unpadded_block_width | uint32_t | Actual data width (may be less at edges) |
| 5 | output_width | uint32_t | Total output width in tiles |
| 6 | block_size | uint32_t | Total tiles per block (height * width) |
| 7 | start_id | uint32_t | Starting tile ID |

## Kernel Implementations

### Reader Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader_binary_interleaved_start_id | BRISC (data movement) | NOC0 | DRAM/L1 src0, src1 | CB c_0, CB c_1 | Read tile pairs from both inputs |

- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/dataflow/reader_binary_interleaved_start_id.cpp`
- **Key Logic**: Two code paths based on `block_or_width_sharded`:
  - **False (interleaved/height-sharded)**: Simple linear loop from `start_id` to `start_id + num_tiles`, reading one tile at a time from each input.
  - **True (block/width sharded)**: 2D loop over `block_height` x `block_width` with row stride of `num_cores_y * block_width`.
  - **Sharded inputs**: When `IN0_SHARDED` or `IN1_SHARDED` is defined, the corresponding reads are skipped -- only `cb_reserve_back` + `cb_push_back` is done to notify the CB that data is available.

### Compute Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| eltwise_binary_sfpu_kernel | TRISC (compute) | N/A | CB c_0 (or c_3), CB c_1 (or c_4) | CB c_2 | copy_tile to DST, SFPU mul, pack_tile |

- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/eltwise_binary_sfpu_kernel.cpp`
- **Key Logic**:
  1. Waits for `per_core_block_size` tiles in both input CBs
  2. Reserves space in output CB
  3. Acquires tile registers (`tile_regs_acquire` + `tile_regs_wait`)
  4. Copies input A tiles to even DST slots: `copy_tile(cb_inp0, i, i*2)`
  5. Copies input B tiles to odd DST slots: `copy_tile(cb_inp1, i, i*2+1)`
  6. For each tile, executes the appropriate init + SFPU op:
     - FLOAT32: `BINOP_INIT` = `mul_binary_tile_init()`, `BINARY_SFPU_OP` = `mul_binary_tile(i*2+1, i*2, i*2+1)`
     - INT32/UINT32/UINT16: `MUL_INT_INIT` = `mul_int_tile_init<Format>()`, `BINARY_SFPU_OP` = `mul_int_tile<Format>(i*2+1, i*2, i*2+1)`
  7. Packs result from DST slot `i*2` to output CB
  8. Commits and releases tile registers
  9. Pops both input CBs, pushes output CB

### Writer Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| writer_unary_interleaved_start_id | BRISC (data movement) | NOC1 | CB c_2 | DRAM/L1 dst | Write result tiles sequentially |

- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
- **Key Logic**: Simple sequential write loop using `TensorAccessor`. One tile at a time: `cb_wait_front` -> `noc_async_write_page` -> `noc_async_writes_flushed` -> `cb_pop_front`. When `OUT_SHARDED` is defined, only `cb_wait_front` is called (data is already in the correct L1 location).

**Alternative writer** (block/width sharded input with interleaved output): `writer_unary_sharded_blocks_interleaved_start_id.cpp` handles the 2D block-to-interleaved conversion during writes.

## Implementation Notes

1. **UnpackToDestFp32**: For all non-POWER binary SFPU ops, all input CBs use `UnpackToDestMode::UnpackToDestFp32`. This means data is unpacked directly into the DST accumulator registers in FP32 format, bypassing the SRCA/SRCB registers. This is essential for SFPU operations which operate on DST register data.

2. **DST register interleaving**: The compute kernel places input A in even DST slots and input B in odd DST slots. This is a standard pattern for binary SFPU ops that need both operands accessible in DST simultaneously.

3. **No pre-scaling for MUL**: Unlike operations such as LOGADDEXP (which applies `exp()` to inputs first), plain MUL does not use the pre-scaling path. CBs c_3 and c_4 are not allocated.

4. **Fused activations**: The program factory supports optional fused activations (`operation_attributes.activations`), which can chain additional SFPU operations after the multiply via `SFPU_OP_CHAIN_0` defines.

5. **Integer multiplication variants**: The operation seamlessly handles INT32, UINT32, and UINT16 types via separate `mul_int_tile` specializations, each with appropriate data format templates.

6. **Block size optimization**: `find_max_block_size` finds the largest power-of-2 that divides the shard tile count (capped at 8), allowing the compute kernel to process multiple tiles per inner loop iteration for better efficiency.

7. **Zero-start grid optimization**: When the worker grid is a single rectangle starting at (0,0) and any shard grids also start at (0,0), a faster work distribution algorithm is used instead of the generic `CoreRangeSet`-based approach.

## External Knowledge Sources

### DeepWiki Queries
1. **Query**: "How does the binary SFPU element-wise program factory work? What kernels does it use (reader, compute, writer)? How are circular buffers configured for binary SFPU operations?"
   **Reason**: Initial architectural understanding of the SFPU binary operation program factory pattern.
   **Key Findings**: Confirmed the three-kernel architecture (reader, compute, writer), the CB configuration pattern (c_0/c_1 for inputs, c_2 for output, c_3/c_4 for optional intermediates), and the use of `UnpackToDestFp32` mode for SFPU operations.

### Documentation References
1. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary/common/binary_op_utils.cpp` (lines 228-241)
   **Reason**: Needed to determine exactly which defines and SFPU functions are used for MUL across different data types.
   **Key Information**: MUL routes to `mul_binary_tile` for floats (with `BINOP_INIT`) and `mul_int_tile<Format>` for integers (with `MUL_INT_INIT`). The BINARY_SFPU_OP macro is constructed as `op_name(i*2+1, i*2, i*2+1)`.

2. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/binary_device_operation.cpp` (lines 22-66)
   **Reason**: Needed to understand when the SFPU path is selected over the FPU path.
   **Key Information**: MUL uses the SFPU factory when `a.dtype() == b.dtype()` and dtype is FLOAT32, INT32, UINT32, or UINT16.

3. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/eltwise_multi_core_program_factory_common.hpp`
   **Reason**: Understanding runtime argument setup and core distribution logic.
   **Key Information**: Work splitting uses `split_work_to_cores()` for interleaved, shard grid for sharded. Two-group load balancing for remainder tiles. Detailed start_id calculation for sharded layouts.

## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel functions that the compute kernel dispatches to. The MUL operation has three distinct SFPU kernel paths: floating-point multiplication (`calculate_sfpu_binary_mul`), 32-bit integer multiplication (`mul_int32`), and 16-bit unsigned integer multiplication (`_mul_int_`). Each is analyzed in full below.

### SFPU Abstraction Layers

**Path 1: FLOAT32 (mul_binary_tile)**

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_binary_sfpu.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_binop.h` |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/ckernel_sfpu_binary.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{arch}/llk_lib/llk_math_eltwise_binary_sfpu_params.h` |

**Path 2: INT32/UINT32 (mul_int_tile)**

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/mul_int_sfpu.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_mul_int.h` |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/ckernel_sfpu_mul_int32.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{arch}/llk_lib/llk_math_eltwise_binary_sfpu_params.h` |

**Path 3: UINT16 (mul_int_tile\<UInt16\>)**

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/mul_int_sfpu.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_mul_int.h` |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_{arch}/common/inc/sfpu/ckernel_sfpu_mul_int.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{arch}/llk_lib/llk_math_eltwise_binary_sfpu_params.h` |

### Call Chain

**FLOAT32 path**:
1. The compute kernel invokes `mul_binary_tile(idst0, idst1, odst)` (API header in `eltwise_binary_sfpu.h`), which wraps `MATH((llk_math_eltwise_binary_sfpu_binop_mul<APPROX, BinaryOp::MUL, DST_ACCUM_MODE>(idst0, idst1, odst)))`.
2. `llk_math_eltwise_binary_sfpu_binop_mul` (in `llk_math_eltwise_binary_sfpu_binop.h`) calls `_llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>()` with `ckernel::sfpu::calculate_sfpu_binary_mul<APPROXIMATE, BinaryOp::MUL, 8, is_fp32_dest_acc_en>` as the function pointer.
3. `_llk_math_eltwise_binary_sfpu_params_` (in `llk_math_eltwise_binary_sfpu_params.h`) calls `_llk_math_eltwise_binary_sfpu_start_()` to set up the DST write address and stall for SFPU readiness, then iterates 4 times (one per 16x16 face in RC mode), calling the SFPU function each time with a `TTI_SETRWC` to advance the DST read/write counter by 16 rows between faces.
4. `calculate_sfpu_binary_mul` (in `ckernel_sfpu_binary.h`) loops 8 iterations (8 rows per face), loading two `vFloat` values from DST, multiplying them, optionally applying bf16 RNE rounding, and storing the result back.

**INT32/UINT32 path**:
1. The compute kernel invokes `mul_int_tile<DataFormat::Int32>(idst0, idst1, odst)` (API header in `mul_int_sfpu.h`), which wraps `MATH((llk_math_eltwise_binary_sfpu_mul_int<APPROX, DataFormat::Int32>(idst0, idst1, odst)))`.
2. `llk_math_eltwise_binary_sfpu_mul_int` (in `llk_math_eltwise_binary_sfpu_mul_int.h`) calls `_llk_math_eltwise_binary_sfpu_params_` with `sfpu::mul_int32<APPROXIMATE>` as the function pointer.
3. The params dispatch iterates 4 faces in RC mode, calling `mul_int32` each time.
4. `mul_int32` (in `ckernel_sfpu_mul_int32.h`) implements full 32-bit integer multiplication using different algorithms on Wormhole vs Blackhole (see annotated source below).

**UINT16 path**:
1. Same API entry point but with `DataFormat::UInt16` template argument.
2. `llk_math_eltwise_binary_sfpu_mul_int` dispatches to `sfpu::_mul_int_<APPROXIMATE, ITERATIONS>` instead of `mul_int32`.
3. `_mul_int_` (in `ckernel_sfpu_mul_int.h`) uses `SFPLOADMACRO` and `SFPMUL24` for efficient 16-bit unsigned multiplication.

**Init path (FLOAT32)**: `mul_binary_tile_init()` calls `llk_math_eltwise_binary_sfpu_binop_init<APPROX, BinaryOp::MUL>()`, which calls `llk_math_eltwise_binary_sfpu_init<SfpuType::unused, APPROXIMATE>(sfpu_binary_init<APPROXIMATE, BinaryOp::MUL>)`. The `_llk_math_eltwise_binary_sfpu_init_<SfpuType::unused>()` configures ADDR_MOD_7 with all-zero increments (no auto-increment), initializes the SFPU config register, and resets counters. `sfpu_binary_init<APPROX, BinaryOp::MUL>()` is a no-op for MUL (only DIV, POW, and XLOGY require initialization).

**Init path (INT32/UINT32)**: `mul_int_tile_init<DataFormat::Int32>()` calls `llk_math_eltwise_binary_sfpu_mul_int_init<APPROX, DataFormat::Int32>()`, which calls `_llk_math_eltwise_binary_sfpu_init_<SfpuType::mul_int32>()` (configuring both ADDR_MOD_7 with dest.incr=0 and ADDR_MOD_6 with dest.incr=2), followed by `sfpu::mul_int32_init<APPROXIMATE>()` which sets up programmable constants and SFPLOADMACRO instruction templates.

**Init path (UINT16)**: Same as INT32 but with `SfpuType::mul_uint16` and `sfpu::_init_mul_int_<APPROXIMATE>()` which configures different SFPLOADMACRO macros and programmable constants.

### Annotated SFPU Kernel Source

#### Path 1: FLOAT32 -- `calculate_sfpu_binary_mul` and `float32_to_bf16_rne`

```cpp
// File: tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/ckernel_sfpu_binary.h
// NOTE: Wormhole and Blackhole implementations are identical for this function.

sfpi_inline sfpi::vFloat float32_to_bf16_rne(sfpi::vFloat in) {
    sfpi::vUInt bits = sfpi::reinterpret<sfpi::vUInt>(in);
    sfpi::vUInt lsb = (bits >> 16) & 1; // extract bf16 mantissa LSB for tie-breaking
    bits = bits + 0x7fffU + lsb; // RNE: add rounding bias + LSB for even tie-break
    bits = bits & 0xFFFF0000U; // truncate lower 16 bits to get bf16 in fp32 container
    return sfpi::reinterpret<sfpi::vFloat>(bits);
}

template <bool APPROXIMATION_MODE, BinaryOp BINOP, int ITERATIONS, bool is_fp32_dest_acc_en>
inline void calculate_sfpu_binary_mul(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    // APPROXIMATION_MODE=true (from APPROX), BINOP=BinaryOp::MUL, ITERATIONS=8, is_fp32_dest_acc_en=DST_ACCUM_MODE
    constexpr uint dst_tile_size_sfpi = 32; // 64 / SFP_DESTREG_STRIDE = 32 rows per tile via SFPI
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi]; // load from DST[in0_tile + current_row]
        sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi]; // load from DST[in1_tile + current_row]

        sfpi::vFloat result = in0 * in1; // SFPU multiply: compiles to SFPMAD or SFPMUL

        if constexpr (!is_fp32_dest_acc_en) {
            // When not in fp32 accumulation mode, round to bf16 to match FPU behavior
            result = float32_to_bf16_rne(result);

            // FPU behavior: 0 * anything = 0 (including inf, nan)
            v_if(in0 == 0 || in1 == 0) { result = 0.0f; }
            v_endif;
        }

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result; // store result to DST[out_tile + current_row]
        sfpi::dst_reg++; // advance to next row within the face
    }
}

template <bool APPROXIMATION_MODE, BinaryOp BINOP>
inline void sfpu_binary_init() {
    _sfpu_binary_init_<APPROXIMATION_MODE, BINOP>();
}
```

The `_sfpu_binary_init_` function (from `ckernel_sfpu_binary.h` in the tt_llk submodule) is a no-op for `BinaryOp::MUL`:

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_{arch}/common/inc/sfpu/ckernel_sfpu_binary.h

template <bool APPROXIMATION_MODE, BinaryOp BINOP>
inline void _sfpu_binary_init_()
{
    if constexpr (BINOP == BinaryOp::DIV || BINOP == BinaryOp::POW)
    {
        _init_sfpu_reciprocal_<false>();
    }
    else if constexpr (BINOP == BinaryOp::XLOGY)
    {
        _init_log_<APPROXIMATION_MODE>();
    }
    // MUL, ADD, SUB, RSUB: no initialization needed
}
```

#### Path 2: INT32/UINT32 -- `mul_int32` (Wormhole B0)

```cpp
// File: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_mul_int32.h

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void mul_int32(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    // APPROXIMATION_MODE not used
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        constexpr uint dst_tile_size = 64; // raw DST stride (not halved by SFPI)

        // Implementation notes, see the original file for more details
        // Splits 32-bit integers into three 11-bit chunks:
        //   a = (a2 << 22) | (a1 << 11) | a0
        //   b = (b2 << 22) | (b1 << 11) | b0
        // Then computes: a*b = (top << 22) + (mid << 11) + low
        //   top = a0*b2 + a1*b1 + a2*b0
        //   mid = a0*b1 + a1*b0
        //   low = a0*b0
        // Uses fp32_to_u23 trick: mantissa_bits(x + 2**23) extracts integer value.

        TT_SFPLOAD(p_sfpu::LREG0, INT32, ADDR_MOD_3, dst_index_in0 * dst_tile_size); // a0 = load int from DST
        TTI_SFPSHFT2(p_sfpu::LREG0, p_sfpu::LREG13, p_sfpu::LREG2, 5); // a1 = a0 >> 11 (LREG13 = -11)
        TTI_SFPSHFT2(p_sfpu::LREG2, p_sfpu::LREG13, p_sfpu::LREG4, 5); // a2 = a1 >> 11

        TTI_SFPAND(0, p_sfpu::LREG12, p_sfpu::LREG2, 0); // a1 &= 0x7ff (LREG12 = 0x7ff)
        TTI_SFPCAST(p_sfpu::LREG2, p_sfpu::LREG2, 0);     // a1 = int_to_fp32(a1)

        TTI_SFPCAST(p_sfpu::LREG4, p_sfpu::LREG4, 0);     // a2 = int_to_fp32(a2)

        TTI_SFPAND(0, p_sfpu::LREG12, p_sfpu::LREG0, 0); // a0 &= 0x7ff
        TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG0, 0);     // a0 = int_to_fp32(a0)

        TT_SFPLOAD(p_sfpu::LREG1, INT32, ADDR_MOD_3, dst_index_in1 * dst_tile_size); // b0 = load int from DST
        TTI_SFPSHFT2(p_sfpu::LREG1, p_sfpu::LREG13, p_sfpu::LREG3, 5); // b1 = b0 >> 11
        TTI_SFPSHFT2(p_sfpu::LREG3, p_sfpu::LREG13, p_sfpu::LREG5, 5); // b2 = b1 >> 11

        TTI_SFPCAST(p_sfpu::LREG5, p_sfpu::LREG5, 0);     // b2 = int_to_fp32(b2)

        // top = a0*b2 + 2**23 (LREG14 = 2**23 magic constant for mantissa extraction)
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG5, p_sfpu::LREG14, p_sfpu::LREG5, 0);

        TTI_SFPAND(0, p_sfpu::LREG12, p_sfpu::LREG3, 0); // b1 &= 0x7ff
        TTI_SFPCAST(p_sfpu::LREG3, p_sfpu::LREG3, 0);     // b1 = int_to_fp32(b1)

        // top += a1*b1
        TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG3, p_sfpu::LREG5, p_sfpu::LREG5, 0);

        TTI_SFPAND(0, p_sfpu::LREG12, p_sfpu::LREG1, 0); // b0 &= 0x7ff
        TTI_SFPCAST(p_sfpu::LREG1, p_sfpu::LREG1, 0);     // b0 = int_to_fp32(b0)

        // top += a2*b0
        TTI_SFPMAD(p_sfpu::LREG4, p_sfpu::LREG1, p_sfpu::LREG5, p_sfpu::LREG5, 0);

        // mid = a0*b1 + 2**23
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG3, p_sfpu::LREG14, p_sfpu::LREG6, 0);

        // low = a0*b0 + 2**23
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG14, p_sfpu::LREG0, 0);

        // mid += a1*b0
        TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG1, p_sfpu::LREG6, p_sfpu::LREG6, 0);

        // extract integer values from fp32 mantissa bits
        TTI_SFPEXMAN(0, p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPEXMAN_MOD1_PAD9); // low = mantissa(low)
        TTI_SFPEXMAN(0, p_sfpu::LREG6, p_sfpu::LREG6, sfpi::SFPEXMAN_MOD1_PAD9); // mid = mantissa(mid)
        TTI_SFPEXMAN(0, p_sfpu::LREG5, p_sfpu::LREG5, sfpi::SFPEXMAN_MOD1_PAD9); // top = mantissa(top)

        TTI_SFPSHFT(22, 0, p_sfpu::LREG5, 1); // top <<= 22 (SFPSHFT_MOD1_ARG_IMM)
        TTI_SFPSHFT(11, 0, p_sfpu::LREG6, 1); // mid <<= 11

        TTI_SFPIADD(0, p_sfpu::LREG6, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_CC_NONE); // low += mid
        TTI_SFPIADD(0, p_sfpu::LREG5, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_CC_NONE); // low += top

        TT_SFPSTORE(p_sfpu::LREG0, INT32, ADDR_MOD_2, dst_index_out * dst_tile_size); // store result to DST
    }
}

template <bool APPROXIMATION_MODE>
inline void mul_int32_init() {
    sfpi::vConstIntPrgm0 = 0x7ff;        // LREG12: 11-bit mask
    sfpi::vConstIntPrgm1 = -11;           // LREG13: shift amount for 11-bit chunking
    sfpi::vConstFloatPrgm2 = 8388608.0f;  // LREG14: 2**23 magic constant for mantissa extraction
}
```

#### Path 2: INT32/UINT32 -- `mul_int32` (Blackhole)

The Blackhole implementation uses `SFPLOADMACRO` for significantly higher throughput (8 cycles per row vs many more on Wormhole).

```cpp
// File: tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_mul_int32.h

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void mul_int32(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    // APPROXIMATION_MODE not used
    // Implementation notes, see the original file for more details
    // Uses SFPLOADMACRO for 8-cycle-per-row throughput.
    // Algorithm: splits 32-bit values using >>23 to get hi/lo parts,
    // computes cross products and lo product via SFPMUL24,
    // then reconstructs: result = ((hi + cross0 + cross1) << 23) + lo

    constexpr uint dst_tile_size = 64;

    uint offset_in0 = dst_index_in0 * dst_tile_size;
    uint offset_in1 = dst_index_in1 * dst_tile_size;
    uint offset_out = dst_index_out * dst_tile_size;

    constexpr uint a0 = p_sfpu::LREG0;
    constexpr uint b0 = p_sfpu::LREG0; // aliases a0 (reused after a0 is consumed)
    constexpr uint a1 = p_sfpu::LREG1;
    constexpr uint b1 = p_sfpu::LREG2;
    constexpr uint b2 = p_sfpu::LREG3;
    constexpr uint c = p_sfpu::LREG4;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TT_SFPLOAD(b0, INT32, ADDR_MOD_7, offset_in1);                                    // b0 = load b from DST
        TT_SFPLOADMACRO((0 << 2) | (a1 & 3), INT32, ADDR_MOD_7, offset_in0 | (a1 >> 2)); // Macro 0: load a, schedule a1=a>>23, then a1=mul24_lo(b0,a1)
        TT_SFPLOADMACRO((1 << 2) | (b1 & 3), INT32, ADDR_MOD_7, offset_in1 | (b1 >> 2)); // Macro 1: load b, schedule b1=b>>23, then b1=mul24_lo(a0,b1)
        TT_SFPLOAD(a0, INT32, ADDR_MOD_7, offset_in0);                                    // a0 = load a from DST
        TT_SFPLOADMACRO((2 << 2) | (b2 & 3), INT32, ADDR_MOD_7, offset_in1 | (b2 >> 2)); // Macro 2: load b as b2, schedule b2=mul24_lo(a0,b2)
        TTI_SFPMUL24(a0, b2, p_sfpu::LCONST_0, c, sfpi::SFPMUL24_MOD1_UPPER);             // c = mul24_hi(a0, b2)
        TTI_SFPIADD(0, a1, b1, sfpi::SFPIADD_MOD1_CC_NONE);                                // b1 = b1 + a1 (cross terms)
        TT_SFPLOADMACRO((3 << 2) | (c & 3), INT32, ADDR_MOD_6, offset_out | (c >> 2));    // Macro 3: schedule c=shft(b1+c, 23), then store b2+c
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP; // drain pipeline
}

template <bool APPROXIMATION_MODE>
inline void mul_int32_init() {
    // Implementation notes, see the original file for more details
    // Loads instruction templates i=0-3 via backdoor (VD=12+i),
    // then configures 4 SFPLOADMACRO macros (Sequence[0]-[3]) via SFPCONFIG.

    constexpr uint b1 = p_sfpu::LREG2;
    constexpr uint c = p_sfpu::LREG4;

    // InstructionTemplate[0]: SFPSHFT2 with immediate -23 (extract high bits)
    TTI_SFPSHFT2(-23 & 0xfff, 0, 12, sfpi::SFPSHFT2_MOD1_SHFT_IMM);
    // InstructionTemplate[1]: SFPMUL24 lower 24 bits
    TTI_SFPMUL24(0, 0, p_sfpu::LCONST_0, 13, sfpi::SFPMUL24_MOD1_LOWER);
    // InstructionTemplate[2]: SFPSHFT with immediate 23 (shift cross terms back), uses VC as shift count source
    TTI_SFPSHFT(23, b1, 14, 1 | 4);
    // InstructionTemplate[3]: SFPIADD to accumulate final result
    TTI_SFPIADD(0, c, 15, sfpi::SFPIADD_MOD1_CC_NONE);

    // Macro 0: round=shft2(-23), mad=mul24_lo (delay 1)
    {
        constexpr uint simple_bits = 0;
        constexpr uint mad_bits = 0x80 | 0x00 | (1 << 3) | (4 + 1);    // VB=VD, delay=1, template=1
        constexpr uint round_bits = 0x80 | 0x00 | (0 << 3) | (4 + 0);  // VB=VD, delay=0, template=0
        constexpr uint store_bits = 0;
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4, 0); // Sequence[0]
    }
    // Macro 1: simple=SFPSHFT(b1,23), mad=mul24_lo (delay 1), round=shft2(-23)
    {
        constexpr uint simple_bits = 0x80 | 0x00 | (4 << 3) | (4 + 3); // VB=VD, delay=4, template=3 (SFPIADD)
        constexpr uint mad_bits = 0x80 | 0x00 | (1 << 3) | (4 + 1);
        constexpr uint round_bits = 0x80 | 0x00 | (0 << 3) | (4 + 0);
        constexpr uint store_bits = 0;
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4+1, 0); // Sequence[1]
    }
    // Macro 2: simple=iadd+VD=16(store), mad=mul24_lo
    {
        constexpr uint simple_bits = 0x80 | 0x40 | (4 << 3) | (4 + 3); // VB=VD, VD=16(store), delay=4, template=3
        constexpr uint mad_bits = 0x80 | 0x00 | (1 << 3) | (4 + 1);
        TTI_SFPCONFIG((mad_bits << 8) | simple_bits, 4+2, 1); // Sequence[2]
    }
    // Macro 3: simple=shft(b1,23), store=SFPSTORE LO16 (delay 2)
    {
        constexpr uint simple_bits = 0x80 | 0x00 | (0 << 3) | (4 + 2); // template=2 (SFPSHFT)
        constexpr uint mad_bits = 0;
        constexpr uint round_bits = 0;
        constexpr uint store_bits = 0x00 | 0x40 | (2 << 3) | 3;        // VD=16(store), delay=2, SFPSTORE
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4+3, 0); // Sequence[3]
    }
    // Misc config: StoreMod0=MOD0_FMT_SRCB, UsesLoadMod0ForStore={1,1,1,1}, UnitDelayKind=WaitForElapsed
    TTI_SFPCONFIG(0xff0, 8, 1);
}
```

#### Path 3: UINT16 -- `_mul_int_` (Blackhole)

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_mul_int.h
// NOTE: Blackhole uses SFPLOADMACRO + SFPMUL24 for 1-3 cycles per row throughput.

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _mul_int_(const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out)
{
    // APPROXIMATION_MODE not used
    // Implementation notes, see the original file for more details
    // Throughput: 1 cycle if all indices same, 2 if out matches one input, 3 otherwise.
    // Uses SFPLOADMACRO to schedule SFPMUL24 (24-bit multiply) and SFPSTORE in pipeline.

    int offset0 = (dst_index_in0 * 32) << 1; // dst tile offset, scaled for LO16 addressing
    int offset1 = (dst_index_in1 * 32) << 1;

    constexpr int a = p_sfpu::LREG0;
    constexpr int b = p_sfpu::LREG1;

    if (dst_index_out == dst_index_in0 && dst_index_out == dst_index_in1)
    {
        // All same tile: 1 cycle/row -- SFPLOADMACRO loads, multiplies, and stores in one shot
#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++)
        {
            TT_SFPLOADMACRO((0 << 2) | (a & 3), LO16, ADDR_MOD_6, offset1 | (a >> 2));
        }
    }
    else if (dst_index_out == dst_index_in0)
    {
        // Out == in0: 2 cycles/row
#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++)
        {
            TT_SFPLOAD(a, LO16, ADDR_MOD_7, offset1);  // load b (no auto-incr)
            TT_SFPLOADMACRO((0 << 2) | (b & 3), LO16, ADDR_MOD_6, offset0 | (b >> 2)); // macro: load a, mul24(a,b), store
        }
    }
    else if (dst_index_out == dst_index_in1)
    {
        // Out == in1: 2 cycles/row
#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++)
        {
            TT_SFPLOAD(a, LO16, ADDR_MOD_7, offset0);  // load a (no auto-incr)
            TT_SFPLOADMACRO((0 << 2) | (b & 3), LO16, ADDR_MOD_6, offset1 | (b >> 2)); // macro: load b, mul24(a,b), store
        }
    }
    else
    {
        // General case: 3 cycles/row
        int offset2 = (dst_index_out * 32) << 1;

#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++)
        {
            TT_SFPLOAD(a, LO16, ADDR_MOD_7, offset0);  // load a
            TT_SFPLOADMACRO((1 << 2) | (b & 3), LO16, ADDR_MOD_7, offset2 | (b >> 2)); // macro 1: schedule store to out
            TT_SFPLOAD(b, LO16, ADDR_MOD_6, offset1);   // load b, ADDR_MOD_6 increments dest by 2
        }
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP; // drain pipeline
}

template <bool APPROXIMATION_MODE>
inline void _init_mul_int_()
{
    // InstructionTemplate[0]: SFPMUL24 lower 24 bits, with LREG0 as both operands
    TTI_SFPMUL24(p_sfpu::LREG0, 0, p_sfpu::LCONST_0, 12, 0);

    // Macro 0: mad=mul24(VB=VD,VD=16/store), store=SFPSTORE LO16
    {
        constexpr std::uint32_t simple_bits = 0;
        constexpr std::uint32_t mad_bits    = 0x80 | 0x40 | (0 << 3) | 4; // VB=VD, VD=16(store), delay=0, template=0
        constexpr std::uint32_t round_bits  = 0;
        constexpr std::uint32_t store_bits  = 0x00 | 0x40 | (2 << 3) | 3; // VD=16(store), delay=2, SFPSTORE
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 0, 0); // Sequence[0]
    }

    // Macro 1: mad=mul24(VB=VD), store=SFPSTORE LO16 (different VD routing for general case)
    {
        constexpr std::uint32_t simple_bits = 0;
        constexpr std::uint32_t mad_bits    = 0x80 | 0x40 | (1 << 3) | 4; // VB=VD, VD=16(store), delay=1
        constexpr std::uint32_t round_bits  = 0;
        constexpr std::uint32_t store_bits  = 0x00 | 0x40 | (3 << 3) | 3; // delay=3
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 1, 0); // Sequence[1]
    }

    // Misc: UsesLoadMod0ForStore={1,1}, UnitDelayKind=WaitForElapsed
    TTI_SFPCONFIG(0x330, 8, 1);
}
```

#### Path 3: UINT16 -- `_mul_int_` (Wormhole B0)

The Wormhole implementation uses a different algorithm based on splitting u16 inputs into two u8 chunks and using FMA with a 2^23 magic constant for mantissa extraction.

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_mul_int.h

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _mul_int_(const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out)
{
    // APPROXIMATION_MODE not used
    // Implementation notes, see the original file for more details
    // Split u16 inputs: a = (a1 << 8) | a0; b = (b1 << 8) | b0
    // Compute via FMA: hi = a0*b1 + 2^23, then hi += a1*b0; lo = a0*b0 + 2^23
    // Result = mantissa(lo) + (mantissa(hi) << 8)
    // Uses SFPLOADMACRO for 12 cycles/row throughput.

    int offset0    = (dst_index_in0 * 32) << 1;
    int offset1    = (dst_index_in1 * 32) << 1;
    int offset_out = (dst_index_out * 32) << 1;

    constexpr int a0  = p_sfpu::LREG0;
    constexpr int b0  = p_sfpu::LREG1;
    constexpr int a1  = p_sfpu::LREG2;
    constexpr int b1  = p_sfpu::LREG3;
    constexpr int out = p_sfpu::LREG4;
    constexpr int tmp = p_sfpu::LREG5;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        // Macro-scheduled loads with cast and shift operations
        TT_SFPLOADMACRO((0 << 2) | (b1 & 3), InstrModLoadStore::LO16, ADDR_MOD_3, offset1 | (b1 >> 2));   // load b, schedule b1=b>>8, cast
        TT_SFPLOADMACRO((1 << 2) | (a0 & 3), InstrModLoadStore::LO16, ADDR_MOD_3, offset0 | (a0 >> 2));   // load a as a0, schedule cast
        TT_SFPLOADMACRO((1 << 2) | (b0 & 3), InstrModLoadStore::LO16, ADDR_MOD_3, offset1 | (b0 >> 2));   // load b as b0, schedule cast

        TTI_SFPAND(0, p_sfpu::LREG12, a0, 0); // a0 &= 0xff (LREG12 = 0xff from init)
        TTI_SFPAND(0, p_sfpu::LREG12, b0, 0); // b0 &= 0xff

        TT_SFPLOADMACRO((2 << 2) | (a1 & 3), InstrModLoadStore::LO16, ADDR_MOD_3, offset0 | (a1 >> 2));   // load a, schedule a1=a>>8, cast
        TT_SFPLOADMACRO((3 << 2) | (out & 3), InstrModLoadStore::LO16, ADDR_MOD_2, offset_out | (out >> 2)); // schedule final store

        TTI_SFPMAD(a0, b1, p_sfpu::LREG13, b1, 0);  // hi = a0*b1 + 2^23 (LREG13 = 8388608.0f from init)
        TTI_SFPMAD(a0, b0, p_sfpu::LREG13, a0, 0);  // lo = a0*b0 + 2^23
        TTI_SFPMAD(a1, b0, b1, b1, 0);               // hi += a1*b0

        TTI_SFPEXMAN(0, a0, tmp, sfpi::SFPEXMAN_MOD1_PAD9); // tmp = mantissa_bits(lo)
        TTI_SFPEXMAN(0, b1, out, sfpi::SFPEXMAN_MOD1_PAD9); // out = mantissa_bits(hi)
        // Macro 3 completes: out <<= 8, then out = tmp + out (via SFPIADD), then store
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP; // drain pipeline
}

template <bool APPROXIMATION_MODE>
inline void _init_mul_int_()
{
    sfpi::vConstIntPrgm0   = 0xff;      // LREG12: 8-bit mask for u8 extraction
    sfpi::vConstFloatPrgm1 = 8388608.0; // LREG13: 2^23 magic constant for mantissa trick

    constexpr int tmp = p_sfpu::LREG5;

    TTI_SFPSHFT2(-8 & 0xfff, 0, 12, 6); // InstructionTemplate[0]: shift right by 8 (extract high byte)
    TTI_SFPCAST(0, 13, 0);               // InstructionTemplate[1]: int_to_fp32 cast
    TTI_SFPSHFT2(8, 0, 14, 6);           // InstructionTemplate[2]: shift left by 8 (reconstruct)
    TTI_SFPIADD(0, tmp, 15, sfpi::SFPIADD_MOD1_CC_NONE); // InstructionTemplate[3]: integer add

    // Macros 0-3 configure SFPLOADMACRO pipeline scheduling
    // Implementation notes, see the original file for more details

    // Macro 0: round=shft2(-8), simple=cast
    {
        constexpr std::uint32_t simple_bits = 0x00 | 0x00 | (1 << 3) | (4 + 1);
        constexpr std::uint32_t mad_bits    = 0;
        constexpr std::uint32_t round_bits  = 0x80 | 0x00 | (0 << 3) | (4 + 0);
        constexpr std::uint32_t store_bits  = 0;
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 0, 0);
    }
    // Macro 1: simple=cast (delay 3)
    {
        constexpr std::uint32_t simple_bits = 0x00 | 0x00 | (3 << 3) | (4 + 1);
        constexpr std::uint32_t mad_bits    = 0;
        TTI_SFPCONFIG((mad_bits << 8) | simple_bits, 4 + 1, 1);
    }
    // Macro 2: simple=cast (delay 2), round=shft2(-8)
    {
        constexpr std::uint32_t simple_bits = 0x00 | 0x00 | (2 << 3) | (4 + 1);
        constexpr std::uint32_t mad_bits    = 0;
        constexpr std::uint32_t round_bits  = 0x80 | 0x00 | (1 << 3) | (4 + 0);
        constexpr std::uint32_t store_bits  = 0;
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 2, 0);
    }
    // Macro 3: simple=iadd(VD=16/store), round=shft2(<<8), store=SFPSTORE LO16
    {
        constexpr std::uint32_t simple_bits = 0x80 | 0x40 | (6 << 3) | (4 + 3); // VB=VD, VD=16, delay=6
        constexpr std::uint32_t mad_bits    = 0;
        constexpr std::uint32_t round_bits  = 0x80 | 0x00 | (5 << 3) | (4 + 2); // shft2(<<8), delay=5
        constexpr std::uint32_t store_bits  = 0x00 | 0x40 | (7 << 3) | 3;       // VD=16, delay=7, SFPSTORE
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 3, 0);
    }
    // Misc: UsesLoadMod0ForStore={1,1,1,1}, UnitDelayKind=WaitForElapsed
    TTI_SFPCONFIG(0xff0, 8, 1);
}
```

### SFPU Instructions Used

**FLOAT32 path (`calculate_sfpu_binary_mul`)**:
- `SFPLOAD` (via `dst_reg[]` read) -- loads a 32-bit float value from a DST register row into an SFPU local register (LREG)
- `SFPMAD` or `SFPMUL` (via `in0 * in1`) -- performs floating-point multiply-accumulate or multiply; the SFPI compiler selects the appropriate instruction for `vFloat * vFloat`
- `SFPSHFT` (via `bits >> 16`, `bits & mask`) -- integer right-shift used in the bf16 RNE rounding helper
- `SFPIADD` (via `bits + 0x7fffU + lsb`) -- integer addition used in bf16 RNE rounding
- `SFPAND` (via `bits & 0xFFFF0000U`) -- bitwise AND for masking in bf16 rounding
- `SFPSETCC` / `SFPENCC` (via `v_if`, `v_endif`) -- condition code manipulation for predicated execution (zero-check path)
- `SFPSTORE` (via `dst_reg[] =`) -- stores result from LREG back to DST register row
- `SFPLOADI` (via `vFloat(0.0f)` constant) -- loads immediate values

**INT32/UINT32 path (`mul_int32` Wormhole)**:
- `TT_SFPLOAD` -- loads 32-bit integer from DST into LREG
- `TTI_SFPSHFT2` -- two-operand shift: extracts 11-bit chunks via right-shift by 11
- `TTI_SFPAND` -- masks to 11 bits (AND with 0x7ff)
- `TTI_SFPCAST` -- converts integer to fp32 representation (lossless for 11-bit values)
- `TTI_SFPMAD` -- fused multiply-add in fp32: computes partial products and adds 2^23 bias
- `TTI_SFPEXMAN` -- extracts mantissa bits from fp32 value (inverse of the 2^23 trick)
- `TTI_SFPSHFT` -- integer left-shift to reconstruct final result (<<22, <<11)
- `TTI_SFPIADD` -- integer addition to combine partial products
- `TT_SFPSTORE` -- stores final 32-bit integer result back to DST

**INT32/UINT32 path (`mul_int32` Blackhole)**:
- `TT_SFPLOAD` -- loads integer from DST
- `TT_SFPLOADMACRO` -- macro-scheduled load that triggers pre-configured instruction sequences (shift, multiply, store)
- `TTI_SFPMUL24` -- 24-bit integer multiply (upper or lower half); hardware-native instruction on Blackhole
- `TTI_SFPIADD` -- integer addition for accumulating cross terms
- `TTI_SFPSHFT2` (via instruction templates) -- shift for extracting high bits (>>23)
- `TTI_SFPSHFT` (via instruction templates) -- shift for reconstructing result (<<23)
- `TTI_SFPCONFIG` -- configures SFPLOADMACRO instruction sequences and miscellaneous settings
- `TTI_SFPLOADI` -- loads immediate values into LREG0 for macro configuration
- `TTI_SFPNOP` -- pipeline drain (3 NOPs after the loop)

**UINT16 path (`_mul_int_` Blackhole)**:
- `TT_SFPLOADMACRO` -- macro-scheduled load triggering SFPMUL24 + SFPSTORE pipeline
- `TT_SFPLOAD` -- plain loads for operands that cannot use macros
- `TTI_SFPMUL24` (via instruction template) -- 24-bit multiply for u16*u16
- `TTI_SFPCONFIG` / `TTI_SFPLOADI` -- macro and instruction template configuration
- `TTI_SFPNOP` -- pipeline drain

**UINT16 path (`_mul_int_` Wormhole)**:
- `TT_SFPLOADMACRO` -- loads with scheduled cast and shift operations
- `TTI_SFPAND` -- masks to 8 bits for u8 chunk extraction
- `TTI_SFPMAD` -- fused multiply-add for partial products with 2^23 bias
- `TTI_SFPEXMAN` -- mantissa extraction for integer recovery
- `TTI_SFPSHFT2` (via templates) -- shift operations for byte splitting/reconstruction
- `TTI_SFPCAST` (via templates) -- integer-to-float conversion
- `TTI_SFPIADD` (via templates) -- integer addition for final accumulation

### SFPU Register Usage

**FLOAT32 path**:
- `dst_reg[dst_index_in0 * 32]` -- reads input A from DST at tile offset `in0`, current row within face
- `dst_reg[dst_index_in1 * 32]` -- reads input B from DST at tile offset `in1`, current row within face
- `dst_reg[dst_index_out * 32]` -- writes result to DST at tile offset `out`, current row within face
- `dst_reg++` -- auto-increments the implicit row counter after each iteration
- SFPU local registers (LREGs) are used implicitly by the SFPI compiler for intermediate values (`in0`, `in1`, `result`, `bits`, `lsb`)

**INT32/UINT32 path (Wormhole)**:
- `LREG0` (p_sfpu::LREG0) -- holds `a0` (11-bit chunk of input A), later reused for `low` product
- `LREG1` (p_sfpu::LREG1) -- holds `b0` (11-bit chunk of input B)
- `LREG2` (p_sfpu::LREG2) -- holds `a1` (middle 11-bit chunk of A)
- `LREG3` (p_sfpu::LREG3) -- holds `b1` (middle 11-bit chunk of B)
- `LREG4` (p_sfpu::LREG4) -- holds `a2` (high bits of A)
- `LREG5` (p_sfpu::LREG5) -- holds `b2` (high bits of B), later reused for `top`
- `LREG6` -- holds `mid` partial product
- `LREG12` (vConstIntPrgm0) -- programmable constant `0x7ff` (11-bit mask)
- `LREG13` (vConstIntPrgm1) -- programmable constant `-11` (shift amount)
- `LREG14` (vConstFloatPrgm2) -- programmable constant `8388608.0f` (2^23)

**INT32/UINT32 path (Blackhole)**:
- `LREG0` (a0/b0) -- alternately holds a and b values (aliased since reused after consumption)
- `LREG1` (a1) -- high bits of input A (a >> 23)
- `LREG2` (b1) -- high bits of input B (b >> 23), accumulates cross products
- `LREG3` (b2) -- holds b for low-part multiply
- `LREG4` (c) -- holds mul24_hi result and final accumulated value
- DST registers accessed at offsets `dst_index * 64` for INT32 format

**UINT16 path (Blackhole)**:
- `LREG0` (a) -- input A value
- `LREG1` (b) -- input B value
- SFPLOADMACRO implicitly manages intermediate register routing via VD parameters

**UINT16 path (Wormhole)**:
- `LREG0` (a0) -- low byte of A, later reused for `lo` product
- `LREG1` (b0) -- low byte of B
- `LREG2` (a1) -- high byte of A
- `LREG3` (b1) -- high byte of B, later holds `hi` accumulator
- `LREG4` (out) -- output accumulator, holds mantissa of `hi`
- `LREG5` (tmp) -- temporary for mantissa of `lo`
- `LREG12` (vConstIntPrgm0) -- programmable constant `0xff` (8-bit mask)
- `LREG13` (vConstFloatPrgm1) -- programmable constant `8388608.0f` (2^23)

### Address Mode Configuration

**ADDR_MOD_7** -- Configured by `eltwise_binary_sfpu_configure_addrmod<SfpuType>()` for all SFPU types:
- `srca.incr = 0`, `srcb.incr = 0`, `dest.incr = 0`
- Used for SFPLOAD instructions where no auto-increment is desired (the SFPU function handles row advancement explicitly via `dst_reg++` or `TTI_SETRWC`).
- Configuration is **identical** on Wormhole and Blackhole.

**ADDR_MOD_6** -- Additionally configured for `SfpuType::mul_int32` and `SfpuType::mul_uint16`:
- `srca.incr = 0`, `srcb.incr = 0`, `dest.incr = 2`
- The `dest.incr = 2` causes the DST write address to auto-increment by 2 after each SFPSTORE, which is used by the SFPLOADMACRO store operations to advance through the output tile rows.
- Configuration is **identical** on Wormhole and Blackhole.

**ADDR_MOD_3** and **ADDR_MOD_2** -- Used by the Wormhole `mul_int32` and `_mul_int_` kernels for load/store addressing. These are not explicitly configured in the binary SFPU init function (they are set up by other parts of the system, typically the A2D / unpack-to-dest path that runs before the SFPU). ADDR_MOD_3 is used for loads with auto-increment, ADDR_MOD_2 for stores with auto-increment.

**Face iteration** -- Between faces, `_llk_math_eltwise_binary_sfpu_params_` uses `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` twice per face (advancing by 16 rows total = one 16x16 face) in RC vector mode.

**Wormhole vs Blackhole difference in `_start_`/`_done_`**: On Wormhole, `_llk_math_eltwise_binary_sfpu_start_` calls `math::set_addr_mod_base()` and `_done_` calls `math::clear_addr_mod_base()` + `TTI_STALLWAIT(STALL_CFG, WAIT_SFPU)`. On Blackhole, these calls are absent -- the address mode base is not explicitly managed.

## External Knowledge Sources

### DeepWiki Queries
1. **Query**: "How does the binary SFPU element-wise program factory work? What kernels does it use (reader, compute, writer)? How are circular buffers configured for binary SFPU operations?"
   **Reason**: Initial architectural understanding of the SFPU binary operation program factory pattern.
   **Key Findings**: Confirmed the three-kernel architecture (reader, compute, writer), the CB configuration pattern (c_0/c_1 for inputs, c_2 for output, c_3/c_4 for optional intermediates), and the use of `UnpackToDestFp32` mode for SFPU operations.

2. **Query**: "How does mul_binary_tile work in the SFPU binary operations? What is the call chain from mul_binary_tile through LLK to the ckernel SFPU implementation? What files contain the SFPU kernel for binary multiplication (mul_binary_tile and mul_int_tile)?"
   **Reason**: Needed to trace the full call chain from the compute API down to the SFPU kernel implementation and locate all relevant source files.
   **Key Findings**: Identified the three-layer architecture: API header -> LLK dispatch (llk_math_eltwise_binary_sfpu_binop.h) -> core SFPU (ckernel_sfpu_binary.h for floats, ckernel_sfpu_mul_int32.h for integers). Confirmed that `calculate_sfpu_binary_mul` is a separate, MUL-specific function distinct from the generic `_calculate_sfpu_binary_`.

3. **Query**: "How does mul_binary_tile work? Trace the call chain from the compute API through LLK dispatch to the ckernel SFPU implementation. What files contain ckernel_sfpu_mul_binary and the LLK math eltwise binary sfpu functions for multiplication? Also explain mul_int_tile." (tenstorrent/tt-llk)
   **Reason**: Cross-referenced with tt-llk repo for authoritative LLK-level details on the integer multiplication paths.
   **Key Findings**: Confirmed the Wormhole mul_int32 uses an 11-bit chunking algorithm with SFPMAD + SFPEXMAN, while Blackhole uses SFPLOADMACRO + SFPMUL24 for much higher throughput. The UINT16 path (`_mul_int_`) exists in the tt_llk submodule and uses SFPLOADMACRO on both architectures but with different algorithms (24-bit multiply on Blackhole, 8-bit chunking with FMA on Wormhole).

### Documentation References
1. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary/common/binary_op_utils.cpp` (lines 228-241)
   **Reason**: Needed to determine exactly which defines and SFPU functions are used for MUL across different data types.
   **Key Information**: MUL routes to `mul_binary_tile` for floats (with `BINOP_INIT`) and `mul_int_tile<Format>` for integers (with `MUL_INT_INIT`). The BINARY_SFPU_OP macro is constructed as `op_name(i*2+1, i*2, i*2+1)`.

2. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/binary_device_operation.cpp` (lines 22-66)
   **Reason**: Needed to understand when the SFPU path is selected over the FPU path.
   **Key Information**: MUL uses the SFPU factory when `a.dtype() == b.dtype()` and dtype is FLOAT32, INT32, UINT32, or UINT16.

3. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/eltwise_multi_core_program_factory_common.hpp`
   **Reason**: Understanding runtime argument setup and core distribution logic.
   **Key Information**: Work splitting uses `split_work_to_cores()` for interleaved, shard grid for sharded. Two-group load balancing for remainder tiles. Detailed start_id calculation for sharded layouts.

### Confluence References
No Confluence page sections were consulted for this analysis. The SFPU instructions used in the MUL kernels (SFPLOAD, SFPMAD, SFPMUL24, SFPLOADMACRO, SFPSHFT, SFPIADD, SFPAND, SFPCAST, SFPEXMAN, SFPSTORE, SFPCONFIG, SFPLOADI, SFPNOP) were sufficiently understood from the source code comments and DeepWiki context.

### Glean References
No Glean searches were performed for this analysis. The source code provided sufficient detail on all SFPU instruction usage and register manipulation patterns.
