# MUL (binary_ng) Implementation Analysis

## Overview

The MUL operation in the `binary_ng` framework performs element-wise multiplication of two tensors (or a tensor and a scalar). It supports both **FPU** and **SFPU** execution paths, multiple broadcasting modes (none, scalar, row, column, mixed), and both interleaved and sharded memory layouts. The `binary_ng` framework is a generalized binary operation engine -- MUL is one of many operations dispatched through the same program factory by configuring defines and kernel selection.

**Program factory path**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_program_factory.cpp`

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | tile |
| **Unit size** | 1 tile (32x32 elements) |
| **Total units** | `c.physical_volume() / tile_hw` (total output tiles) |
| **Loop structure** | 6-level nested loop: nD -> D -> N -> C -> Ht -> Wt; each innermost iteration processes one tile |

One work unit is a single output tile. The compute kernel processes one tile per cycle (`num_tiles_per_cycle = 1`), reading one tile from each input, performing the multiply, and writing one output tile.

## Tensor Format and Layout

### Input Tensors

| Property | Input Tensor A | Input Tensor B |
|----------|---------------|---------------|
| **Logical shape** | Up to rank 6+: [..., D, N, C, H, W] | Up to rank 6+: [..., D, N, C, H, W] (or scalar) |
| **Dimension convention** | Last 5 dims: D, N, C, H, W; dims > 5 collapsed into nD | Same as A, or 1x1 for scalar |
| **Tensor layout** | TILE_LAYOUT | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED or SHARDED (height, width, block) | INTERLEAVED or SHARDED; or absent (scalar from host) |
| **Buffer type** | DRAM or L1 | DRAM or L1 (or none for scalar) |
| **Data type** | BFLOAT16, FLOAT32, INT32, UINT32 | Same as A (or inferred for scalar mode) |

### Output Tensor

| Property | Output Tensor C |
|----------|----------------|
| **Logical shape** | Broadcast-expanded shape of A and B |
| **Dimension convention** | Same [..., D, N, C, H, W] convention |
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED or SHARDED |
| **Buffer type** | DRAM or L1 |
| **Data type** | BFLOAT16, FLOAT32, INT32, UINT32 |

### Layout Transformations

No explicit tilize/untilize is performed. Both inputs and outputs must already be in TILE_LAYOUT. When `a_dtype != c_dtype` and the operation is not a quantization or integer division, a `TYPECAST` unary post-activation is appended to the compute kernel's post-processing chain.

## Data Flow Pattern

### Two-Tensor Path (b is a tensor)

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader (ng) | DRAM/L1 (tensor A and tensor B) | CB c_0 (src_a), CB c_1 (src_b) | reserve_back, noc_async_read_page, push_back |
| 2 | Compute | CB c_0, CB c_1 | CB c_2 (output) | wait_front, pop_front, reserve_back, push_back |
| 3 | Writer (ng) | CB c_2 | DRAM/L1 (output C) | wait_front, noc_async_write_page, pop_front |

Key: The reader kernel reads **both** input tensors (A and B). The writer kernel only writes the output.

### Scalar Path (b is a scalar value)

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader (old) | DRAM/L1 (tensor A only) | CB c_0 (src_a) | reserve_back, noc_async_read_page, push_back |
| 1b | Writer (scalar) | Host scalar value | CB c_1 (src_b, filled once) | reserve_back, fill_with_val, push_back |
| 2 | Compute | CB c_0, CB c_1 | CB c_2 (output) | wait_front, pop_front, reserve_back, push_back |
| 3 | Writer (scalar) | CB c_2 | DRAM/L1 (output C) | wait_front, noc_async_write_page, pop_front |

Key: In scalar mode the writer fills CB c_1 with the scalar value once, then acts as the output writer. The reader only reads tensor A.

## Circular Buffer Configuration

### Base Configuration (no broadcast, interleaved)

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_src_a | Input A staging | 2 tiles | 1 tile | Double | Reader | Compute | Program |
| c_1 | cb_src_b | Input B staging | 2 tiles (tensor) / 1 tile (scalar) | 1 tile | Double / Single | Reader or Writer(scalar) | Compute | Program |
| c_2 | cb_out | Output staging | 2 tiles | 1 tile | Double | Compute | Writer | Program |
| c_3 | cb_lhs_interim | LHS activation intermediate | 1 tile | 1 tile | Single | Compute | Compute | Block |
| c_4 | cb_rhs_interim | RHS activation intermediate | 1 tile | 1 tile | Single | Compute | Compute | Block |

Notes:
- c_3 is only created if `PROCESS_LHS_ACTIVATIONS` is non-empty. For plain MUL, it is empty (no LHS pre-processing).
- c_4 is only created if `PROCESS_RHS_ACTIVATIONS` is non-empty. For plain MUL, it is empty.
- When sharded, CB capacity equals the shard volume in tiles (not 2), and the CB is backed by the sharded buffer directly.

### Broadcast-Specific CBs

| CB ID | Name | Purpose | Capacity | Created When |
|-------|------|---------|----------|-------------|
| c_5 | cb_row_bcast_a | Row broadcast buffer for A | 2 tiles | ROW_A or ROW_A_COL_B broadcast |
| c_6 | cb_row_bcast_b | Row broadcast buffer for B | 2 tiles | ROW_B or ROW_B_COL_A broadcast |

## Pipeline Pattern Summary

- **Interleaved mode**: CB c_0 and c_1 have capacity 2 tiles with block size 1 tile = **Double-buffered**. CB c_2 has capacity 2 tiles with block size 1 tile = **Double-buffered**. This allows the reader and writer to overlap with compute.
- **Sharded mode**: CBs are sized to the full shard volume. The entire shard is pushed/popped at once = effectively **Single-buffered** per shard.
- **Scalar mode**: CB c_1 is 1 tile = **Single-buffered** (filled once, consumed repeatedly without popping until the end).

## Index Calculations

The program factory decomposes the output tensor's tile-space coordinates using a 6-level hierarchy:

```
padded_shape -> [nD, D, N, C, Ht, Wt]
where:
  Ht = shape[-2] / tile_height
  Wt = shape[-1] / tile_width
  nD = product of all dims beyond rank 5 (collapsed)
```

**Stride computation for broadcasting**: Each input tensor has independent strides for each dimension level. A stride is set to 0 when a dimension is 1 (broadcast), otherwise it equals the product of inner dimensions:

```cpp
aHt * aWt * aC * aN * aD * (aND > 1)   // nD_stride: 0 if aND==1
aHt * aWt * aC * aN * (aD > 1)          // d_stride: 0 if aD==1
aHt * aWt * aC * (aN > 1)               // n_stride: 0 if aN==1
aHt * aWt * (aC > 1)                    // c_stride: 0 if aC==1
```

This zero-stride pattern implements broadcasting: when a dimension is 1, its stride is 0, so the tile offset does not advance along that dimension.

**TensorAccessor** is used for both reader and writer to map tile indices to physical addresses (handling interleaved bank distribution).

## Memory Access Patterns

### Read Pattern
- **Interleaved**: Tiles read one at a time via `noc_async_read_page` with immediate barrier. Sequential within each tile row (tw loop), then advancing through th, c, n, d, nd. When broadcasting, the input tile offset wraps due to zero strides.
- **Sharded**: All shard tiles are made available instantly via `cb_reserve_back` / `cb_push_back` on the full shard.

### Write Pattern
- **Interleaved**: Tiles written one at a time via `noc_async_write_page` with immediate barrier. Same nested loop order as reader.
- **Sharded**: Output is written directly to the sharded CB; no explicit NoC writes needed.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 2D (prefers rectangular grid starting at (0,0)) |
| **Grid dimensions** | Determined by `operation_attributes.worker_grid` |
| **Total cores** | `compute_with_storage_grid.x * compute_with_storage_grid.y` (zero-start) or `all_device_cores.num_cores()` |
| **Work per core** | `num_tiles_per_core_group_1` or `num_tiles_per_core_group_2` tiles |
| **Load balancing** | Two-group split: group 1 gets `ceil(total_tiles / num_cores)`, group 2 gets `floor(total_tiles / num_cores)` |

Work splitting is done via `tt::tt_metal::split_work_to_cores()`. Cores not assigned any tiles receive zero-filled runtime arguments and effectively no-op. When sharded, the core grid is determined by the shard spec, and each core processes its own shard.

## Arguments

### Compile-Time Arguments

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | num_tiles_per_cycle | uint32_t | Always 1 -- tiles produced per read-compute-write cycle |

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0..M | TensorAccessor args (A) | uint32_t | Bank mapping and addressing metadata for tensor A |
| M+1..N | TensorAccessor args (B) | uint32_t | Bank mapping and addressing metadata for tensor B |
| N+1 | has_sharding | uint32_t | 1 if native L1 sharding is active, 0 otherwise |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0..M | TensorAccessor args (C) | uint32_t | Bank mapping and addressing metadata for output tensor C |
| M+1 | has_sharding | uint32_t | 1 if native L1 sharding is active, 0 otherwise |

### Runtime Arguments

#### Reader Kernel (two-tensor path, 21 args)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | DRAM/L1 address of tensor A buffer |
| 1 | start_tile_id | uint32_t | Starting output tile index for this core (c_start_id) |
| 2 | src_num_tiles | uint32_t | Number of A shard tiles (sharded only, else 0) |
| 3 | dst_num_tiles | uint32_t | Number of output tiles assigned to this core |
| 4 | dst_shard_width | uint32_t | Width of output shard in tiles (sharded only, else 0) |
| 5 | nD_stride | uint32_t | A's stride for collapsed nD dimension (0 if broadcast) |
| 6 | d_stride | uint32_t | A's stride for D dimension |
| 7 | n_stride | uint32_t | A's stride for N dimension |
| 8 | c_stride | uint32_t | A's stride for C dimension |
| 9 | D | uint32_t | Output D dimension |
| 10 | N | uint32_t | Output N dimension |
| 11 | C | uint32_t | Output C dimension |
| 12 | Ht | uint32_t | Output height in tiles |
| 13 | Wt | uint32_t | Output width in tiles |
| 14 | cND | uint32_t | Collapsed nD dimension of output |
| 15 | src_addr_b | uint32_t | DRAM/L1 address of tensor B buffer |
| 16 | nD_stride_b | uint32_t | B's stride for collapsed nD dimension |
| 17 | d_stride_b | uint32_t | B's stride for D dimension |
| 18 | n_stride_b | uint32_t | B's stride for N dimension |
| 19 | c_stride_b | uint32_t | B's stride for C dimension |
| 20 | src_num_tiles_b | uint32_t | Number of B shard tiles (sharded only, else 0) |

#### Writer Kernel (two-tensor path, 11 args)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | DRAM/L1 address of output buffer C |
| 1 | start_tile_id | uint32_t | Starting output tile index for this core |
| 2 | dst_num_tiles | uint32_t | Number of output tiles for this core |
| 3 | dst_shard_width | uint32_t | Width of output shard in tiles |
| 4 | D | uint32_t | Output D dimension |
| 5 | N | uint32_t | Output N dimension |
| 6 | C | uint32_t | Output C dimension |
| 7 | Ht | uint32_t | Output height in tiles |
| 8 | Wt | uint32_t | Output width in tiles |
| 9 | cND | uint32_t | Collapsed nD dimension |
| 10 | (reserved) | uint32_t | Always 0 |

#### Writer Kernel (scalar path, 11 args)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | packed_scalar | uint32_t | Scalar value packed into uint32 (bf16 pair or float bits) |
| 1 | dst_addr | uint32_t | DRAM/L1 address of output buffer C |
| 2 | start_tile_id | uint32_t | Starting output tile index |
| 3 | dst_num_tiles | uint32_t | Number of output tiles |
| 4 | dst_shard_width | uint32_t | Width of output shard in tiles |
| 5 | D | uint32_t | Output D dimension |
| 6 | N | uint32_t | Output N dimension |
| 7 | C | uint32_t | Output C dimension |
| 8 | Ht | uint32_t | Output height in tiles |
| 9 | Wt | uint32_t | Output width in tiles |
| 10 | cND | uint32_t | Collapsed nD dimension |

#### Compute Kernel (4 args)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | num_tiles | uint32_t | Total tiles to process on this core |
| 1 | freq | uint32_t | Broadcast repeat frequency (1 for no bcast, Wt for col bcast, Ht*Wt for scalar bcast) |
| 2 | counter | uint32_t | Initial counter offset for broadcast cycling |
| 3 | compute_scalar_value | uint32_t | Quantization zero point (0 if not quant op) |

## Kernel Implementations

### FPU Path (non-SFPU MUL)

For MUL, the FPU path sets `binary_op = FpuBinaryOp::MUL`, which generates defines:
- `BINARY_OP` = `mul_tiles`
- `BINARY_OP_TYPE` = `EltwiseBinaryType::ELWMUL`

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader (ng, no_bcast) | RISCV_0 | NOC0 | DRAM (A, B) | CB c_0, c_1 | Read A and B tiles via TensorAccessor |
| compute (no_bcast) | RISCV_2 | N/A | CB c_0, c_1 | CB c_2 | `mul_tiles` via FPU matrix engine |
| writer (ng, no_bcast) | RISCV_1 | NOC1 | CB c_2 | DRAM (C) | Write output tiles via TensorAccessor |

- **File (compute)**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_no_bcast.cpp`
- **Key Logic**: Calls `binary_op_init_common`, then per tile: `binary_tiles_init<true, BINARY_OP_TYPE>`, `tile_regs_acquire`, `BINARY_OP(cb_lhs, cb_rhs, 0, 0, 0)`, optional post-activations, `pack_tile`, `tile_regs_release`.

### SFPU Path (SFPU MUL)

For MUL, the SFPU path sets `binary_op = SfpuBinaryOp::MUL`, generating:
- `BINARY_SFPU_INIT` = `mul_binary_tile_init();` (or `mul_int_tile_init<DataFormat::Int32>();` for INT32)
- `BINARY_SFPU_OP` = `mul_binary_tile` (or `mul_int_tile<DataFormat::Int32>`)

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader (ng, no_bcast) | RISCV_0 | NOC0 | DRAM (A, B) | CB c_0, c_1 | Read A and B tiles via TensorAccessor |
| compute (sfpu, no_bcast) | RISCV_2 | N/A | CB c_0, c_1 | CB c_2 | `copy_tile` both operands to dest, then `mul_binary_tile` on SFPU |
| writer (ng, no_bcast) | RISCV_1 | NOC1 | CB c_2 | DRAM (C) | Write output tiles via TensorAccessor |

- **File (compute)**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_sfpu_no_bcast.cpp`
- **Key Logic**: Uses `copy_tile` to load both operands into dest registers at even/odd indices (i*2, i*2+1), then calls `BINARY_SFPU_OP(i*2, i*2+1, i*2)`. The SFPU operates on dest register pairs. `unpack_to_dest_mode` is set to `UnpackToDestFp32` for SFPU ops (except POWER).

### Scalar Path

- **File (compute)**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_sfpu_scalar.cpp`
- **Key Logic**: RHS is preprocessed once before the loop (`cb_wait_front` on cb_post_rhs outside the loop). The scalar tile stays in the CB throughout, popped only after the main loop completes.

### Reader Kernels

- **File (two-tensor, no bcast)**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/dataflow/reader_interleaved_no_bcast.cpp`
- **Key Logic**: 6-level nested loop iterating over output tile coordinates. Computes separate `tile_offset` and `tile_offset_b` for inputs A and B using per-dimension strides. Strides of 0 implement broadcasting. Uses `TensorAccessor` for address translation. When sharded, simply calls `cb_reserve_back` / `cb_push_back` on the entire shard.

### Writer Kernels

- **File (two-tensor)**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/dataflow/writer_interleaved_no_bcast.cpp`
- **File (scalar)**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/writer_interleaved_scalar.cpp`
- **Key Logic**: Same 6-level nested loop as reader. The scalar writer additionally fills CB c_1 with the packed scalar value before the write loop.

## Implementation Notes

1. **FPU vs SFPU selection**: The `is_sfpu` flag in operation attributes determines the execution path. For MUL, both paths are available. The FPU path uses the hardware matrix multiply unit (`mul_tiles`), while the SFPU path uses the vector unit (`mul_binary_tile`). The FPU path is preferred for BFLOAT16 as it uses the more efficient matrix engine.

2. **Broadcast handling**: The framework supports 9 subtile broadcast types. Broadcast is handled at two levels:
   - **Reader level**: Different reader kernels handle filling/repeating tiles for scalar, row, and column broadcasts.
   - **Compute level**: Different compute kernels handle LLK-level broadcast optimizations (e.g., `eltwise_binary_row_bcast.cpp` for row broadcasts with BFLOAT16).

3. **LLK broadcast optimization**: When all operands are BFLOAT16 and the broadcast type is ROW or ROW_COL, a specialized LLK broadcast kernel is used (`ComputeRowBcastNg` or `ComputeRowColBcastNg`). This avoids tile filling in the reader and instead uses hardware-level broadcast during the FPU operation.

4. **Zero-start grid optimization**: When the core grid is a single rectangle starting at (0,0) and any sharded tensors also start at (0,0), the factory uses optimized `grid_to_cores` functions rather than the generic `corerange_to_cores` path, for faster core iteration.

5. **Sharding fallback**: If the output is unevenly sharded or the native L1 sharding conditions are not met, the factory falls back to treating sharded tensors as interleaved (using TensorAccessor for address translation), avoiding kernel deadlocks from mismatched shard sizes across cores.

6. **Scalar packing**: For BFLOAT16, the scalar is truncated and packed as two bf16 values in a uint32. For FLOAT32, the raw float bits are stored. For INT32/UINT32, the integer value is stored directly.

7. **Pre/post activations**: The MUL operation itself has no pre/post activations, but the framework supports them. Operations like LOGICAL_AND use MUL with NEZ pre-processing on both inputs. The activation chain is compiled into defines (`PROCESS_LHS_ACTIVATIONS`, `PROCESS_RHS_ACTIVATIONS`, `PROCESS_POST_ACTIVATIONS`).

8. **Rank > 5 support**: Dimensions beyond rank 5 are collapsed into a single `nD` dimension, allowing the 6-level loop structure to handle arbitrary-rank tensors.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the binary_ng operation work? What is the binary_ng program factory structure, what kernels does it use, and how does it handle different broadcast modes?"
   **Reason**: Initial reconnaissance to understand the overall architecture before diving into source code.
   **Key Findings**: Confirmed the kernel selection logic via `BinaryNgKernelConfig`, the broadcast type enum, and the separation of reader/compute/writer kernels. Identified the `kernels_ng` directory for newer kernel variants.

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_utils.hpp`
   **Reason**: Understand the kernel name enum, OpConfig structure, and helper function signatures.
   **Key Information**: KernelName enum with 13 variants, OpConfig supporting both FPU and SFPU binary ops, AllShardSpecs/AllShardVolumes for sharding support.

2. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_utils.cpp`
   **Reason**: Understand kernel file path mapping, OpConfig construction for MUL, and SFPU init/op function names.
   **Key Information**: MUL maps to `FpuBinaryOp::MUL` or `SfpuBinaryOp::MUL`. FPU defines: `mul_tiles` / `EltwiseBinaryType::ELWMUL`. SFPU defines: `mul_binary_tile_init()` / `mul_binary_tile`. Integer variant: `mul_int_tile_init<DataFormat::Int32>()` / `mul_int_tile<DataFormat::Int32>`.

3. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_device_operation.hpp`
   **Reason**: Understand SubtileBroadcastType enum values and their meanings.
   **Key Information**: 9 broadcast types covering all combinations of scalar, row, column, and mixed broadcasts.

## SFPU Kernel Implementation

This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to for the MUL (binary_ng) SFPU path.

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_binary_sfpu.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_binop.h` |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_binary.h` (overrides in tt-metal repo) and `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/common/inc/sfpu/ckernel_sfpu_binary.h` (canonical tt-llk source) |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_binary_sfpu_params.h` |

### Call Chain

1. The compute kernel calls `mul_binary_tile(i*2, i*2+1, i*2)` (defined via `BINARY_SFPU_OP` macro).
2. `mul_binary_tile()` in `eltwise_binary_sfpu.h` calls `MATH((llk_math_eltwise_binary_sfpu_binop_mul<APPROX, BinaryOp::MUL, DST_ACCUM_MODE>(idst0, idst1, odst)))`.
3. `llk_math_eltwise_binary_sfpu_binop_mul()` in `llk_math_eltwise_binary_sfpu_binop.h` invokes `_llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::calculate_sfpu_binary_mul<APPROXIMATE, BinaryOp::MUL, 8, is_fp32_dest_acc_en>, dst_index0, dst_index1, odst, VectorMode::RC)`.
4. `_llk_math_eltwise_binary_sfpu_params_()` in `llk_math_eltwise_binary_sfpu_params.h` calls `_llk_math_eltwise_binary_sfpu_start_()`, then invokes the passed `sfpu_func` once per face (4 faces for RC mode), using `TTI_SETRWC` to advance the destination register pointer between faces.
5. `calculate_sfpu_binary_mul()` in `ckernel_sfpu_binary.h` executes the SFPU microcode: loads two values from DEST registers, multiplies them, optionally rounds to bf16, and stores the result.

The init path follows a parallel chain: `mul_binary_tile_init()` -> `llk_math_eltwise_binary_sfpu_binop_init<APPROX, BinaryOp::MUL>()` -> `llk_math_eltwise_binary_sfpu_init<SfpuType::unused, APPROXIMATE>(sfpu_binary_init<APPROX, BinaryOp::MUL>)` -> `_llk_math_eltwise_binary_sfpu_init_<SfpuType::unused>()` (configures SFPU config reg, address modes, resets counters) followed by `_sfpu_binary_init_<APPROX, BinaryOp::MUL>()` (no-op for MUL since it needs no special LUT or register setup).

### Annotated SFPU Kernel Source

```cpp
// File: tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_binary.h
// (Also canonical in tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/common/inc/sfpu/ckernel_sfpu_binary.h)

// Helper: Convert float32 to bfloat16 using IEEE 754 Round-to-Nearest-Even (RNE)
sfpi_inline sfpi::vFloat float32_to_bf16_rne(sfpi::vFloat in) {
    sfpi::vUInt bits = sfpi::reinterpret<sfpi::vUInt>(in);
    // Extract bit 16 (LSB of bf16 mantissa) for tie-breaking
    sfpi::vUInt lsb = (bits >> 16) & 1;
    // Implementation notes, see the original file for more details
    bits = bits + 0x7fffU + lsb;
    bits = bits & 0xFFFF0000U;
    return sfpi::reinterpret<sfpi::vFloat>(bits);
}

template <bool APPROXIMATION_MODE, BinaryOp BINOP, int ITERATIONS, bool is_fp32_dest_acc_en>
inline void calculate_sfpu_binary_mul(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    // APPROXIMATION_MODE=true (APPROX define), BINOP=BinaryOp::MUL, ITERATIONS=8, is_fp32_dest_acc_en=DST_ACCUM_MODE
    constexpr uint dst_tile_size_sfpi = 32; // 64 rows / SFP_DESTREG_STRIDE(2) = 32 sfpi rows per tile
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi]; // Load from DEST tile at input0 offset
        sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi]; // Load from DEST tile at input1 offset

        sfpi::vFloat result = in0 * in1; // SFPMUL: element-wise multiply across SIMD lanes

        if constexpr (!is_fp32_dest_acc_en) {
            result = float32_to_bf16_rne(result); // Software RNE truncation to bf16
            // Zero-preservation to match FPU: 0 * x = 0, x * 0 = 0 (prevents NaN from 0 * inf)
            v_if(in0 == 0 || in1 == 0) { result = 0.0f; } // Uses SFPU condition codes (SFPSETCC + SFPENCC)
            v_endif;
        }

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result; // Store result to output DEST tile
        sfpi::dst_reg++; // Advance DEST row pointer by SFP_DESTREG_STRIDE for next iteration
    }
}

template <bool APPROXIMATION_MODE /*unused*/, BinaryOp BINOP>
inline void _sfpu_binary_init_() {
    // For MUL: no-op. No LUT or special register initialization needed.
    // (DIV/POW would init reciprocal LUT, XLOGY would init log LUT)
    if constexpr (BINOP == BinaryOp::DIV || BINOP == BinaryOp::POW) {
        _init_sfpu_reciprocal_<false>();
    } else if constexpr (BINOP == BinaryOp::XLOGY) {
        _init_log_<APPROXIMATION_MODE>();
    }
}
```

### SFPU Instructions Used

| Instruction / Intrinsic | Description |
|--------------------------|-------------|
| `sfpi::dst_reg[index]` (load) | **SFPLOAD**: Loads a 32-bit float value from the DEST register file at the specified row offset into an SFPU vector register (vFloat). Operates on all SIMD lanes simultaneously. |
| `sfpi::dst_reg[index] = val` (store) | **SFPSTORE**: Stores a 32-bit float value from an SFPU vector register back to the DEST register file at the specified row offset. |
| `in0 * in1` (vFloat multiply) | **SFPMUL**: Performs element-wise floating-point multiplication across all SIMD lanes. The SFPU multiplier computes a full 32-bit float product. |
| `sfpi::dst_reg++` | **SFPINCRWC** (or SETRWC variant): Advances the internal DEST register row counter by `SFP_DESTREG_STRIDE` (2 rows), moving to the next pair of rows for the next SFPU iteration. |
| `sfpi::reinterpret<vUInt>(val)` | **SFPCAST** / bitwise reinterpret: Reinterprets float bits as unsigned integer without conversion. Used in the bf16 rounding logic. |
| `bits >> 16` | **SFPSHFT**: Right-shifts the integer representation by 16 bits to isolate the bf16 mantissa LSB. |
| `bits & mask` | **SFPAND**: Bitwise AND for masking operations (extracting LSB, clearing lower 16 bits). |
| `bits + constant` | **SFPIADD** or **SFPADDI**: Integer addition used in the RNE rounding bias calculation (`bits + 0x7fff + lsb`). |
| `v_if(in0 == 0 \|\| in1 == 0)` | **SFPSETCC** + **SFPENCC**: Sets condition codes by comparing vFloat values against zero. The `v_if` construct uses condition code registers to enable/disable lanes for subsequent operations. The `\|\|` is implemented as two separate comparisons combined via condition code logic. |
| `result = 0.0f` (conditional) | **SFPLOADI** (conditional): Loads an immediate constant into the vector register, but only for lanes where the condition code is active. |
| `TTI_SETRWC` | **SETRWC**: Adjusts the DEST register write counter between tile faces. Called by the params dispatch layer (not inside the SFPU kernel itself) to advance by 16 rows (two `+8` increments) between each face call. |
| `TTI_STALLWAIT` | **STALLWAIT**: Stalls the instruction pipeline until the SFPU has completed all pending operations. Used in `_start_` and `_done_` functions. |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **DEST registers (input0)** | Tile at index `dst_index_in0` (typically `i*2 = 0`). Contains the LHS operand loaded via `copy_tile`. Each tile occupies 32 sfpi-addressable rows (64 physical rows / stride 2). |
| **DEST registers (input1)** | Tile at index `dst_index_in1` (typically `i*2+1 = 1`). Contains the RHS operand loaded via `copy_tile`. Placed at the next adjacent tile slot in DEST. |
| **DEST registers (output)** | Tile at index `dst_index_out` (typically `i*2 = 0`, same as input0). The result overwrites the LHS operand in-place. |
| **SFPU vector registers (LREGs)** | `in0`, `in1`, `result` are allocated to SFPU local registers (LREGs 0-3). `float32_to_bf16_rne` uses additional LREGs for `bits` and `lsb`. The SFPU has 4 local vector registers available per thread. |
| **Condition code register** | Used by `v_if(in0 == 0 || in1 == 0)` to mask lanes where either input is zero. SFPSETCC sets the CC, and subsequent stores only affect enabled lanes. |
| **DEST row counter (RWC_D)** | Auto-incremented by `dst_reg++` each iteration (by SFP_DESTREG_STRIDE=2). Between faces, the params layer uses `TTI_SETRWC` to advance by 16 rows total (2 increments of 8). |

### Address Mode Configuration

The init function `_llk_math_eltwise_binary_sfpu_init_<SfpuType::unused>()` configures the address mode as follows:

**ADDR_MOD_7** (used for all standard binary SFPU ops including MUL):
- `srca.incr = 0` -- Source A not auto-incremented (SFPU reads directly from DEST, not via SrcA)
- `srcb.incr = 0` -- Source B not auto-incremented
- `dest.incr = 0` -- DEST not auto-incremented via addr_mod (the SFPU kernel manages DEST advancement explicitly via `dst_reg++` and `TTI_SETRWC`)

This is the same on both **Wormhole B0** and **Blackhole**. The ADDR_MOD_7 slot is chosen to avoid conflicts with ADDR_MOD_0 and ADDR_MOD_2, which are used by the A2D (unpack-to-DEST) copy_tile operations that precede the SFPU work.

**ADDR_MOD_6** is NOT configured for MUL. It is only set for `mul_int32`, `mul_uint16`, and max/min operations, which need `dest.incr = 2` for their different iteration patterns.

**Hardware generation differences** in the `_start_` and `_done_` functions:
- **Wormhole B0**: `_start_` calls `math::set_addr_mod_base()` (selects the addr_mod base register); `_done_` calls `TTI_STALLWAIT(STALL_CFG, WAIT_SFPU)` and `math::clear_addr_mod_base()`.
- **Blackhole**: `_start_` omits `set_addr_mod_base()` (Blackhole handles this differently); `_done_` only calls `math::clear_dst_reg_addr()` without the extra STALLWAIT or addr_mod_base cleanup.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "What is the SFPU kernel implementation for mul_binary_tile? Where is the ckernel_sfpu_mul.h file and how does the binary SFPU multiply work through the LLK layers?"
   **Reason**: Needed to locate the exact file paths and understand the full call chain from the compute API to the core SFPU function.
   **Key Findings**: Confirmed there is no `ckernel_sfpu_mul.h` -- floating-point binary multiply is in `ckernel_sfpu_binary.h`. The MUL path uses a dedicated `calculate_sfpu_binary_mul` function (separate from the generic `_calculate_sfpu_binary_`). Integer multiply uses a separate `ckernel_sfpu_mul_int32.h`.

2. **Query**: "How does the SFPU binary multiply (mul_binary_tile) work? What is the call chain from the compute API through llk_math_eltwise_binary_sfpu to ckernel_sfpu_mul?"
   **Reason**: Cross-referenced with tt-llk repo to get the canonical LLK source paths and understand the params dispatch layer.
   **Key Findings**: Confirmed the `_llk_math_eltwise_binary_sfpu_params_` function handles face iteration (4 faces for VectorMode::RC), calling the SFPU function once per face with 8 iterations = 32 rows per face = 128 rows total for a 32x32 tile. Found that `_sfpu_binary_init_` is a no-op for MUL.

### Confluence References

No Confluence references were needed for this analysis. The SFPU instructions used (`SFPLOAD`, `SFPSTORE`, `SFPMUL`, `SFPSHFT`, `SFPAND`, `SFPSETCC`) are standard SFPI intrinsics documented via the DeepWiki `tenstorrent/sfpi` and `tenstorrent/tt-isa-documentation` repos.

### Glean References

No Glean references were needed for this analysis.
