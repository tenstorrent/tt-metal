# MUL (binary_ng) Implementation Analysis

## Overview

The MUL operation in the `binary_ng` framework performs element-wise multiplication of two tensors: `c = a * b`. It supports both FPU and SFPU execution paths, tensor-tensor and tensor-scalar modes, N-dimensional broadcasting (scalar, row, column, mixed row-column), and interleaved or sharded memory layouts. The `binary_ng` ("next generation") framework is a unified program factory that handles all binary operations (ADD, SUB, MUL, DIV, etc.) through compile-time define substitution, selecting the appropriate LLK function at kernel compile time.

**Program factory path**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_program_factory.cpp`

For MUL specifically:
- **FPU path**: `binary_op = FpuBinaryOp::MUL`, which expands to `BINARY_OP = mul_tiles` and `BINARY_OP_TYPE = EltwiseBinaryType::ELWMUL`
- **SFPU path**: `binary_op = SfpuBinaryOp::MUL`, which expands to `BINARY_SFPU_INIT = mul_binary_tile_init();` and `BINARY_SFPU_OP = mul_binary_tile` (or `mul_int_tile<DataFormat::Int32>` for integer types)

## Work Unit Definition

One work unit is **one 32x32 tile**. The compute kernel processes exactly 1 output tile per read-compute-write cycle (`num_tiles_per_cycle = 1`). Each cycle reads one tile from input A and one tile from input B (or reuses a scalar/broadcast tile), performs the multiply, and produces one output tile.

## Tensor Format and Layout

### Input Tensor A

| Property | Value |
|---|---|
| **Dimension Convention** | Up to N-D; internally viewed as `[nD, D, N, C, Ht, Wt]` where dimensions > 5 are collapsed into `nD` |
| **Tensor Layout** | Tiled (32x32 tiles) |
| **Memory Layout** | Interleaved (DRAM or L1) or Sharded (Height, Width, or Block) |
| **Buffer Type** | DRAM or L1 |
| **Data Types** | BFLOAT16, FLOAT32, INT32, UINT32 |

### Input Tensor B

| Property | Value |
|---|---|
| **Dimension Convention** | Same as A, or scalar (no tensor, just a packed value) |
| **Tensor Layout** | Tiled (32x32 tiles), or N/A if scalar |
| **Memory Layout** | Interleaved or Sharded |
| **Buffer Type** | DRAM or L1, or N/A if scalar |
| **Data Types** | Same as A (BFLOAT16 by default for FPU scalar path; matches A for SFPU scalar path) |

### Output Tensor C

| Property | Value |
|---|---|
| **Dimension Convention** | Broadcast-expanded shape of A and B |
| **Tensor Layout** | Tiled (32x32 tiles) |
| **Memory Layout** | Interleaved or Sharded |
| **Buffer Type** | DRAM or L1 |
| **Data Types** | BFLOAT16, FLOAT32, INT32, UINT32 |

### Layout Transformations

No explicit tilize/untilize within the operation. All tensors must already be in tiled layout. If `a_dtype != c_dtype` (and not a quant or integer-division op), a `TYPECAST` post-activation is automatically appended to the compute kernel defines.

## Data Flow Pattern

### Tensor-Tensor Mode (b is a tensor)

1. **Reader kernel** (runs on BRISC/NoC0): Reads tiles of A into CB0 and tiles of B into CB1 from DRAM/L1 (or marks sharded data as available). For interleaved memory, reads one tile at a time using `noc_async_read_page` via TensorAccessor. For sharded memory, issues `cb_reserve_back` + `cb_push_back` to make the pre-existing L1 data visible to the compute kernel.

2. **Compute kernel** (runs on MATH/PACK RISC-V):
   - **FPU path**: Waits for 1 tile in CB0 (LHS) and 1 tile in CB1 (RHS), calls `mul_tiles(cb_lhs, cb_rhs, 0, 0, 0)` which uses the FPU matrix engine to perform element-wise multiplication, then packs the result to CB2.
   - **SFPU path**: Copies LHS tile to DST register slot 0, RHS tile to DST register slot 1, calls `mul_binary_tile(0, 1, 0)` which runs on the SFPU vector engine, then packs result from DST slot 0 to CB2.

3. **Writer kernel** (runs on NCRISC/NoC1): Waits for 1 tile in CB2, writes it to DRAM/L1 output buffer using `noc_async_write_page`. For sharded output, the data is already in L1 so no writes occur.

### Tensor-Scalar Mode (b is a scalar value)

1. **Reader kernel**: Only reads tiles of A into CB0 (same as above, but no B reads).

2. **Writer kernel**: Additionally fills CB1 with a single tile containing the scalar value (using `fill_with_val` or `fill_with_val_bfloat16`), then writes output tiles from CB2 to DRAM. The scalar tile is created once and persists in CB1 for the entire operation.

3. **Compute kernel**: Same as tensor-tensor but uses the "Scalar" variant that waits for RHS only once (before the loop) and pops it only after all tiles are processed.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity (tiles) | Data Format | Buffering | Producer | Consumer |
|---|---|---|---|---|---|---|---|
| `c_0` | `cb_src_a` / `cb_pre_lhs` | Input tensor A | 2 (interleaved) or shard volume (sharded) | Same as A dtype | Double-buffered (interleaved) | Reader | Compute |
| `c_1` | `cb_src_b` / `cb_pre_rhs` | Input tensor B or scalar | 2 (tensor interleaved), 1 (scalar), or shard volume (sharded) | Same as B dtype | Double-buffered (tensor) / Single (scalar) | Reader (tensor) or Writer (scalar) | Compute |
| `c_2` | `cb_out` | Output tensor C | 2 (interleaved) or shard volume (sharded) | Same as C dtype | Double-buffered (interleaved) | Compute | Writer |
| `c_3` | `cb_post_lhs` (optional) | LHS after pre-activation | 1 | Same as A (SFPU) or Float16_b (FPU+exp ops) | Single-buffered | Compute (PREPROCESS) | Compute (main op) |
| `c_4` | `cb_post_rhs` (optional) | RHS after pre-activation | 1 | Same as B (SFPU) or Float16_b (FPU+exp ops) | Single-buffered | Compute (PREPROCESS) | Compute (main op) |
| `c_5` | (optional) | Row broadcast buffer for A | 2 | Same as A dtype | Double-buffered | Reader | Compute |
| `c_6` | (optional) | Row broadcast buffer for B | 2 | Same as B dtype | Double-buffered | Reader | Compute |

**Note for MUL**: CBs c_3 and c_4 are only created if `lhs_activations` or `rhs_activations` are non-empty. For a plain `MUL` with no pre/post activations, only c_0, c_1, and c_2 are used. CBs c_5 and c_6 are only used for row-broadcast subtypes (ROW_A, ROW_B, ROW_A_COL_B, ROW_B_COL_A).

## Pipeline Pattern Summary

- **Interleaved mode**: CB0, CB1, CB2 each have capacity=2, enabling double-buffered overlap between reader and compute, and between compute and writer.
- **Sharded mode**: CB0, CB1, CB2 have capacity equal to the full shard volume. All data is pre-loaded in L1, so there is no reader-compute overlap (the entire shard is made available at once).
- **Scalar mode**: CB1 has capacity=1 (single scalar tile, loaded once and reused for all iterations).

## Index Calculations

The operation uses a 6-level nested loop structure to iterate over output tiles. For tensors with rank > 5, dimensions beyond 5 are collapsed into a single `nD` dimension.

### Output tile ID to multi-dimensional index decomposition:
```
tiles_per_nd = D * N * C * Ht * Wt
start_nd = start_tile_id / tiles_per_nd
start_d  = (start_tile_id % tiles_per_nd) / tiles_per_d
start_n  = ... / tiles_per_n
start_c  = ... / HtWt
start_th = ... / Wt
start_tw = ... % Wt
```

### Input tile offset calculation (for broadcasting):
The reader computes input tile offsets using stride values that encode broadcasting. Each stride is either the product of inner dimensions (if that dimension exists in the input) or 0 (if the dimension is broadcast). The key formula:
```
tile_offset = start_nd * nD_stride + start_d * d_stride + start_n * n_stride + start_c * c_stride + start_th * Wt
```

The stride for dimension X is computed as `product_of_inner_dims * (dim_X > 1)`, which evaluates to 0 when the dimension is 1 (broadcast) and the correct stride otherwise.

## Memory Access Patterns

### Read Pattern
- **Interleaved**: Sequential tile-by-tile reads using `noc_async_read_page` with `noc_async_read_barrier` after each tile. Tiles are read in row-major order within each 2D slice (th, tw), then across C, N, D, nD dimensions.
- **Sharded**: No actual reads. The reader issues `cb_reserve_back` + `cb_push_back` to signal that sharded data already in L1 is available to compute.

### Write Pattern
- **Interleaved**: Sequential tile-by-tile writes using `noc_async_write_page` with `noc_async_write_barrier` after each tile. Same iteration order as reads.
- **Sharded**: No actual writes. Output data is packed directly into the sharded output buffer in L1.

## Core Distribution Strategy

| Property | Value |
|---|---|
| **Grid Topology** | Rectangular grid from `operation_attributes.worker_grid` |
| **Work Splitting** | `split_work_to_cores` divides total output tiles across cores |
| **Load Balancing** | Two core groups: group 1 gets `ceil(total_tiles / num_cores)` tiles, group 2 gets `floor(total_tiles / num_cores)` tiles |
| **Remainder Handling** | Cores not in group 1 or group 2 receive zero-initialized runtime args (no-op) |
| **Iteration Order** | Row-major by default |

### Sharded Mode Distribution
When sharded, the core grid comes from the shard spec rather than work splitting. Each core processes exactly its shard. The `ShardShapeGenerator` computes per-core shard dimensions, handling edge cores that may have smaller shards.

### Zero-Start Grid Optimization
When the grid is a single rectangle starting at (0,0) and sharded tensors also start at (0,0), a fast path (`zero_start_grid = true`) is used that avoids the overhead of generic `CoreRangeSet` iteration.

## Arguments

### Compile-Time Arguments

#### Reader Kernel
| Index | Name | Type | Description |
|---|---|---|---|
| 0..N | TensorAccessor args (A) | uint32_t | Tensor accessor compile-time parameters for input A |
| N+1..M | TensorAccessor args (B) | uint32_t | Tensor accessor compile-time parameters for input B |
| M+1 | has_sharding | uint32_t | 1 if any tensor is sharded, 0 otherwise |

#### Writer Kernel
| Index | Name | Type | Description |
|---|---|---|---|
| 0..N | TensorAccessor args (C) | uint32_t | Tensor accessor compile-time parameters for output C |
| N+1 | has_sharding | uint32_t | 1 if any tensor is sharded, 0 otherwise |

#### Compute Kernel
| Index | Name | Type | Description |
|---|---|---|---|
| 0 | num_tiles_per_cycle | uint32_t | Always 1 -- tiles produced per read-compute-write cycle |

### Runtime Arguments

#### Reader Kernel (21 args)
| Index | Name | Type | Description |
|---|---|---|---|
| 0 | src_addr | uint32_t | DRAM/L1 address of input tensor A |
| 1 | start_tile_id | uint32_t | Starting output tile ID for this core (c_start_id) |
| 2 | src_num_tiles | uint32_t | Number of A tiles in shard (sharded) or 0 (interleaved) |
| 3 | dst_num_tiles | uint32_t | Number of output tiles this core processes (c_num_tiles) |
| 4 | dst_shard_width | uint32_t | Width of output shard in tiles (sharded) or 0 |
| 5 | nD_stride | uint32_t | Stride for collapsed dims > 5 in A (0 if broadcast) |
| 6 | d_stride | uint32_t | Stride for D dimension in A (0 if broadcast) |
| 7 | n_stride | uint32_t | Stride for N dimension in A (0 if broadcast) |
| 8 | c_stride | uint32_t | Stride for C dimension in A (0 if broadcast) |
| 9 | D | uint32_t | Output D dimension (from output shape) |
| 10 | N | uint32_t | Output N dimension |
| 11 | C | uint32_t | Output C dimension |
| 12 | Ht | uint32_t | Output height in tiles |
| 13 | Wt | uint32_t | Output width in tiles |
| 14 | cND | uint32_t | Collapsed dimensions > 5 for output |
| 15 | src_addr_b | uint32_t | DRAM/L1 address of input tensor B (0 if scalar) |
| 16 | nD_stride_b | uint32_t | Stride for collapsed dims > 5 in B |
| 17 | d_stride_b | uint32_t | Stride for D dimension in B |
| 18 | n_stride_b | uint32_t | Stride for N dimension in B |
| 19 | c_stride_b | uint32_t | Stride for C dimension in B |
| 20 | src_num_tiles_b | uint32_t | Number of B tiles in shard (sharded) or 0 |

#### Writer Kernel -- Tensor-Tensor Mode (11 args)
| Index | Name | Type | Description |
|---|---|---|---|
| 0 | dst_addr | uint32_t | DRAM/L1 address of output tensor C |
| 1 | start_tile_id | uint32_t | Starting output tile ID for this core |
| 2 | dst_num_tiles | uint32_t | Number of output tiles this core writes |
| 3 | dst_shard_width | uint32_t | Width of output shard in tiles |
| 4 | D | uint32_t | Output D dimension |
| 5 | N | uint32_t | Output N dimension |
| 6 | C | uint32_t | Output C dimension |
| 7 | Ht | uint32_t | Output height in tiles |
| 8 | Wt | uint32_t | Output width in tiles |
| 9 | cND | uint32_t | Collapsed dimensions > 5 |
| 10 | (reserved) | uint32_t | Always 0 |

#### Writer Kernel -- Scalar Mode (11 args)
| Index | Name | Type | Description |
|---|---|---|---|
| 0 | packed_scalar | uint32_t | Packed scalar value (bfloat16 pair or float bits) |
| 1 | dst_addr | uint32_t | DRAM/L1 address of output tensor C |
| 2 | start_tile_id | uint32_t | Starting output tile ID for this core |
| 3 | dst_num_tiles | uint32_t | Number of output tiles this core writes |
| 4 | dst_shard_width | uint32_t | Width of output shard in tiles |
| 5 | D | uint32_t | Output D dimension |
| 6 | N | uint32_t | Output N dimension |
| 7 | C | uint32_t | Output C dimension |
| 8 | Ht | uint32_t | Output height in tiles |
| 9 | Wt | uint32_t | Output width in tiles |
| 10 | cND | uint32_t | Collapsed dimensions > 5 |

#### Compute Kernel (4 args)
| Index | Name | Type | Description |
|---|---|---|---|
| 0 | num_tiles | uint32_t | Total number of tiles to process on this core |
| 1 | freq | uint32_t | Broadcast frequency (1 for NONE, Wt for COL, Ht*Wt for SCALAR) |
| 2 | counter | uint32_t | Starting broadcast counter |
| 3 | compute_scalar_value | uint32_t | 0 for MUL (used for quant/where ops only) |

## Kernel Implementations

### Reader Kernel -- Tensor-Tensor, No Broadcast

- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/dataflow/reader_interleaved_no_bcast.cpp`
- **Key Logic**: Reads A tiles into CB0 and B tiles into CB1 in lockstep. Uses a 6-level nested loop (nD, D, N, C, Ht, Wt) iterating over output tile coordinates. For each output tile, computes the corresponding input A and B tile offsets using stride-based broadcasting. When sharded, simply marks shard data as available via `cb_reserve_back`/`cb_push_back`.

### Reader Kernel -- Scalar Mode (A only)

- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/reader_interleaved_no_bcast.cpp`
- **Key Logic**: Same structure as tensor-tensor reader but only reads A tiles into CB0. B is handled by the scalar writer.

### Compute Kernel -- FPU No-Broadcast

- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_no_bcast.cpp`
- **Key Logic**: For MUL, the define `BINARY_OP` expands to `mul_tiles`. The loop processes one tile per iteration: waits for LHS and RHS, acquires dest registers, calls `mul_tiles(cb_lhs, cb_rhs, 0, 0, 0)`, applies optional post-activations, packs result to CB2. The FPU path does NOT copy tiles to DST first -- the `mul_tiles` LLK reads directly from source CBs.

### Compute Kernel -- SFPU No-Broadcast

- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_sfpu_no_bcast.cpp`
- **Key Logic**: For MUL, `BINARY_SFPU_INIT` = `mul_binary_tile_init()` and `BINARY_SFPU_OP` = `mul_binary_tile`. The SFPU path must explicitly copy both operand tiles into DST registers (`copy_tile(cb_lhs, i, i*2)` for LHS into even slots, `copy_tile(cb_rhs, i, i*2+1)` for RHS into odd slots), then calls `mul_binary_tile(i*2, i*2+1, i*2)` to multiply them in-place.

### Compute Kernel -- FPU Scalar

- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_scalar.cpp`
- **Key Logic**: Same as FPU no-broadcast but the RHS tile (scalar) is waited on once before the loop and popped after the loop ends. Each iteration only waits for a new LHS tile.

### Compute Kernel -- SFPU Scalar

- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_sfpu_scalar.cpp`
- **Key Logic**: Same pattern as SFPU no-broadcast but RHS (scalar tile) is waited once and popped after the loop.

### Writer Kernel -- Tensor-Tensor

- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/dataflow/writer_interleaved_no_bcast.cpp`
- **Key Logic**: Waits for output tiles in CB2 and writes them to DRAM/L1 using `noc_async_write_page`. Same 6-level nested loop structure as the reader. For sharded output, the `#if !DST_SHARDED` guard skips all write logic.

### Writer Kernel -- Scalar (also writes output)

- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/writer_interleaved_scalar.cpp`
- **Key Logic**: First fills one tile in CB1 with the packed scalar value using `fill_with_val_bfloat16` (or `fill_with_val<1024, float>` for float32). Then writes output tiles from CB2 to DRAM/L1 using the same nested loop pattern.

## Implementation Notes

1. **Unified Framework**: The `binary_ng` program factory handles ALL binary operations through a single code path. MUL is differentiated only by compile-time defines (`BINARY_OP`, `BINARY_OP_TYPE` for FPU; `BINARY_SFPU_INIT`, `BINARY_SFPU_OP` for SFPU).

2. **FPU vs SFPU Selection**: The `is_sfpu` flag determines the execution path. MUL supports both. FPU MUL uses the dedicated matrix engine (`EltwiseBinaryType::ELWMUL`), while SFPU MUL uses the vector engine (`mul_binary_tile`). Integer multiplication (INT32/UINT32) uses `mul_int_tile<DataFormat::Int32>` on the SFPU.

3. **Broadcast Stride Encoding**: Broadcasting is elegantly encoded in the stride values. For a dimension of size 1, the stride is set to 0 (via `dim * (dim > 1)`), causing the reader to re-read the same slice for that dimension regardless of the output index.

4. **Pre/Post Activations**: The framework supports chaining unary operations before (LHS/RHS activations) and after (POST activations) the binary operation. For plain MUL, these are empty. For compound ops like LOGICAL_AND, MUL is the core op with NEZ pre/post-activations.

5. **Scalar Packing**: For BFLOAT16, the scalar is packed as two bfloat16 values in a single uint32_t. For FLOAT32, it is bit-cast. For INT32/UINT32, it is cast directly.

6. **fp32 Dest Accumulation**: Enabled when output or both inputs are FLOAT32, INT32, or UINT32. This affects DST register precision.

7. **UnpackToDestMode**: For SFPU ops (except POWER), both source CBs use `UnpackToDestFp32` mode for maximum precision during the copy-to-DST step.

8. **Sharding Constraints**: Native L1 sharding requires: (a) both inputs have identical shapes and memory configs, (b) no uneven shards on output, (c) no DRAM buffers, and (d) all shard grids match. When these conditions are not met, the operation falls back to interleaved (tensor accessor) mode even if inputs are sharded.

9. **No-op Cores**: Cores outside the active core groups receive all-zero runtime args, causing them to skip computation entirely (the `continue` statement in the runtime args loop).

## External Knowledge Sources

### DeepWiki Queries
1. **Query**: "How does the binary_ng operation work? What is the architecture of binary_ng program factory, including its kernel types, circular buffer setup, and core distribution strategy?"
   **Reason**: Needed initial architectural context before reading source code.
   **Key Findings**: Confirmed the dynamic kernel selection based on SubtileBroadcastType, the three kernel types (reader/compute/writer), CB allocation strategy, and core distribution approach. Identified that the operation supports scalar, row, column, and mixed broadcast types.

### Documentation References
1. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_utils.cpp`
   **Reason**: Maps BinaryOpType::MUL to specific LLK functions and defines.
   **Key Information**: FPU MUL uses `mul_tiles` / `EltwiseBinaryType::ELWMUL`; SFPU MUL uses `mul_binary_tile_init()` / `mul_binary_tile` (or `mul_int_tile<DataFormat::*>` for integers).

2. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_utils.hpp`
   **Reason**: Understanding kernel name enums and OpConfig structure.
   **Key Information**: Lists all KernelName variants and the FpuBinaryOp/SfpuBinaryOp enums. MUL exists in both FPU and SFPU variants.

3. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_device_operation.hpp`
   **Reason**: Understanding SubtileBroadcastType enum and its semantics.
   **Key Information**: Nine broadcast types covering all combinations of scalar, row, column, and full tensors.

## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to for the MUL (binary_ng) operation.

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_binary_sfpu.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_binop.h` |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_binary.h` (overrides in tt-metal repo) and `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/common/inc/sfpu/ckernel_sfpu_binary.h` (upstream tt_llk) |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_binary_sfpu_params.h` |

### Call Chain

1. The compute kernel calls `mul_binary_tile(i*2, i*2+1, i*2)` (defined in `eltwise_binary_sfpu.h`).
2. `mul_binary_tile` wraps `MATH((llk_math_eltwise_binary_sfpu_binop_mul<APPROX, BinaryOp::MUL, DST_ACCUM_MODE>(idst0, idst1, odst)))`, dispatching to the LLK layer.
3. `llk_math_eltwise_binary_sfpu_binop_mul` calls `_llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>` with `ckernel::sfpu::calculate_sfpu_binary_mul<APPROX, BinaryOp::MUL, 8, is_fp32_dest_acc_en>` as the callable.
4. `_llk_math_eltwise_binary_sfpu_params_` sets the DST write address, stalls until SFPU is ready, then invokes the callable once per face (4 times for `VectorMode::RC`), advancing the DST read/write counter by 16 rows between faces via `TTI_SETRWC`.
5. The callable `calculate_sfpu_binary_mul` executes the core SFPU multiply loop: loads operands from DST, multiplies, optionally applies bf16 RNE rounding, and stores the result back to DST.

Note: MUL uses a **dedicated** `calculate_sfpu_binary_mul` function rather than the generic `_calculate_sfpu_binary_` used by ADD/SUB. This is because MUL requires special bf16 rounding (RNE) and zero-handling logic to match FPU behavior.

### Annotated SFPU Kernel Source

```cpp
// File: tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_binary.h

// Helper: Convert float32 to bfloat16 using IEEE 754 Round-to-Nearest-Even (RNE)
sfpi_inline sfpi::vFloat float32_to_bf16_rne(sfpi::vFloat in) {
    sfpi::vUInt bits = sfpi::reinterpret<sfpi::vUInt>(in);
    sfpi::vUInt lsb = (bits >> 16) & 1; // bit 16 of float32 = LSB of bf16 mantissa
    // 0x7fff + lsb implements RNE: ties round to even
    bits = bits + 0x7fffU + lsb;
    bits = bits & 0xFFFF0000U; // truncate lower 16 bits
    return sfpi::reinterpret<sfpi::vFloat>(bits);
}

template <bool APPROXIMATION_MODE, BinaryOp BINOP, int ITERATIONS, bool is_fp32_dest_acc_en>
inline void calculate_sfpu_binary_mul(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    // APPROXIMATION_MODE=true (from APPROX define), BINOP=BinaryOp::MUL, ITERATIONS=8, is_fp32_dest_acc_en=DST_ACCUM_MODE
    constexpr uint dst_tile_size_sfpi = 32; // 64 rows / SFP_DESTREG_STRIDE(2) = 32 addressable rows per tile face
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi]; // SFPLOAD from DST[in0_tile_offset]
        sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi]; // SFPLOAD from DST[in1_tile_offset]

        sfpi::vFloat result = in0 * in1; // SFPMUL: element-wise multiply across SIMD lanes

        if constexpr (!is_fp32_dest_acc_en) {
            result = float32_to_bf16_rne(result); // software bf16 RNE rounding to match FPU precision
            // Force 0 * x = 0 and x * 0 = 0 to match FPU behaviour (SFPU may produce -0 or NaN for 0*inf)
            v_if(in0 == 0 || in1 == 0) { result = 0.0f; } // SFPSETCC + conditional SFPSTORE
            v_endif;
        }

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result; // SFPSTORE to DST[out_tile_offset]
        sfpi::dst_reg++; // advance DST row pointer by SFP_DESTREG_STRIDE (implicit INCRWC)
    }
}
```

The initialization function for MUL is a no-op since MUL does not require any special SFPU configuration (no reciprocal LUT, no log constants):

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/common/inc/sfpu/ckernel_sfpu_binary.h

template <bool APPROXIMATION_MODE, BinaryOp BINOP>
inline void _sfpu_binary_init_() {
    // For MUL: BINOP == BinaryOp::MUL, so none of the constexpr branches are taken.
    // No initialization is needed -- this is effectively a no-op.
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
| `sfpi::dst_reg[offset]` (read) | **SFPLOAD**: Loads a vector of elements from DST register file at the given offset into an SFPU vector register (vFloat). Operates on all SIMD lanes simultaneously. |
| `sfpi::dst_reg[offset] = val` (write) | **SFPSTORE**: Stores a vector from an SFPU register back into the DST register file at the given offset. |
| `in0 * in1` (vFloat * vFloat) | **SFPMUL**: Performs element-wise floating-point multiplication across all SIMD lanes. The SFPU multiplier produces a full float32 result internally. |
| `sfpi::reinterpret<vUInt>(val)` | **SFPCAST** (or bitwise reinterpret): Reinterprets the bit pattern of a vFloat as vUInt without conversion. Used for the RNE rounding bit manipulation. |
| `bits >> 16`, `bits & mask`, `bits + val` | **SFPSHFT** / **SFPAND** / **SFPIADD**: Integer shift, bitwise AND, and integer addition on vector registers. These implement the software bf16 RNE rounding algorithm. |
| `v_if(cond) { ... } v_endif` | **SFPSETCC** + conditional execution: Sets the SFPU condition code based on the comparison, then conditionally executes the enclosed block only on lanes where the condition is true. |
| `sfpi::dst_reg++` | **INCRWC** (implicit): Increments the DST register row pointer by `SFP_DESTREG_STRIDE` (2), advancing to the next pair of rows within the current face. |
| `TTI_SETRWC` | **SETRWC**: Issued by the params dispatcher between face iterations to advance the DST counter by 8 rows (one quarter-face), repositioning the SFPU's read/write window to the next face. |
| `TTI_STALLWAIT(STALL_SFPU, MATH)` | **STALLWAIT**: Stalls execution until the SFPU is ready to accept new instructions. Issued at the start of the SFPU operation. |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **DST register file** | Contains both input tiles and the output tile. LHS tile is at slot `idst0` (e.g., 0), RHS tile at slot `idst1` (e.g., 1), output overwrites slot `odst` (e.g., 0, same as LHS). Each tile occupies 32 addressable rows in DST (64 physical rows / stride 2). |
| **SFPU vFloat registers (LREGs)** | `in0`, `in1`, `result` are mapped to SFPU local registers (LREGs 0-3). These hold the vector values loaded from DST during each iteration. The SFPU has 4 local registers available. |
| **SFPU vUInt registers** | `bits`, `lsb` in the `float32_to_bf16_rne` helper reuse the same LREGs reinterpreted as unsigned integers for the RNE bit manipulation. |
| **Condition code register** | Set by `v_if(in0 == 0 \|\| in1 == 0)` via SFPSETCC. Controls per-lane conditional execution for the zero-clamping logic. Cleared/restored by `v_endif`. |
| **DST write address register** | Set by `math::set_dst_write_addr` at the start to point to tile index 0. Incremented by `dst_reg++` (per iteration) and `TTI_SETRWC` (per face). |

### Address Mode Configuration

The init function `_llk_math_eltwise_binary_sfpu_init_` configures address modes via `eltwise_binary_sfpu_configure_addrmod<SfpuType::unused>()`.

For floating-point MUL (`SfpuType::unused`), only **ADDR_MOD_7** is configured:

| Field | Value | Description |
|-------|-------|-------------|
| `srca.incr` | 0 | No source A auto-increment (SFPU reads from DST, not SrcA) |
| `srcb.incr` | 0 | No source B auto-increment |
| `dest.incr` | 0 | No destination auto-increment (DST advancement is handled explicitly by `dst_reg++` and `TTI_SETRWC`) |

This configuration is **identical for Wormhole B0 and Blackhole**. The `ADDR_MOD_6` variant (with `dest.incr = 2`) is only configured for integer mul/min/max operations (`SfpuType::mul_int32`, `SfpuType::mul_uint16`, etc.), which are not used by the floating-point MUL path.

The reason ADDR_MOD_7 uses zero increments is that the SFPU binary kernel manages its own DST addressing through explicit `dst_reg++` calls within the inner loop and `TTI_SETRWC` calls between faces. This avoids conflicts with ADDR_MOD_0 and ADDR_MOD_2, which are used by the A2D (unpack-to-dest) copy_tile operations that precede the SFPU work.

**Wormhole vs Blackhole difference**: The `_llk_math_eltwise_binary_sfpu_start_` function on Wormhole includes an additional `math::set_addr_mod_base()` call, and `_llk_math_eltwise_binary_sfpu_done_` includes `TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::WAIT_SFPU)` followed by `math::clear_addr_mod_base()`. On Blackhole, these calls are absent -- the start function only does `set_dst_write_addr` + `STALLWAIT`, and done only does `clear_dst_reg_addr`. This reflects Blackhole's simplified SFPU synchronization model.

## External Knowledge Sources

### DeepWiki Queries
1. **Query**: "Where is the mul_binary_tile function defined in the compute API? What is the call chain from mul_binary_tile through LLK down to the ckernel SFPU implementation for binary multiplication?"
   **Reason**: Needed to trace the complete SFPU call chain from API to implementation.
   **Key Findings**: Confirmed that `mul_binary_tile` is in `eltwise_binary_sfpu.h`, routes through `llk_math_eltwise_binary_sfpu_binop_mul` in the LLK layer, and ultimately calls `calculate_sfpu_binary_mul` in `ckernel_sfpu_binary.h`. MUL has a dedicated function separate from the generic `_calculate_sfpu_binary_`.

2. **Query**: "How is mul_binary_tile implemented in the LLK layer? What is the SFPU kernel function for binary multiplication?"
   **Reason**: Needed details on the LLK dispatch mechanism, address mode configuration, and the params function that manages face iteration.
   **Key Findings**: Confirmed the `_llk_math_eltwise_binary_sfpu_params_` function handles VectorMode dispatch (R/C/RC), calling the SFPU function once per face with `TTI_SETRWC` between faces. The `eltwise_binary_sfpu_configure_addrmod` function sets ADDR_MOD_7 with zero increments for floating-point operations.

### Confluence References
No Confluence references were needed for this analysis. The SFPU instructions used (SFPLOAD, SFPSTORE, SFPMUL, SFPSETCC, SFPSHFT, SFPAND, SFPIADD) are standard SFPI intrinsics well-documented in DeepWiki.

### Glean References
No Glean references were needed for this analysis.
