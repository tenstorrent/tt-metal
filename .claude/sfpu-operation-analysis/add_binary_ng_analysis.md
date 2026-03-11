# ADD (binary_ng) Implementation Analysis

## Overview

The ADD operation is implemented within the `binary_ng` (next-generation binary) framework, a unified program factory that handles all binary element-wise operations (ADD, SUB, MUL, DIV, and many more) through compile-time macro configuration rather than separate program factories per operation. For ADD specifically, the operation computes `c = a + b` element-wise, where `b` can be either a tensor or a scalar.

**Program factory path**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_program_factory.cpp`

The binary_ng framework supports two execution paths:
- **FPU path**: Uses the hardware matrix unit (FPU) for ADD/SUB/MUL on floating-point types. ADD maps to `EltwiseBinaryType::ELWADD` via the `add_tiles` LLK.
- **SFPU path**: Uses the vector unit (SFPU) for ADD on integer types or when explicitly requested. ADD maps to `add_binary_tile` (float) or `add_int_tile<DataFormat::Int32>` (int32).

## Work Unit Definition

One work unit is **one tile** (32x32 elements). The operation processes `num_tiles_per_cycle = 1` output tile per read-compute-write cycle. The total number of output tiles (`c_num_tiles`) is divided across cores, with each core processing its assigned tile range sequentially.

## Tensor Format and Layout

### Input Tensor A

| Property | Value |
|---|---|
| Dimension Convention | Up to rank 10+; internally decomposed as `[nD, D, N, C, Ht, Wt]` where dims beyond rank 5 are collapsed into `nD` |
| Tensor Layout | TILE (32x32) |
| Memory Layout | INTERLEAVED or SHARDED (HEIGHT, WIDTH, or BLOCK) |
| Buffer Type | DRAM or L1 |
| Data Type | BFLOAT16, FLOAT32, INT32, UINT32 |

### Input Tensor B (optional -- may be a scalar)

| Property | Value |
|---|---|
| Dimension Convention | Same as A, with broadcasting support |
| Tensor Layout | TILE (32x32) |
| Memory Layout | INTERLEAVED or SHARDED (HEIGHT, WIDTH, or BLOCK) |
| Buffer Type | DRAM or L1 |
| Data Type | BFLOAT16, FLOAT32, INT32, UINT32; when scalar, inherits A's dtype (SFPU) or BFLOAT16 (FPU) |

### Output Tensor C

| Property | Value |
|---|---|
| Dimension Convention | Broadcasted shape of A and B |
| Tensor Layout | TILE (32x32) |
| Memory Layout | INTERLEAVED or SHARDED |
| Buffer Type | DRAM or L1 |
| Data Type | Configurable; defaults to A's dtype |

### Layout Transformations

No tilize/untilize is performed within the operation. All tensors must already be in TILE layout. When input and output data types differ and the operation is not a quantization op, an automatic `TYPECAST` post-activation is appended to the compute kernel defines.

## Data Flow Pattern

### Tensor + Tensor Path (b is a tensor)

1. **Reader kernel** (runs on BRISC/NoC0): Reads tiles from both tensor A and tensor B into CB0 (`c_0`) and CB1 (`c_1`) respectively. For interleaved memory, tiles are read one at a time using `noc_async_read_page` via TensorAccessor. For sharded memory, all shard tiles are made available immediately via `cb_reserve_back`/`cb_push_back`.

2. **Compute kernel** (runs on MATH/PACK RISC-Vs):
   - **FPU path** (`eltwise_binary_no_bcast.cpp`): Waits for one tile each from CB0 and CB1, performs `add_tiles(cb_lhs, cb_rhs, 0, 0, 0)` using hardware FPU, then packs result into CB2 (`c_2`).
   - **SFPU path** (`eltwise_binary_sfpu_no_bcast.cpp`): Copies tiles from CB0 and CB1 into destination registers (even/odd slots), then calls `add_binary_tile(dst0, dst1, dst0)` on the SFPU, packs to CB2.

3. **Writer kernel** (runs on NCRISC/NoC1): Reads computed tiles from CB2 and writes them to the output tensor in DRAM/L1 using `noc_async_write_page`.

### Tensor + Scalar Path (b is a scalar)

1. **Writer kernel** (`writer_interleaved_scalar.cpp`): Fills a single tile in CB1 with the scalar value using `fill_with_val` (repeated across all 1024 elements). Also handles writing output tiles from CB2 to DRAM/L1.

2. **Reader kernel** (`reader_interleaved_no_bcast.cpp`): Reads only tensor A tiles into CB0.

3. **Compute kernel** (`eltwise_binary_scalar.cpp`): The scalar tile in CB1 is read once and reused for every input tile. The RHS CB is never popped until all tiles are processed (scalar is persistent).

### Broadcasting Variants

The reader kernel is selected based on `SubtileBroadcastType`:
- **NONE**: `reader_interleaved_no_bcast.cpp` -- reads both A and B tile-for-tile
- **ROW_A/ROW_B**: `reader_interleaved_row_bcast.cpp` -- one operand has H=1 in tiles
- **COL_A/COL_B**: `reader_interleaved_col_bcast.cpp` -- one operand has W=1 in tiles
- **SCALAR_A/SCALAR_B**: `reader_interleaved_scalar_bcast.cpp` -- one operand is 1x1 tile
- **ROW_B_COL_A / ROW_A_COL_B**: `reader_interleaved_row_col_mixed_bcast.cpp` -- mixed broadcast

## Circular Buffer Configuration

| CB ID | Purpose | Capacity (tiles) | Block Size (tiles) | Buffering | Producer | Consumer | Data Format |
|---|---|---|---|---|---|---|---|
| `c_0` | Input A | 2 (interleaved) or shard_volume (sharded) | 1 | Double-buffered (interleaved) / Single (sharded) | Reader | Compute | A's data format |
| `c_1` | Input B / Scalar | 2 (tensor interleaved), 1 (scalar), or shard_volume (sharded) | 1 | Double-buffered (tensor) / Single (scalar/sharded) | Reader (tensor) or Writer (scalar) | Compute | B's data format |
| `c_2` | Output C | 2 (interleaved) or shard_volume (sharded) | 1 | Double-buffered (interleaved) / Single (sharded) | Compute | Writer | C's data format |
| `c_3` | LHS intermediate | 1 | 1 | Single-buffered | Compute (preprocessing) | Compute | A's format (SFPU) or Float16_b (FPU with exp ops) |
| `c_4` | RHS intermediate | 1 | 1 | Single-buffered | Compute (preprocessing) | Compute | B's format (SFPU) or Float16_b (FPU with exp ops) |
| `c_5` | Row bcast buffer for A | 2 | 1 | Double-buffered | Reader | Compute | A's data format |
| `c_6` | Row bcast buffer for B | 2 | 1 | Double-buffered | Reader | Compute | B's data format |

**Notes**: CB3 and CB4 are only allocated when LHS or RHS pre-activations exist (e.g., `LOGADDEXP` applies `EXP` to both inputs). CB5/CB6 are only allocated for ROW_A/ROW_B or mixed row-col broadcast types. For plain ADD without activations, only CB0, CB1, CB2 are used.

## Pipeline Pattern Summary

- **Interleaved mode**: CB0, CB1, CB2 each have capacity=2 tiles with block_size=1, enabling **double-buffering**. The reader can fill the next tile while compute processes the current one.
- **Sharded mode**: CBs are sized to the full shard volume and function as **single-buffered** bulk transfers -- all tiles are available at once.
- **Scalar path**: CB1 has capacity=1 (single tile filled once and held persistent).

## Index Calculations

The operation uses a 6-level nested index decomposition to map a flat `start_tile_id` to multi-dimensional coordinates:

```
tiles_per_nd = D * N * C * Ht * Wt
tiles_per_d  = N * C * Ht * Wt
tiles_per_n  = C * Ht * Wt
HtWt         = Ht * Wt
```

From `start_tile_id`, the kernel computes:
- `start_nd = start_tile_id / tiles_per_nd` (collapsed high dims)
- `start_d`, `start_n`, `start_c` (batch dimensions)
- `start_th`, `start_tw` (tile row/col within the 2D plane)

For broadcasting, stride values encode whether a dimension is broadcast. A stride of 0 means the input dimension is 1 (broadcast), computed as `aHt * aWt * aC * aN * (aD > 1)`. The `(dim > 1)` expression produces 0 when the dimension is 1, zeroing out the stride and causing the same tiles to be re-read.

**TensorAccessor** is used for physical address translation: `TensorAccessorArgs` encodes buffer metadata as compile-time args, and `noc_async_read_page(page_id, accessor, l1_addr)` translates the logical page ID to the correct DRAM bank and address.

## Memory Access Patterns

### Read Pattern

- **Interleaved**: Sequential tile reads within each tile-row (`tw` loop), advancing through `th`, `c`, `n`, `d`, `nd` dimensions. Each tile read issues a `noc_async_read_page` followed by `noc_async_read_barrier` (synchronous per tile).
- **Sharded**: All shard tiles are already in L1 -- no NoC reads. The CB is immediately made available.
- **Broadcasting**: For broadcast dimensions, stride is 0, so the same tile pages are re-read from DRAM (or the same L1 shard position is reused).

### Write Pattern

- **Interleaved**: Sequential tile writes matching the output tensor's tile ordering. Each tile is written via `noc_async_write_page` with a barrier per tile.
- **Sharded**: Output CB is backed by the output buffer in L1 -- no explicit NoC writes. Results are written directly to the shard's memory region.

## Core Distribution Strategy

| Property | Value |
|---|---|
| Grid Topology | Determined by `worker_grid` (from device or user override) |
| Work Unit | Output tiles (`c_num_tiles = physical_volume / tile_hw`) |
| Splitting Method | `split_work_to_cores()` for interleaved; shard grid for sharded |
| Load Balancing | Two core groups: group 1 gets `ceil(tiles/cores)` tiles, group 2 gets `floor(tiles/cores)` tiles |
| Remainder Handling | Excess tiles go to core_group_1; remaining cores in core_group_2 get one fewer tile |
| No-op Cores | Cores outside both groups receive zero-filled runtime args and exit immediately |
| Sharded Mode | Core grid comes from the shard spec; each core processes its shard's tiles (variable per-core tile count for edge shards) |

**Optimization**: When the grid is a single rectangular range starting at (0,0) (`zero_start_grid`), a faster work distribution path is used with `grid_to_cores` instead of generic `corerange_to_cores`.

## Arguments

### Compile-Time Arguments

#### Reader Kernel
| Index | Name | Type | Description |
|---|---|---|---|
| 0..N | TensorAccessorArgs (A) | uint32_t[] | Buffer metadata for tensor A (bank count, page size, etc.) |
| N+1..M | TensorAccessorArgs (B) | uint32_t[] | Buffer metadata for tensor B (or A if scalar) |
| M+1 | has_sharding | uint32_t | 1 if any tensor uses native L1 sharding, 0 otherwise |

#### Writer Kernel
| Index | Name | Type | Description |
|---|---|---|---|
| 0..N | TensorAccessorArgs (C) | uint32_t[] | Buffer metadata for output tensor C |
| N+1 | has_sharding | uint32_t | 1 if native L1 sharding active |

#### Compute Kernel
| Index | Name | Type | Description |
|---|---|---|---|
| 0 | num_tiles_per_cycle | uint32_t | Always 1 -- tiles processed per iteration |

### Runtime Arguments

#### Reader Kernel (tensor+tensor path, 21 args)
| Index | Name | Type | Description |
|---|---|---|---|
| 0 | src_addr | uint32_t | Base address of tensor A buffer |
| 1 | start_tile_id (c_start_id) | uint32_t | First output tile ID this core processes |
| 2 | a_num_tiles | uint32_t | Number of A shard tiles (0 if interleaved) |
| 3 | c_num_tiles | uint32_t | Number of output tiles for this core |
| 4 | c_current_shard_width | uint32_t | Shard width in tiles (0 if interleaved) |
| 5 | nD_stride | uint32_t | A's stride for collapsed high dims (0 if broadcast) |
| 6 | d_stride | uint32_t | A's stride for D dimension |
| 7 | n_stride | uint32_t | A's stride for N dimension |
| 8 | c_stride | uint32_t | A's stride for C dimension |
| 9 | D (output) | uint32_t | Output D dimension |
| 10 | N (output) | uint32_t | Output N dimension |
| 11 | C (output) | uint32_t | Output C dimension |
| 12 | Ht (output) | uint32_t | Output height in tiles |
| 13 | Wt (output) | uint32_t | Output width in tiles |
| 14 | cND | uint32_t | Output collapsed dims > 5 |
| 15 | src_addr_b | uint32_t | Base address of tensor B buffer |
| 16 | nD_stride_b | uint32_t | B's stride for collapsed high dims |
| 17 | d_stride_b | uint32_t | B's stride for D dimension |
| 18 | n_stride_b | uint32_t | B's stride for N dimension |
| 19 | c_stride_b | uint32_t | B's stride for C dimension |
| 20 | b_num_tiles | uint32_t | Number of B shard tiles (0 if interleaved) |

#### Writer Kernel (tensor+tensor path, 11 args)
| Index | Name | Type | Description |
|---|---|---|---|
| 0 | dst_addr | uint32_t | Base address of output C buffer |
| 1 | c_start_id | uint32_t | First output tile ID for this core |
| 2 | c_num_tiles | uint32_t | Number of output tiles for this core |
| 3 | c_current_shard_width | uint32_t | Shard width in tiles (0 if interleaved) |
| 4 | cD | uint32_t | Output D dimension |
| 5 | cN | uint32_t | Output N dimension |
| 6 | cC | uint32_t | Output C dimension |
| 7 | cHt | uint32_t | Output height in tiles |
| 8 | cWt | uint32_t | Output width in tiles |
| 9 | cND | uint32_t | Output collapsed dims > 5 |
| 10 | (reserved) | uint32_t | Always 0 |

#### Writer Kernel (scalar path, 11 args)
| Index | Name | Type | Description |
|---|---|---|---|
| 0 | packed_scalar | uint32_t | Scalar value packed into uint32 (bf16 pair or f32 bits) |
| 1 | dst_addr | uint32_t | Base address of output C buffer |
| 2 | c_start_id | uint32_t | First output tile ID |
| 3 | c_num_tiles | uint32_t | Number of output tiles |
| 4 | c_current_shard_width | uint32_t | Shard width in tiles |
| 5-10 | D, N, C, Ht, Wt, cND | uint32_t | Output shape dimensions |

#### Compute Kernel (4 args)
| Index | Name | Type | Description |
|---|---|---|---|
| 0 | c_num_tiles | uint32_t | Total tiles to process on this core |
| 1 | freq | uint32_t | Broadcast frequency (1 for no-bcast, Wt for col-bcast, Ht*Wt for scalar-bcast) |
| 2 | counter | uint32_t | Starting position within the broadcast cycle |
| 3 | compute_scalar_value | uint32_t | Zero-point value for quantization ops (0 for ADD) |

## Kernel Implementations

### Reader Kernel (tensor+tensor, no broadcast)

- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/dataflow/reader_interleaved_no_bcast.cpp`
- **Key Logic**: Reads tiles from both A and B in lockstep using a 6-deep nested loop over `[nD, D, N, C, Ht, Wt]`. Uses separate stride calculations for A and B to support broadcasting at any dimension level. The two reads are issued to different CBs but share a single `noc_async_read_barrier`. For sharded inputs, the kernel simply marks the shard tiles as available and exits.

### Reader Kernel (scalar path)

- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/reader_interleaved_no_bcast.cpp`
- **Key Logic**: Reads only tensor A tiles into CB0. The B-side scalar is handled by the writer kernel.

### Writer Kernel (tensor+tensor)

- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/dataflow/writer_interleaved_no_bcast.cpp`
- **Key Logic**: Writes output tiles from CB2 to DRAM/L1 using the same 6-deep loop structure. For sharded outputs, the kernel is a no-op (CB2 is backed by the output shard buffer).

### Writer Kernel (scalar path)

- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/writer_interleaved_scalar.cpp`
- **Key Logic**: First fills a single tile in CB1 with the packed scalar value using `fill_with_val`. Then writes output tiles from CB2 to DRAM. This kernel serves double duty: it produces the scalar B input AND writes the output.

### Compute Kernel (FPU, no broadcast)

- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_no_bcast.cpp`
- **Key Logic**: For each tile: waits on CB0 (LHS) and CB1 (RHS), acquires destination registers, calls `BINARY_OP(cb_lhs, cb_rhs, 0, 0, 0)` which expands to `add_tiles(...)` for ADD, packs result to CB2. The `binary_tiles_init` is called once outside the loop when no activations are present (optimization to avoid repeated init overhead).

### Compute Kernel (SFPU, no broadcast)

- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_sfpu_no_bcast.cpp`
- **Key Logic**: For each tile: copies LHS to even destination register slots and RHS to odd slots using `copy_tile`, then calls `BINARY_SFPU_OP(even, odd, even)` which expands to `add_binary_tile(i*2, i*2+1, i*2)`. The SFPU operates on destination register data, not directly on CB data, requiring the explicit `copy_tile` steps. Uses `unpack_to_dest_mode = UnpackToDestFp32` for all non-POWER operations.

### Compute Kernel (FPU, scalar)

- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_scalar.cpp`
- **Key Logic**: The RHS scalar tile is waited on once before the loop and popped after all tiles are processed. Each iteration only waits on a new LHS tile, reusing the persistent RHS.

### Compute Kernel (SFPU, bcast)

- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_sfpu.cpp`
- **Key Logic**: Uses `freq` and `tile_start` runtime args to manage broadcast cycling. The broadcast operand is loaded once per `freq` iterations, while the non-broadcast operand is loaded fresh each iteration. Handles SCALAR, COL, and mixed ROW_COL broadcast patterns through the `BCAST_INPUT` compile-time define.

## Implementation Notes

1. **Unified framework**: The same program factory handles ADD, SUB, MUL, DIV, POWER, bitwise ops, comparisons, and ~30+ other binary operations. The operation-specific behavior is entirely controlled through compile-time `#define` macros (`BINARY_OP`, `BINARY_SFPU_OP`, `BINARY_SFPU_INIT`).

2. **FPU vs SFPU selection**: For ADD, the FPU path is preferred when both operands are floating-point (BFLOAT16). The FPU uses the dedicated matrix engine which can perform ADD more efficiently. SFPU is used for integer types or when explicitly requested.

3. **Broadcast stride trick**: Broadcasting is implemented through stride manipulation rather than tile duplication. When a dimension has size 1, its stride is set to 0 (via the `(dim > 1)` multiplication), causing the reader to re-read the same tiles without extra memory.

4. **LLK broadcast optimization**: For ROW and ROW_COL broadcast patterns with BFLOAT16 inputs/outputs, a specialized LLK broadcast path is used (`is_llk_bcast` returns true), enabling hardware-level broadcast support in the FPU.

5. **Activation fusion**: Pre-activations (applied to inputs before the binary op) and post-activations (applied to output) are fused into the same compute kernel. For ADD specifically, `BIAS_GELU` uses ADD as its core operation with a GELU post-activation, and `LOGADDEXP` applies EXP to both inputs before ADD with LOG post-activation.

6. **PACK_RELU optimization**: When the only post-activation is RELU, it is applied during the pack phase (hardware RELU in packer) rather than as a separate SFPU operation, saving cycles.

7. **Sharding constraints**: Native L1 sharding is only used when: (a) both tensors have the same shape, (b) tensors are in L1, (c) output is not unevenly sharded, and (d) grids are compatible. Otherwise, the operation falls back to treating sharded tensors as interleaved (using TensorAccessor for address translation).

8. **Program caching**: The `override_runtime_arguments` method enables efficient re-execution by updating only runtime args (buffer addresses, tile counts, shape dimensions) without recompiling kernels.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the binary_ng (next generation binary) operation work? What is its program factory structure, what kernel variants does it support, and how does it handle broadcasting between tensors of different shapes?"
   **Reason**: Needed architectural overview of the binary_ng framework before diving into source code.
   **Key Findings**: Confirmed single ProgramFactory design, 9 SubtileBroadcastType variants, FPU/SFPU dual paths, kernel naming convention with `_ng` suffix for next-gen kernels, and NumPy-like broadcasting rules.

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_utils.hpp` and `.cpp`
   **Reason**: Needed to understand kernel name mapping, OpConfig construction for ADD, and how defines are generated.
   **Key Information**: ADD maps to `FpuBinaryOp::ADD` (FPU) or `SfpuBinaryOp::ADD` (SFPU). FPU defines: `BINARY_OP = add_tiles`, `BINARY_OP_TYPE = EltwiseBinaryType::ELWADD`. SFPU defines: `BINARY_SFPU_INIT = add_binary_tile_init()`, `BINARY_SFPU_OP = add_binary_tile`. Integer ADD uses `add_int_tile<DataFormat::Int32>`.

2. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_device_operation.hpp`
   **Reason**: Needed to understand operation_attributes_t structure and SubtileBroadcastType enum.
   **Key Information**: 9 broadcast types, operation attributes include binary_op_type, activations, scalar, is_sfpu flag, worker_grid.

3. **Source**: CLAUDE.md (repository instructions)
   **Reason**: Architectural context for hardware (Tensix cores, FPU/SFPU, circular buffers, NoC).
   **Key Information**: 5 RISC-V CPUs per core, reader/compute/writer kernel model, CB synchronization APIs.

## SFPU Kernel Implementation

This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to for the ADD operation.

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_binary_sfpu.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{blackhole,wormhole_b0}/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_binop.h` |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_{blackhole,wormhole_b0}/common/inc/sfpu/ckernel_sfpu_binary.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{blackhole,wormhole_b0}/llk_lib/llk_math_eltwise_binary_sfpu_params.h` |

### Call Chain

1. The compute kernel calls `add_binary_tile(i*2, i*2+1, i*2)` (macro `BINARY_SFPU_OP`), defined in `eltwise_binary_sfpu.h`, which wraps the call in a `MATH(...)` macro to ensure it runs on the MATH RISC-V.

2. Inside `MATH`, it calls `llk_math_eltwise_binary_sfpu_binop<APPROX, ckernel::BinaryOp::ADD>(idst0, idst1, odst)` in `llk_math_eltwise_binary_sfpu_binop.h`.

3. This function calls `_llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::calculate_sfpu_binary<APPROXIMATE, BinaryOp::ADD, 8, false>, dst_index0, dst_index1, odst, VectorMode::RC)` in `llk_math_eltwise_binary_sfpu_params.h`, which handles the tile face iteration (4 faces in RC mode), stalling until the SFPU is ready, and advancing the destination register write counter between faces via `TTI_SETRWC`.

4. For each face, `calculate_sfpu_binary<APPROXIMATE, BinaryOp::ADD, 8, false>` is called (in `ckernel_sfpu_binary.h` -- the metal overlay), which directly delegates to `_calculate_sfpu_binary_<APPROXIMATION_MODE, BinaryOp::ADD, 8>` in the tt-llk submodule's `ckernel_sfpu_binary.h`.

5. `_calculate_sfpu_binary_` iterates 8 times (one per row of a 16x16 face, processing 64 elements per SFPU vector op), loading operands from destination registers, performing `result = in0 + in1`, and storing back.

### Annotated SFPU Kernel Source

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_binary.h
// (Wormhole version is identical)

template <bool APPROXIMATION_MODE, BinaryOp BINOP, int ITERATIONS = 8>
inline void _calculate_sfpu_binary_(const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out)
{ // For ADD: APPROXIMATION_MODE=true (unused), BINOP=BinaryOp::ADD, ITERATIONS=8
    static constexpr float nan = std::numeric_limits<float>::quiet_NaN();
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++)
    {
        // size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
        constexpr std::uint32_t dst_tile_size_sfpi = 32;
        sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi]; // SFPLOAD from DEST[in0_tile_offset + current_row]
        sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi]; // SFPLOAD from DEST[in1_tile_offset + current_row]
        sfpi::vFloat result = 0.0f; // SFPLOADI -- load immediate 0.0

        if constexpr (BINOP == BinaryOp::ADD)
        {
            result = in0 + in1; // SFPADD -- element-wise floating-point addition
        }
        // SUB, MUL, DIV, RSUB, POW, XLOGY branches omitted (dead code for ADD)

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result; // SFPSTORE to DEST[out_tile_offset + current_row]
        sfpi::dst_reg++; // INCRWC -- advance dest register write counter by SFP_DESTREG_STRIDE (2)
    }
}

template <bool APPROXIMATION_MODE /*unused*/, BinaryOp BINOP>
inline void _sfpu_binary_init_()
{
    // For ADD: no initialization needed (no reciprocal tables or log constants required)
    if constexpr (BINOP == BinaryOp::DIV || BINOP == BinaryOp::POW)
    {
        _init_sfpu_reciprocal_<false>();
    }
    else if constexpr (BINOP == BinaryOp::XLOGY)
    {
        _init_log_<APPROXIMATION_MODE>();
    }
    // ADD/SUB/MUL/RSUB: no-op init
}
```

The intermediate wrapper in the metal overlay that delegates to the above:

```cpp
// File: tt_metal/hw/ckernels/{blackhole,wormhole_b0}/metal/llk_api/llk_sfpu/ckernel_sfpu_binary.h

template <bool APPROXIMATION_MODE, BinaryOp BINOP, int ITERATIONS = 8, bool is_fp32_dest_acc_en = false>
inline void calculate_sfpu_binary(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    _calculate_sfpu_binary_<APPROXIMATION_MODE, BINOP, ITERATIONS>(dst_index_in0, dst_index_in1, dst_index_out);
}

template <bool APPROXIMATION_MODE /*unused*/, BinaryOp BINOP>
inline void sfpu_binary_init() {
    _sfpu_binary_init_<APPROXIMATION_MODE, BINOP>();
}
```

The parameters dispatch function that handles face iteration:

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_binary_sfpu_params.h
// (Wormhole version is identical)

template <bool APPROXIMATE, typename Callable, typename... Args>
inline void _llk_math_eltwise_binary_sfpu_params_(
    Callable&& sfpu_func,
    std::uint32_t dst_index_in0,
    std::uint32_t dst_index_in1,
    std::uint32_t dst_index_out,
    int vector_mode = static_cast<int>(VectorMode::RC),
    Args&&... args)
{
    _llk_math_eltwise_binary_sfpu_start_<DST_SYNC_MODE>(0); // set DEST write addr, stall until SFPU ready

    VectorMode mode = static_cast<VectorMode>(vector_mode);

    // For ADD with VectorMode::RC (default): process all 4 faces of the 32x32 tile
    if (mode == VectorMode::RC)
    {
#pragma GCC unroll 0
        for (int face = 0; face < 4; face++)
        {
            std::forward<Callable>(sfpu_func)(dst_index_in0, dst_index_in1, dst_index_out, std::forward<Args>(args)...);
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D); // advance DEST counter by 8 rows
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D); // advance DEST counter by 8 more rows (total 16 = one face)
        }
    }
    // R and C modes omitted (not used for standard ADD)

    _llk_math_eltwise_binary_sfpu_done_(); // clear DEST register address
}
```

### SFPU Instructions Used

| Instruction | SFPI Syntax | Description |
|-------------|-------------|-------------|
| **SFPLOAD** | `sfpi::dst_reg[offset]` (as rvalue) | Loads a 64-element vector from the DEST register file at the specified row offset into an SFPU local register (LREG). |
| **SFPLOADI** | `sfpi::vFloat result = 0.0f` | Loads an immediate floating-point constant into an SFPU local register. |
| **SFPADD** | `in0 + in1` | Performs element-wise floating-point addition of two SFPU local registers. This is the core operation for binary ADD. Compiles to the `SFPADD` hardware instruction (opcode 0x85). |
| **SFPSTORE** | `sfpi::dst_reg[offset] = result` (as lvalue) | Stores a 64-element vector from an SFPU local register back to the DEST register file at the specified row offset. |
| **INCRWC** | `sfpi::dst_reg++` | Increments the DEST register write counter by `SFP_DESTREG_STRIDE` (2), advancing to the next row pair for the next iteration. Compiles to `__builtin_rvtt_ttincrwc`. |
| **SETRWC** | `TTI_SETRWC(...)` | Directly sets the Read-Write Counter for the DEST register. Used between faces to advance the DEST pointer by 8 rows per call (two calls = 16 rows = one 16x16 face). |
| **STALLWAIT** | `TTI_STALLWAIT(...)` | Stalls the MATH RISC-V until the SFPU is ready (used in `_start_`) or until SFPU completes (used in `_done_` on Wormhole). |

### SFPU Register Usage

**DEST Register File**: The SFPU reads from and writes to the shared DEST register file. For binary ADD:
- **Input tile 0** (LHS) is at DEST offset `dst_index_in0 * 32` (e.g., slot 0 = rows 0..31). This tile was placed there by `copy_tile(cb_post_lhs, 0, 0)` in the compute kernel.
- **Input tile 1** (RHS) is at DEST offset `dst_index_in1 * 32` (e.g., slot 1 = rows 32..63). This tile was placed there by `copy_tile(cb_post_rhs, 0, 1)`.
- **Output** overwrites the LHS slot at DEST offset `dst_index_out * 32` (same as in0, i.e., slot 0 = rows 0..31), since `odst == idst0`.

**SFPU Local Registers (LREGs)**: The SFPU has a small set of vector local registers. Within each iteration of the inner loop:
- `in0`: LREG holding the loaded LHS row vector (64 elements).
- `in1`: LREG holding the loaded RHS row vector (64 elements).
- `result`: LREG holding the computed sum, then stored back to DEST.

**Write Counter (RWC)**: The DEST write counter is auto-incremented by `SFP_DESTREG_STRIDE` (2) on each `dst_reg++` call within the 8-iteration inner loop, advancing through 8 row-pairs (16 rows) of one face. Between faces, `TTI_SETRWC` advances the counter by 8+8=16 more rows to reach the next face.

### Address Mode Configuration

The address mode for binary SFPU operations is configured in `eltwise_binary_sfpu_configure_addrmod()`, called during `_llk_math_eltwise_binary_sfpu_init_<SfpuType::unused>()`.

For ADD (and all binary ops except mul_int32, mul_uint16, max, min, and their int/uint variants):

| Field | ADDR_MOD_7 |
|-------|------------|
| `srca.incr` | 0 |
| `srcb.incr` | 0 |
| `dest.incr` | 0 |

This means the hardware does not auto-increment any source or destination addresses between SFPU instructions. All address advancement is handled explicitly by `dst_reg++` (INCRWC) in the inner loop and `TTI_SETRWC` between faces.

**Wormhole vs Blackhole**: The `eltwise_binary_sfpu_configure_addrmod()` function is identical across both architectures for ADD. The only difference in the overall flow is that Wormhole's `_llk_math_eltwise_binary_sfpu_start_` calls `math::set_addr_mod_base()` and `_done_` calls `math::clear_addr_mod_base()` with an additional `TTI_STALLWAIT(STALL_CFG, WAIT_SFPU)` stall. Blackhole omits these addr_mod_base calls, relying on a simpler synchronization model.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the add_binary_tile SFPU operation work in the binary_ng framework? What is the call chain from add_binary_tile through the LLK layers down to the ckernel SFPU implementation?"
   **Reason**: Needed to identify the full call chain and file paths for the SFPU kernel implementation of binary ADD.
   **Key Findings**: Confirmed the call chain: `add_binary_tile` -> `llk_math_eltwise_binary_sfpu_binop<APPROX, BinaryOp::ADD>` -> `_llk_math_eltwise_binary_sfpu_params_` -> `calculate_sfpu_binary` -> `_calculate_sfpu_binary_`. Identified that the core implementation lives in the tt-llk submodule.

2. **Query**: "What is the implementation of add_binary_tile in the LLK? Trace the call chain and identify SFPU instructions used." (tenstorrent/tt-llk)
   **Reason**: Needed LLK-specific details about the binary SFPU dispatch mechanism.
   **Key Findings**: Confirmed `SFPADD` (opcode 0x85) is the hardware instruction, `ADDR_MOD_7` is used for most binary ops, and `ADDR_MOD_6` is conditionally set only for mul/max/min integer operations.

3. **Query**: "How does sfpi::dst_reg work? What does dst_reg[index] compile to in terms of SFPU instructions?" (tenstorrent/sfpi)
   **Reason**: Needed to understand the SFPI abstraction for DEST register access and what hardware instructions the C++ syntax compiles to.
   **Key Findings**: `dst_reg[index]` compiles to SFPLOAD/SFPSTORE, `dst_reg++` compiles to INCRWC (`__builtin_rvtt_ttincrwc`), and `vFloat + vFloat` compiles to `SFPADD` via `__builtin_rvtt_sfpadd`.

4. **Query**: "What does the SETRWC instruction do in the Tensix ISA?" (tenstorrent/tt-isa-documentation)
   **Reason**: Needed to understand the TTI_SETRWC calls used between faces in the params dispatch function.
   **Key Findings**: SETRWC directly sets the Read-Write Counters. The `CR_D` parameter uses carry register mode for destination, and the value 8 advances the DEST counter by 8 rows per call.

### Confluence References

Not consulted -- DeepWiki and source code provided sufficient detail for the ADD operation's straightforward SFPU usage.

### Glean References

Not consulted -- the ADD SFPU kernel is simple enough that public documentation and source code were sufficient.
