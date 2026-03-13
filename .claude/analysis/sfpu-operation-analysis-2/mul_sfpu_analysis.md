# MUL (SFPU) Implementation Analysis

## Overview
The MUL (SFPU) operation performs element-wise multiplication of two tensors using the SFPU (Special Function Processing Unit) vector engine on Tenstorrent hardware. This path is selected when both input tensors have matching height and width dimensions, matching data types, and the data type is one of FLOAT32, INT32, UINT32, or UINT16. For non-matching types or mixed-precision inputs, a different (FPU-based) program factory is used instead.

**Program factory path**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/element_wise_multi_core_sfpu_pgm_factory.cpp`

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | tile (32x32 elements) |
| **Unit size** | 1 tile (interleaved) or `max_block_size` tiles (sharded) |
| **Total units** | `physical_volume / TILE_HW` tiles total, divided across cores |
| **Loop structure** | Outer loop over `per_core_block_cnt` blocks, inner loop over `per_core_block_size` tiles per block. For interleaved mode, block_size=1 so blocks=tiles. For sharded mode, block_size=`find_max_block_size(num_tiles_per_shard)`. |

## Tensor Format and Layout

### Input Tensors

| Property | Input Tensor A | Input Tensor B |
|----------|---------------|----------------|
| **Logical shape** | [N, C, H, W] | [N, C, H, W] (same H, W as A) |
| **Dimension convention** | NCHW | NCHW |
| **Tensor layout** | TILE_LAYOUT | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED or SHARDED | INTERLEAVED or SHARDED |
| **Buffer type** | DRAM or L1 | DRAM or L1 |
| **Data type** | FLOAT32, INT32, UINT32, or UINT16 | Same as A |

### Output Tensor

| Property | Output Tensor |
|----------|---------------|
| **Logical shape** | Same as input A |
| **Dimension convention** | NCHW |
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED or SHARDED |
| **Buffer type** | DRAM or L1 |
| **Data type** | Matches output configuration (typically same as inputs) |

### Layout Transformations
No tilize/untilize or reshard operations are performed within this program factory. Both inputs and the output must already be in TILE_LAYOUT.

## Data Flow Pattern

### Interleaved Path (default)

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader | DRAM/L1 (src0_buffer) | CB c_0 | `cb_reserve_back(c_0, 1)`, `noc_async_read_tile`, `cb_push_back(c_0, 1)` |
| 1 | Reader | DRAM/L1 (src1_buffer) | CB c_1 | `cb_reserve_back(c_1, 1)`, `noc_async_read_tile`, `cb_push_back(c_1, 1)` |
| 2 | Compute | CB c_0, CB c_1 | CB c_2 | `cb_wait_front(c_0/c_1)`, `copy_tile` to DST, SFPU `mul_binary_tile`, `pack_tile` to c_2, `cb_pop_front`, `cb_push_back(c_2)` |
| 3 | Writer | CB c_2 | DRAM/L1 (dst_buffer) | `cb_wait_front(c_2, 1)`, `noc_async_write_page`, `cb_pop_front(c_2, 1)` |

### Sharded Path
When inputs are sharded, the reader simply calls `cb_reserve_back` + `cb_push_back` for the full shard size (the CB is backed by the tensor buffer directly via `set_globally_allocated_address`). Similarly, when the output is sharded, the writer just calls `cb_wait_front` on the full shard. Data does not traverse the NoC for sharded tensors.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity (interleaved) | Capacity (sharded) | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------------------|-------------------|------------|-----------|----------|----------|----------|
| c_0 | cb_src0 | Input A staging | 2 tiles | num_tiles_per_shard | 1 tile | Double (interleaved) / Single (sharded) | Reader | Compute | Program |
| c_1 | cb_src1 | Input B staging | 2 tiles | num_tiles_per_shard | 1 tile | Double (interleaved) / Single (sharded) | Reader | Compute | Program |
| c_2 | cb_output | Output staging | 2 tiles | num_tiles_per_shard | 1 tile | Double (interleaved) / Single (sharded) | Compute | Writer | Program |
| c_3 | cb_interim0 | Pre-processing scratchpad for input A | max_block_size tiles | max_block_size tiles | max_block_size | Single | Compute | Compute | Block |
| c_4 | cb_interim1 | Pre-processing scratchpad for input B | max_block_size tiles | max_block_size tiles | max_block_size | Single | Compute | Compute | Block |

**Note**: CB c_3 and c_4 are only created when `SFPU_OP_INIT_PRE_IN0_0` or `SFPU_OP_INIT_PRE_IN1_0` defines are present. For a standard MUL with FLOAT32 inputs, these are NOT created (no pre-processing needed). They would be created for operations like DIV (which pre-applies RECIP to input B) or LOGICAL_AND (which pre-applies NEZ).

## Pipeline Pattern Summary

**Interleaved mode**: CB c_0, c_1, and c_2 each have capacity = 2 tiles with block size = 1 tile, enabling **double-buffering**. The reader can write the next tile while compute processes the current one, and compute can write the next output tile while the writer drains the current one.

**Sharded mode**: CBs are backed by the tensor buffer directly (globally allocated address). The entire shard is available at once. No pipelining is needed since data is already in L1.

## Index Calculations

For **interleaved** tensors, `TensorAccessor` handles the mapping from a linear tile ID to a physical DRAM/L1 bank address. The reader iterates tile IDs from `start_id` to `start_id + num_tiles`, and `noc_async_read_tile(tile_id, accessor, l1_addr)` resolves the bank and offset.

For **block/width sharded** tensors with interleaved output, the reader uses a 2D loop:
- `row_start_tile_id = start_id`
- Inner loop: `tile_id = row_start_tile_id` to `row_start_tile_id + block_width`
- Row advancement: `row_start_tile_id += num_cores_y * block_width`

The `start_id` for sharded layouts is computed as:
```
start_id = (core_index / num_shards_per_width) * (block_height * block_width * num_shards_per_width)
         + (core_index % num_shards_per_width) * block_width
```

## Memory Access Patterns

### Read Pattern
- **Interleaved**: Sequential tile-by-tile reads from DRAM. One tile of A and one tile of B are read per iteration, with a `noc_async_read_barrier()` after each pair.
- **Block/width sharded with interleaved input**: Strided reads. Tiles are read in row-major order within a block, then stride by `num_cores_y * block_width` tiles to reach the next row of the shard.
- **Sharded input**: No NoC reads. Data is already in L1 via globally allocated CB.

### Write Pattern
- **Interleaved output**: Sequential tile-by-tile writes. Each tile is written with `noc_async_write_page`, flushed, then the CB slot is freed.
- **Sharded output**: No NoC writes. Output CB is backed by the output tensor buffer in L1.
- **Block/width sharded to interleaved**: Uses a specialized writer (`writer_unary_sharded_blocks_interleaved_start_id.cpp`) that handles block-to-interleaved layout conversion with unpadded dimensions.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 2D (from `operation_attributes.worker_grid`) |
| **Grid dimensions** | Device-dependent, from `worker_grid` |
| **Total cores** | Grid area (e.g., 8x8 = 64) |
| **Work per core** | `num_tiles / num_cores` (interleaved) or `num_tiles_per_shard` (sharded) |
| **Load balancing** | Two-group split: group 1 gets `ceil(num_tiles/num_cores)` tiles, group 2 gets `floor(num_tiles/num_cores)` tiles. Sharded mode: equal per-core work. |

The runtime args helper `set_eltwise_binary_runtime_args` uses `split_work_to_cores` (from `work_split.hpp`) for interleaved mode, which creates two core groups to handle remainder tiles. For sharded mode, the shard grid directly defines the core assignment with equal tiles per core.

An optimization detects "zero-start grids" (single rectangular grid starting at (0,0)) to use faster work distribution algorithms.

## Arguments

### Compile-Time Arguments

#### Reader Kernel
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | block_or_width_sharded | uint32_t | 1 if any tensor uses block or width sharding, 0 otherwise |
| 1+ | src0_args (TensorAccessor) | varies | TensorAccessor parameters for input A (only if A not sharded) |
| N+ | src1_args (TensorAccessor) | varies | TensorAccessor parameters for input B (only if B not sharded) |

**Reader Defines**: `IN0_SHARDED=1` if A is sharded, `IN1_SHARDED=1` if B is sharded.

#### Writer Kernel
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | output_cb_index | uint32_t | CB index for output (always c_2) |
| 1+ | dst_args (TensorAccessor) | varies | TensorAccessor parameters for output buffer |

**Writer Defines**: `OUT_SHARDED=1` if output is sharded.

#### Compute Kernel
Compile-time arguments are delivered via `#define` macros:
| Define | Value for MUL (float) | Description |
|--------|----------------------|-------------|
| `BINOP_INIT` | `mul_binary_tile_init();` | Initializes SFPU multiply |
| `BINARY_SFPU_OP` | `mul_binary_tile(0, 1, 0);` | Executes element-wise multiply: DST[0] = DST[0] * DST[1] |

For integer types (INT32, UINT32, UINT16), `MUL_INT_INIT` is used instead of `BINOP_INIT`, and `BINARY_SFPU_OP` calls `mul_int_tile<DataFormat::...>`.

### Runtime Arguments

#### Reader Kernel
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src0_addr | uint32_t | Base address of input tensor A buffer |
| 1 | src1_addr | uint32_t | Base address of input tensor B buffer |
| 2 | num_tiles | uint32_t | Total tiles for this core to process |
| 3 | start_id | uint32_t | Starting tile ID for this core |
| 4 | block_height | uint32_t | Shard block height in tiles (0 if interleaved) |
| 5 | block_width | uint32_t | Shard block width in tiles (0 if interleaved) |
| 6 | num_cores_y | uint32_t | Number of shards per width dimension (0 if interleaved) |

#### Compute Kernel
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_cnt | uint32_t | Number of blocks to process on this core |
| 1 | per_core_block_size | uint32_t | Tiles per block |

#### Writer Kernel (interleaved output, standard path)
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Base address of output buffer |
| 1 | num_pages | uint32_t | Number of tiles to write |
| 2 | start_id | uint32_t | Starting tile ID for writes |

#### Writer Kernel (block/width sharded to interleaved)
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Base address of output buffer |
| 1 | block_height | uint32_t | Block height in tiles |
| 2 | block_width | uint32_t | Block width in tiles |
| 3 | unpadded_block_height | uint32_t | Actual (unpadded) block height for edge cores |
| 4 | unpadded_block_width | uint32_t | Actual (unpadded) block width for edge cores |
| 5 | output_width | uint32_t | Output tensor width in tiles |
| 6 | block_size | uint32_t | Total tiles per block (height * width) |
| 7 | start_id | uint32_t | Starting tile ID for this core's output |
| 8 | (unused) | uint32_t | Always 0 |

## Kernel Implementations

### Reader Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader | RISCV_0 | NOC0 | DRAM/L1 (src0, src1) | CB c_0, CB c_1 | Read input tiles via TensorAccessor |

- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/dataflow/reader_binary_interleaved_start_id.cpp`
- **Key Logic**: Handles both interleaved and sharded inputs via conditional compilation (`IN0_SHARDED`, `IN1_SHARDED`). For sharded inputs, simply reserves and pushes the full shard. For interleaved inputs, reads one tile at a time with `noc_async_read_tile`. Supports a block/width-sharded 2D traversal pattern when `block_or_width_sharded` is set.

### Compute Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| compute | RISCV_2 (MATH) | N/A | CB c_0, CB c_1 | CB c_2 | SFPU mul_binary_tile |

- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/eltwise_binary_sfpu_kernel.cpp`
- **Key Logic**: Per block iteration: waits for `per_core_block_size` tiles on both input CBs, acquires tile registers, copies input A tiles to even DST slots (`i*2`) and input B tiles to odd DST slots (`i*2+1`), then for each tile pair calls `BINOP_INIT` (= `mul_binary_tile_init()`) and `BINARY_SFPU_OP` (= `mul_binary_tile(0, 1, 0)`), packs result from DST[i*2] to output CB. The kernel supports optional pre-processing stages via `SFPU_OP_INIT_PRE_IN0_0` / `SFPU_OP_INIT_PRE_IN1_0` (not used for plain MUL). Also supports fused post-activations via `SFPU_OP_CHAIN_0` and `PACK_RELU`.

**FP32 accumulation**: Enabled when output dtype is FLOAT32, INT32, or UINT32. For MUL (non-POWER), `UnpackToDestMode::UnpackToDestFp32` is set on all input CBs.

### Writer Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| writer | RISCV_1 | NOC1 | CB c_2 | DRAM/L1 (dst_buffer) | Write output tiles via TensorAccessor |

- **File (standard)**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
- **File (block-to-interleaved)**: `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded_blocks_interleaved_start_id.cpp`
- **Key Logic**: Standard path writes one tile at a time with `noc_async_write_page` and flushes after each tile. For sharded output, simply calls `cb_wait_front` to wait for all tiles (output buffer is globally allocated). The block-to-interleaved variant handles the case where compute produces sharded blocks but output must be written to interleaved memory.

## Implementation Notes

1. **SFPU vs FPU path selection**: The SFPU path is specifically selected for `BinaryOpType::MUL` when both inputs share the same dtype and that dtype is FLOAT32, INT32, UINT32, or UINT16. For BFLOAT16 inputs, the non-SFPU `ElementWiseMultiCore` factory is used instead, which uses the FPU matrix engine for the multiplication.

2. **Integer multiplication variants**: For INT32/UINT32/UINT16, the operation uses `mul_int_tile<DataFormat::...>` with `MUL_INT_INIT` instead of the floating-point `mul_binary_tile` with `BINOP_INIT`. These are separate SFPU implementations optimized for integer arithmetic.

3. **DST register layout**: The compute kernel interleaves inputs in DST registers: input A occupies even slots (0, 2, 4, ...) and input B occupies odd slots (1, 3, 5, ...). The SFPU operation reads from DST[i*2] and DST[i*2+1] and writes the result to DST[i*2], which is then packed to the output CB.

4. **No broadcasting**: This program factory requires both inputs to have the same height and width. Broadcasting is handled by separate program factories (`BroadcastHeightMultiCore`, `BroadcastWidthMultiCore`, `BroadcastHeightAndWidthMultiCore`).

5. **Program caching**: The `override_runtime_arguments` method allows efficient re-execution with different tensor addresses without re-creating the program, since all structural decisions (kernel selection, CB sizing, core grid) are fixed at program creation time.

6. **Block size optimization**: `find_max_block_size` finds the largest divisor of `num_tiles_per_shard` (capped at a hardware limit), allowing the compute kernel to process multiple tiles per iteration to reduce loop overhead.

## External Knowledge Sources

### DeepWiki Queries
1. **Query**: "How does the binary SFPU element-wise program factory work? What kernels does it use (reader, compute, writer)? How does it handle interleaved vs sharded tensors and broadcasting?"
   **Reason**: Initial architectural understanding of the binary SFPU program factory and its kernel structure.
   **Key Findings**: Confirmed three kernels (reader, compute, writer), sharded vs interleaved handling via defines and globally allocated CBs, and that this factory is specifically for non-broadcast element-wise operations. Broadcasting uses separate factories.

### Documentation References
1. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/binary_device_operation.cpp` (lines 21-33)
   **Reason**: Needed to confirm when the SFPU path is selected for MUL.
   **Key Information**: `is_binary_sfpu_op` returns true for MUL when both dtypes match and are FLOAT32/INT32/UINT32/UINT16. Same-height-same-width check in `select_program_factory` gates entry to `ElementWiseMultiCoreSfpu`.

2. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary/common/binary_op_utils.cpp` (lines 228-241, 535)
   **Reason**: Needed to determine which SFPU defines are generated for MUL.
   **Key Information**: For float MUL: `BINOP_INIT` = `mul_binary_tile_init()`, `BINARY_SFPU_OP` = `mul_binary_tile(0, 1, 0)`. For integer MUL: `MUL_INT_INIT` = `mul_int_tile_init<DataFormat::...>()`, same BINARY_SFPU_OP pattern.

3. **Source**: `tt_metal/hw/inc/api/compute/eltwise_binary_sfpu.h` (lines 43-44, 70)
   **Reason**: Needed to trace the actual SFPU function invoked.
   **Key Information**: `mul_binary_tile` calls `llk_math_eltwise_binary_sfpu_binop_mul<APPROX, BinaryOp::MUL, DST_ACCUM_MODE>`, a specialized MUL variant (distinct from ADD/SUB which use the generic `binop` path).

## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_binary_sfpu.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_binop.h` |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/ckernel_sfpu_binary.h` (arch-local wrapper) and `tt_metal/third_party/tt_llk/tt_llk_{arch}/common/inc/sfpu/ckernel_sfpu_binary.h` (shared core logic) |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{arch}/llk_lib/llk_math_eltwise_binary_sfpu_params.h` |

### Call Chain

1. The compute kernel calls `mul_binary_tile_init()` which invokes `llk_math_eltwise_binary_sfpu_binop_init<APPROX, BinaryOp::MUL>()`, which in turn calls `_llk_math_eltwise_binary_sfpu_init_<SfpuType::unused>()` (configures SFPU config reg, address modes, and resets counters) followed by `sfpu_binary_init<APPROX, BinaryOp::MUL>()` which calls `_sfpu_binary_init_<APPROX, BinaryOp::MUL>()` -- for MUL this is a no-op since MUL needs no special constant initialization (unlike DIV/POW which init reciprocal constants).

2. The compute kernel calls `mul_binary_tile(0, 1, 0)` which invokes `llk_math_eltwise_binary_sfpu_binop_mul<APPROX, BinaryOp::MUL, DST_ACCUM_MODE>(0, 1, 0)`.

3. `llk_math_eltwise_binary_sfpu_binop_mul` calls `_llk_math_eltwise_binary_sfpu_params_<APPROX>()` with the function pointer `calculate_sfpu_binary_mul<APPROX, BinaryOp::MUL, 8, is_fp32_dest_acc_en>` and dst indices (0, 1, 0) in RC vector mode.

4. `_llk_math_eltwise_binary_sfpu_params_` asserts tile indices are valid, calls `_llk_math_eltwise_binary_sfpu_start_` (sets DST write address, stalls until SFPU is ready), then iterates over all 4 faces of the tile (RC mode), calling `calculate_sfpu_binary_mul` once per face, with `TTI_SETRWC` advancing the DST row counter by 16 rows (2 x 8) between faces.

5. `calculate_sfpu_binary_mul` is the core SFPU function that executes 8 iterations (one per row within the 16x16 face half), loading from two DST tile locations, multiplying, optionally rounding to BF16, and storing back. Note: the SFPU processes 2 rows per iteration (SFPU lanes = 32 elements = 2 rows of 16), so 8 iterations cover the 16 rows of one face.

6. After all 4 faces, `_llk_math_eltwise_binary_sfpu_done_` clears the DST register address.

### Annotated SFPU Kernel Source

The MUL operation uses a **dedicated** `calculate_sfpu_binary_mul` function rather than the generic `_calculate_sfpu_binary_` template. This is because MUL needs special handling for BF16 rounding and zero-input edge cases when not in FP32 accumulation mode.

The arch-local file (`tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/ckernel_sfpu_binary.h`) contains the full implementation. Both Blackhole and Wormhole B0 versions are identical for `calculate_sfpu_binary_mul`. The tt_llk shared file (`tt_metal/third_party/tt_llk/tt_llk_{arch}/common/inc/sfpu/ckernel_sfpu_binary.h`) contains the `_sfpu_binary_init_` function.

```cpp
// File: tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/ckernel_sfpu_binary.h

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
    // APPROXIMATION_MODE=true, BINOP=BinaryOp::MUL, ITERATIONS=8, is_fp32_dest_acc_en depends on dtype
    constexpr uint dst_tile_size_sfpi = 32; // 64/SFP_DESTREG_STRIDE = 32 rows per tile in SFPI addressing
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi]; // SFPLOAD from DST[0*32] = input A
        sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi]; // SFPLOAD from DST[1*32] = input B

        sfpi::vFloat result = in0 * in1; // SFPMUL: FP32 multiply (alias for SFPMAD with addend=0)

        if constexpr (!is_fp32_dest_acc_en) {
            // When not in FP32 dest mode, round result to BF16 via software RNE
            result = float32_to_bf16_rne(result);

            // To match FPU behaviour: 0 * x = 0 and x * 0 = 0 (SFPU multiply may produce NaN for 0*inf)
            v_if(in0 == 0 || in1 == 0) { result = 0.0f; } // SFPSETCC + SFPENCC + conditional SFPMOV
            v_endif;
        }

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result; // SFPSTORE to DST[0*32]
        sfpi::dst_reg++; // Advance SFPU row pointer by SFP_DESTREG_STRIDE (=2 rows)
    }
}
```

The initialization function (from the tt_llk shared file):

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_{arch}/common/inc/sfpu/ckernel_sfpu_binary.h

template <bool APPROXIMATION_MODE, BinaryOp BINOP>
inline void _sfpu_binary_init_() {
    // For MUL: BINOP == BinaryOp::MUL, so none of the constexpr branches are taken.
    // No special initialization is needed for floating-point multiply.
    if constexpr (BINOP == BinaryOp::DIV || BINOP == BinaryOp::POW) {
        _init_sfpu_reciprocal_<false>();
    } else if constexpr (BINOP == BinaryOp::XLOGY) {
        _init_log_<APPROXIMATION_MODE>();
    }
    // MUL falls through with no action.
}
```

The parameters dispatch function (face iteration and DST advancement):

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_{arch}/llk_lib/llk_math_eltwise_binary_sfpu_params.h

template <bool APPROXIMATE, typename Callable, typename... Args>
inline void _llk_math_eltwise_binary_sfpu_params_(
    Callable&& sfpu_func,
    std::uint32_t dst_index_in0,
    std::uint32_t dst_index_in1,
    std::uint32_t dst_index_out,
    int vector_mode = static_cast<int>(VectorMode::RC),
    Args&&... args)
{
    _llk_math_eltwise_binary_sfpu_start_<DST_SYNC_MODE>(0); // Sets DST write addr, stalls until SFPU ready

    // RC mode: iterate over all 4 faces of the 32x32 tile
    // Each face is 16x16 = 256 elements, processed in 8 SFPU iterations of 32 lanes
    for (int face = 0; face < 4; face++) {
        sfpu_func(dst_index_in0, dst_index_in1, dst_index_out, std::forward<Args>(args)...);
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D); // Advance DST counter by 8
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D); // Advance DST counter by 8 more (total +16 rows)
    }
    _llk_math_eltwise_binary_sfpu_done_(); // Clears DST register address
}
```

### SFPU Instructions Used

| Instruction | Description |
|-------------|-------------|
| **SFPLOAD** | Loads a value from the Destination Register File into an SFPU local register (LREG). Used by `sfpi::dst_reg[index]` reads. Format conversion (e.g., FP16_B to FP32) happens inline. Opcode 0x70, IPC=1, latency=1. |
| **SFPMUL** (alias of SFPMAD) | Performs FP32 multiply: `result = A * B + 0.0`. SFPMUL is an alias of SFPMAD where the addend register (RG[VC]) is 0. Used by the `in0 * in1` expression. Opcode 0x86, IPC=1, latency=2. Flushes subnormals, sets exception flags. |
| **SFPSTORE** | Stores a value from an SFPU local register back to the Destination Register File. Used by `sfpi::dst_reg[index] = result`. |
| **SFPSETCC** | Sets the Condition Code Result based on a register value. Used by `v_if(in0 == 0 \|\| in1 == 0)` to test for zero inputs. Opcode 0x7B, IPC=1, latency=1. The `== 0` comparison uses InstrMod=6 (set CC if value is zero). |
| **SFPENCC** | Enables/disables predicated execution. Used by `v_if`/`v_endif` to bracket conditional blocks. |
| **SFPCOMPC** | Complements the Condition Code Result. Used internally by the `\|\|` operator in `v_if` to combine two conditions. |
| **SFPPUSHC / SFPPOPC** | Push/pop Condition Code state onto the CC stack. Used by nested `v_if` with `\|\|` to save/restore CC state between the two comparisons. |
| **SFPMOV** | Moves/loads an immediate value into an LREG, used to set `result = 0.0f` in the conditional zero-handling path. |
| **SFPSHFT** | Logical shift right, used by `bits >> 16` in `float32_to_bf16_rne`. |
| **SFPAND** | Bitwise AND, used by `bits & 1` and `bits & 0xFFFF0000U` in the BF16 rounding logic. |
| **SFPIADD** | Integer add, used by `bits + 0x7fffU + lsb` in the BF16 rounding logic. |
| **SFPCONFIG** | Writes to the SFPU configuration register. Called once during init via `TTI_SFPCONFIG(0, 0xF, 1)` to reset the SFPU config state. Opcode 0x91, IPC=0.5, latency=1. |
| **SETRWC** | Sets Read/Write Counters. Used between faces to advance the DST row pointer by 16 rows (two `TTI_SETRWC(..., 8, ...)` calls per face). Not an SFPU instruction per se but a Tensix math engine instruction that controls DST addressing. |
| **STALLWAIT** | Stalls until the SFPU pipeline is idle. Used at the start (`_llk_math_eltwise_binary_sfpu_start_`) and on Wormhole at the end (`_llk_math_eltwise_binary_sfpu_done_`) to synchronize the math engine with the SFPU. |

**Note**: When `is_fp32_dest_acc_en=true` (the typical FLOAT32 MUL path), the BF16 rounding and zero-check instructions (SFPSHFT, SFPAND, SFPIADD, SFPSETCC, SFPENCC, SFPCOMPC, SFPPUSHC, SFPPOPC, SFPMOV) are **not emitted**. The kernel reduces to just SFPLOAD, SFPMUL, SFPSTORE per iteration -- a very tight inner loop.

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **DST[dst_index_in0 * 32 + row]** | Source: input A tile data. Read via SFPLOAD into LREG. `dst_index_in0=0`, so DST rows 0-15 for face 0, offset by SETRWC for subsequent faces. |
| **DST[dst_index_in1 * 32 + row]** | Source: input B tile data. Read via SFPLOAD into LREG. `dst_index_in1=1`, so DST rows 32-47 for face 0 (offset 1*32 from base). |
| **DST[dst_index_out * 32 + row]** | Destination: output tile data. Written via SFPSTORE. `dst_index_out=0`, so writes back to the same DST rows as input A. |
| **LREG[0..3]** (RG[VA..VD]) | SFPU local registers. `in0` and `in1` are loaded into LREGs, the multiply result goes to an LREG, and SFPSTORE writes it back. The SFPI compiler manages LREG allocation; typically `in0` -> LREG[0], `in1` -> LREG[1], `result` -> LREG[2] or reuses LREG[0]. |
| **LREG[7]** (RG[7]) | Index register, not directly used by this kernel's MUL path but available for indirect addressing modes. |
| **DST row counter** | Managed by `sfpi::dst_reg++` (increments by SFP_DESTREG_STRIDE=2 each iteration) and `TTI_SETRWC` (advances by 8 rows per call, 2 calls = 16 rows between faces). Over 4 faces x 8 iterations x 2 rows/iteration = 64 rows = full 32x32 tile. |

### Address Mode Configuration

The address mode is configured during `_llk_math_eltwise_binary_sfpu_init_` via `eltwise_binary_sfpu_configure_addrmod<SfpuType::unused>()`.

For floating-point MUL (`SfpuType::unused`), only **ADDR_MOD_7** is configured:

| Field | Value | Description |
|-------|-------|-------------|
| `srca.incr` | 0 | No auto-increment for SrcA register addressing |
| `srcb.incr` | 0 | No auto-increment for SrcB register addressing |
| `dest.incr` | 0 | No auto-increment for Dest register addressing |

This is set to ADDR_MOD_7 specifically to avoid conflicting with ADDR_MOD_0 and ADDR_MOD_2, which are used by the A2D (unpack-to-DST) copy operations that precede the SFPU work.

The `if constexpr` branch that configures **ADDR_MOD_6** with `dest.incr=2` is only taken for integer multiplication types (`SfpuType::mul_int32`, `SfpuType::mul_uint16`) and min/max operations. For float MUL, ADDR_MOD_6 is not configured.

The address mode configuration is **identical between Wormhole B0 and Blackhole** for this operation. The only difference between the two architectures in the binary SFPU flow is that Wormhole B0 calls `math::set_addr_mod_base()` in `_llk_math_eltwise_binary_sfpu_start_` and `math::clear_addr_mod_base()` + `TTI_STALLWAIT(STALL_CFG, WAIT_SFPU)` in `_llk_math_eltwise_binary_sfpu_done_`, while Blackhole omits these calls.

## External Knowledge Sources

### DeepWiki Queries
1. **Query**: "How does the binary SFPU element-wise program factory work? What kernels does it use (reader, compute, writer)? How does it handle interleaved vs sharded tensors and broadcasting?"
   **Reason**: Initial architectural understanding of the binary SFPU program factory and its kernel structure.
   **Key Findings**: Confirmed three kernels (reader, compute, writer), sharded vs interleaved handling via defines and globally allocated CBs, and that this factory is specifically for non-broadcast element-wise operations. Broadcasting uses separate factories.

2. **Query**: "How does the SFPU binary multiply operation work? Trace the call chain from mul_binary_tile_init and mul_binary_tile through LLK to the ckernel SFPU implementation."
   **Reason**: Needed to understand the full call chain and identify the specific SFPU kernel files for MUL.
   **Key Findings**: Confirmed that MUL uses a dedicated `calculate_sfpu_binary_mul` (not the generic `_calculate_sfpu_binary_`), identified file paths across both tt-metal and tt_llk repos, and learned about the BF16 RNE rounding and zero-check special handling.

3. **Query**: "How is the SFPU binary multiply implemented in tt_llk? What is the call chain from mul_binary_tile through llk_math_eltwise_binary_sfpu to the ckernel_sfpu level? What SFPU instructions does it use?"
   **Reason**: Needed tt_llk-specific details on the LLK dispatch mechanism and SFPU instruction mapping.
   **Key Findings**: Confirmed that `vFloat * vFloat` maps to the SFPMUL instruction (alias of SFPMAD), identified the `_llk_math_eltwise_binary_sfpu_params_` face iteration pattern, and confirmed SFPLOAD/SFPSTORE for DST register access.

### Confluence References
1. **Section**: "SFPMUL" from the Tensix SFPU Instruction Set Architecture page (page ID 1170505767)
   **Key Information**: SFPMUL (opcode 0x86) is an alias of SFPMAD with RG[VC]=0. It performs FP32 multiply with IPC=1, latency=2. It flushes subnormals and sets exception flags but does not set condition codes.

2. **Section**: "SFPMAD" from the same page
   **Key Information**: SFPMAD (opcode 0x84) performs `(A * B) + C` as a fused multiply-add. InstrMod bits control source/destination selection and sign inversion. Subnormal inputs are flushed to signed zero before computation. NaN/Inf/Overflow/Denorm flags are set on the result.

3. **Section**: "SFPSETCC" from the same page
   **Key Information**: SFPSETCC (opcode 0x7B) sets CC.Res based on register value. InstrMod=6 tests if RG[VC] is zero. Used by the `v_if(in0 == 0 || in1 == 0)` conditional in the non-FP32 path.

4. **Section**: "SFPCONFIG" from the same page
   **Key Information**: SFPCONFIG (opcode 0x91) writes to configuration registers. `TTI_SFPCONFIG(0, 0xF, 1)` initializes the SFPU config register during `_init_sfpu_config_reg()`.

### Documentation References
1. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/binary_device_operation.cpp` (lines 21-33)
   **Reason**: Needed to confirm when the SFPU path is selected for MUL.
   **Key Information**: `is_binary_sfpu_op` returns true for MUL when both dtypes match and are FLOAT32/INT32/UINT32/UINT16. Same-height-same-width check in `select_program_factory` gates entry to `ElementWiseMultiCoreSfpu`.

2. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary/common/binary_op_utils.cpp` (lines 228-241, 535)
   **Reason**: Needed to determine which SFPU defines are generated for MUL.
   **Key Information**: For float MUL: `BINOP_INIT` = `mul_binary_tile_init()`, `BINARY_SFPU_OP` = `mul_binary_tile(0, 1, 0)`. For integer MUL: `MUL_INT_INIT` = `mul_int_tile_init<DataFormat::...>()`, same BINARY_SFPU_OP pattern.

3. **Source**: `tt_metal/hw/inc/api/compute/eltwise_binary_sfpu.h` (lines 43-44, 70)
   **Reason**: Needed to trace the actual SFPU function invoked.
   **Key Information**: `mul_binary_tile` calls `llk_math_eltwise_binary_sfpu_binop_mul<APPROX, BinaryOp::MUL, DST_ACCUM_MODE>`, a specialized MUL variant (distinct from ADD/SUB which use the generic `binop` path).

### Glean References
No Glean queries were needed for this analysis. The SFPU instruction details were sufficiently covered by Confluence and DeepWiki.
