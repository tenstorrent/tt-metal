# Typecast (Sharded) Implementation Analysis

## Overview

The typecast sharded operation converts tensor data from one data type to another using the SFPU, operating on sharded tensors that already reside in L1 memory. Since both input and output are sharded in L1, no DRAM reads or writes are required -- the operation is purely an in-place-style format conversion performed by the compute kernel. The reader kernel merely signals data availability (since sharded data is already present in L1-backed circular buffers), and there is no writer kernel at all.

**Program factory path**: `ttnn/cpp/ttnn/operations/copy/typecast/device/typecast_sharded_program_factory.cpp`

## Path Selection: FPU vs SFPU

The typecast sharded program factory uses **only an SFPU path**. There is no FPU variant within this factory. The compute kernel (`eltwise_typecast.cpp`) calls `init_sfpu()`, `copy_tile()` (unpack to DEST), then invokes `TYPECAST_LLK_INIT()` and `TYPECAST_LLK()` which expand to `typecast_tile_init<IN_DTYPE, OUT_DTYPE>()` and `typecast_tile<IN_DTYPE, OUT_DTYPE>(0)` -- both of which dispatch to `llk_math_eltwise_unary_sfpu_typecast`, a pure SFPU operation.

The broader typecast device operation has **four program factories** selected by `TypecastDeviceOperation::select_program_factory()`:

1. **TypecastShardedProgramFactory** (this analysis) -- selected when:
   - Input tensor is sharded
   - Input and output tile sizes match (`tile_size(act_df) == tile_size(out_df)`)
   - Both input and output buffers are in L1 (not DRAM)
   - Shard width is L1-aligned (for non-BFP types)
   - Shard size is a multiple of tile size (for non-BFP types)
   - No `sub_core_grids` are specified
2. **TypecastProgramFactory** -- fallback for sharded inputs that don't meet the above constraints, or non-sharded tile-layout inputs
3. **TypecastSubgridProgramFactory** -- when `sub_core_grids` is specified
4. **TypecastRowMajorChunkedProgramFactory** -- for row-major layout inputs

## Work Unit Definition

One work unit is **one tile** (32x32 elements). The compute kernel processes tiles one at a time: unpack one tile from input CB to DEST, apply SFPU typecast, pack one tile to output CB. The total number of tiles per core is `num_tile_per_core`, computed from shard dimensions.

## Tensor Format and Layout

### Input Tensor

| Property           | Value                                                                                         |
|--------------------|-----------------------------------------------------------------------------------------------|
| Dimension Convention | NHWC-style; shard_spec.shape is [height, width]                                              |
| Tensor Layout      | TILE (32x32)                                                                                  |
| Memory Layout      | Sharded (height, width, or block)                                                             |
| Buffer Type        | L1 (enforced via `TT_FATAL(src_is_dram == 0)`)                                               |
| Data Type          | Any supported input dtype: Float16_b, Float32, Int32, UInt16, UInt32, UInt8, Bfp8_b, Bfp4_b  |
| Shard Shape        | From `shard_spec.shape` -- [shard_height, shard_width]                                        |
| Core Grid          | From `shard_spec.grid` (`all_cores`)                                                          |
| Shard Orientation  | Inherited from input tensor's shard spec                                                      |

### Output Tensor

| Property           | Value                                                                                         |
|--------------------|-----------------------------------------------------------------------------------------------|
| Dimension Convention | Same as input                                                                                |
| Tensor Layout      | TILE (32x32)                                                                                  |
| Memory Layout      | Sharded (same memory layout as input, enforced by validation)                                 |
| Buffer Type        | L1 (enforced via `TT_FATAL(dst_is_dram == 0)`)                                               |
| Data Type          | Target output dtype (same supported types as input)                                           |
| Shard Shape        | Same core count as input; output shard spec must match `ncores`                               |
| Core Grid          | Same as input (`all_cores`)                                                                   |

### Layout Transformations

No tilize/untilize or reshard operations. The operation is strictly a data format conversion (e.g., Float16_b to Float32) while preserving the tile layout and sharding configuration. The constraint `tile_size(act_df) == tile_size(out_df)` ensures input and output occupy the same number of bytes per tile.

## Data Flow Pattern

1. **Data already in L1**: Input tensor is sharded, so each core's shard data already resides in L1 memory at the address backing CB c_0.
2. **Reader signals availability**: The reader kernel (`reader_unary_sharded.cpp`) executes `cb_push_back(cb_id_in0, num_tiles_per_core)`, making all tiles in the input CB visible to the compute kernel. No NoC transfers occur.
3. **Compute processes tile-by-tile**: For each tile:
   - `cb_wait_front(input_cb, 1)` -- wait for one input tile
   - `copy_tile(input_cb, 0, 0)` -- unpack tile from CB to DEST register 0
   - `TYPECAST_LLK_INIT()` / `TYPECAST_LLK(0)` -- apply SFPU typecast on DEST register 0
   - `pack_tile(0, output_cb)` -- pack DEST register 0 into output CB
   - `cb_pop_front(input_cb, 1)` -- free the input tile slot
4. **Output written to sharded CB**: The output CB (c_2) is backed by the output tensor's L1 buffer (`set_globally_allocated_address`). After all tiles in a block are processed, `cb_push_back(output_cb, per_core_block_dim)` publishes them. Since the CB is globally allocated to the output buffer, no explicit writer kernel is needed.

## Circular Buffer Configuration

| CB ID | Purpose         | Data Format | Page Size                     | Num Pages            | Total Size                        | Buffering | Producer        | Consumer        |
|-------|-----------------|-------------|-------------------------------|----------------------|-----------------------------------|-----------|-----------------|-----------------|
| c_0   | Input (sharded) | `act_df`    | `round_up_to_mul32(tile_size(act_df))` | `num_tile_per_core` | page_size * num_tile_per_core     | Single    | Reader (signal) | Compute (unpack)|
| c_2   | Output (sharded)| `out_df`    | `round_up_to_mul32(tile_size(out_df))` | `num_tile_per_core` | page_size * num_tile_per_core     | Single    | Compute (pack)  | N/A (L1-backed) |

Both CBs use `set_globally_allocated_address` to bind directly to the input/output tensor buffers in L1. The buffering factor is explicitly 1 (single-buffered), which is appropriate because sharded data is fully resident in L1 and no streaming from DRAM is needed.

## Pipeline Pattern Summary

- **Single-buffered**: Both c_0 and c_2 have `buffering_factor = 1`. The entire shard is available at once in L1. There is no overlap between read and compute because the reader merely signals; the compute kernel processes tiles sequentially.

## Index Calculations

No tensor accessor is used. The tile count per core (`num_tile_per_core`) is calculated in the host program factory:

- **For BFP8_B/BFP4_B inputs**: `num_tile_per_core = ceil(shard_width / TILE_WIDTH) * ceil(shard_height / TILE_HEIGHT)` -- standard tile counting.
- **For other formats**: `num_tile_per_core = ceil(shard_height * shard_width * datum_size(act_df) / tile_size(act_df))` -- byte-based calculation dividing total shard bytes by tile size.

Within the compute kernel, tiles are accessed sequentially (tile 0, 1, 2, ...) via `copy_tile(input_cb, 0, 0)` where offset 0 advances implicitly through `cb_pop_front`.

## Memory Access Patterns

### Read Pattern

Sequential tile-by-tile reads from the input CB. Since the CB is backed by the sharded L1 buffer, reads are local L1 accesses with no NoC involvement. Each `copy_tile` unpacks one tile from L1 to DEST registers.

### Write Pattern

Sequential tile-by-tile writes to the output CB. Each `pack_tile` writes one tile from DEST registers into the output CB, which is backed by the output tensor's L1 buffer. No NoC writes occur.

## Core Distribution Strategy

| Property             | Value                                                                |
|----------------------|----------------------------------------------------------------------|
| Grid Topology        | Determined by `shard_spec.grid` (the input tensor's shard grid)      |
| Total Cores          | `shard_spec.num_cores()` (must equal output shard's core count)      |
| Work per Core        | `num_tile_per_core` tiles (uniform across all cores in the shard grid)|
| Remainder Handling   | None -- shard spec guarantees equal distribution per core            |
| Load Balancing       | Uniform; every core processes the same number of tiles               |

All kernels (reader, compute) are deployed to `all_cores`, which is the full shard grid from the input tensor's shard spec.

## Arguments

### Compile-Time Arguments

**Reader Kernel** (`reader_unary_sharded.cpp`):

| Index | Name       | Type     | Description                       |
|-------|------------|----------|-----------------------------------|
| 0     | `cb_id_in0`| uint32_t | Input circular buffer ID (c_0)    |

**Compute Kernel** (`eltwise_typecast.cpp`):

| Index | Name                  | Type     | Description                                             |
|-------|-----------------------|----------|---------------------------------------------------------|
| 0     | `per_core_block_cnt`  | uint32_t | Number of blocks per core (always 1)                    |
| 1     | `per_core_block_dim`  | uint32_t | Number of tiles per block (= `num_tile_per_core`)       |
| 2     | `input_cb`            | uint32_t | Input circular buffer ID (c_0)                          |
| 3     | `output_cb`           | uint32_t | Output circular buffer ID (c_2)                         |

**Compute Kernel Defines**:

| Define Name         | Expansion                                              | Description                                        |
|---------------------|--------------------------------------------------------|----------------------------------------------------|
| `TYPECAST_LLK_INIT` | `typecast_tile_init<IN_DTYPE_u, OUT_DTYPE_u>()`        | SFPU typecast initialization with format enum values|
| `TYPECAST_LLK`      | `typecast_tile<IN_DTYPE_u, OUT_DTYPE_u>(tile_index)`   | SFPU typecast operation with format enum values     |

### Runtime Arguments

**Reader Kernel**:

| Index | Name                | Type     | Description                             |
|-------|---------------------|----------|-----------------------------------------|
| 0     | `num_tile_per_core` | uint32_t | Number of tiles to signal as available  |

**Compute Kernel**: No runtime arguments.

## Kernel Implementations

| Kernel   | File                                                                                          | Type     | Assigned Cores |
|----------|-----------------------------------------------------------------------------------------------|----------|----------------|
| Reader   | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp`     | Dataflow | `all_cores`    |
| Compute  | `ttnn/cpp/ttnn/operations/copy/typecast/device/kernels/compute/eltwise_typecast.cpp`          | Compute  | `all_cores`    |

### Reader Kernel

| Property       | Value                                                                                      |
|----------------|--------------------------------------------------------------------------------------------|
| File           | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp`  |
| Assigned Cores | `all_cores` (full shard grid)                                                              |

**Key Logic**:
- This is a minimal sharded reader: the only action is `cb_push_back(cb_id_in0, num_tiles_per_core)`.
- Since the input CB is globally allocated to the sharded tensor's L1 buffer, data is already present. The push_back merely signals to the compute kernel that all tiles are available.
- No NoC reads, no loops, no data movement.
- **Synchronization**: Pushes `num_tiles_per_core` pages to `cb_id_in0` (c_0), making them visible to the compute kernel's `cb_wait_front`.

### Compute Kernel

| Property       | Value                                                                                              |
|----------------|----------------------------------------------------------------------------------------------------|
| File           | `ttnn/cpp/ttnn/operations/copy/typecast/device/kernels/compute/eltwise_typecast.cpp`               |
| Assigned Cores | `all_cores` (full shard grid)                                                                      |

**Key Logic**:
- Calls `init_sfpu(input_cb, output_cb)` once to configure unpacker/packer for the input and output data formats.
- Outer loop iterates `per_core_block_cnt` times (always 1 in the sharded factory).
- Reserves space for `per_core_block_dim` tiles in the output CB at the start of each block.
- Inner loop processes tiles one at a time:
  1. `tile_regs_acquire()` -- acquire DEST register file
  2. `cb_wait_front(input_cb, 1)` -- wait for one tile from reader
  3. `copy_tile(input_cb, 0, 0)` -- unpack tile from input CB offset 0 into DEST register 0
  4. `TYPECAST_LLK_INIT()` -- initialize SFPU typecast (configures SFPU for the specific format pair)
  5. `TYPECAST_LLK(0)` -- execute SFPU typecast on DEST register 0
  6. `tile_regs_commit()` -- signal DEST is ready for packing
  7. `tile_regs_wait()` -- wait for pack permission
  8. `pack_tile(0, output_cb)` -- pack DEST register 0 into output CB
  9. `cb_pop_front(input_cb, 1)` -- free the consumed input tile
  10. `tile_regs_release()` -- release DEST registers
- After the inner loop, `cb_push_back(output_cb, per_core_block_dim)` publishes all processed tiles.
- **Synchronization**: Consumes from c_0 via `cb_wait_front`/`cb_pop_front` (one tile at a time). Produces to c_2 via `cb_reserve_back` (bulk) / `pack_tile` / `cb_push_back` (bulk). The `tile_regs_acquire/commit/wait/release` sequence coordinates unpacker and packer access to DEST registers.
- Note: `TYPECAST_LLK_INIT()` is called inside the inner loop on every tile iteration. This is slightly unusual (typically init is called once before the loop), but the SFPU init for typecast may need to reset state per tile.

## Implementation Notes

- **Program factory variants**: Four factories exist (see Path Selection section). The sharded factory is selected by `can_use_sharded_optimized_factory()` which checks: input is sharded, tile sizes match, both buffers in L1, alignment constraints met, no sub_core_grids.

- **Type-based operation variants**: Supports a wide range of type conversions via SFPU. The input/output data format enums are passed as template parameters to `typecast_tile<IN_DTYPE, OUT_DTYPE>`. Supported conversions include: Float16_b, Float32, Int32, UInt16, UInt32, UInt8, Bfp8_b, Bfp4_b (see typecast.h for the full matrix). The program factory encodes `input_dtype` and `output_dtype` as compile-time defines.

- **UnpackToDestFP32 mode**: Enabled when `args.preserve_fp32_precision` is true. Sets `UnpackToDestMode::UnpackToDestFp32` on the input CB (`in_cb_id`), which configures the unpacker to use 32-bit precision when loading tiles into DEST. This is critical for conversions involving 32-bit types (Float32, Int32, UInt32) where DEST must be in 32-bit mode.

- **Broadcast type selection**: N/A. This is a unary element-wise operation with no broadcasting.

- **Sharding support and constraints**: This factory is exclusively for sharded tensors. Supports height, width, and block sharding (whatever the input shard spec provides). Constraints: input and output tile sizes must match, both buffers must be in L1, shard width must be L1-aligned for non-BFP types, shard total bytes must be a multiple of tile size for non-BFP types.

- **FP32 dest accumulation**: Controlled by `args.fp32_dest_acc_en`, passed directly to `ComputeConfig.fp32_dest_acc_en`. When enabled, DEST registers operate in 32-bit mode, which is required for conversions involving 32-bit data types.

## SFPU Kernel Implementation

This section provides a dedicated deep dive into the underlying SFPU kernel functions that the compute kernel dispatches to. The typecast operation is unique in that it selects among **15 distinct SFPU kernel functions** at compile time based on the `IN_DTYPE`/`OUT_DTYPE` template parameters, plus several conversions that require no SFPU work at all (handled entirely by unpacker/packer). Each kernel function implements a specific format conversion algorithm.

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/typecast.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_typecast.h` |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/common/inc/sfpu/ckernel_sfpu_typecast.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |

### Call Chain

1. The compute kernel calls `TYPECAST_LLK(0)` which expands to `typecast_tile<IN_DTYPE, OUT_DTYPE>(0)` in `typecast.h`.
2. `typecast_tile` calls `llk_math_eltwise_unary_sfpu_typecast<APPROX, IN_DTYPE, OUT_DTYPE>(idst)` wrapped in the `MATH()` macro (runs on TRISC_MATH only).
3. `llk_math_eltwise_unary_sfpu_typecast` uses a compile-time `if constexpr` chain to select the correct `_calculate_typecast_*` function based on `IN_DTYPE`/`OUT_DTYPE`, then calls `_llk_math_eltwise_unary_sfpu_params_<APPROX>(selected_function, dst_index, vector_mode)`.
4. `_llk_math_eltwise_unary_sfpu_params_` sets up the DEST write address, stalls for SFPU readiness, then invokes the selected function once per face (4 times for `VectorMode::RC`), with `SETRWC` advancing the DEST address by 16 rows between faces.

Similarly, `TYPECAST_LLK_INIT()` expands to `typecast_tile_init<IN_DTYPE, OUT_DTYPE>()` which calls `llk_math_eltwise_unary_sfpu_typecast_init`, which selects a conversion-specific init function (or uses the default no-op init) and calls `llk_math_eltwise_unary_sfpu_init<SfpuType::typecast, APPROX>(init_fn)`. This configures address modes and SFPLOADMACRO templates.

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` (the default). All 4 faces of the 32x32 tile are processed -- the SFPU function is called 4 times in a loop.
- **Operation invocation**: For each of the 4 faces, the selected `_calculate_typecast_*<APPROX, 8>()` function is called. The `ITERATIONS=8` template parameter means each invocation processes 8 rows (one 16x16 face = 8 SFPU iterations, since the SFPU processes 2 rows per "row" in some modes, but fundamentally iterates 8 times to cover 16 rows of a face, with each SFPU row being a 32-element vector).
- **DEST address progression**: Between face invocations, `TTI_SETRWC` (Wormhole) or `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_` (Blackhole) advances the DEST read/write pointer by 16 rows (two `inc_dst_addr<8>()` calls). Within the SFPU function, the DEST address auto-increments by 2 per iteration via `ADDR_MOD_6` (dest.incr=2) for SFPLOADMACRO-based kernels, or by 1 per iteration via `ADDR_MOD_7` (dest.incr=0, used with explicit `SFPLOAD`/`SFPSTORE` that use their own addressing).

### Conversion Dispatch Table

The LLK dispatch layer (`llk_math_eltwise_unary_sfpu_typecast`) uses compile-time `if constexpr` to select the SFPU kernel function. The following table maps each supported conversion to its SFPU function. Note that many input formats (Float16_b, Bfp8_b, Bfp4_b) are first unpacked to Float32 in DEST, so they reuse the same `fp32_to_*` kernels.

| Input Format | Output Format | SFPU Function | Notes |
|---|---|---|---|
| Float16_b | UInt16 | `calculate_typecast_fp32_to_uint16` | Unpacked as FP32 in DEST |
| UInt16 | Float16_b | `calculate_typecast_uint16_to_fp16b` | |
| Int32 | Float16_b | `calculate_typecast_int32_to_fp16b` | |
| Float16_b | Int32 | `calculate_typecast_fp32_to_int32` | |
| Float16_b | Float32 | *(no SFPU kernel)* | Handled by packer |
| Float32 | Float16_b | `calculate_typecast_fp32_to_fp16b` | |
| Float32 | UInt16 | `calculate_typecast_fp32_to_uint16` | |
| UInt16 | Float32 | `calculate_typecast_uint16_to_fp32` | |
| Float32 | Int32 | `calculate_typecast_fp32_to_int32` | |
| Int32 | Float32 | `calculate_typecast_int32_to_fp32` | |
| Float16_b | UInt32 | `calculate_typecast_fp32_to_uint32` | |
| UInt32 | Float16_b | `calculate_typecast_uint32_to_fp16b` | |
| Float32 | UInt32 | `calculate_typecast_fp32_to_uint32` | |
| UInt32 | Float32 | `calculate_typecast_uint32_to_fp32` | |
| UInt16 | UInt32 | `calculate_typecast_uint16_to_uint32` | |
| UInt16 | Int32 | `calculate_typecast_uint16_to_uint32` | Same kernel as UInt32 output |
| UInt32 | UInt16 | `calculate_typecast_uint32_to_uint16` | |
| Int32 | UInt16 | `calculate_typecast_int32_to_uint16` | |
| Bfp8_b/Bfp4_b | Float16_b | *(no SFPU kernel)* | Handled by unpacker |
| Float16_b | Bfp8_b/Bfp4_b | *(no SFPU kernel)* | Handled by packer |
| Bfp8_b/Bfp4_b | Float32 | *(no SFPU kernel)* | Handled by unpacker/packer |
| Float32 | Bfp8_b/Bfp4_b | *(no SFPU kernel)* | Handled by packer |
| Bfp8_b/Bfp4_b | UInt16/Int32/UInt32 | Reuses `fp32_to_*` kernels | Unpacked as FP32 in DEST |
| UInt16/Int32/UInt32 | Bfp8_b/Bfp4_b | Reuses `*_to_fp16b` kernels | Packed from FP16B/FP32 in DEST |

### Annotated SFPU Kernel Source

The typecast operation has **15 distinct SFPU kernel functions** across two hardware variants (Wormhole B0 and Blackhole). The implementations differ between architectures primarily in which `ADDR_MOD` indices they reference (`ADDR_MOD_2`/`ADDR_MOD_3` on Wormhole vs `ADDR_MOD_6`/`ADDR_MOD_7` on Blackhole) and in the `_calculate_typecast_fp32_to_int32_` function (Wormhole uses SFPI abstractions while Blackhole uses raw TTI_ instructions). The Blackhole source is presented below as the primary reference, with Wormhole differences noted.

**Kernel style determination**: The majority of these kernels use raw `TT_`/`TTI_` instructions and `SFPLOADMACRO` macro scheduling. Two kernels -- `_calculate_typecast_fp32_to_int32_` (Blackhole only, where it uses raw TTI_) and `_calculate_typecast_fp32_to_uint32_` -- use complex CC manipulation with `SFPSETCC`, `SFPEXEXP`, `SFPIADD` CC side effects, and `SFPENCC`. These use **Style B** with CC State Machine diagrams. The remaining kernels use **Style A** since they either use SFPLOADMACRO (where CC is not directly manipulated) or SFPI abstractions (`v_if`/`v_endif`).

#### Style A Kernels: SFPLOADMACRO-based and SFPI-based

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_typecast.h

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_fp32_to_uint16_() // APPROXIMATION_MODE=false, ITERATIONS=8
{
    // Implementation notes, see the original file for more details
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        int v = d & 1; // alternate between p_sfpu::LREG0 and p_sfpu::LREG1
        TT_SFPLOADMACRO((0 << 2) | (v & 3), InstrModLoadStore::DEFAULT, ADDR_MOD_6, v >> 2); // Load row from DEST, schedule max(v,0) + rnd(uint16) + store via macro
        TTI_SFPNOP; // Pipeline bubble required between macro invocations
    }
    TTI_SFPNOP; // Drain pipeline (3 NOPs for remaining scheduled operations)
    TTI_SFPNOP;
    TTI_SFPNOP;
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_uint16_to_fp16b_() // APPROXIMATION_MODE=false, ITERATIONS=8
{
    // Implementation notes, see the original file for more details
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        int v = d & 1;
        TT_SFPLOADMACRO((0 << 2) | (v & 3), InstrModLoadStore::LO16, ADDR_MOD_6, v >> 2); // Load LO16 from DEST, schedule cast + rnd(fp16b) + store
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_int32_to_fp16b_() // APPROXIMATION_MODE=false, ITERATIONS=8
{
    // Implementation notes, see the original file for more details
    constexpr int t = p_sfpu::LREG4; // Temporary register for abs/cast results

    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_USHORT, 0);       // L0 = 0.0 (used for positive values)
    TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB, 0xcf00);  // L1 = -2^31 (used for overflow)

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        int v = 2 + (d & 1); // alternate between LREG2 and LREG3
        TT_SFPLOADMACRO((0 << 2) | (v & 3), InstrModLoadStore::INT32, ADDR_MOD_6, v >> 2); // Load int32 row from DEST
        TT_SFPABS(0, v, t, 0);                                         // t = abs(v)
        TTI_SFPSHFT2(t, p_sfpu::LREG12, p_sfpu::LREG7, 5);            // L7 = t >> 31 (extract sign bit for indirect MAD VA selection)
        TTI_SFPCAST(t, t, 0);                                          // t = cast(t) - converts unsigned int to float
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_fp32_to_fp16b_() // APPROXIMATION_MODE=false, ITERATIONS=8
{
    // Implementation notes, see the original file for more details
    constexpr int b = p_sfpu::LREG2;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        int a = d & 1; // alternate between LREG0 and LREG1
        TT_SFPLOADMACRO((0 << 2) | (a & 3), 0, ADDR_MOD_7, a >> 2);           // Load [a], schedule >>= 16 (round unit)
        TTI_SFPLOADMACRO((1 << 2) | (b & 3), 0, ADDR_MOD_6, b >> 2);          // Load [b], schedule += 0x7fff, store
        TT_SFPAND(0, p_sfpu::LREG12, a, 0);                                    // a &= 1 (extract LSB for round-to-even)
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_uint16_to_fp32_() // APPROXIMATION_MODE=false, ITERATIONS=8
{
    // Implementation notes, see the original file for more details
    constexpr int v = p_sfpu::LREG0;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        TTI_SFPLOADMACRO((0 << 2) | (v & 3), InstrModLoadStore::LO16, ADDR_MOD_6, v >> 2); // Load LO16, schedule cast + store(FP32)
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_int32_to_fp32_() // APPROXIMATION_MODE=false, ITERATIONS=8
{
    // Implementation notes, see the original file for more details
    constexpr int t = p_sfpu::LREG4;

    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_USHORT, 0);       // L0 = 0.0
    TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB, 0xcf00);  // L1 = -2^31

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        int v = 2 + (d & 1); // alternate between LREG2 and LREG3
        TT_SFPLOADMACRO((0 << 2) | (v & 3), InstrModLoadStore::INT32, ADDR_MOD_6, v >> 2); // Load int32 from DEST
        TT_SFPABS(0, v, t, 0);                                         // t = abs(v)
        TTI_SFPSHFT2(t, p_sfpu::LREG12, p_sfpu::LREG7, 5);            // L7 = t >> 31 (sign bit -> L7 for indirect MAD)
        TTI_SFPCAST(t, t, 0);                                          // t = cast unsigned int to float
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_uint32_to_fp16b_() // APPROXIMATION_MODE=false, ITERATIONS=8
{
    // Implementation notes, see the original file for more details
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_USHORT, 0);       // L0 = 0.0
    TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB, 0x4f00);  // L1 = 2^31

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        int v = 2 + (d & 1); // alternate between LREG2 and LREG3
        TT_SFPLOADMACRO((0 << 2) | (v & 3), InstrModLoadStore::INT32, ADDR_MOD_6, v >> 2); // Load int32 from DEST
        TT_SFPSHFT2(v, p_sfpu::LREG12, p_sfpu::LREG7, 5);            // L7 = v >> 31 (sign/high bit for indirect MAD)
        TT_SFPSETSGN(0, v, v, 1);                                     // v = setsgn(v, 0) -- clear sign bit, mod1=1 means ARG_IMM
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_uint32_to_fp32_() // APPROXIMATION_MODE=false, ITERATIONS=8
{
    // Implementation notes, see the original file for more details
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_USHORT, 0);       // L0 = 0.0
    TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB, 0x4f00);  // L1 = 2^31

    constexpr int a  = p_sfpu::LREG2;
    constexpr int b  = p_sfpu::LREG3;
    constexpr int L7 = p_sfpu::LREG7;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        TTI_SFPLOADMACRO((0 << 2) | (a & 3), InstrModLoadStore::INT32, ADDR_MOD_7, a >> 2);  // Load [a] from DEST
        TTI_SFPLOADMACRO((1 << 2) | (b & 3), InstrModLoadStore::INT32, ADDR_MOD_7, b >> 2);  // Load [b] from DEST (cast of a)
        TTI_SFPLOADMACRO((2 << 2) | (L7 & 3), InstrModLoadStore::INT32, ADDR_MOD_6, L7 >> 2); // Load [L7], schedule >>31 + MAD + store
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_uint16_to_uint32_() // APPROXIMATION_MODE=false, ITERATIONS=8
{
    // Implementation notes, see the original file for more details
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        TTI_SFPLOADMACRO((0 << 2) | 0, InstrModLoadStore::LO16, ADDR_MOD_6, 0); // Load LO16, schedule direct INT32 store
    }
    TTI_SFPNOP;
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_uint32_to_uint16_() // APPROXIMATION_MODE=false, ITERATIONS=8
{
    // Implementation notes, see the original file for more details
    constexpr int a = p_sfpu::LREG0;
    constexpr int b = p_sfpu::LREG1;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        TTI_SFPLOADMACRO((0 << 2) | (a & 3), InstrModLoadStore::LO16, ADDR_MOD_7, a >> 2);  // Load high 16 bits
        TTI_SFPLOADMACRO((1 << 2) | (b & 3), InstrModLoadStore::INT32, ADDR_MOD_6, b >> 2);  // Load low 16 bits, OR with high, store
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_int32_to_uint16_() // APPROXIMATION_MODE=false, ITERATIONS=8
{
    // Implementation notes, see the original file for more details
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        int a = d & 1;
        TT_SFPLOADMACRO((0 << 2) | (a & 3), InstrModLoadStore::INT32, ADDR_MOD_6, a >> 2); // Load int32 from DEST
        TT_SFPCAST(a, a, 0);  // cast int32 to fp32 (allows SFPSTOCHRND to convert to uint16)
        TTI_SFPNOP;           // Pipeline gap for cast latency
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}
```

#### Style A Kernel: SFPI-based (Wormhole `_calculate_typecast_fp32_to_int32_`)

The Wormhole B0 variant of `_calculate_typecast_fp32_to_int32_` uses SFPI abstractions (`v_if`/`v_endif`, `sfpi::vFloat`, `sfpi::vInt`), making CC flow explicit:

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_typecast.h

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_fp32_to_int32_() // APPROXIMATION_MODE=false, ITERATIONS=8
{
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::vFloat in = sfpi::dst_reg[0];          // Load FP32 value from DEST

        sfpi::vInt exp = sfpi::exexp(in);              // Extract debiased exponent (exp=0 means 1.x)
        sfpi::vUInt man = sfpi::exman8(in);            // Extract mantissa with implicit 1 at bit 23

        sfpi::vInt shift_amt = exp - 23;               // Shift amount: how far to move mantissa
        sfpi::vInt result = sfpi::reinterpret<sfpi::vInt>(sfpi::shft(man, shift_amt)); // result = man << shift_amt

        v_if (exp >= 31) {                             // Overflow: |value| >= 2^31
            result = 0x80000000;                       // Clamp to INT_MIN
        }
        v_endif;

        v_if (exp < 0) {                               // Underflow: |value| < 1
            result = 0;                                // Round to zero
        }
        v_endif;

        v_if (in < 0.0f) {                             // Negative input: apply two's complement
            result = ~result + 1;                      // -x = ~x + 1
        }
        v_endif;

        sfpi::dst_reg[0] = result;                     // Store result back to DEST
        sfpi::dst_reg++;                               // Advance DEST pointer
    }
}
```

#### Style B Kernels: Raw TTI_ with CC manipulation

##### `_calculate_typecast_fp32_to_int32_` (Blackhole)

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_typecast.h

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_fp32_to_int32_()
{
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);
        TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_USHORT, 0);

        TTI_SFPEXEXP(0, p_sfpu::LREG0, p_sfpu::LREG2, sfpi::SFPEXEXP_MOD1_SET_CC_SGN_EXP | sfpi::SFPEXEXP_MOD1_SET_CC_COMP_EXP);
        TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB, 0x8000);
        TTI_SFPIADD(-31 & 0xfff, p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_LT0);
        TTI_SFPIADD(8, p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
        TTI_SFPEXMAN(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);
        TTI_SFPSHFT(0, p_sfpu::LREG2, p_sfpu::LREG1, 0);
        TTI_SFPENCC(0, 0, 0, 0);

        TTI_SFPSETCC(0, p_sfpu::LREG0, 0, sfpi::SFPSETCC_MOD1_LREG_LT0);
        TTI_SFPIADD(0, p_sfpu::LCONST_0, p_sfpu::LREG1, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
        TTI_SFPENCC(0, 0, 0, 0);

        TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_6, 0);
    }
}
```

```
_calculate_typecast_fp32_to_int32_ (Blackhole) -- CC State Transitions
════════════════════════════════════════════════════════════════

  CC State: ALL_ENABLED                   <-- initial state
       |
       |  SFPLOAD L0 = DEST[row]          (no CC effect)
       |  SFPLOADI L1 = 0                 (no CC effect)
       |
       v
  +-------------------------------------------+
  | SFPEXEXP  SET_CC_SGN_EXP |                |
  |           SET_CC_COMP_EXP                 |
  |                                           |
  | L2 = debiased exponent of L0             |
  | CC <- (exp >= 0)                         |
  +-------------------+-----------------------+
                      |
                      v
  CC State: ENABLED where exp >= 0
       |
       |  SFPLOADI L1 = 0x80000000        (CC-guarded: only exp>=0 lanes write; for exp<0, L1 stays 0)
       |
       v
  +-------------------------------------------+
  | SFPIADD  -31, L2, L2                     |
  |          CC_LT0                           |
  |                                           |
  | L2 = exp - 31                            |
  | CC <- CC_prev AND (result < 0)           |
  |    = (exp >= 0) AND (exp - 31 < 0)       |
  |    = (0 <= exp < 31)                     |
  +-------------------+-----------------------+
                      |
                      v
  CC State: ENABLED where 0 <= exp < 31
       |
       |  SFPIADD +8, L2, L2  CC_NONE    (CC-guarded: only 0<=exp<31 lanes; result: L2 = exp-23)
       |  SFPEXMAN L0 -> L1               (CC-guarded: mantissa with implicit 1 at bit 23)
       |  SFPSHFT L1 <<= L2              (CC-guarded: result = mantissa << (exp-23))
       |
       v
  +-------------------------------------------+
  | SFPENCC                                   |
  |                                           |
  | CC <- ALL_ENABLED                        |
  +-------------------+-----------------------+
                      |
                      v
  CC State: ALL_ENABLED
       |
       v
  +-------------------------------------------+
  | SFPSETCC  LREG_LT0  (on L0)             |
  |                                           |
  | CC <- (input < 0)                        |
  +-------------------+-----------------------+
                      |
                      v
  CC State: ENABLED where input < 0
       |
       |  SFPIADD 2SCOMP L1, CC_NONE     (CC-guarded: negate result for negative inputs)
       |
       v
  +-------------------------------------------+
  | SFPENCC                                   |
  |                                           |
  | CC <- ALL_ENABLED                        |
  +-------------------+-----------------------+
                      |
                      v
  CC State: ALL_ENABLED
       |
       |  SFPSTORE L1, INT32              (all lanes: store result to DEST)
       v
```

The algorithm works in two CC blocks:
1. **Block 1 (magnitude computation)**: SFPEXEXP extracts the exponent and enables lanes with exp >= 0. For disabled lanes (exp < 0, meaning |value| < 1), L1 remains 0. SFPIADD further narrows to exp < 31 (overflow lanes get L1 = INT_MIN from the SFPLOADI). The mantissa is extracted, shifted by (exp-23), producing the integer magnitude.
2. **Block 2 (sign application)**: SFPSETCC enables lanes where input < 0, and SFPIADD with 2SCOMP negates the result (two's complement) for those lanes only.

##### `_calculate_typecast_fp32_to_uint32_` (Blackhole)

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_typecast.h

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_fp32_to_uint32_()
{
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);
        TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_USHORT, 0);

        TTI_SFPSETCC(0, p_sfpu::LREG0, 0, sfpi::SFPSETCC_MOD1_LREG_GTE0);
        TTI_SFPEXEXP(0, p_sfpu::LREG0, p_sfpu::LREG2, sfpi::SFPEXEXP_MOD1_SET_CC_SGN_EXP | sfpi::SFPEXEXP_MOD1_SET_CC_COMP_EXP);
        TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_SHORT, 0xffff);
        TTI_SFPIADD(-32 & 0xfff, p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_LT0);
        TTI_SFPIADD(9, p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
        TTI_SFPEXMAN(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);
        TTI_SFPSHFT(0, p_sfpu::LREG2, p_sfpu::LREG1, 0);
        TTI_SFPENCC(0, 0, 0, 0);

        TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_6, 0);
    }
}
```

```
_calculate_typecast_fp32_to_uint32_ (Blackhole) -- CC State Transitions
════════════════════════════════════════════════════════════════

  CC State: ALL_ENABLED                   <-- initial state
       |
       |  SFPLOAD L0 = DEST[row]          (no CC effect)
       |  SFPLOADI L1 = 0                 (no CC effect, L1=0 is result for negative inputs)
       |
       v
  +-------------------------------------------+
  | SFPSETCC  LREG_GTE0  (on L0)            |
  |                                           |
  | CC <- (input >= 0)                       |
  +-------------------+-----------------------+
                      |
                      v
  CC State: ENABLED where input >= 0
       |
       v
  +-------------------------------------------+
  | SFPEXEXP  SET_CC_SGN_EXP |                |
  |           SET_CC_COMP_EXP                 |
  |                                           |
  | L2 = debiased exponent of L0             |
  | CC <- CC_prev AND (exp >= 0)             |
  |    = (input >= 0) AND (exp >= 0)         |
  +-------------------+-----------------------+
                      |
                      v
  CC State: ENABLED where input >= 0 AND exp >= 0
       |
       |  SFPLOADI L1 = 0xFFFFFFFF        (CC-guarded: overflow sentinel for out-of-range)
       |
       v
  +-------------------------------------------+
  | SFPIADD  -32, L2, L2                     |
  |          CC_LT0                           |
  |                                           |
  | L2 = exp - 32                            |
  | CC <- CC_prev AND (result < 0)           |
  |    = (input>=0) AND (exp>=0) AND (exp<32)|
  +-------------------+-----------------------+
                      |
                      v
  CC State: ENABLED where input >= 0, 0 <= exp < 32
       |
       |  SFPIADD +9, L2, L2  CC_NONE    (CC-guarded: L2 = exp-23)
       |  SFPEXMAN L0 -> L1               (CC-guarded: mantissa with implicit 1)
       |  SFPSHFT L1 <<= L2              (CC-guarded: result = mantissa << (exp-23))
       |
       v
  +-------------------------------------------+
  | SFPENCC                                   |
  |                                           |
  | CC <- ALL_ENABLED                        |
  +-------------------+-----------------------+
                      |
                      v
  CC State: ALL_ENABLED
       |
       |  SFPSTORE L1, INT32              (all lanes: store result to DEST)
       v
```

The uint32 variant differs from int32 by: (1) starting with `SFPSETCC GTE0` to clamp negative inputs to 0 (L1 stays 0 for negative lanes), (2) using 0xFFFFFFFF as overflow sentinel (UINT_MAX) instead of 0x80000000 (INT_MIN), (3) testing `exp < 32` instead of `exp < 31` (uint32 range is 0 to 2^32-1), and (4) adding 9 instead of 8 to the exponent adjustment.

### SFPU Instructions Used

The typecast operation uses a wide variety of SFPU instructions due to its many conversion paths:

| Instruction | Description |
|---|---|
| `SFPLOADMACRO` | Loads a row from DEST into an LREG and schedules pre-configured operations (Simple, MAD, Round, Store) across pipeline sub-units. Central to most typecast kernels for achieving high throughput. |
| `SFPLOAD` | Loads a row from DEST into an LREG. Used in the fp32_to_int32 and fp32_to_uint32 kernels where explicit instruction sequencing is needed instead of macro scheduling. |
| `SFPSTORE` | Stores an LREG row back to DEST. Used with `InstrModLoadStore::INT32` for integer output formats. |
| `SFPLOADI` | Loads an immediate value into an LREG. Used to set constants (0.0, -2^31, 2^31, 0xFFFFFFFF, 0x7FFF, etc.). |
| `SFPEXEXP` | Extracts the debiased exponent from a float in an LREG. With `SET_CC_SGN_EXP | SET_CC_COMP_EXP`, also sets CC based on exponent sign. |
| `SFPEXMAN` | Extracts the mantissa with implicit leading 1 at bit 23 from a float. |
| `SFPSHFT` | Barrel shifts an LREG by the amount in another LREG (signed: positive=left, negative=right). |
| `SFPSHFT2` | Shifts an LREG by an immediate or LREG-sourced amount. Used with `MOD1_SHFT_LREG` to extract sign bit (>> 31) and with `MOD1_SHFT_IMM` for constant shifts (>> 16). |
| `SFPCAST` | Converts between unsigned integer and float representations in LREGs. |
| `SFPABS` | Computes absolute value of an LREG. |
| `SFPSETSGN` | Sets or clears the sign bit of a float. With `MOD1_ARG_IMM`, clears sign (makes positive). |
| `SFPSETCC` | Sets the condition code based on an LREG value. `LREG_LT0` enables lanes where value < 0; `LREG_GTE0` enables lanes where value >= 0. |
| `SFPENCC` | Resets CC to ALL_ENABLED (unconditional). |
| `SFPIADD` | Integer add with immediate or register source. With `CC_LT0`, narrows CC to lanes where result < 0. With `2SCOMP_LREG_DST`, performs two's complement negation. |
| `SFPAND` | Bitwise AND between two LREGs. Used in fp32_to_fp16b for round-to-even LSB extraction. |
| `SFPOR` | Bitwise OR between two LREGs. Used in uint32_to_uint16 for combining high and low 16-bit halves. |
| `SFPSWAP` | Swaps/min/max between an LREG and a constant. Used to clamp negative values to 0 for uint16 conversions. |
| `SFP_STOCH_RND` | Stochastic/deterministic rounding. Used for FP32-to-FP16B and FP32-to-UINT16 conversions with various modes. |
| `SFPMAD` | Multiply-accumulate. With `MOD1_INDIRECT_VA`, uses L7 to index VA (selects between L0 and L1), enabling sign-dependent correction for int32/uint32 conversions. |
| `SFPCONFIG` | Configures SFPLOADMACRO templates, macros, and misc settings (store format, delay kind). |
| `SFPGT` | Greater-than comparison (Blackhole only). Used in uint32_to_uint16 init to produce a mask (VD = value > 0 ? -1 : 0). |
| `SFPNOP` | No-operation. Required for pipeline draining after SFPLOADMACRO sequences and between dependent operations. |

### SFPU Register Usage

| Register | Usage |
|---|---|
| **LREG0** | Frequently used as input value register or constant (0.0 in many conversions). In alternating schemes, paired with LREG1. |
| **LREG1** | Paired with LREG0 for register alternation. Also holds constants (-2^31 or 2^31 in signed/unsigned conversions). In fp32_to_int32, holds the result value. |
| **LREG2** | Used as exponent register in fp32_to_int32/uint32. In alternating schemes with LREG3 for int32 conversions. Also used as temporary `b` in fp32_to_fp16b. |
| **LREG3** | Paired with LREG2 for register alternation in int32/uint32 conversions. |
| **LREG4** | Temporary register `t` for abs/cast intermediate results in int32_to_fp16b and int32_to_fp32. |
| **LREG7** | Stores the sign bit (result of `>> 31`) for indirect MAD VA selection in signed/unsigned integer conversions. L7 indexes into L0/L1 to select the correction constant. |
| **LREG12** | Programmable constant register (`vConstIntPrgm0`). Holds values like 1, -31, or serves as a source for AND/shift operations. |
| **LREG13** | Programmable constant register (`vConstIntPrgm1`). Holds 0x7FFF in fp32_to_fp16b for round-to-nearest-even. |
| **DEST** | Source/destination for tile data. SFPLOAD reads from DEST; SFPSTORE writes back. Address auto-increments via ADDR_MOD between iterations. |

### Address Mode Configuration

Two address modes are configured for the typecast operation via `eltwise_unary_sfpu_configure_addrmod<SfpuType::typecast>()`. The configuration is **identical** for both Wormhole B0 and Blackhole:

| Address Mode | srca.incr | srcb.incr | dest.incr | Purpose |
|---|---|---|---|---|
| **ADDR_MOD_7** | 0 | 0 | 0 | No auto-increment. Used by `SFPLOAD`/`SFPSTORE` in kernels that manage addressing explicitly (fp32_to_int32, fp32_to_uint32), and by some SFPLOADMACRO calls that need a non-incrementing load (e.g., the first load in fp32_to_fp16b). |
| **ADDR_MOD_6** | 0 | 0 | 2 | Auto-increments DEST address by 2 after each SFPLOAD/SFPSTORE. Used by most SFPLOADMACRO-based kernels to advance through DEST rows. The increment of 2 (rather than 1) is because each SFPLOADMACRO typically processes a pair of operations per DEST row, and the even/odd alternation in the macro scheduling handles the intermediate rows. |

Note: The Blackhole SFPU kernel source references `ADDR_MOD_6` and `ADDR_MOD_7`, while the Wormhole source references `ADDR_MOD_2` and `ADDR_MOD_3` for the same purpose. Despite different index numbers, the configured values are the same (dest.incr=2 and dest.incr=0 respectively). The difference in indices is because Wormhole and Blackhole have different default usage patterns for ADDR_MOD slots -- Wormhole reserves ADDR_MOD_6/7 for other purposes while Blackhole reserves ADDR_MOD_2/3.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the typecast operation work in TTNN? What is the sharded typecast program factory and what kernels does it use?"
   **Reason**: Initial reconnaissance to understand the typecast operation architecture and factory selection logic.
   **Key Findings**: Four program factories exist; the sharded factory is an optimized path for L1-sharded inputs with matching tile sizes. Confirmed the SFPU-based typecast LLK functions.

2. **Query**: "How does the typecast_tile LLK function work? What does it do in the SFPU?"
   **Reason**: Understand the SFPU-level typecast implementation details.
   **Key Findings**: `typecast_tile` dispatches to `_calculate_typecast_` functions using SFPU instructions (SFPLOADMACRO, SFPCAST, SFPSHFT, SFPSTORE). Template parameters are IN_DTYPE and OUT_DTYPE as data format enum values. Operations are unrolled 8x for performance.

3. **Query**: "What does init_sfpu do? What does copy_tile do? How does reader_unary_sharded.cpp work?"
   **Reason**: Understand the compute kernel helper functions and the sharded reader pattern.
   **Key Findings**: `init_sfpu` configures unpacker/packer for CB data formats. `copy_tile` unpacks a tile from a CB into DEST registers. The sharded reader just does `cb_push_back` since data is already in L1.

4. [SFPU] **Query**: "What is SFPLOADMACRO? How does it work in the SFPU pipeline? What are macros, instruction templates, and how do they schedule operations across the Load, Simple, MAD, Round, and Store sub-units?"
   **Reason**: SFPLOADMACRO is the primary instruction used in most typecast SFPU kernels. Understanding its pipeline scheduling mechanism is critical to interpreting the throughput tables in the kernel comments.
   **Key Findings**: SFPLOADMACRO loads from DEST to an LREG and then schedules up to 4 additional instructions across the Simple, MAD, Round, and Store sub-units using pre-configured `LoadMacroConfig` templates (set via SFPCONFIG). Each macro invocation can achieve multi-operation throughput per cycle. Instruction templates and sequence bits control which operations run on which sub-units with configurable delays. Automatic stalling does NOT apply to SFPLOADMACRO-scheduled instructions -- software must manually ensure correct pipeline gaps.

### Documentation References

1. **Source**: `tt_metal/hw/inc/api/compute/eltwise_unary/typecast.h`
   **Reason**: Verify the typecast_tile API signature and supported type conversions.
   **Key Information**: Template parameters are `<uint32_t IN_DTYPE, uint32_t OUT_DTYPE>`, dispatches to `llk_math_eltwise_unary_sfpu_typecast`. Full conversion matrix documented in comments (20+ type pairs supported).

2. **Source**: `ttnn/cpp/ttnn/operations/copy/typecast/device/typecast_device_op.cpp`
   **Reason**: Understand factory selection logic and validation constraints.
   **Key Information**: `can_use_sharded_optimized_factory()` checks: input is sharded, tile sizes match, both L1, alignment OK, no sub_core_grids. Input/output memory layouts must match.

3. **Source**: `ttnn/cpp/ttnn/operations/copy/typecast/device/typecast_device_op_types.hpp`
   **Reason**: Identify TypecastParams fields that control operation behavior.
   **Key Information**: Params include `input_dtype`, `output_dtype`, `output_memory_config`, `fp32_dest_acc_en`, `preserve_fp32_precision`, `bfp8_pack_precise`, `sub_core_grids`.

### Confluence References

[SFPU] No Confluence page was consulted for this analysis. The SFPU instructions used in the typecast kernels were sufficiently documented through DeepWiki and the source code comments.

### Glean References

[SFPU] No Glean searches were performed for this analysis.
