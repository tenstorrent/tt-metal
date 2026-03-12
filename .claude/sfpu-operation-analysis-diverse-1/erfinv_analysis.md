# ERFINV Implementation Analysis

## Overview

ERFINV computes the element-wise inverse error function of each element in the input tensor. Given an input value `x` in the range (-1, 1), it returns the value `y` such that `erf(y) = x`. The implementation uses Winitzki's 2008 analytical approximation formula involving logarithms and square roots, executed entirely on the SFPU vector engine.

**Program factory path**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp`

## Path Selection: FPU vs SFPU

The unary program factory (`UnaryProgramFactory::create`) serves all unary element-wise operations through a single code path. The compute kernel is selected via `utils::get_compute_kernel_path(ops_chain[0].type(), input.dtype())`. For ERFINV, this function falls through to the `default` case (line 1009 of `unary_op_utils.cpp`) and returns `"eltwise_sfpu.cpp"` -- the generic SFPU compute kernel. Only a small number of operations (LGAMMA, MISH, TANHSHRINK, IDENTITY, WHERE_TSS, LOGIT, HARDSHRINK, HARDSWISH, LOGSIGMOID) have dedicated compute kernel files; all other unary operations, including ERFINV, use the shared SFPU dispatch kernel. The SFPU path is unconditionally selected for ERFINV regardless of data type or other parameters.

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Tile (32x32 elements) |
| **Unit size** | 1 tile |
| **Total units** | `num_pages` = total number of tiles in the input tensor |
| **Loop structure** | Outer loop over blocks (per_core_block_cnt tiles), inner loop over tiles within block (per_core_block_dim = 1), processing one tile at a time |

## Tensor Format and Layout

| Property | Input Tensor | Output Tensor |
|----------|--------------|---------------|
| **Logical shape** | Any shape (flattened to pages) | Same as input |
| **Dimension convention** | N/A (element-wise) | N/A (element-wise) |
| **Tensor layout** | TILE_LAYOUT (or ROW_MAJOR) | TILE_LAYOUT (or ROW_MAJOR) |
| **Memory layout** | INTERLEAVED | INTERLEAVED |
| **Buffer type** | DRAM (or L1) | DRAM (or L1) |
| **Data type** | BFLOAT16 / FLOAT32 | BFLOAT16 / FLOAT32 |

### Layout Transformations

No layout transformations (tilize/untilize) are performed. Input and output share the same layout. The factory supports both TILE and ROW_MAJOR layouts; page size is derived from tile size for TILE layout or buffer page size for ROW_MAJOR.

## Data Flow Pattern

1. **Reader** reads one tile from DRAM into CB c_0 (input staging).
2. **Compute** waits for a tile in CB c_0, copies it to DST register, executes the SFPU erfinv operation in-place on DST, packs the result into CB c_2 (output staging), then pops the input tile from CB c_0.
3. **Writer** waits for a tile in CB c_2, writes it to DRAM, then pops it from CB c_2.
4. Steps 1-3 repeat for all tiles assigned to the core, with double-buffering enabling overlap between reader and compute.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_input | Input staging | 2 tiles | 1 tile | Double | Reader | Compute | Program |
| c_2 | cb_output | Output staging | 2 tiles | 1 tile | Double | Compute | Writer | Program |

Note: CB c_1 (tmp0) is only allocated for HARDSHRINK and LOGIT operations. ERFINV does not use it.

## Pipeline Pattern Summary

Both CB c_0 and CB c_2 are double-buffered (capacity = 2 x block size). This allows the reader to fill one slot while the compute kernel processes from the other, and similarly the compute kernel can write to one output slot while the writer drains the other. This provides standard double-buffered pipeline overlap across all three kernels.

## Index Calculations

The reader and writer use `TensorAccessor` to map a linear page index to a physical DRAM address. The page index starts at `start_id` (a runtime argument computed from the core's position in the work distribution) and increments sequentially through `start_id + num_pages - 1`. The `TensorAccessor` handles bank-interleaved address translation internally, converting the logical page index to the appropriate DRAM bank and offset.

## Memory Access Patterns

### Read Pattern

Sequential tile reads. Each core reads a contiguous range of tile indices `[start_id, start_id + num_pages)`. Within the range, tiles are read one at a time in order via `noc_async_read_page`, with an `noc_async_read_barrier` after each tile to ensure completion before pushing to CB.

### Write Pattern

Sequential tile writes matching the read order. Each core writes tiles one at a time via `noc_async_write_page`, with `noc_async_writes_flushed` after each tile and a final `noc_async_write_barrier` after all tiles.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (linearized from 2D grid) |
| **Grid dimensions** | Up to `compute_with_storage_grid_size` (device-dependent) |
| **Total cores** | Determined by `split_work_to_cores` |
| **Work per core** | `num_pages_per_core_group_1` tiles for core_group_1; `num_pages_per_core_group_2` tiles for core_group_2 |
| **Load balancing** | Two-group split: group 1 gets `ceil(num_pages / num_cores)` tiles, group 2 gets `floor(num_pages / num_cores)` tiles (remainder handling) |

Cores are enumerated in column-major order: `core = {i / num_cores_y, i % num_cores_y}`.

## Arguments

### Compile-Time Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0+ | TensorAccessorArgs | uint32_t[] | Buffer type, page size, and bank mapping info for source tensor (appended by `TensorAccessorArgs(*src_buffer)`) |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | cb_id_out | uint32_t | Output circular buffer index (c_2 = 2) |
| 1+ | TensorAccessorArgs | uint32_t[] | Buffer type, page size, and bank mapping info for destination tensor |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_cnt | uint32_t | Number of tile blocks to process on this core |
| 1 | per_core_block_dim | uint32_t | Number of tiles per block (always 1 for ERFINV) |

### Runtime Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Source buffer DRAM address |
| 1 | num_pages | uint32_t | Number of pages (tiles) to read |
| 2 | start_id | uint32_t | Starting page index for this core |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Destination buffer DRAM address |
| 1 | num_pages | uint32_t | Number of pages (tiles) to write |
| 2 | start_id | uint32_t | Starting page index for this core |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | packed_scalar1 | uint32_t | Unused for ERFINV (always 0) |
| 1 | packed_scalar2 | uint32_t | Unused for ERFINV (always 0) |

## Kernel Implementations

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| Reader | BRISC (RISCV_0) | NOC0 | DRAM | CB c_0 | Sequential tile reads via TensorAccessor |
| Compute | TRISC (math RISCV) | N/A | CB c_0 | CB c_2 | SFPU erfinv: copy_tile, erfinv_tile, pack_tile |
| Writer | NCRISC (RISCV_1) | NOC1 | CB c_2 | DRAM | Sequential tile writes via TensorAccessor |

### Reader Kernel

| Property | Value |
|----------|-------|
| **File** | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp` |
| **Assigned cores** | all_cores (both core_group_1 and core_group_2) |

**Key Logic**:
- Reads `num_pages` tiles starting from `start_id`, one tile at a time
- For each tile: `cb_reserve_back(c_0, 1)` to wait for space, `noc_async_read_page` to initiate DMA, `noc_async_read_barrier` to wait for completion, `cb_push_back(c_0, 1)` to signal compute
- Supports optional `BACKWARDS` mode (compile-time define) for reverse iteration, not used in standard ERFINV
- Page size is derived from the CB interface at runtime, making the kernel layout-agnostic

### Compute Kernel

| Property | Value |
|----------|-------|
| **File** | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp` |
| **Assigned cores** | core_group_1 gets `num_pages_per_core_group_1` as `per_core_block_cnt`; core_group_2 gets `num_pages_per_core_group_2` |

**Key Logic**:
- Calls `init_sfpu(c_0, c_2)` once to initialize SFPU hardware with input/output CB configuration
- Outer loop iterates `per_core_block_cnt` times (one iteration per tile since `per_core_block_dim = 1`)
- Inner loop (single iteration): `tile_regs_acquire()`, `cb_wait_front(c_0, 1)`, `copy_tile(c_0, 0, 0)` to move tile from CB to DST register
- The `SFPU_OP_CHAIN_0` macro expands to `erfinv_tile_init(); erfinv_tile(0);` via compile-time defines set by `get_block_defines`
- `erfinv_tile(0)` invokes SFPU microcode that processes 8 iterations (32 rows of a 32x32 tile, 4 rows per SFPU vector operation)
- After SFPU execution: `tile_regs_commit()`, `tile_regs_wait()`, `pack_tile(0, c_2)` to write from DST to output CB
- `cb_pop_front(c_0, 1)` frees the input tile, `tile_regs_release()` releases DST
- `cb_push_back(c_2, per_core_block_dim)` signals writer (executed once per outer loop iteration)
- **Synchronization**: Waits on CB c_0 via `cb_wait_front`, pops from c_0 via `cb_pop_front`, reserves/pushes to CB c_2 via `cb_reserve_back`/`cb_push_back`

**SFPU Algorithm (erfinv)**:
- Based on Winitzki (2008) approximation: `erfinv(x) = sign(x) * sqrt(-2/(pi*a) - log(1-x^2)/2 + sqrt((2/(pi*a) + log(1-x^2)/2)^2 - log(1-x^2)/a))`
- Constant `a = 0.147`, giving `2/(pi*a) = 4.3307...` and `1/a = 6.8027...`
- Uses custom fast inverse square root (Quake III algorithm with magic number `0x5f37`) followed by two Newton-Raphson refinement iterations, then multiplies by the original value to get sqrt
- Calls `calculate_log_body` in FP32 mode to compute `log(1 - x^2)` avoiding intermediate rounding
- Edge cases: `|x| == 1` returns infinity, `|x| > 1` returns NaN, sign is restored via `setsgn`
- Initialization: `erfinv_init` calls `log_init` since the log function is used internally

### Writer Kernel

| Property | Value |
|----------|-------|
| **File** | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` |
| **Assigned cores** | all_cores (both core_group_1 and core_group_2) |

**Key Logic**:
- Writes `num_pages` tiles starting from `start_id`, one tile at a time
- For each tile: `cb_wait_front(c_2, 1)` to wait for compute output, `noc_async_write_page` to initiate DMA write, `noc_async_writes_flushed` to ensure write is dispatched, `cb_pop_front(c_2, 1)` to free the output slot
- Final `noc_async_write_barrier` ensures all writes complete before kernel exits
- Supports `OUT_SHARDED` mode (compile-time define) where writer simply waits for all pages in CB without writing to DRAM -- not used in standard interleaved path

## Implementation Notes

- **Program factory variants**: Two factories exist: `UnaryProgramFactory` (standard, uses `split_work_to_cores` for automatic grid sizing) and `UnarySubCoreGridProgramFactory` (uses a user-specified sub-core grid). Both follow the same SFPU kernel dispatch pattern. Factory selection depends on whether `sub_core_grids` is specified in the operation parameters.
- **Type-based operation variants**: ERFINV supports BFLOAT16 and FLOAT32 inputs. The data type affects CB data formats and compile-time defines (`INP_FLOAT32` vs `INP_FLOAT`). No type-specific code paths exist within the SFPU kernel itself; the SFPU operates in its native format.
- **UnpackToDestFP32 mode**: Enabled when `args.preserve_fp32_precision` is true. Sets `UnpackToDestMode::UnpackToDestFp32` for CB c_0 and c_1, causing the unpacker to convert input data to FP32 in the DST register before SFPU processing.
- **Broadcast type selection**: N/A -- ERFINV is a pure element-wise unary operation with no broadcasting.
- **Sharding support and constraints**: The standard `UnaryProgramFactory` operates on interleaved buffers. The writer kernel has an `OUT_SHARDED` compile-time path but it is not activated for ERFINV in this factory. Sharded unary operations are handled by a separate sharded program factory (not analyzed here).
- **FP32 dest accumulation**: Controlled by `args.fp32_dest_acc_en` and passed to `ComputeConfig`. When enabled, DST registers use FP32 precision. The SFPU erfinv kernel internally uses FP32 for the log computation (`calculate_log_body<false, false, true>`) regardless of this setting.

## SFPU Kernel Implementation

This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/erfinv.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h` (macro definitions) and `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (params dispatch) |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_erfinv.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |

### Call Chain

1. The compute kernel calls `erfinv_tile(0)` (from the API header `erfinv.h`).
2. `erfinv_tile` wraps via `SFPU_UNARY_NO_PARAM_KERNEL_FN(calculate_erfinv, RC, APPROX, idst)`, which expands to `_llk_math_eltwise_unary_sfpu_params_<APPROX>(ckernel::sfpu::calculate_erfinv<APPROX>, idst, (int)VectorMode::RC)`.
3. `_llk_math_eltwise_unary_sfpu_params_` (in `llk_math_eltwise_unary_sfpu_params.h`) sets the DST write address, switches to ADDR_MOD base 1 (mods 4-7), stalls until SFPU is ready, then loops over 4 faces (VectorMode::RC) calling `calculate_erfinv<false>()` once per face, advancing the DST address by 16 rows between faces via `TTI_SETRWC`.
4. `calculate_erfinv` (in `ckernel_sfpu_erfinv.h`) runs 8 inner iterations per face, each processing one 4-row SFPU vector from `dst_reg[0]`, computing the erfinv approximation, and writing back to `dst_reg[0]` with `dst_reg++` to advance to the next 4 rows.
5. The erfinv body calls `calculate_log_body<false, false, true>` (from `ckernel_sfpu_log.h`) for the natural logarithm, and `calculate_sqrt_custom<false>` (local helper) for two square root computations.

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` -- all 4 faces of the 32x32 tile are processed (face 0, 1, 2, 3).
- **Operation invocation**: The params dispatch loops 4 times (once per face). Each iteration calls `calculate_erfinv<false>()` which internally loops 8 times (8 iterations x 4 rows per SFPU vector = 32 rows per face). Total: 4 faces x 8 iterations = 32 SFPU vector operations per tile.
- **DEST address progression**: Between faces, the DST address is advanced by 16 rows via two `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` calls (each advancing by 8 rows). Within each face, `dst_reg++` in the SFPU kernel auto-increments the DEST read/write pointer by 1 (4 rows) after each of the 8 iterations.

### Annotated SFPU Kernel Source

This kernel uses SFPI abstractions (`sfpi::vFloat`, `sfpi::vInt`, `dst_reg`, `v_if`/`v_endif`), so Style A (inline-commented source) is used.

```cpp
// File: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_erfinv.h

template <bool APPROXIMATION_MODE>
sfpi_inline sfpi::vFloat calculate_sqrt_custom(sfpi::vFloat in) { // APPROXIMATION_MODE=false (always called with <false>)
    sfpi::vFloat val = in;
    sfpi::vFloat out;
    v_if(val != 0.0f) {
        // Fast inverse square root (Quake III algorithm) with magic constant 0x5f37 for bfloat16
        sfpi::vUInt magic = sfpi::reinterpret<sfpi::vUInt>(sfpi::vFloat(sfpi::s2vFloat16b(0x5f37)));
        sfpi::vFloat approx = sfpi::reinterpret<sfpi::vFloat>(magic - (sfpi::reinterpret<sfpi::vUInt>(val) >> 1)); // initial 1/sqrt(val) estimate
        sfpi::vFloat neg_half_val = val * -0.5f;
        approx = ((approx * approx) * neg_half_val + 1.5f) * approx; // Newton-Raphson iteration 1
        approx = ((approx * approx) * neg_half_val + 1.5f) * approx; // Newton-Raphson iteration 2
        out = approx * val; // convert 1/sqrt(val) to sqrt(val) by multiplying by val
    }
    v_else { out = val; } // sqrt(0) = 0
    v_endif;
    return out;
}

template <bool APPROXIMATION_MODE>
sfpi_inline sfpi::vFloat calculate_erfinv_body(sfpi::vFloat in) { // APPROXIMATION_MODE=true
    // Implementation notes, see the original file for more details

    // Compute log(1 - x^2)
    sfpi::vFloat log_value = in * in;
    log_value = 1 - log_value;
    log_value = calculate_log_body<false, false, true>(log_value, 0); // FAST_APPROX=false, HAS_BASE_SCALING=false, is_fp32_dest_acc_en=true

    sfpi::vFloat temp = log_value * 0.5;

    constexpr float TwoPiA = 4.330746750799873f;   // 2 / (pi * 0.147)
    constexpr float OneDivA = 6.802721088435375f;   // 1 / 0.147

    // tmp = -(2/(pi*a) + log(1-x^2)/2)
    temp = TwoPiA + temp;
    temp = -temp;

    // inner_sqrt = sqrt(tmp^2 - log(1-x^2)/a)
    sfpi::vFloat calculated_value = (temp * temp) - (log_value * OneDivA);
    sfpi::vFloat intermediate_result = calculate_sqrt_custom<false>(calculated_value);
    calculated_value = temp + intermediate_result;

    // result = sqrt(tmp + inner_sqrt)
    sfpi::vFloat result = calculate_sqrt_custom<false>(calculated_value);

    return result;
}

template <bool APPROXIMATION_MODE>
inline void calculate_erfinv() { // APPROXIMATION_MODE=false (APPROX=false for erfinv, set by get_op_approx_mode)
    constexpr int ITERATIONS = 8; // 8 iterations x 4 rows/vector = 32 rows per face
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0]; // load 4-element vector from current DST position
        sfpi::vFloat result;

        sfpi::vFloat abs_v = sfpi::abs(v); // exploit symmetry: erfinv(-x) = -erfinv(x)

        v_if(abs_v == 1.0f) { result = std::numeric_limits<float>::infinity(); } // erfinv(+/-1) = +/-inf
        v_elseif(abs_v > 1.0f) {
            result = std::numeric_limits<float>::quiet_NaN(); // domain error for |x| > 1
        }
        v_else { result = calculate_erfinv_body<true>(abs_v); } // main computation on |x|
        v_endif;

        result = sfpi::setsgn(result, v); // restore original sign from input

        sfpi::dst_reg[0] = result; // write back to DST
        sfpi::dst_reg++; // advance DST pointer by 4 rows
    }
}
```

**Dependency: `calculate_log_body` (from `ckernel_sfpu_log.h`)**

The erfinv kernel calls `calculate_log_body<false, false, true>` to compute the natural logarithm. With template arguments `FAST_APPROX=false`, `HAS_BASE_SCALING=false`, `is_fp32_dest_acc_en=true`, this function:

1. Normalizes the input to [1, 2) via `setexp(in, 127)`.
2. Evaluates a degree-5 minimax polynomial approximation of `log(x)` over [1, 2] using `PolynomialEvaluator::eval` with coefficients loaded into programmable constant registers (`vConstFloatPrgm1`, `vConstFloatPrgm2`) and literal constants.
3. Extracts the debiased exponent via `exexp(in)`, converts it to float via `int32_to_float`, and combines: `result = exponent * ln(2) + series_result`.
4. Handles edge cases: `ln(0) = -inf`, positive infinity input returns infinity, negative/NaN inputs return NaN.
5. Because `is_fp32_dest_acc_en=true`, the final fp16b truncation step is skipped, preserving full FP32 precision in the result.

### SFPU Instructions Used

The erfinv kernel uses SFPI abstractions that compile down to SFPU instructions. The following SFPU operations are invoked:

| Instruction/Intrinsic | Usage | Description |
|----------------------|-------|-------------|
| `SFPLOAD` (via `dst_reg[0]` read) | Load input vector | Loads a 4-element vector from the current DST register position into an SFPU LREG |
| `SFPSTORE` (via `dst_reg[0] = result`) | Store result vector | Stores a 4-element vector from an SFPU LREG back to the current DST register position |
| `SFPMUL` / `SFPMAD` (via `*`, `*` + `+`) | Arithmetic | Multiply and multiply-accumulate operations for polynomial evaluation, Newton-Raphson, and the Winitzki formula |
| `SFPADD` (via `+`, `-`) | Arithmetic | Addition and subtraction of vectors and scalars |
| `SFPNEG` (via unary `-`) | Negation | Negates a vector (used for `-temp`) |
| `SFPABS` (via `sfpi::abs()`) | Absolute value | Computes element-wise absolute value to exploit erfinv symmetry |
| `SFPSETSGN` (via `sfpi::setsgn()`) | Sign manipulation | Copies the sign bit from the original input to the result, restoring sign after computing on absolute values |
| `SFPSETEXP` (via `sfpi::setexp()`) | Exponent manipulation | Sets the exponent field of a float to a specified value; used in log to normalize input to [1, 2) range |
| `SFPEXEXP` (via `sfpi::exexp()`) | Exponent extraction | Extracts the debiased exponent from a float; used in log computation |
| `SFPINT32_TO_FLOAT` (via `sfpi::int32_to_float()`) | Type conversion | Converts integer exponent to floating point for the log correction term |
| `SFPSETCC` / `SFPENCC` (via `v_if`, `v_elseif`, `v_else`, `v_endif`) | Conditional execution | Per-lane predication for edge case handling (|x|==1, |x|>1, x==0) and sqrt zero guard |
| `SFPLOADI` (via `s2vFloat16b(0x5f37)`) | Immediate load | Loads the magic constant for fast inverse square root |
| `SFPSHFT` (via `>> 1`) | Bit shift | Right-shifts the integer reinterpretation of the float by 1 bit as part of the fast inverse sqrt initial estimate |
| `SFPIADD` (via integer `magic - (...)`) | Integer arithmetic | Integer subtraction for the fast inverse sqrt bit manipulation |
| `SFPLUT` (via `PolynomialEvaluator::eval`) | Polynomial evaluation | Evaluates the minimax polynomial for log approximation using Horner's method with multiply-accumulate chains |
| `SFPREINTERPRET` (via `sfpi::reinterpret<>()`) | Type reinterpretation | Reinterprets between vFloat and vUInt for bit-level manipulation in the fast inverse sqrt algorithm |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **DST register (DEST)** | Input tile data is loaded from DST and results are written back to DST. The SFPU reads/writes via `dst_reg[0]` which maps to the current DST row offset. Each face occupies 16 x 16 elements = 16 rows in DST address space. |
| **LREG[0-3]** | SFPU local registers used implicitly by SFPI compiler for intermediate values. The kernel uses many temporaries (`val`, `approx`, `neg_half_val`, `log_value`, `temp`, `calculated_value`, `intermediate_result`, `result`, `abs_v`) which the compiler maps to LREGs with spilling to DST as needed. |
| **vConstFloatPrgm0** | Programmed during `log_init` to `0.69314718246459961` (ln(2)). Used by `calculate_log_body` as the `vConstLn2` multiplier for exponent correction. |
| **vConstFloatPrgm1** | Programmed during `log_init` to `-2.0069785118103027`. Used as a coefficient in the minimax polynomial for log approximation. |
| **vConstFloatPrgm2** | Programmed during `log_init` to `3.767500400543213`. Used as a coefficient in the minimax polynomial for log approximation. |
| **CC (Condition Code) register** | Used implicitly by `v_if`/`v_elseif`/`v_else`/`v_endif` for per-lane predication. Controls which SIMD lanes execute edge case paths (|x|==1 -> inf, |x|>1 -> NaN) vs the main computation path, and guards the zero check in `calculate_sqrt_custom`. |

### Address Mode Configuration

ADDR_MOD_7 is configured during `_llk_math_eltwise_unary_sfpu_init_<SfpuType::erfinv>()` via `eltwise_unary_sfpu_configure_addrmod`:

```
ADDR_MOD_7: srca.incr = 0, srcb.incr = 0, dest.incr = 0
```

All increments are zero because the SFPU kernel manages DST address progression explicitly: within each face, `dst_reg++` advances the pointer by 4 rows (1 SFPU vector width), and between faces, the params dispatch issues `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` twice to advance by 16 rows total.

The `set_addr_mod_base()` call at the start of the params dispatch switches to the upper address mode bank (ADDR_MOD 4-7) to avoid conflicts with the A2D (Accumulate-to-DEST) pipeline which uses ADDR_MOD 0 and 2. After SFPU completion, `clear_addr_mod_base()` restores the lower bank.

ERFINV does not use ADDR_MOD_6 (only `topk_local_sort`, `typecast`, and min/max operations configure it). The configuration is identical between Wormhole B0 and Blackhole, as the `eltwise_unary_sfpu_configure_addrmod` function has the same default ADDR_MOD_7 setup on both architectures. The Blackhole params dispatch uses helper functions (`_llk_math_eltwise_unary_sfpu_start_`, `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_`, `_llk_math_eltwise_unary_sfpu_done_`) instead of inline `TTI_SETRWC` calls, but the net effect is identical: each face boundary advances DST by 16 rows via two `inc_dst_addr<8>()` calls.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the unary SFPU program factory work in ttnn? What kernels does it use for reader, compute, and writer? How are SFPU unary operations like erfinv dispatched?"
   **Reason**: Initial architectural understanding of the unary SFPU dispatch mechanism.
   **Key Findings**: Confirmed that the factory uses `reader_unary_interleaved_start_id.cpp`, `writer_unary_interleaved_start_id.cpp`, and `eltwise_sfpu.cpp`. Operations are dispatched via compile-time `SFPU_OP_CHAIN_0` macro. The `get_block_defines` utility generates the macro definitions, and `sfpu_split_includes.h` conditionally includes operation headers based on `SFPU_OP_ERFINV_INCLUDE`.

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Trace how ERFINV maps to its compute kernel path and macro definitions.
   **Key Information**: `get_compute_kernel_path` returns `"eltwise_sfpu.cpp"` for ERFINV (default case). `get_block_defines` generates `SFPU_OP_ERFINV_INCLUDE=1` and `SFPU_OP_CHAIN_0` expanding to `erfinv_tile_init(); erfinv_tile(0);`. `get_op_approx_mode` returns `false` for ERFINV.

2. **Source**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_erfinv.h`
   **Reason**: Understand the SFPU-level implementation of the erfinv algorithm.
   **Key Information**: Uses Winitzki's 2008 approximation with `a=0.147`. Processes 8 iterations per tile (4 rows per SFPU vector = 32 rows total). Depends on `calculate_log_body` (from `ckernel_sfpu_log.h`) and a custom fast sqrt using Quake III's inverse square root trick with Newton-Raphson refinement. Edge cases handle |x|=1 (infinity) and |x|>1 (NaN). Sign is preserved via `setsgn`.

3. **Source**: `tt_metal/hw/inc/api/compute/eltwise_unary/erfinv.h`
   **Reason**: Understand the API-level wrapper for the SFPU erfinv function.
   **Key Information**: `erfinv_tile` dispatches via `SFPU_UNARY_NO_PARAM_KERNEL_FN(calculate_erfinv, RC, APPROX, idst)`. `erfinv_tile_init` calls `SFPU_INIT_KERNEL_CALL(erfinv, sfpu::erfinv_init, APPROX)` which initializes the log subsystem.

4. [SFPU] **Source**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_erfinv.h` and `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_erfinv.h`
   **Reason**: Verify that Wormhole B0 and Blackhole implementations are identical.
   **Key Information**: Both files are byte-identical. The erfinv SFPU kernel has no architecture-specific differences.

5. [SFPU] **Source**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_log.h`
   **Reason**: Understand the `calculate_log_body` function called by erfinv with template args `<false, false, true>`.
   **Key Information**: Uses minimax polynomial approximation over [1, 2], exponent extraction via `exexp`, and programmable constants (`vConstFloatPrgm0/1/2`). With `is_fp32_dest_acc_en=true`, skips the final fp16b truncation, preserving FP32 precision.

6. [SFPU] **Source**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Understand the params dispatch layer that wraps the SFPU function call with DST address setup, stalling, and face iteration.
   **Key Information**: For VectorMode::RC, iterates 4 faces, calling the SFPU function once per face, advancing DST by 16 rows between faces via two `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` calls.

7. [SFPU] **Source**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: Understand ADDR_MOD configuration for erfinv and the init/start/done sequence.
   **Key Information**: `eltwise_unary_sfpu_configure_addrmod` sets ADDR_MOD_7 with all increments = 0 for the default SFPU case. `set_addr_mod_base()` switches to the upper addr mod bank (4-7) during SFPU execution.

### DeepWiki Queries

8. [SFPU] **Query**: "How is the erfinv SFPU kernel implemented? What is the call chain from compute kernel API through LLK to the ckernel SFPU implementation for erfinv? What files contain the erfinv SFPU kernel?" (asked to `tenstorrent/tt-metal`)
   **Reason**: Locate all files in the erfinv call chain and understand the dispatch mechanism.
   **Key Findings**: Confirmed the 3-layer call chain: API header -> LLK macro dispatch -> ckernel SFPU implementation. Identified `erfinv_init` calls `log_init` for programmable constant setup. Implementation is identical between Wormhole B0 and Blackhole.

9. [SFPU] **Query**: "How is the erfinv SFPU kernel implemented in the LLK layer? What is the ckernel_sfpu function for erfinv, and what SFPU instructions does it use?" (asked to `tenstorrent/tt-llk`)
   **Reason**: Get LLK-specific details on the SFPU dispatch and iteration pattern.
   **Key Findings**: Confirmed the standard LLK unary SFPU pattern: init -> start -> call sfpu function per face -> done. The erfinv function processes 8 iterations per face invocation.
