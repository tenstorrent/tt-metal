## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### Unary Dispatch Summary
- **UnaryOpType**: `CBRT`
- **Compute kernel**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`
- **SFPU_OP_CHAIN_0 expansion**: `cbrt_tile_init(); cbrt_tile(0);`

#### Approximation Mode Resolution
There are TWO independent controls for approximation. Both are reported below.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(CBRT)` in `unary_op_utils.cpp` -- falls through to `default: return false` |
| Template parameter (SFPU_OP_CHAIN) | none (non-parameterized) | `get_op_init_and_func_default()` -- `cbrt_tile_init()` / `cbrt_tile(idst)` with no explicit template args |
| Effective SFPU path | `APPROXIMATION_MODE=false`, takes the `!is_fp32_dest_acc_en` branch (fp16b output path) by default | `if constexpr (is_fp32_dest_acc_en)` branch in `calculate_cube_root` -- the `else` branch is taken for standard (non-fp32-accumulation) mode |

Note: `APPROX` in the API header `cbrt.h` maps directly to `math_approx_mode` from the compute config. Since `cbrt_tile_init()` has no template arguments, it uses the default `APPROX` value (`false`). The `DST_ACCUM_MODE` compile-time define controls `is_fp32_dest_acc_en`.

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/cbrt.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_cbrt.h` (WH and BH are identical) |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_cbrt.h` (WH and BH are identical) |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |

### Call Chain

1. **`cbrt_tile(idst)`** (API header `cbrt.h`) -- calls `llk_math_eltwise_unary_sfpu_cbrt<APPROX, DST_ACCUM_MODE>(idst)` inside the `MATH()` wrapper, which executes only on the math RISC.
2. **`llk_math_eltwise_unary_sfpu_cbrt<APPROXIMATE, fp32_dest_acc_en>(dst_index)`** (LLK dispatch) -- calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>()` passing `sfpu::calculate_cube_root<APPROXIMATE, fp32_dest_acc_en, ITERATIONS=8>` as the SFPU functor, `dst_index`, and `VectorMode::RC`.
3. **`_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(sfpu_func, dst_index, vector_mode)`** (parameters dispatch) -- sets DEST write address, stalls until SFPU is free, then iterates over 4 faces in `VectorMode::RC` mode, calling the SFPU functor once per face and advancing the DEST face address between faces via `SETRWC`/`inc_dst_addr<8>` x2.
4. **`sfpu::calculate_cube_root<APPROXIMATE, is_fp32_dest_acc_en, ITERATIONS=8>()`** (core SFPU) -- the innermost function, processes 8 iterations (one face) per call, computing cube root using a magic-constant initial estimate followed by polynomial refinement.

Similarly, `cbrt_tile_init()` calls `llk_math_eltwise_unary_sfpu_cbrt_init<APPROX>()`, which calls `llk_math_eltwise_unary_sfpu_init<SfpuType::cbrt, APPROXIMATE>(sfpu::cube_root_init<APPROXIMATE>)`. This initializes the SFPU config register, configures `ADDR_MOD_7`, resets counters, and then calls `cube_root_init()` to program the three polynomial coefficients into `vConstFloatPrgm{0,1,2}`.

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` (all 4 faces of the tile are processed).
- **Operation invocation**: The parameters dispatch calls `calculate_cube_root()` once per face (4 times total). Each invocation processes `ITERATIONS=8` sfpi rows (one complete 16x16 face = 256 elements).
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, `SETRWC`/`inc_dst_addr` between faces). On Wormhole, the params dispatch uses `TTI_SETRWC` with `CR_D, 8` twice to advance by 16 physical rows (= 1 face). On Blackhole, it calls `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` which calls `math::inc_dst_addr<8>()` twice, achieving the same effect.

### Annotated SFPU Kernel Source

The kernel uses SFPI abstractions (`sfpi::vFloat`, `sfpi::vInt`, `sfpi::dst_reg`, `sfpi::abs`, `sfpi::setsgn`, etc.), so Style A (inline-commented source) is used.

```cpp
// File: tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_cbrt.h

// Implementation notes, see the original file for more details
// Based on "Fast Calculation of Cube and Inverse Cube Roots Using a Magic
// Constant and Its Implementation on Microcontrollers" by Moroz et al.

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_cube_root() { // APPROXIMATION_MODE=false, is_fp32_dest_acc_en=false (default), ITERATIONS=8
    sfpi::vFloat negative_third_256 = -0x1.555556p-10f; // -1/3 / 256 = -(1/768), loaded via SFPLOADI pair

    // Magic constant 0x548c2b4b / 256 + 2^23
    sfpi::vFloat magic = 1418472267.0f / 256.0f + 8388608.0f; // precomputed additive constant for initial estimate

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) { // 8 iterations per face
        sfpi::vFloat a = sfpi::dst_reg[0]; // SFPLOAD from current DEST row pair (32 elements)
        sfpi::vFloat x = sfpi::abs(a);     // SFPABS -- take absolute value, preserve sign in 'a' for later

        // Implementation notes, see the original file for more details
        // Computes initial approximation y0 = reinterpret_as_float(0x548c2b4b - int_bits(|x|)/3)
        // using floating-point arithmetic to avoid integer division

        sfpi::vFloat f = sfpi::int32_to_float(sfpi::reinterpret<sfpi::vInt>(x), 0); // SFPCAST: reinterpret x bits as int, cast to float (round-to-nearest-even)

        f = f * negative_third_256 + magic; // SFPMAD: f = f * (-1/768) + (magic_constant/256 + 2^23)

        // Now, left-shift by 8 to restore integer result.
        sfpi::vFloat y = sfpi::reinterpret<sfpi::vFloat>(sfpi::reinterpret<sfpi::vInt>(f) << 8); // SFPSHFT: left-shift by 8 to undo the /256 scaling

        if constexpr (is_fp32_dest_acc_en) {
            // FP32 accumulation path: two-step Newton refinement for higher precision
            sfpi::vFloat c = (x * y) * (y * y);     // SFPMAD x2: c = x * y^3
            y = y * (c * (sfpi::vConstFloatPrgm2 * c + sfpi::vConstFloatPrgm1) + sfpi::vConstFloatPrgm0); // SFPMAD x3: first refinement step using polynomial in c

            sfpi::vFloat d = x * (y * y);            // SFPMAD: d = x * y^2
            c = d * y + sfpi::vConstNeg1;             // SFPMAD: c = d*y - 1 = x*y^3 - 1 (residual)
            sfpi::vFloat negative_third = sfpi::addexp(negative_third_256, 8); // SFPDIVP2: restore -1/3 from -1/768 by adding 8 to exponent
            sfpi::vFloat t = c * negative_third + sfpi::vConst1; // SFPMAD: t = 1 - (residual)/3 (second Newton correction factor)
            d = sfpi::setsgn(d, a);                   // SFPSETSGN: restore original sign from input
            y = d * (t * t);                          // SFPMAD x2: y = sign(a) * x*y^2 * t^2

            sfpi::dst_reg[0] = y;                     // SFPSTORE: write result back to DEST
        } else {
            // FP16B path (default): single-step polynomial refinement
            sfpi::vFloat d = x * (y * y);             // SFPMAD x2: d = x * y^2 (should be close to cbrt(x)^2 * cbrt(x)^{-2} * x ...)
            sfpi::vFloat c = d * y;                   // SFPMAD: c = x * y^3 (should be close to 1.0)
            sfpi::vFloat t = c * (sfpi::vConstFloatPrgm2 * c + sfpi::vConstFloatPrgm1) + sfpi::vConstFloatPrgm0; // SFPMAD x3: polynomial correction t = P(c)
            d = sfpi::setsgn(d, a);                   // SFPSETSGN: restore original sign from input
            y = d * (t * t);                          // SFPMAD x2: y = sign(a) * d * t^2

            sfpi::dst_reg[0] = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(y, 0)); // SFP_STOCH_RND: convert fp32 to fp16b (round-to-nearest-even), then SFPSTORE
        }
        sfpi::dst_reg++;                              // advance to next sfpi row (2 physical DEST rows = 32 elements)
    }
}

template <bool APPROXIMATION_MODE>
inline void cube_root_init() { // APPROXIMATION_MODE=false
    sfpi::vConstFloatPrgm0 = 0x1.c09806p0f;  // SFPCONFIG: ~1.7523 -- constant term of refinement polynomial
    sfpi::vConstFloatPrgm1 = -0x1.403e6cp0f;  // SFPCONFIG: ~-1.2509 -- linear coefficient of refinement polynomial
    sfpi::vConstFloatPrgm2 = 0x1.04cdb2p-1f;  // SFPCONFIG: ~0.5094 -- quadratic coefficient of refinement polynomial
}
```

### SFPU Instructions Used

| SFPU Instruction | SFPI Abstraction | Description | Count (fp16b path) |
|------------------|-----------------|-------------|---------------------|
| **SFPLOAD** | `dst_reg[0]` (read) | Load 32 elements from current DEST row pair into LREG | 1 |
| **SFPABS** | `sfpi::abs(a)` | Compute absolute value of input (clear sign bit) | 1 |
| **SFPCAST** | `sfpi::int32_to_float(...)` | Reinterpret FP32 bit pattern as INT32, then cast to FP32 (INT32->FP32 conversion with round-to-nearest-even) | 1 |
| **SFPMAD** | `*`, `* +` operators on `vFloat` | Fused multiply-add: used for the initial estimate computation (`f * neg_third_256 + magic`), polynomial evaluation, and all float multiplications/additions | ~8 (per iteration in fp16b path) |
| **SFPSHFT** | `vInt << 8` | Left-shift integer by 8 bits to undo the /256 scaling of the magic constant trick | 1 |
| **SFPSETSGN** | `sfpi::setsgn(d, a)` | Copy sign from original input `a` onto magnitude result `d` to handle negative inputs | 1 |
| **SFP_STOCH_RND** | `sfpi::float_to_fp16b(y, 0)` | Convert FP32 to FP16B with round-to-nearest-even (mode 0) -- only in fp16b path | 1 (fp16b only) |
| **SFPSTORE** | `dst_reg[0] = ...` (write) | Store result back to DEST row pair | 1 |
| **SFPDIVP2** | `sfpi::addexp(neg_third_256, 8)` | Add 8 to the exponent of `negative_third_256` to recover `-1/3` from `-1/768` -- only in fp32 path | 1 (fp32 only) |
| **SFPLOADI** | implicit from `vFloat` literal assignments | Load 16-bit immediate halves to construct float constants in LREGs | multiple (for `negative_third_256`, `magic`) |
| **SFPCONFIG** | `vConstFloatPrgm{0,1,2} = ...` in init | Program the three polynomial coefficients into the SFPU constant registers | 3 (init only) |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **DEST row pairs** | Input tile data is read via `dst_reg[0]` (SFPLOAD) and results written back via `dst_reg[0] = ...` (SFPSTORE). The `dst_reg++` advances through the face, covering 8 sfpi rows = 16 physical rows = 1 face. |
| **LREG0-LREG3** | Used as scratch registers by the compiler to hold intermediate values: `a` (original input), `x` (absolute value), `f` (cast-to-float intermediate), `y` (initial estimate and refined result), `d`, `c`, `t` (polynomial evaluation temporaries). The SFPI compiler allocates these automatically. |
| **vConstFloatPrgm0** | Polynomial constant: `0x1.c09806p0f` (~1.7523). This is the constant term of the refinement polynomial P(c). Programmed via SFPCONFIG during init. |
| **vConstFloatPrgm1** | Polynomial constant: `-0x1.403e6cp0f` (~-1.2509). Linear coefficient of P(c). Programmed via SFPCONFIG during init. |
| **vConstFloatPrgm2** | Polynomial constant: `0x1.04cdb2p-1f` (~0.5094). Quadratic coefficient of P(c). Programmed via SFPCONFIG during init. |
| **vConstNeg1** | Fixed constant register: `-1.0`. Used in fp32 path to compute residual `c = d*y - 1`. |
| **vConst1** | Fixed constant register: `1.0`. Used in fp32 path for Newton correction `t = 1 + c*(-1/3)`. |

### Address Mode Configuration

The address mode for CBRT is configured by `eltwise_unary_sfpu_configure_addrmod<SfpuType::cbrt>()`, which is called during `_llk_math_eltwise_unary_sfpu_init_<SfpuType::cbrt>()`.

Since `SfpuType::cbrt` does not match any special-case `if constexpr` branches (those handle `topk_local_sort`, `typecast`, `unary_max/min`, etc.), only the default configuration applies:

| Configuration | ADDR_MOD slot | srca.incr | srcb.incr | dest.incr |
|---------------|--------------|-----------|-----------|-----------|
| Default (all ops) | `ADDR_MOD_7` | 0 | 0 | 0 |

This is identical on both Wormhole B0 and Blackhole. The `ADDR_MOD_7` with all-zero increments means the SFPU hardware does not auto-increment the DEST address between SFPU instructions -- instead, the within-face progression is handled by the SFPI abstraction's `dst_reg++` (which emits `SETRWC` internally to advance by one sfpi row = 2 physical DEST rows), and the between-face progression is handled by the parameters dispatch via `inc_dst_addr<8>()` x2 (= advance by 16 physical rows = 1 face).

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Determine compute kernel path, SFPU_OP_CHAIN_0 expansion, and approximation mode for CBRT
   **Key Findings**: CBRT uses `eltwise_sfpu.cpp`, expands to `cbrt_tile_init()` / `cbrt_tile(idst)`, `get_op_approx_mode()` returns `false` for all ops (default case), include guard is `SFPU_OP_CBRT_INCLUDE`

2. **File**: `tt_metal/hw/inc/api/compute/eltwise_unary/cbrt.h`
   **Reason**: API header exposing `cbrt_tile()` and `cbrt_tile_init()`
   **Key Findings**: `cbrt_tile(idst)` calls `llk_math_eltwise_unary_sfpu_cbrt<APPROX, DST_ACCUM_MODE>(idst)` inside `MATH()` wrapper

3. **File**: `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_cbrt.h`
   **Reason**: LLK dispatch layer bridging API to core SFPU function
   **Key Findings**: WH and BH are identical. Calls `_llk_math_eltwise_unary_sfpu_params_` with `sfpu::calculate_cube_root<APPROXIMATE, fp32_dest_acc_en, ITERATIONS=8>` and `VectorMode::RC`. Init calls `sfpu::cube_root_init<APPROXIMATE>`

4. **File**: `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_cbrt.h`
   **Reason**: Core SFPU implementation of cube root
   **Key Findings**: WH and BH implementations are identical. Uses magic constant method from Moroz et al. paper. Two code paths: fp32 (two-step Newton refinement with SFPDIVP2 for higher precision) and fp16b (single polynomial refinement + SFP_STOCH_RND for output conversion). No condition code manipulation (no v_if/v_else). Fully SFPI-based (Style A kernel).

5. **File**: `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Parameters dispatch function that orchestrates face iteration and DEST addressing
   **Key Findings**: WH version uses `TTI_SETRWC` directly; BH version uses helper functions (`_llk_math_eltwise_unary_sfpu_start_`, `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_`, `_llk_math_eltwise_unary_sfpu_done_`). Both iterate 4 faces in VectorMode::RC, calling the SFPU functor once per face.

6. **File**: `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: Address mode configuration and SFPU init/start/done functions
   **Key Findings**: `SfpuType::cbrt` uses only the default `ADDR_MOD_7` (all increments = 0). WH and BH address mode configs are identical for cbrt.

7. **File**: `runtime/sfpi/include/sfpi_lib.h`
   **Reason**: Map SFPI library functions to underlying SFPU instructions
   **Key Findings**: `int32_to_float` -> `SFPCAST`, `abs` -> `SFPABS`, `setsgn(v, sgn)` -> `SFPSETSGN_V` (copies sign), `addexp` -> `SFPDIVP2` (add to exponent), `float_to_fp16b` -> `SFP_STOCH_RND` (FP32->FP16B conversion)

8. **File**: `runtime/sfpi/include/sfpi.h`
   **Reason**: Map SFPI operator overloads to underlying SFPU instructions
   **Key Findings**: `vInt << N` -> `SFPSHFT_I` (integer left shift by immediate), `vFloat * vFloat` -> `SFPMAD` (multiply-add with zero addend), `vFloat * vFloat + vFloat` -> `SFPMAD` (fused multiply-add)

9. **File**: `.claude/references/sfpu-hardware-model.md`
   **Reason**: Authoritative reference for SFPU architecture, instruction semantics, tile/face geometry
   **Key Findings**: Confirmed stride-2 addressing model, 8 iterations per face, 32 elements per iteration, SFPMAD as the universal arithmetic instruction for float operations
