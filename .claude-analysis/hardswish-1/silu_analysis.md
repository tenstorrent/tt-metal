## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### Unary Dispatch Summary
- **UnaryOpType**: `SILU`
- **Compute kernel**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`
- **SFPU_OP_CHAIN_0 expansion**: `silu_tile(0)`

#### Approximation Mode Resolution
There are TWO independent controls for approximation.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(SILU)` in `unary_op_utils.cpp` -- falls through to `default: return false` |
| Template parameter (SFPU_OP_CHAIN) | none (non-parameterized) | `get_op_init_and_func_default()` returns `"silu_tile_init();"` and `"silu_tile({idst});"` -- no template arguments in the macro strings |
| Effective SFPU path | `APPROX=false`; WH init sets reciprocal polynomial coefficients (k0, k1, k2); BH init sets `vConstFloatPrgm0 = 2.0f` for Newton-Raphson. Compute path uses `_sfpu_sigmoid_<is_fp32_dest_acc_en>` which selects exp precision based on dest accumulation mode, and reciprocal with 1 or 2 Newton-Raphson iterations. | `silu_init<APPROX>()` in `ckernel_sfpu_silu.h`; `_sfpu_sigmoid_` in `ckernel_sfpu_sigmoid.h` |

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/compute_kernel_api.h` (lines 463-465) |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_silu.h` (build-generated from tt_llk) |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_silu.h` (build-generated); original source in `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/common/inc/sfpu/ckernel_sfpu_silu.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |

**Important note on source discrepancy**: The tt_llk source files (`tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_silu.h` and `tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_silu.h`) contain a different implementation that uses `_sigmoid_piecewise_linear_positive_()` (a polynomial approximation). However, the **build-generated** versions in `build_Debug/libexec/tt-metalium/tt_metal/hw/ckernels/*/metal/llk_api/llk_sfpu/ckernel_sfpu_silu.h` use a different, higher-accuracy implementation based on `_sfpu_sigmoid_()` (which computes `1 / (1 + exp(-x))`). The build-generated versions are what actually run on hardware. This analysis documents the **build-generated** (deployed) implementation.

### Call Chain

1. `silu_tile(idst)` in `compute_kernel_api.h` calls `llk_math_eltwise_unary_sfpu_silu<APPROX, DST_ACCUM_MODE>(idst)`.
2. `llk_math_eltwise_unary_sfpu_silu()` in `llk_math_eltwise_unary_sfpu_silu.h` calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::calculate_silu<is_fp32_dest_acc_en, 8>, dst_index, vector_mode)`.
3. `_llk_math_eltwise_unary_sfpu_params_()` in `llk_math_eltwise_unary_sfpu_params.h` sets up the DEST address, stalls for SFPU readiness, then invokes `calculate_silu<is_fp32_dest_acc_en, 8>()` once per face (4 faces for VectorMode::RC), with SETRWC between faces.
4. `calculate_silu()` in `ckernel_sfpu_silu.h` iterates 8 times per face, each iteration loading from `dst_reg[0]`, computing `x * _sfpu_sigmoid_<is_fp32_dest_acc_en>(x)`, optionally rounding to bfloat16, and storing back.
5. `_sfpu_sigmoid_()` in `ckernel_sfpu_sigmoid.h` computes `1 / (1 + exp(-x))` by calling `_sfpu_exp_improved_()` or `_sfpu_exp_21f_()` for the exponential, then `_sfpu_reciprocal_()` for the division.

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` (default). All 4 faces of the tile are processed.
- **Operation invocation**: `calculate_silu<is_fp32_dest_acc_en, 8>()` is called once per face. Each call iterates 8 times (ITERATIONS=8), processing 32 elements per iteration (2 physical DEST rows). Total: 4 faces x 8 iterations x 32 elements = 1024 elements = full tile.
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, SETRWC between faces). On Wormhole, `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` is called twice between faces (advancing by 16 physical rows = 1 face). On Blackhole, `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` calls `math::inc_dst_addr<8>()` twice for the same effect. Within a face, `dst_reg++` advances 1 sfpi row = 2 physical DEST rows per iteration.

### Annotated SFPU Kernel Source

The kernel uses SFPI abstractions (`sfpi::vFloat`, `sfpi::dst_reg`, `v_if`/`v_endif`), so Style A (inline-commented source) is used.

**Primary kernel: `calculate_silu` (Wormhole B0 / Blackhole -- identical)**:

```cpp
// File: tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_silu.h

#include "ckernel_sfpu_sigmoid.h"

namespace ckernel::sfpu {

template <bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_silu() { // is_fp32_dest_acc_en depends on DST_ACCUM_MODE, ITERATIONS=8
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0]; // SFPLOAD: load 32 elements from current DEST position

        // silu(x) = x * sigmoid(x)
        sfpi::vFloat result = x * _sfpu_sigmoid_<is_fp32_dest_acc_en>(x); // SFPMAD/SFPMUL: multiply x by sigmoid(x)

        // Round to bfloat16 if not in fp32 accumulation mode
        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0)); // SFP_STOCH_RND: round-to-nearest-even to bf16
        }

        sfpi::dst_reg[0] = result; // SFPSTORE: write result back to DEST
        sfpi::dst_reg++; // advance to next sfpi row (2 physical DEST rows)
    }
}

template <bool APPROXIMATION_MODE>
inline void silu_init() { // APPROXIMATION_MODE=false
    if constexpr (!APPROXIMATION_MODE) {
        _init_sfpu_reciprocal_<false>(); // WH: sets vConstFloatPrgm0/1/2 for quadratic estimate; BH: sets vConstFloatPrgm0=2.0
    } else {
        _init_sfpu_reciprocal_<true>(); // same on WH; BH: empty body
    }
}

}  // namespace ckernel::sfpu
```

**Helper function: `_sfpu_sigmoid_` (Wormhole B0 / Blackhole -- identical)**:

```cpp
// File: tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_sigmoid.h

template <bool is_fp32_acc_to_dest_mode = true>
sfpi_inline sfpi::vFloat _sfpu_sigmoid_(sfpi::vFloat x) {
    // Compute sigmoid as: sigmoid(x) = 1 / (1 + exp(-x))

    sfpi::vFloat exp_neg_x;
    if constexpr (is_fp32_acc_to_dest_mode) {
        exp_neg_x = _sfpu_exp_improved_<true>(-x); // FP32 path: Cody-Waite range reduction + 7th-order Taylor
    } else {
        exp_neg_x = _sfpu_exp_21f_<true>(-x); // BF16 path: exp_21f algorithm (~1 ULP on bfloat16)
    }

    sfpi::vFloat denominator = sfpi::vConst1 + exp_neg_x; // SFPMAD: 1.0 + exp(-x)

    sfpi::vFloat result;
    if constexpr (is_fp32_acc_to_dest_mode) {
        result = _sfpu_reciprocal_<2>(denominator); // 2 Newton-Raphson iterations for FP32 precision
    } else {
        result = _sfpu_reciprocal_<1>(denominator); // 1 Newton-Raphson iteration for BF16 precision
    }

    return result;
}
```

**Helper function: `_sfpu_reciprocal_` (Wormhole B0)**:

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_recip.h

template <int max_iter = 2>
sfpi_inline sfpi::vFloat _sfpu_reciprocal_(const sfpi::vFloat in) { // max_iter=2 for fp32, max_iter=1 for bf16
    // Scale input to [1.0, 2.0) and negate
    sfpi::vFloat negative_x = sfpi::setman(sfpi::vConstNeg1, sfpi::reinterpret<sfpi::vInt>(in)); // SFPSETMAN: combine sign/exp of -1.0 with mantissa of in

    // Quadratic initial estimate: y = k2 - k1*x + k0*x^2
    sfpi::vFloat y = sfpi::vConstFloatPrgm1 + sfpi::vConstFloatPrgm0 * negative_x; // SFPMAD: k1 + k0*(-x)

    // Scale factor: 255 - in.Exp, computed via bitwise NOT
    sfpi::vInt scale_bits = ~sfpi::reinterpret<sfpi::vUInt>(in); // SFPNOT: bitwise complement

    // Continue quadratic estimate
    y = sfpi::vConstFloatPrgm2 + y * negative_x; // SFPMAD: k2 + y*(-x)

    // Clear mantissa of scale factor
    sfpi::vFloat scale = sfpi::setman(sfpi::reinterpret<sfpi::vFloat>(scale_bits), 0); // SFPSETMAN: zero mantissa

    // First Newton-Raphson iteration: t = 1.0 - x*y
    sfpi::vFloat t = sfpi::vConst1 + negative_x * y; // SFPMAD: 1.0 + (-x)*y

    // Scale factor adjustment: handle inf/zero edge cases
    scale *= 0.5f; // SFPMUL: halve scale

    // Continue NR: y = y + y*t
    y = y + y * t; // SFPMAD: y*t + y

    if constexpr (max_iter > 1) { // FP32 path: second NR iteration
        t = sfpi::vConst1 + negative_x * y; // SFPMAD
        y = y + y * t; // SFPMAD
    }

    // Apply scale and restore sign
    y = y * scale; // SFPMUL: apply scaling
    y = sfpi::setsgn(y, in); // SFPSETSGN: restore original sign

    return y;
}
```

**Helper function: `_sfpu_reciprocal_` (Blackhole)**:

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_recip.h

template <int max_iter = 2>
sfpi_inline sfpi::vFloat _sfpu_reciprocal_(const sfpi::vFloat x) { // max_iter=2 for fp32, max_iter=1 for bf16
    // Hardware-accelerated initial approximation
    sfpi::vFloat y = sfpi::approx_recip(x); // SFPARECIP: ~7-bit initial estimate

    if constexpr (max_iter > 0) {
        // t = x*y - 2.0  (negated NR error term for NaN detection via sign check)
        sfpi::vFloat t = x * y - sfpi::vConstFloatPrgm0; // SFPMAD: x*y - 2.0, where vConstFloatPrgm0=2.0

        if constexpr (max_iter > 1) { // FP32 path: two NR iterations
            sfpi::vFloat y1 = y * -t - sfpi::vConst0; // SFPMAD: y*(-t) - 0.0
            v_if (t < 0) { // CC guard: skip if t=NaN (NaN >= 0)
                t = x * y1 - sfpi::vConstFloatPrgm0; // SFPMAD: second NR error
                y = y1 * -t - sfpi::vConst0; // SFPMAD: second NR correction
            }
            v_endif;
        } else { // BF16 path: one NR iteration
            v_if (t < 0) { // CC guard: skip if t=NaN
                y = y * -t - sfpi::vConst0; // SFPMAD: NR correction
            }
            v_endif;
        }
    }

    return y;
}
```

**Helper function: `_sfpu_exp_21f_` (used in BF16 path, Wormhole B0 / Blackhole -- identical)**:

```cpp
// File: tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_exp.h

template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _sfpu_exp_21f_(sfpi::vFloat val) {
    // Implementation notes, see the original file for more details
    constexpr float ONE_LN2 = 1.4426950216293334961f;
    sfpi::vFloat xlog2 = (val * ONE_LN2 + 127.f); // SFPMAD: scale to base-2 and add IEEE754 bias

    // Clamp to [0, 255] to prevent overflow
    sfpi::vFloat threshold_low = 0.f; // SFPLOADI
    sfpi::vFloat threshold_high = sfpi::vFloat(255.f); // SFPLOADI
    sfpi::vec_min_max(threshold_low, xlog2); // SFPSWAP: min/max pair
    sfpi::vec_min_max(xlog2, threshold_high); // SFPSWAP: min/max pair

    sfpi::vInt z = _float_to_int32_for_exp21f_(xlog2); // SFPEXEXP + SFPEXMAN + SFPSHFT: float-to-int conversion

    sfpi::vInt exponential_part = exexp_nodebias(sfpi::reinterpret<sfpi::vFloat>(z)); // SFPEXEXP: extract exponent (2^integer_part)
    sfpi::vInt fractional_part = sfpi::exman9(sfpi::reinterpret<sfpi::vFloat>(z)); // SFPEXMAN: extract mantissa (fractional part)

    sfpi::vFloat frac = sfpi::int32_to_float(fractional_part, 0); // SFPCAST: int32 -> float32

    // 2nd degree polynomial for 2^frac approximation
    frac = PolynomialEvaluator::eval(frac, 1.0017248f, 7.839635491371155e-08f, 4.791750143340323e-15f); // chain of SFPMAD (Horner's method)

    // Recombine: 2^int_part * 2^frac_part
    sfpi::vFloat y = sfpi::setexp(frac, exponential_part); // SFPSETEXP: set exponent field

    if constexpr (!is_fp32_dest_acc_en) {
        y = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(y, 0)); // SFP_STOCH_RND: round to bf16
    }

    return y;
}
```

**Helper function: `_sfpu_exp_f32_accurate_` (used in FP32 path via `_sfpu_exp_improved_<true>`)**:

```cpp
// File: tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_exp.h

sfpi_inline sfpi::vFloat _sfpu_exp_f32_accurate_(sfpi::vFloat val) {
    sfpi::vFloat result = sfpi::vConst0;

    // Implementation notes, see the original file for more details
    constexpr float OVERFLOW_THRESHOLD = 128.0f;
    constexpr float UNDERFLOW_THRESHOLD = -127.0f;

    constexpr float INV_LN2 = 1.4426950408889634f;
    sfpi::vFloat z = val * INV_LN2; // SFPMAD: x / ln(2)

    sfpi::vInt exp_bits = sfpi::exexp(z); // SFPEXEXP: extract exponent for special case check

    v_if(z >= OVERFLOW_THRESHOLD) { // CC: overflow check
        result = std::numeric_limits<float>::infinity(); // SFPLOADI: +inf
    }
    v_elseif(z <= UNDERFLOW_THRESHOLD) { // CC: underflow check
        result = sfpi::vConst0; // 0.0
    }
    v_elseif(exp_bits == 255) { // CC: NaN check
        result = std::numeric_limits<float>::quiet_NaN(); // SFPLOADI: NaN
    }
    v_else {
        sfpi::vInt k_int;
        sfpi::vFloat k = _sfpu_round_nearest_int32_(z, k_int); // round z to nearest int

        // Cody-Waite range reduction: r = x - k*ln2_hi - k*ln2_lo
        constexpr float LN2_HI = -0.6931152343750000f;
        constexpr float LN2_LO = -3.19461832987e-05f;
        sfpi::vFloat r_hi = k * LN2_HI + val; // SFPMAD
        sfpi::vFloat r = k * LN2_LO + r_hi; // SFPMAD

        // 7th-order Taylor polynomial for exp(r) on the reduced range
        sfpi::vFloat p = PolynomialEvaluator::eval(
            r,
            sfpi::vConst1, sfpi::vConst1, 0.5f, 1.0f / 6.0f,
            1.0f / 24.0f, 1.0f / 120.0f, 1.0f / 720.0f, 1.0f / 5040.0f
        ); // chain of 7 SFPMAD instructions (Horner's method)

        // Scale by 2^k via exponent manipulation
        sfpi::vInt p_exp = sfpi::exexp_nodebias(p); // SFPEXEXP
        sfpi::vInt new_exp = p_exp + k_int; // SFPIADD
        result = sfpi::setexp(p, new_exp); // SFPSETEXP
    }
    v_endif;

    return result;
}
```

### SFPU Instructions Used

The silu kernel composes several sub-operations. The instructions listed here are those emitted by the full `calculate_silu` -> `_sfpu_sigmoid_` -> `_sfpu_exp_*` / `_sfpu_reciprocal_` call tree.

| Instruction | Description | Used in |
|-------------|-------------|---------|
| `SFPLOAD` | Load 32 elements from DEST into LREG | `calculate_silu`: load `x` from `dst_reg[0]` |
| `SFPSTORE` | Store LREG contents back to DEST | `calculate_silu`: store final result to `dst_reg[0]` |
| `SFPMAD` | Fused multiply-add: `VD = VA * VB + VC` | Pervasive: all additions (1+exp(-x)), all multiplications (x*sigmoid), polynomial evaluations, Newton-Raphson iterations |
| `SFPMUL` | Multiply with immediate | `_sfpu_reciprocal_` (WH): `scale *= 0.5f` |
| `SFPLOADI` | Load 16-bit immediate to LREG | Loading float constants (thresholds, polynomial coefficients, inf, NaN) |
| `SFPABS` | Absolute value | Not directly used in build version (only in tt_llk source version) |
| `SFPSETCC` | Set condition code from LREG comparison | Emitted by `v_if (val < 0.0f)`, `v_if (z >= ...)`, etc. |
| `SFPCOMPC` | Complement condition code (else branch) | Emitted by `v_elseif` / `v_else` constructs |
| `SFPENCC` | Enable/disable condition code | Emitted by `v_if` (enable) and `v_endif` (disable) |
| `SFPPUSHC` | Push CC state onto stack | Emitted by nested `v_if` / `v_elseif` for CC stack management |
| `SFPPOPC` | Pop CC state from stack | Emitted by `v_endif` to restore CC state |
| `SFPNOT` | Bitwise NOT | `_sfpu_reciprocal_` (WH): `~in` for scale factor computation |
| `SFPSETMAN` | Set mantissa field | `_sfpu_reciprocal_` (WH): scale input to [1,2), clear scale mantissa |
| `SFPSETEXP` | Set exponent field | `_sfpu_exp_21f_` and `_sfpu_exp_f32_accurate_`: recombine `2^k * frac` |
| `SFPSETSGN` | Set sign field | `_sfpu_reciprocal_` (WH): restore original sign of result |
| `SFPEXEXP` | Extract exponent | `_sfpu_exp_21f_`: extract integer part; `_sfpu_exp_f32_accurate_`: special case detection |
| `SFPEXMAN` | Extract mantissa (8 or 9 bit) | `_sfpu_exp_21f_`: extract fractional part |
| `SFPSHFT` | Shift operations | `_float_to_int32_for_exp21f_`: shift mantissa by exponent |
| `SFPSWAP` | Conditional min/max swap | `_sfpu_exp_21f_`: clamp `xlog2` via `vec_min_max` |
| `SFPIADD` | Integer add | `_sfpu_exp_f32_accurate_`: `p_exp + k_int` for exponent scaling |
| `SFPCAST` | Format conversion | `_sfpu_exp_21f_`: `int32_to_float` for fractional part |
| `SFP_STOCH_RND` | Stochastic/deterministic rounding | `calculate_silu` and `_sfpu_exp_21f_`: `float_to_fp16b` for bf16 rounding |
| `SFPARECIP` | Hardware approximate reciprocal (BH only) | `_sfpu_reciprocal_` (BH): initial ~7-bit estimate via `approx_recip(x)` |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **DEST (via dst_reg)** | Source and destination for tile data. Each iteration processes 32 elements (2 physical rows x 16 cols). `dst_reg[0]` reads/writes the current position; `dst_reg++` advances. |
| **LREG0-LREG3** | Used as temporaries throughout the computation. In the SFPI model, `vFloat` variables map to LREGs dynamically. Key allocations: `x` (input value), `exp_neg_x` (exponential), `denominator`, `result` (sigmoid output), `y` (reciprocal estimate), `t` (NR error term), `scale` (reciprocal scaling). |
| **vConstFloatPrgm0** | WH: Reciprocal polynomial coefficient k0 = 0.3232325 (set by `_init_sfpu_reciprocal_`). BH: Newton-Raphson constant 2.0 (set by `_init_sfpu_reciprocal_`). |
| **vConstFloatPrgm1** | WH: Reciprocal polynomial coefficient k1 = 1.4545460 (set by `_init_sfpu_reciprocal_`). BH: Not set by silu init. |
| **vConstFloatPrgm2** | WH: Reciprocal polynomial coefficient k2 = 2.1212125 (set by `_init_sfpu_reciprocal_`). BH: Not set by silu init. |
| **vConst0** | Hardware constant 0.0f. Used as addend in SFPMAD for pure multiply. |
| **vConst1** | Hardware constant 1.0f. Used in `1.0 + exp(-x)` and NR iteration `1.0 + (-x)*y`. |
| **vConstNeg1** | Hardware constant -1.0f. Used by WH reciprocal to create negative scaled input. |

### Address Mode Configuration

For `SfpuType::silu`, the address mode configuration is identical on Wormhole B0 and Blackhole:

**ADDR_MOD_7** (configured by `eltwise_unary_sfpu_configure_addrmod<SfpuType::silu>()`):
```
addr_mod_t {
    .srca = {.incr = 0},
    .srcb = {.incr = 0},
    .dest = {.incr = 0},
}
```

All increments are zero. DEST address advancement is handled by the SFPI `dst_reg++` abstraction within the kernel loop, not by hardware auto-increment. ADDR_MOD_7 is chosen to avoid conflicts with ADDR_MOD_0 and ADDR_MOD_2 which are used by the A2D (unpack-to-DEST) pipeline.

No additional ADDR_MODs (e.g., ADDR_MOD_6) are configured for `SfpuType::silu` since it does not appear in any of the special-case `if constexpr` blocks in `eltwise_unary_sfpu_configure_addrmod`.

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Determine compute kernel path, init/func macros, and approximation mode for SILU
   **Key Findings**: SILU uses `eltwise_sfpu.cpp`, non-parameterized init `silu_tile_init()` / func `silu_tile(idst)`, approx_mode=false (default case)

2. **File**: `tt_metal/hw/inc/api/compute/compute_kernel_api.h`
   **Reason**: Trace API header for silu_tile / silu_tile_init
   **Key Findings**: `silu_tile(idst)` calls `llk_math_eltwise_unary_sfpu_silu<APPROX, DST_ACCUM_MODE>(idst)`; `silu_tile_init()` calls `llk_math_eltwise_unary_sfpu_silu_init<APPROX>()`

3. **File**: `build_Debug/libexec/tt-metalium/tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_silu.h` (build-generated)
   **Reason**: Trace LLK dispatch layer for silu
   **Key Findings**: LLK dispatches to `calculate_silu<is_fp32_dest_acc_en, 8>` via `_llk_math_eltwise_unary_sfpu_params_`; init calls `silu_init<APPROXIMATE>` which calls `_init_sfpu_reciprocal_`

4. **File**: `build_Debug/libexec/tt-metalium/tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_silu.h` (build-generated)
   **Reason**: Read the core SFPU kernel for silu (the actually deployed version)
   **Key Findings**: `calculate_silu` computes `x * _sfpu_sigmoid_<is_fp32_dest_acc_en>(x)` with optional bf16 rounding. This differs from the tt_llk source which uses a piecewise polynomial approximation.

5. **File**: `build_Debug/libexec/tt-metalium/tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_sigmoid.h` (build-generated)
   **Reason**: Trace _sfpu_sigmoid_ helper used by silu
   **Key Findings**: sigmoid(x) = 1/(1+exp(-x)); uses _sfpu_exp_improved_ (fp32) or _sfpu_exp_21f_ (bf16) for exp, then _sfpu_reciprocal_ with 2 or 1 NR iterations

6. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_recip.h`
   **Reason**: Read WH reciprocal implementation
   **Key Findings**: Uses quadratic initial estimate (k2 - k1*x + k0*x^2) + Newton-Raphson refinement; scale factor via bitwise NOT; sign restoration via SFPSETSGN

7. **File**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_recip.h`
   **Reason**: Read BH reciprocal implementation
   **Key Findings**: Uses hardware `approx_recip` (SFPARECIP) for ~7-bit initial estimate + Newton-Raphson; negated error term for NaN detection via sign check

8. **File**: `build_Debug/libexec/tt-metalium/tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_exp.h` (build-generated)
   **Reason**: Read exp implementations used by sigmoid
   **Key Findings**: _sfpu_exp_21f_ uses Moroz et al. 2022 algorithm (2nd degree poly on fractional part); _sfpu_exp_f32_accurate_ uses Cody-Waite range reduction + 7th-order Taylor polynomial

9. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Read parameters dispatch for VectorMode::RC face iteration
   **Key Findings**: 4-face loop with SETRWC between faces; callable invoked once per face with ITERATIONS=8

10. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`
    **Reason**: Read address mode configuration for silu
    **Key Findings**: ADDR_MOD_7 = all zeros (no auto-increment); silu not in special-case ADDR_MOD_6 list

11. **File**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu.h`
    **Reason**: Compare BH address mode configuration
    **Key Findings**: Identical ADDR_MOD_7 config; BH adds `SfpuType::reciprocal` to ADDR_MOD_6 special cases but silu is not affected

12. **File**: `.claude/references/sfpu-hardware-model.md`
    **Reason**: SFPU architecture reference for instruction semantics, register layout, addressing model
    **Key Findings**: Stride-2 addressing, 8 iterations per face, SFPMAD for float add/mul, SFPIADD for integer only

13. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_silu.h` (tt_llk source)
    **Reason**: Compare with build-generated version to understand implementation differences
    **Key Findings**: tt_llk source uses `_sigmoid_piecewise_linear_positive_` (POLYVAL5 polynomial approx), which is DIFFERENT from the build version that uses `_sfpu_sigmoid_` (exp+reciprocal). The build version is what runs on hardware.

14. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_polyval.h`
    **Reason**: Understand PolynomialEvaluator used in exp implementations
    **Key Findings**: Horner's method via recursive variadic template; each coefficient adds one SFPMAD instruction
