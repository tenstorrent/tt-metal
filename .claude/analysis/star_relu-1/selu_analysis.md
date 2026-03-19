## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### Unary Dispatch Summary
- **UnaryOpType**: `SELU`
- **Compute kernel**: `eltwise_sfpu.cpp` (default path from `get_compute_kernel_path()`)
- **SFPU_OP_CHAIN_0 expansion**: `selu_tile({idst}, {scale_hex}u, {alpha_hex}u)` where `scale` and `alpha` are passed as bit-cast `uint32_t` from the two float parameters

#### Approximation Mode Resolution
There are TWO independent controls for approximation. Both are reported below.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(SELU)` in `unary_op_utils.cpp` -- falls through to `default: return false` |
| Template parameter (SFPU_OP_CHAIN) | none | `get_op_init_and_func()` produces `selu_tile_init()` and `selu_tile({idst}, ...)` with no explicit template arguments; `selu_tile_init()` calls `llk_math_eltwise_unary_sfpu_selu_init<APPROX>()` where `APPROX` is the kernel-level `APPROX` define (which equals `math_approx_mode`, i.e., `false`) |
| Effective SFPU path | `APPROXIMATION_MODE=false` in `calculate_selu`. However, `_sfpu_exp_21f_<true>` is called with `is_fp32_dest_acc_en=true` hardcoded (to avoid premature rounding), meaning the exp helper skips the final `float_to_fp16b` conversion. The `APPROXIMATION_MODE` template parameter of `calculate_selu` is not used within the selu function body itself -- the exp21f algorithm is always used regardless. | See `ckernel_sfpu_unary_selu.h` line 26 |

### SFPU Abstraction Layers
| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/selu.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_selu.h` |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_unary_selu.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |

### Call Chain
1. `selu_tile(idst, param0, param1)` (API header) calls `llk_math_eltwise_unary_sfpu_selu<APPROX, DST_ACCUM_MODE>(idst, param0, param1)` guarded by `MATH((...))`.
2. `llk_math_eltwise_unary_sfpu_selu` (LLK dispatch) calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(calculate_selu<APPROXIMATE, is_fp32_dest_acc_en, ITERATIONS>, dst_index, vector_mode, scale, alpha)`.
3. `_llk_math_eltwise_unary_sfpu_params_` (params dispatch) sets DEST write address, activates ADDR_MOD base, stalls for SFPU readiness, then loops over faces calling the SFPU function and advancing DEST with `TTI_SETRWC` between faces.
4. `calculate_selu(scale, alpha)` (core SFPU) runs 8 iterations per face, loading from `dst_reg[0]`, branching on sign, computing `scale * x` for positive or `(exp(x) - 1) * alpha * scale` for negative, and writing back to `dst_reg[0]`.
5. For negative inputs, `_sfpu_exp_21f_<true>(v)` is called, implementing the Moroz et al. 2022 "exp_21f" polynomial approximation of `exp(x)`.

### Parameters Dispatch Summary
- **Vector mode**: `VectorMode::RC` (default). All 4 faces of the tile are processed.
- **Operation invocation**: The params dispatch loops over 4 faces. For each face, it calls `calculate_selu(scale, alpha)` once (which internally iterates 8 times over sfpi rows), then advances DEST by 2 `TTI_SETRWC` calls (each advancing by 8 sfpi rows, totaling 16 physical DEST rows = one face).
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, `TTI_SETRWC` between faces). The `set_addr_mod_base()` call switches to ADDR_MOD 4-7 range; for `SfpuType::selu`, `ADDR_MOD_7` is configured with `dest.incr = 0` (no auto-increment from address mode -- DEST advancement is done explicitly by `dst_reg++` in the SFPI code and `TTI_SETRWC` in the dispatch).

### Annotated SFPU Kernel Source

This kernel uses **SFPI abstractions** (Style A).

```cpp
// File: tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_unary_selu.h
// (Build-resolved version shown; source tree version uses _sfpu_exp_21f_bf16_ which was renamed to _sfpu_exp_21f_)

// SELU(x) = scale * ( max(0, x) + min(0, alpha * (exp(x)-1) ) )
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>
inline void calculate_selu(uint scale, uint alpha) { // APPROXIMATION_MODE=false, is_fp32_dest_acc_en=false, ITERATIONS=8
    sfpi::vFloat scale_value = Converter::as_float(scale);  // reinterpret uint32_t bits as float
    sfpi::vFloat alpha_value = Converter::as_float(alpha);  // reinterpret uint32_t bits as float
#pragma GCC unroll 8

    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];                            // SFPLOAD from current DEST row pair
        v_if(v >= 0.0f) { sfpi::dst_reg[0] = v * scale_value; }      // positive path: SFPSETCC + SFPMAD(v * scale)
        v_else {
            sfpi::vFloat exp_calc = _sfpu_exp_21f_<true>(             // exp_21f with is_fp32_dest_acc_en=true to keep fp32 precision
                v);  // is_fp32_dest_acc_en set to true to avoid rounding as it has to be done at the end of operation
            sfpi::vFloat minus_mul = exp_calc - sfpi::vConst1;        // SFPMAD(exp_calc * 1.0 + (-1.0))
            sfpi::vFloat result = minus_mul * alpha_value * scale_value; // two SFPMAD ops: (exp-1)*alpha, then *scale

            if constexpr (!is_fp32_dest_acc_en) {                     // when DEST is bfloat16 (the common case)
                result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0)); // SFPSTOCHRND: round fp32->bf16
            }
            sfpi::dst_reg[0] = result;                                // SFPSTORE to current DEST row pair
        }
        v_endif;
        sfpi::dst_reg++;                                              // advance to next sfpi row (2 physical DEST rows)
    }
}
```

**Helper function `_sfpu_exp_21f_<true>`** (from the build-resolved `ckernel_sfpu_exp.h`):

```cpp
// File: tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_exp.h (build version)

sfpi_inline sfpi::vInt _float_to_int32_for_exp21f_(sfpi::vFloat val) {
    sfpi::vInt exp = sfpi::exexp(val);       // SFPEXEXP: extract exponent field
    sfpi::vInt man = sfpi::exman8(val);      // SFPEXMAN: extract mantissa with implicit bit
    man = sfpi::reinterpret<sfpi::vInt>(sfpi::shft(sfpi::reinterpret<sfpi::vUInt>(man), exp)); // SFPSHFT: shift mantissa left by exponent
    return man;
}

template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _sfpu_exp_21f_(sfpi::vFloat val) { // is_fp32_dest_acc_en=true when called from selu
    // Implementation notes, see the original file for more details
    constexpr float ONE_LN2 = 1.4426950216293334961f;
    sfpi::vFloat xlog2 = (val * ONE_LN2 + 127.f);            // SFPMAD: val * (1/ln2) + 127

    sfpi::vFloat threshold_low = 0.f;
    sfpi::vFloat threshold_high = sfpi::vFloat(255.f);
    sfpi::vec_min_max(threshold_low, xlog2);                   // SFPSWAP: clamp lower bound to 0
    sfpi::vec_min_max(xlog2, threshold_high);                  // SFPSWAP: clamp upper bound to 255

    sfpi::vInt z = _float_to_int32_for_exp21f_(xlog2);        // convert to int32 (implicitly scaled by 2^23)

    sfpi::vInt exponential_part =
        exexp_nodebias(sfpi::reinterpret<sfpi::vFloat>(z));    // SFPEXEXP: extract integer part of x/ln2
    sfpi::vInt fractional_part =
        sfpi::exman9(sfpi::reinterpret<sfpi::vFloat>(z));      // SFPEXMAN: extract fractional part

    sfpi::vFloat frac = sfpi::int32_to_float(fractional_part, 0); // SFPCAST: int32 -> float

    // 2nd-degree polynomial approximation of 2^(fractional) via Horner's method
    frac = PolynomialEvaluator::eval(frac, 1.0017248f, 7.839635491371155e-08f, 4.791750143340323e-15f); // chain of SFPMAD

    sfpi::vFloat y = sfpi::setexp(frac, exponential_part);     // SFPSETEXP: combine mantissa with exponent = 2^(int) * 2^(frac)

    if constexpr (!is_fp32_dest_acc_en) {                      // skipped when called from selu (is_fp32_dest_acc_en=true)
        y = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(y, 0));
    }

    return y;
}
```

### SFPU Instructions Used

| Instruction / Intrinsic | Description | Used In |
|------------------------|-------------|---------|
| `SFPLOAD` (via `dst_reg[0]` read) | Load 32 elements from current DEST row pair into LREG | `calculate_selu` |
| `SFPSTORE` (via `dst_reg[0]` write) | Store 32 elements from LREG back to DEST row pair | `calculate_selu` |
| `SFPMAD` (via `vFloat * vFloat`, `vFloat + vFloat`, `vFloat - vFloat`) | Fused multiply-add: `a * b + c`. Used for all arithmetic (`v * scale`, `exp - 1`, `result * alpha * scale`, polynomial evaluation) | `calculate_selu`, `_sfpu_exp_21f_` |
| `SFPSETCC` / `SFPENCC` / `SFPCOMPC` (via `v_if`/`v_else`/`v_endif`) | Condition code manipulation for predicated execution based on `v >= 0.0f` | `calculate_selu` |
| `SFPEXEXP` (via `sfpi::exexp()`, `sfpi::exexp_nodebias()`) | Extract biased/unbiased exponent field from float | `_sfpu_exp_21f_`, `_float_to_int32_for_exp21f_` |
| `SFPEXMAN` (via `sfpi::exman8()`, `sfpi::exman9()`) | Extract mantissa (8-bit or 9-bit) with implicit leading bit | `_sfpu_exp_21f_`, `_float_to_int32_for_exp21f_` |
| `SFPSHFT` (via `sfpi::shft()`) | Barrel shift of mantissa by exponent value | `_float_to_int32_for_exp21f_` |
| `SFPSETEXP` (via `sfpi::setexp()`) | Set exponent field of a float (recombine mantissa + exponent) | `_sfpu_exp_21f_` |
| `SFPSWAP` (via `sfpi::vec_min_max()`) | Conditional swap for min/max clamping of intermediate values | `_sfpu_exp_21f_` |
| `SFPCAST` (via `sfpi::int32_to_float()`) | Convert integer to float representation | `_sfpu_exp_21f_` |
| `SFPLOADI` (via `sfpi::vFloat(literal)`, `sfpi::vConst1`) | Load immediate constant into LREG | `calculate_selu`, `_sfpu_exp_21f_` |
| `SFP_STOCH_RND` (via `sfpi::float_to_fp16b()`) | Convert fp32 to bfloat16 with round-to-nearest-even | `calculate_selu` (final rounding) |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| `LREG[0]` (aka `dst_reg[0]`) | Primary working register: loads input from DEST, stores result back |
| `LREG[1-3]` | Intermediate computation registers used by SFPI compiler for temporaries (`exp_calc`, `minus_mul`, `result`, polynomial intermediates) |
| `vConstFloatPrgm0-2` | Not explicitly set by selu init; however, the `_sfpu_exp_21f_` helper does NOT use programmable constants -- it uses inline float literals |
| `vConst1` | Hardcoded constant `1.0f`, used for `exp_calc - vConst1` (the `-1` in `exp(x)-1`) |
| DEST rows | Input/output tile data. Accessed via `dst_reg[0]` with stride-2 addressing. Each iteration processes 2 physical rows (32 elements). |
| Condition Code (CC) | Set by `v_if(v >= 0.0f)` to predicate the positive vs negative branches. `v_else` complements CC, `v_endif` restores it. |

### Address Mode Configuration

For `SfpuType::selu`, the init function `eltwise_unary_sfpu_configure_addrmod<SfpuType::selu>()` configures:

| Address Mode | Field Values | Purpose |
|-------------|--------------|---------|
| `ADDR_MOD_7` | `srca.incr=0, srcb.incr=0, dest.incr=0` | Default SFPU address mode with no auto-increment (DEST advancement is handled by `dst_reg++` in SFPI code and `TTI_SETRWC` in params dispatch) |

This configuration is **identical on Wormhole and Blackhole**. The `SfpuType::selu` does not match any of the special-case `if constexpr` branches that configure `ADDR_MOD_6` (those are for `topk_local_sort`, `typecast`, `unary_max`, `unary_min`, etc.).

During execution, `set_addr_mod_base()` issues `TTI_SETC16(ADDR_MOD_SET_Base_ADDR32, 1)` to switch to the upper address mode bank (ADDR_MOD 4-7). After the SFPU operation completes, `clear_addr_mod_base()` restores the default bank.

## External Knowledge Sources
### DeepWiki Queries
No DeepWiki queries were needed for this analysis. The SFPU kernel implementation was fully traceable from source code.

### Confluence References
No Confluence references were needed. The SELU kernel uses standard SFPI abstractions without complex raw TTI instruction sequences.

### Glean References
No Glean references were needed for this analysis.
