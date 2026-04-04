## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to for the SILU (Sigmoid Linear Unit) operation. SILU computes `silu(x) = x * sigmoid(x)` where `sigmoid(x) = 1 / (1 + exp(-x))`.

### Unary Dispatch Summary
- **UnaryOpType**: `SILU`
- **Compute kernel**: `eltwise_sfpu.cpp` (via `unary_ng` path, default case in `get_compute_kernel_path()`)
- **SFPU_OP_CHAIN_0 expansion**: `silu_tile_init()` / `silu_tile(idst)`

#### Approximation Mode Resolution
There are TWO independent controls for approximation. Both are reported below.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(UnaryOpType)` in `unary_ng_op_utils.cpp` -- returns `false` for all ops (single `return false` statement) |
| Template parameter (SFPU_OP_CHAIN) | none | `get_op_init_and_func()` -- `silu_tile_init()` and `silu_tile({idst})` use no template parameters; the API header `compute_kernel_api.h` forwards the global `APPROX` define |
| Effective SFPU path | Non-approximate sigmoid path: `_sfpu_sigmoid_<is_fp32_dest_acc_en>()` using `_sfpu_exp_accurate_` + `_sfpu_reciprocal_<2>` (fp32) or `_sfpu_reciprocal_<1>` (bf16) | The `calculate_silu()` function always calls `_sfpu_sigmoid_` regardless of APPROXIMATION_MODE; `silu_init()` on WH passes `APPROXIMATION_MODE=false` to `_init_sfpu_reciprocal_<false>()`; on BH explicitly calls `sigmoid_init<false>()` |

### SFPU Abstraction Layers
| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/compute_kernel_api.h` (lines 463-465: `silu_tile()` and `silu_tile_init()`) |
| **LLK Dispatch** | `llk_math_eltwise_unary_sfpu_silu.h` (generated at build time; source in `build_Release/libexec/tt-metalium/tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/`) |
| **Core SFPU Implementation** | `ckernel_sfpu_silu.h` (generated at build time from split includes; the canonical silu kernel is in `build_Release/libexec/.../llk_sfpu/ckernel_sfpu_silu.h`). The third_party sources at `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/common/inc/sfpu/ckernel_sfpu_silu.h` contain an older piecewise-linear version. |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |

### Call Chain
1. **`silu_tile(idst)`** (in `compute_kernel_api.h`) calls `llk_math_eltwise_unary_sfpu_silu<APPROX, DST_ACCUM_MODE>(idst)`.
2. **`llk_math_eltwise_unary_sfpu_silu<APPROX, is_fp32_dest_acc_en>(dst_index)`** (in `llk_math_eltwise_unary_sfpu_silu.h`) calls `_llk_math_eltwise_unary_sfpu_params_<APPROX>(ckernel::sfpu::calculate_silu<is_fp32_dest_acc_en, 8>, dst_index, VectorMode::RC)`.
3. **`_llk_math_eltwise_unary_sfpu_params_`** (in `llk_math_eltwise_unary_sfpu_params.h`) sets DEST write address, stalls for SFPU readiness, then loops over 4 faces calling `calculate_silu<is_fp32_dest_acc_en, 8>()` per face with `SETRWC` between faces.
4. **`calculate_silu<is_fp32_dest_acc_en, ITERATIONS>()`** (in `ckernel_sfpu_silu.h`) iterates 8 times per face, loading from `dst_reg[0]`, computing `x * _sfpu_sigmoid_<is_fp32_dest_acc_en>(x)`, optionally rounding to bf16, and storing back.
5. **`_sfpu_sigmoid_<is_fp32_dest_acc_en>(x)`** (in `ckernel_sfpu_sigmoid.h`) computes `1 / (1 + exp(-x))` by calling `_sfpu_exp_accurate_<is_fp32_dest_acc_en>(-x)` then `_sfpu_reciprocal_<2>` (fp32 mode) or `_sfpu_reciprocal_<1>` (bf16 mode).

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` (default) -- all 4 faces of the tile are processed (Face0 through Face3).
- **Operation invocation**: The params dispatch calls `calculate_silu<is_fp32_dest_acc_en, 8>()` once per face in a loop of 4 iterations. Each call processes 8 sfpi iterations (one full face of 256 elements). Between faces, `TTI_SETRWC` advances the DEST address by the face stride (16 physical rows = 2 SETRWC increments of 8).
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, SETRWC between faces). On Wormhole, `ADDR_MOD_7` (with addr_mod_base=1) is configured with `{srca.incr=0, srcb.incr=0, dest.incr=0}` -- the SFPI `dst_reg++` handles iteration advancement independently of hardware ADDR_MOD auto-increment. On Blackhole, the same `ADDR_MOD_7` configuration applies.

### Annotated SFPU Kernel Source

The SILU kernel uses SFPI abstractions exclusively (Style A). The build-output version is the authoritative implementation, as the third_party source contains an older piecewise-linear sigmoid approximation. The core function delegates heavily to `_sfpu_sigmoid_`, which in turn calls `_sfpu_exp_accurate_` and `_sfpu_reciprocal_`.

#### `calculate_silu` -- the top-level SILU function

```cpp
// File: build_Release/libexec/.../llk_sfpu/ckernel_sfpu_silu.h (identical for WH and BH)

template <bool is_fp32_dest_acc_en, int ITERATIONS> // is_fp32_dest_acc_en from DST_ACCUM_MODE, ITERATIONS=8
inline void calculate_silu() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0]; // SFPLOAD: load 32 elements from current DEST position

        // silu(x) = x * sigmoid(x)
        sfpi::vFloat result = x * _sfpu_sigmoid_<is_fp32_dest_acc_en>(x); // SFPMAD: multiply x by sigmoid(x)

        // Round to bfloat16 if not in fp32 accumulation mode
        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0)); // SFP_STOCH_RND: round to bf16
        }

        sfpi::dst_reg[0] = result; // SFPSTORE: store result back to DEST
        sfpi::dst_reg++;           // advance to next sfpi row (+2 physical DEST rows = 32 elements)
    }
}
```

#### `silu_init` -- initialization (Wormhole B0)

```cpp
// File: build_Release/libexec/.../wormhole_b0/.../ckernel_sfpu_silu.h

template <bool APPROXIMATION_MODE> // APPROXIMATION_MODE=false
inline void silu_init() {
    if constexpr (!APPROXIMATION_MODE) {
        _init_sfpu_reciprocal_<false>(); // sets vConstFloatPrgm0 = 2.0f (used by BH reciprocal NR refinement)
    } else {
        _init_sfpu_reciprocal_<true>();
    }
}
```

#### `silu_init` -- initialization (Blackhole)

```cpp
// File: build_Release/libexec/.../blackhole/.../ckernel_sfpu_silu.h

template <bool APPROXIMATION_MODE> // APPROXIMATION_MODE=false
inline void silu_init() {
    // calculate_silu uses the non-approx sigmoid path via _sfpu_sigmoid_, so we must use non-approx sigmoid_init
    sigmoid_init<false>(); // calls _init_sfpu_reciprocal_<false>(), setting vConstFloatPrgm0 = 2.0f
}
```

#### `_sfpu_sigmoid_` -- the sigmoid helper (shared by WH and BH)

```cpp
// File: build_Release/libexec/.../llk_sfpu/ckernel_sfpu_sigmoid.h (identical for WH and BH)

template <bool is_fp32_acc_to_dest_mode = true>
sfpi_inline sfpi::vFloat _sfpu_sigmoid_(sfpi::vFloat x) {
    // Compute sigmoid as: sigmoid(x) = 1 / (1 + exp(-x))

    sfpi::vFloat exp_neg_x;
    // If fp32 then use higher accuracy exp function
    // Otherwise, use exp_21f (~1 ULP on bfloat16)
    if constexpr (is_fp32_acc_to_dest_mode) {
        exp_neg_x = _sfpu_exp_accurate_<true>(-x); // fp32-accurate exp via Cody-Waite + 7th order Taylor
    } else {
        exp_neg_x = _sfpu_exp_21f_bf16_<true>(-x); // bf16-accurate exp via Moroz et al. exp_21f algorithm
    }

    sfpi::vFloat denominator = sfpi::vConst1 + exp_neg_x; // SFPMAD: 1.0 + exp(-x)

    sfpi::vFloat result;
    if constexpr (is_fp32_acc_to_dest_mode) {
        result = _sfpu_reciprocal_<2>(denominator); // 2 Newton-Raphson iterations for fp32 precision
    } else {
        result = _sfpu_reciprocal_<1>(denominator); // 1 Newton-Raphson iteration for bf16 precision
    }

    return result;
}
```

#### `_sfpu_reciprocal_` -- Wormhole implementation (polynomial + Newton-Raphson)

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_recip.h

template <int max_iter = 2> // max_iter=2 for fp32, max_iter=1 for bf16
sfpi_inline sfpi::vFloat _sfpu_reciprocal_(const sfpi::vFloat in)
{
    // Implementation notes, see the original file for more details
    sfpi::vFloat negative_x = sfpi::setman(sfpi::vConstNeg1, sfpi::reinterpret<sfpi::vInt>(in)); // SFPSETMAN: scale to [-2,-1) range

    // Quadratic initial estimate: y = k2 - k1*x + k0*x**2 (Sollya-optimized coefficients)
    sfpi::vFloat y = sfpi::vConstFloatPrgm1 + sfpi::vConstFloatPrgm0 * negative_x; // SFPMAD

    // Implementation notes, see the original file for more details
    sfpi::vInt scale_bits = ~sfpi::reinterpret<sfpi::vUInt>(in); // SFPNOT: compute ~in for scale exponent

    // Continue with quadratic estimate
    y = sfpi::vConstFloatPrgm2 + y * negative_x; // SFPMAD

    // Scale factor: set mantissa to zero
    sfpi::vFloat scale = sfpi::setman(sfpi::reinterpret<sfpi::vFloat>(scale_bits), 0); // SFPSETMAN

    // First iteration of Newton-Raphson: t = 1.0 - x*y
    sfpi::vFloat t = sfpi::vConst1 + negative_x * y; // SFPMAD

    // Scale factor adjustment: scale = scale*0.5
    scale *= 0.5f; // SFPMAD or SFPMULI

    // Continue Newton-Raphson: y = y + y*t
    y = y + y * t; // SFPMAD

    if constexpr (max_iter > 1)
    {
        // Second iteration of Newton-Raphson: t = 1.0 - x*y; y = y + y*t
        t = sfpi::vConst1 + negative_x * y; // SFPMAD
        y = y + y * t; // SFPMAD
    }

    // Apply scaling factor, and set sign to match input
    y = y * scale; // SFPMAD
    y = sfpi::setsgn(y, in); // SFPSETSGN

    return y;
}
```

#### `_sfpu_reciprocal_` -- Blackhole implementation (hardware-accelerated + Newton-Raphson)

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_recip.h

template <int max_iter = 2> // max_iter=2 for fp32, max_iter=1 for bf16
sfpi_inline sfpi::vFloat _sfpu_reciprocal_(const sfpi::vFloat x)
{
    // sfpi::approx_recip(x) uses SFPARECIP hardware instruction for initial estimate
    sfpi::vFloat y = sfpi::approx_recip(x); // SFPARECIP: hardware-accelerated reciprocal approximation

    // Optionally improve the approximation using Newton-Raphson
    if constexpr (max_iter > 0)
    {
        // Implementation notes, see the original file for more details
        sfpi::vFloat t = x * y - sfpi::vConstFloatPrgm0; // SFPMAD: compute error term (x*y - 2.0)

        if constexpr (max_iter > 1)
        {
            sfpi::vFloat y1 = y * -t - sfpi::vConst0; // SFPMAD: first NR correction
            v_if (t < 0) // SFPSETCC+SFPENCC: guard against NaN (t=NaN implies t>=0)
            {
                t = x * y1 - sfpi::vConstFloatPrgm0; // SFPMAD
                y = y1 * -t - sfpi::vConst0; // SFPMAD: second NR correction
            }
            v_endif; // SFPENCC: restore all lanes
        }
        else
        {
            v_if (t < 0) // SFPSETCC+SFPENCC: guard against NaN
            {
                y = y * -t - sfpi::vConst0; // SFPMAD: single NR correction
            }
            v_endif; // SFPENCC
        }
    }

    return y;
}
```

#### `_sfpu_exp_21f_bf16_` -- bf16 exponential (shared WH/BH)

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_exp.h

template <bool is_fp32_dest_acc_en> // is_fp32_dest_acc_en=true when called from sigmoid in fp32 mode
sfpi_inline sfpi::vFloat _sfpu_exp_21f_bf16_(sfpi::vFloat val)
{
    // Implementation notes, see the original file for more details
    constexpr float ONE_LN2 = 1.4426950216293334961f;
    sfpi::vFloat xlog2      = (val * ONE_LN2 + 127.f); // SFPMAD: x/ln(2) + bias

    // Clamp to avoid overflow in intermediate values
    sfpi::vFloat threshold_low  = 0.f;
    sfpi::vFloat threshold_high = sfpi::vFloat(255.f);
    sfpi::vec_min_max(threshold_low, xlog2); // SFPSWAP: clamp lower bound
    sfpi::vec_min_max(xlog2, threshold_high); // SFPSWAP: clamp upper bound

    sfpi::vInt z = _float_to_int32_for_exp_21f_(xlog2); // SFPEXEXP + SFPEXMAN8 + SFPSHFT2

    sfpi::vInt exponential_part = exexp_nodebias(sfpi::reinterpret<sfpi::vFloat>(z)); // SFPEXEXP: extract exponent
    sfpi::vInt fractional_part  = sfpi::exman9(sfpi::reinterpret<sfpi::vFloat>(z));   // SFPEXMAN: extract mantissa

    sfpi::vFloat frac = sfpi::int32_to_float(fractional_part, 0); // SFPCAST: int32 to float

    // 2nd degree polynomial refinement of 2**(x_f)
    frac = PolynomialEvaluator::eval(frac, 1.0017248f, 7.839635491371155e-08f, 4.791750143340323e-15f); // chain of SFPMAD

    // Recombine exponent and mantissa: 2**(x_i) * 2**(x_f)
    sfpi::vFloat y = sfpi::setexp(frac, exponential_part); // SFPSETEXP

    if constexpr (!is_fp32_dest_acc_en)
    {
        y = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(y, 0)); // SFP_STOCH_RND: round to bf16
    }

    return y;
}
```

#### `_sfpu_exp_fp32_accurate_` -- fp32 exponential (used when `is_fp32_dest_acc_en=true`)

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_exp.h

sfpi_inline sfpi::vFloat _sfpu_exp_fp32_accurate_(sfpi::vFloat val)
{
    sfpi::vFloat result = sfpi::vConst0;

    // Implementation notes, see the original file for more details
    constexpr float OVERFLOW_THRESHOLD  = 128.0f;
    constexpr float UNDERFLOW_THRESHOLD = -127.0f;

    constexpr float INV_LN2 = 1.4426950408889634f;
    sfpi::vFloat z          = val * INV_LN2; // SFPMAD: x * (1/ln2)

    sfpi::vInt exp_bits = sfpi::exexp(z); // SFPEXEXP: extract exponent for NaN check

    v_if (z >= OVERFLOW_THRESHOLD) // SFPSETCC via v_if
    {
        result = std::numeric_limits<float>::infinity(); // SFPLOADI
    }
    v_elseif (z <= UNDERFLOW_THRESHOLD) // SFPCOMPC+SFPPUSHC+SFPSETCC
    {
        result = sfpi::vConst0;
    }
    v_elseif (exp_bits == 255) // NaN detection
    {
        result = std::numeric_limits<float>::quiet_NaN(); // SFPLOADI
    }
    v_else
    {
        sfpi::vInt k_int;
        sfpi::vFloat k = _sfpu_round_to_nearest_int32_(z, k_int); // Hacker's Delight RTE rounding

        // Cody-Waite range reduction for extended precision
        constexpr float LN2_HI = -0.6931152343750000f;
        constexpr float LN2_LO = -3.19461832987e-05f;

        sfpi::vFloat r_hi = k * LN2_HI + val; // SFPMAD
        sfpi::vFloat r = k * LN2_LO + r_hi;   // SFPMAD

        // 7th order Taylor polynomial for exp(r)
        sfpi::vFloat p = PolynomialEvaluator::eval(
            r,
            sfpi::vConst1, sfpi::vConst1, 0.5f, 1.0f/6.0f,
            1.0f/24.0f, 1.0f/120.0f, 1.0f/720.0f, 1.0f/5040.0f
        ); // chain of 7 SFPMAD instructions (Horner's method)

        // Scale by 2^k via exponent manipulation
        sfpi::vInt p_exp = sfpi::exexp_nodebias(p); // SFPEXEXP
        sfpi::vInt new_exp = p_exp + k_int;          // SFPIADD
        result = sfpi::setexp(p, new_exp);            // SFPSETEXP
    }
    v_endif;

    return result;
}
```

### SFPU Instructions Used

The SILU kernel and its sub-functions use the following SFPU instructions (via SFPI abstractions):

| Instruction | Description | Used In |
|------------|-------------|---------|
| `SFPLOAD` | Load 32 elements from DEST row into LREG | `calculate_silu` (via `dst_reg[0]` read) |
| `SFPSTORE` | Store LREG contents back to DEST row | `calculate_silu` (via `dst_reg[0]` write) |
| `SFPMAD` | Fused multiply-add (the primary arithmetic workhorse) | Throughout: `x * sigmoid(x)`, polynomial evaluation (Horner's method), Newton-Raphson iterations, `1.0 + exp_neg_x`, all additions and multiplications |
| `SFPLOADI` | Load 16-bit immediate into LREG | Loading float constants (e.g., `0.5f`, `255.f`, `inf`, `NaN`), polynomial coefficients |
| `SFPSETMAN` | Set mantissa field of a float | `_sfpu_reciprocal_` (WH): scaling input to `[-2,-1)` range and zeroing mantissa of scale factor |
| `SFPSETSGN` | Set sign field of a float | `_sfpu_reciprocal_` (WH): matching output sign to input |
| `SFPNOT` | Bitwise NOT | `_sfpu_reciprocal_` (WH): computing `~in` for scale exponent (255-in.Exp) |
| `SFPSETEXP` | Set exponent field of a float | `_sfpu_exp_21f_bf16_` and `_sfpu_exp_fp32_accurate_`: recombining `2^k * polynomial` |
| `SFPEXEXP` | Extract exponent field | `_sfpu_exp_21f_bf16_` and `_sfpu_exp_fp32_accurate_`: separating integer/fractional parts; also NaN detection |
| `SFPEXMAN` | Extract mantissa field | `_sfpu_exp_21f_bf16_`: extracting fractional part via `exman8`/`exman9` |
| `SFPSHFT2` | Bit shift / cross-lane permute | `_float_to_int32_for_exp_21f_`: shifting mantissa by exponent value |
| `SFPCAST` | Format conversion | `_sfpu_exp_21f_bf16_`: `int32_to_float` conversion |
| `SFP_STOCH_RND` | Stochastic/deterministic rounding | `calculate_silu` and `_sfpu_exp_21f_bf16_`: bf16 rounding via `float_to_fp16b` |
| `SFPSWAP` | Conditional swap (min/max) | `_sfpu_exp_21f_bf16_`: clamping `xlog2` via `vec_min_max` |
| `SFPIADD` | Integer add | `_sfpu_exp_fp32_accurate_`: adding `k_int` to exponent |
| `SFPARECIP` | Hardware-accelerated reciprocal approximation | `_sfpu_reciprocal_` (BH only): `approx_recip(x)` initial estimate |
| `SFPSETCC` | Set condition code from register comparison | `_sfpu_sigmoid_` path: `v_if` guards in exp and reciprocal |
| `SFPENCC` | Enable/disable condition code | `v_if`/`v_endif` blocks: entering/exiting conditional execution |
| `SFPCOMPC` | Complement condition code | `v_elseif`/`v_else` blocks in `_sfpu_exp_fp32_accurate_` |
| `SFPPUSHC` | Push CC onto stack | Nested `v_if`/`v_elseif` blocks in `_sfpu_exp_fp32_accurate_` |
| `SFPPOPC` | Pop CC from stack | Nested `v_if`/`v_elseif` blocks in `_sfpu_exp_fp32_accurate_` |
| `SFPABS` | Absolute value | Not directly in SILU path (the older piecewise-linear third_party version used `sfpi::abs`, but the build version does not) |

### SFPU Register Usage

The SILU kernel and its sub-functions use registers as follows:

| Register | Usage |
|----------|-------|
| **LREG0-LREG3** | General-purpose working registers. Used for intermediate values throughout the computation: input `x`, sigmoid result, exp result, reciprocal working variables, polynomial intermediate terms. The SFPI compiler allocates these automatically from the `vFloat`/`vInt` declarations. |
| **DEST (via `dst_reg`)** | Source of input tile data and destination for output. Each iteration reads 32 elements (2 physical rows x 16 elements) via `SFPLOAD` and writes back via `SFPSTORE`. |
| **vConstFloatPrgm0** | Programmable constant register 0. Set to `2.0f` during `silu_init()` via `_init_sfpu_reciprocal_<false>()`. On Blackhole, used by `_sfpu_reciprocal_` as the Newton-Raphson target value `2.0` in the error computation `x*y - 2.0`. On Wormhole, this register holds the Sollya-optimized coefficient `k0 = 0.3232325017...` for the reciprocal initial quadratic estimate. |
| **vConstFloatPrgm1** | Programmable constant register 1. On Wormhole, set to `k1 = 1.454545974...` (Sollya coefficient for reciprocal quadratic estimate) during init. |
| **vConstFloatPrgm2** | Programmable constant register 2. On Wormhole, set to `k2 = 2.121212482...` (Sollya coefficient for reciprocal quadratic estimate) during init. |
| **vConst1** | Fixed constant = `1.0` (used in sigmoid denominator `1 + exp(-x)`) |
| **vConstNeg1** | Fixed constant = `-1.0` (used in WH reciprocal for `setman` to create negative scaled input) |
| **vConst0** | Fixed constant = `0.0` (used in exp underflow case and BH reciprocal NR correction) |

### Address Mode Configuration

The SILU operation uses the standard unary SFPU address mode configuration, set during `_llk_math_eltwise_unary_sfpu_init_<SfpuType::silu>()`:

**Wormhole B0:**
- `ADDR_MOD_7` is configured with `{srca.incr=0, srcb.incr=0, dest.incr=0}` -- no auto-increment. The `dst_reg++` in the SFPI loop handles DEST address advancement independently via the RWC (Read-Write Counter) mechanism.
- `set_addr_mod_base()` sets the addr_mod base to 1, selecting addr mods 4-7 for SFPU operations (avoiding conflicts with A2D which uses ADDR_MOD_0 and ADDR_MOD_2).
- No additional ADDR_MOD entries are configured for the `silu` SfpuType (it falls through without matching `topk_local_sort`, `typecast`, or `unary_max/min` special cases).

**Blackhole:**
- Same `ADDR_MOD_7` configuration: `{srca.incr=0, srcb.incr=0, dest.incr=0}`.
- Same SfpuType-based dispatch -- `silu` does not match any special-case `if constexpr` branches.
- Note: Blackhole adds `SfpuType::reciprocal` to the `ADDR_MOD_6` special case (with `dest.incr=2`), but this is for standalone reciprocal operations, not when reciprocal is called as a sub-function within sigmoid/SILU.

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary_ng/common/unary_ng_op_utils.cpp`
   **Reason**: Identifies the SFPU_OP_CHAIN_0 expansion for SILU (`silu_tile_init()` / `silu_tile(idst)`) and the compute kernel path (`eltwise_sfpu.cpp`).
   **Key Findings**: SILU dispatches through the `unary_ng` path (not the legacy unary path). `get_op_approx_mode()` returns `false` for all ops. No template parameters are passed to the init/func.

2. **File**: `tt_metal/hw/inc/api/compute/compute_kernel_api.h`
   **Reason**: Contains the tile-level API `silu_tile()` and `silu_tile_init()` that bridge compute kernel calls to LLK dispatch.
   **Key Findings**: `silu_tile(idst)` forwards to `llk_math_eltwise_unary_sfpu_silu<APPROX, DST_ACCUM_MODE>(idst)`. The global `APPROX` and `DST_ACCUM_MODE` defines control template instantiation.

3. **File**: `build_Release/libexec/tt-metalium/tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_silu.h`
   **Reason**: LLK dispatch layer connecting API to core SFPU implementation.
   **Key Findings**: Calls `_llk_math_eltwise_unary_sfpu_params_` with `calculate_silu<is_fp32_dest_acc_en, 8>` as the SFPU function. ITERATIONS is hardcoded to 8.

4. **File**: `build_Release/libexec/tt-metalium/tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_silu.h`
   **Reason**: The authoritative core SFPU implementation for SILU (build-generated from split includes).
   **Key Findings**: Uses `x * _sfpu_sigmoid_<is_fp32_dest_acc_en>(x)` -- a clean composition of sigmoid. The older third_party source uses a different piecewise-linear approach. WH and BH `silu_init` differ: WH passes APPROXIMATION_MODE through; BH always uses `sigmoid_init<false>()`.

5. **File**: `build_Release/libexec/tt-metalium/tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_sigmoid.h`
   **Reason**: Contains `_sfpu_sigmoid_`, the core sigmoid computation used by SILU.
   **Key Findings**: Computes `1/(1+exp(-x))` using `_sfpu_exp_accurate_` for the exponential and `_sfpu_reciprocal_` for the division. fp32 mode uses 2 NR iterations; bf16 mode uses 1.

6. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_exp.h`
   **Reason**: Contains `_sfpu_exp_21f_bf16_` and `_sfpu_exp_fp32_accurate_` used by sigmoid.
   **Key Findings**: bf16 path uses Moroz et al. exp_21f algorithm (polynomial approximation of 2^frac after range reduction). fp32 path uses Cody-Waite range reduction + 7th-order Taylor series with special-case handling for overflow/underflow/NaN.

7. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_recip.h`
   **Reason**: Contains `_sfpu_reciprocal_` (Wormhole) and `_init_sfpu_reciprocal_`.
   **Key Findings**: WH uses Sollya-optimized quadratic initial estimate + Newton-Raphson refinement entirely in software. BH uses `SFPARECIP` hardware instruction for initial estimate + Newton-Raphson. Both set `vConstFloatPrgm0` during init (WH: polynomial coeff `0.323...`; BH: NR target `2.0`).

8. **File**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_recip.h`
   **Reason**: Contains the Blackhole-specific `_sfpu_reciprocal_` implementation.
   **Key Findings**: Uses `sfpi::approx_recip()` (SFPARECIP hardware instruction) for initial approximation, then refines with 1 or 2 Newton-Raphson iterations. Has NaN guard via `v_if (t < 0)` since BH produces `+NaN` for `0*inf`.

9. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Parameters dispatch layer that controls face iteration and DEST address progression.
   **Key Findings**: VectorMode::RC processes all 4 faces. Each face calls the SFPU function once (8 iterations). Between faces, `TTI_SETRWC` advances DEST address by 16 physical rows (2 increments of 8).

10. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`
    **Reason**: Contains `eltwise_unary_sfpu_configure_addrmod` and SFPU init/start/done functions.
    **Key Findings**: `ADDR_MOD_7` configured with all-zero increments for standard unary ops. `set_addr_mod_base()` selects addr mods 4-7. SfpuType::silu has no special ADDR_MOD configuration.

11. **File**: `.claude/references/sfpu-hardware-model.md`
    **Reason**: Authoritative reference for SFPU hardware model, instruction semantics, register layout.
    **Key Findings**: Verified stride-2 addressing (32 elements per dst_reg access), 8 iterations per face, SFPMAD as the universal float arithmetic instruction.

12. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_polyval.h`
    **Reason**: Contains `PolynomialEvaluator::eval` used by exp functions.
    **Key Findings**: Implements Horner's method via recursive variadic template -- each coefficient pair generates one SFPMAD instruction.
