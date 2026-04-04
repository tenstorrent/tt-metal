## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### Unary Dispatch Summary
- **UnaryOpType**: `COSH`
- **Compute kernel**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`
- **SFPU_OP_CHAIN_0 expansion**: `cosh_tile(0)`

#### Approximation Mode Resolution
There are TWO independent controls for approximation.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(COSH)` in `unary_op_utils.cpp` -- falls through to `default: return false` |
| Template parameter (SFPU_OP_CHAIN) | none (non-parameterized) | `get_op_init_and_func_default()` returns `{"cosh_tile_init();", "cosh_tile({idst});"}` -- no template parameters in the call |
| Effective SFPU path | `_sfpu_exp_21f_bf16_<is_fp32_dest_acc_en=false>` is always used; the `APPROXIMATION_MODE` template parameter only affects `cosh_init()` which calls `_init_exponential_<false, false, kCONST_1_FP16B>()` -- this enters the final `else` branch that calls `_init_sfpu_reciprocal_<false>()` (not actually used by the compute path) | `ckernel_sfpu_exp.h` line 923: `else { _init_sfpu_reciprocal_<false>(); }` |

**Note on the init/compute mismatch**: The `cosh_init()` function delegates to `_init_exponential_` which, for `APPROXIMATION_MODE=false` and `FAST_APPROX=false`, initializes a reciprocal lookup. However, `calculate_cosh()` directly calls `_sfpu_exp_21f_bf16_` which does NOT use reciprocals -- it uses polynomial approximation and bit manipulation. The init call is inherited from the exp infrastructure but is effectively a no-op for cosh's actual compute path.

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/cosh.h` |
| **LLK Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (via macro `SFPU_THREE_PARAM_KERNEL_FP32_FIRST` in `llk_math_eltwise_unary_sfpu_macros.h`) |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_cosh.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (function `_llk_math_eltwise_unary_sfpu_params_`) |

**Helper function dependency**: The core SFPU implementation in `ckernel_sfpu_cosh.h` delegates to `_sfpu_exp_21f_bf16_` defined in `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/common/inc/sfpu/ckernel_sfpu_exp.h`. The polynomial evaluator used within is defined in `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/common/inc/sfpu/ckernel_sfpu_polyval.h`.

### Call Chain

1. **`cosh_tile(idst)`** (API header `cosh.h`): Macro-wraps the call with `MATH(SFPU_THREE_PARAM_KERNEL_FP32_FIRST(calculate_cosh, APPROX, DST_ACCUM_MODE, 8, idst, (int)VectorMode::RC))`.
2. **`SFPU_THREE_PARAM_KERNEL_FP32_FIRST`** (macros header): Expands to `_llk_math_eltwise_unary_sfpu_params_<APPROX>(ckernel::sfpu::calculate_cosh<APPROX, DST_ACCUM_MODE, 8>, idst, (int)VectorMode::RC)`.
3. **`_llk_math_eltwise_unary_sfpu_params_<false>`** (LLK params): Sets up DEST addressing for tile `idst`, stalls for SFPU readiness, then loops over 4 faces (VectorMode::RC), calling `calculate_cosh<false, false, 8>()` per face with `SETRWC` between faces.
4. **`calculate_cosh<false, false, 8>()`** (ckernel_sfpu_cosh.h): Iterates 8 times per face, loading `dst_reg[0]`, computing `(exp(v) + exp(-v)) * 0.5`, storing back to `dst_reg[0]`, then advancing `dst_reg++`.
5. **`_sfpu_exp_21f_bf16_<false>(v)`** (ckernel_sfpu_exp.h): Implements the Moroz et al. 2022 exp_21f algorithm -- converts input to base-2 representation, clamps, extracts integer/fractional parts, evaluates a 2nd-degree polynomial, and reconstructs the exponential via `setexp()`.

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` -- all 4 faces of the tile are processed (full tile computation).
- **Operation invocation**: The params dispatch function loops over 4 faces, calling `calculate_cosh<APPROX, DST_ACCUM_MODE, 8>()` once per face. Each invocation of `calculate_cosh` internally loops 8 iterations (ITERATIONS=8), processing all 8 sfpi rows of the face.
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, `SETRWC` between faces). The address mode is `ADDR_MOD_7` on both Wormhole and Blackhole, configured with all-zero increments (`srca=0, srcb=0, dest=0`). The SFPU manages its own addressing via the SFPI `dst_reg++` abstraction within the kernel loop.

### Annotated SFPU Kernel Source

The cosh kernel uses SFPI abstractions (Style A). The Wormhole and Blackhole implementations are identical.

```cpp
// File: tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_cosh.h

// cosh(x) = (exp(x) + exp(-x)) / 2
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>
inline void calculate_cosh() { // APPROXIMATION_MODE=false, is_fp32_dest_acc_en=false, ITERATIONS=8
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];  // SFPLOAD: load 32 elements from current DEST row pair
        sfpi::vFloat result =
            (_sfpu_exp_21f_bf16_<is_fp32_dest_acc_en>(v) +  // exp(x) via Moroz exp_21f algorithm
             _sfpu_exp_21f_bf16_<is_fp32_dest_acc_en>(-v))  // exp(-x): negation is SFPMAD(v, -1.0, 0.0)
            * 0.5f;  // SFPMAD: multiply sum by 0.5
        sfpi::dst_reg[0] = result;  // SFPSTORE: write result back to DEST
        sfpi::dst_reg++;  // advance by 1 sfpi row = 2 physical DEST rows = 32 elements
    }
}
```

The core exponential helper function:

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/common/inc/sfpu/ckernel_sfpu_exp.h

// Implementation notes, see the original file for more details
template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _sfpu_exp_21f_bf16_(sfpi::vFloat val) // is_fp32_dest_acc_en=false
{
    // exp(x) = 2^(x / ln2) = 2^(z_i) * 2^(z_f)
    constexpr float ONE_LN2 = 1.4426950216293334961f;
    sfpi::vFloat xlog2      = (val * ONE_LN2 + 127.f);  // SFPMAD: val * (1/ln2) + 127 (bias)

    // Clamp xlog2 to [0, 255] to prevent overflow/underflow in intermediate values
    sfpi::vFloat threshold_low  = 0.f;                    // SFPLOADI
    sfpi::vFloat threshold_high = sfpi::vFloat(255.f);    // SFPLOADI
    sfpi::vec_min_max(threshold_low, xlog2);              // SFPSWAP: clamp lower bound
    sfpi::vec_min_max(xlog2, threshold_high);             // SFPSWAP: clamp upper bound

    sfpi::vInt z = _float_to_int32_for_exp_21f_(xlog2);  // Convert to integer (see below)

    sfpi::vInt exponential_part = exexp_nodebias(sfpi::reinterpret<sfpi::vFloat>(z)); // SFPEXEXP: extract exponent
    sfpi::vInt fractional_part  = sfpi::exman9(sfpi::reinterpret<sfpi::vFloat>(z));   // SFPEXMAN: extract mantissa

    sfpi::vFloat frac = sfpi::int32_to_float(fractional_part, 0);  // SFPCAST: int32 -> float32

    // 2nd-degree polynomial approximation of 2^(fractional part) via Horner's method
    // eval(frac, c0, c1, c2) = c0 + frac * (c1 + frac * c2) -> chain of SFPMAD instructions
    frac = PolynomialEvaluator::eval(frac, 1.0017248f, 7.839635491371155e-08f, 4.791750143340323e-15f);

    // Recombine: 2^(z_i) * 2^(z_f) by setting exponent of polynomial result
    sfpi::vFloat y = sfpi::setexp(frac, exponential_part);  // SFPSETEXP: set exponent field

    if constexpr (!is_fp32_dest_acc_en) // true: DEST is bfloat16
    {
        // Round to bfloat16 to avoid truncation artifacts from SFPSTORE
        y = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(y, 0));  // SFP_STOCH_RND: fp32 -> bf16
    }

    return y;
}
```

The float-to-int32 conversion helper used within `_sfpu_exp_21f_bf16_`:

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/common/inc/sfpu/ckernel_sfpu_exp.h

sfpi_inline sfpi::vInt _float_to_int32_for_exp_21f_(sfpi::vFloat val)
{
    sfpi::vInt exp = sfpi::exexp(val);   // SFPEXEXP: extract biased exponent
    sfpi::vInt man = sfpi::exman8(val);  // SFPEXMAN: extract mantissa with implicit bit (man8 format)
    man = sfpi::reinterpret<sfpi::vInt>(
        sfpi::shft(sfpi::reinterpret<sfpi::vUInt>(man), exp));  // SFPSHFT: shift mantissa left by exponent
    return man;
}
```

### SFPU Instructions Used

| Instruction | SFPU Opcode | Description | Usage in cosh |
|-------------|-------------|-------------|---------------|
| `SFPLOAD` | 0x70 | Load from DEST row to LREG | Load input element `dst_reg[0]` at start of each iteration |
| `SFPSTORE` | 0x72 | Store LREG to DEST row | Write computed cosh result back to `dst_reg[0]` |
| `SFPMAD` | 0x84 | Fused multiply-add: VD = VA * VB + VC | Core arithmetic: `val * ONE_LN2 + 127`, polynomial evaluation (Horner's chain), `sum * 0.5`, negation of input for exp(-x), addition of exp(x) + exp(-x) |
| `SFPSWAP` | 0x92 | Conditional swap (vec_min_max mode) | Clamping `xlog2` to [0, 255] range (two calls for lower and upper bounds) |
| `SFPEXEXP` | 0x77 | Extract exponent field | Extracting integer part of base-2 representation (`exexp` and `exexp_nodebias`) |
| `SFPEXMAN` | 0x75 | Extract mantissa field | Extracting fractional part (`exman8` for int conversion, `exman9` for fractional part) |
| `SFPSHFT` | 0x7A | Shift operation | Shifting mantissa left by exponent in `_float_to_int32_for_exp_21f_` |
| `SFPCAST` | 0x90 | Format conversion (int32 <-> fp32) | Converting integer fractional part to float for polynomial evaluation |
| `SFPSETEXP` | 0x7C | Set exponent field | Reconstructing `2^(z_i) * 2^(z_f)` by combining polynomial result with integer exponent |
| `SFP_STOCH_RND` | 0x91 | Stochastic/deterministic rounding | Converting fp32 result to bfloat16 (when `is_fp32_dest_acc_en=false`) |
| `SFPLOADI` | 0x71 | Load 16-bit immediate to LREG | Loading constants: thresholds (0.0, 255.0), polynomial coefficients, 0.5 multiplier, 1/ln(2) |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| `LREG0-3` | General-purpose working registers used by SFPI compiler for intermediate values. Holds input value `v`, intermediate exponential results, polynomial coefficients, and final `cosh(x)` result. The SFPI compiler manages register allocation automatically. |
| `dst_reg[0]` | DEST register interface -- reads input and writes output for each sfpi row (32 elements per access, spanning 2 physical DEST rows of 16 elements each) |
| Programmable Constants | `_init_exponential_<false, false, kCONST_1_FP16B>` calls `_init_sfpu_reciprocal_<false>()` which may configure programmable constant registers, but these are NOT used by the `_sfpu_exp_21f_bf16_` compute path. All constants used by cosh are loaded via `SFPLOADI` or embedded as SFPMAD immediate operands. |

### Address Mode Configuration

The address mode for cosh is `ADDR_MOD_7`, configured identically on both Wormhole and Blackhole:

```
ADDR_MOD_7:
  srca.incr  = 0
  srcb.incr  = 0
  dest.incr  = 0
```

This is the standard SFPU address mode for unary operations. The SFPU does NOT use hardware auto-increment for DEST addressing -- instead, the kernel manages DEST row progression explicitly through the SFPI `dst_reg++` abstraction (which advances the SFPU DEST read/write pointer by `SFP_DESTREG_STRIDE=2` physical rows per iteration). Between faces, `SETRWC` instructions in the params dispatch function advance the DEST address by 16 physical rows (2 x `inc_dst_addr<8>()` calls on Wormhole, or `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` on Blackhole).

The `SfpuType::cosh` does not match any of the special-cased types in `eltwise_unary_sfpu_configure_addrmod`, so only `ADDR_MOD_7` is configured (no `ADDR_MOD_6` specialization).

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Determine compute kernel path, init/func names, approx mode, and include guard for COSH
   **Key Findings**: COSH uses `eltwise_sfpu.cpp`, `cosh_tile_init()` / `cosh_tile({idst})`, `math_approx_mode=false`, include guard `SFPU_OP_COSH_INCLUDE`

2. **File**: `tt_metal/hw/inc/api/compute/eltwise_unary/cosh.h`
   **Reason**: API header defining `cosh_tile()` and `cosh_tile_init()` -- entry point for SFPU dispatch
   **Key Findings**: Uses `SFPU_THREE_PARAM_KERNEL_FP32_FIRST(calculate_cosh, APPROX, DST_ACCUM_MODE, 8, idst, (int)VectorMode::RC)` and `SFPU_INIT_KERNEL_CALL(cosh, ckernel::sfpu::cosh_init, APPROX)`

3. **File**: `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_cosh.h`
   **Reason**: Core SFPU implementation of `calculate_cosh` and `cosh_init`
   **Key Findings**: Implements `cosh(x) = (exp(x) + exp(-x)) / 2` using `_sfpu_exp_21f_bf16_` helper. Both architectures have identical source. Init delegates to `_init_exponential_<false, false, kCONST_1_FP16B>()`

4. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_exp.h`
   **Reason**: Contains `_sfpu_exp_21f_bf16_`, `_float_to_int32_for_exp_21f_`, and `_init_exponential_` definitions
   **Key Findings**: exp_21f algorithm from Moroz et al. 2022 -- uses base-2 conversion, clamping, integer/fractional decomposition, 2nd-degree polynomial approximation, and exponent recombination. For APPROXIMATION_MODE=false, init falls to `_init_sfpu_reciprocal_<false>()` (not used by compute path)

5. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_polyval.h`
   **Reason**: Contains `PolynomialEvaluator::eval` used for polynomial approximation in exp_21f
   **Key Findings**: Horner's method via recursive variadic template. `eval(x, c0, c1, c2)` = `c0 + x * (c1 + x * c2)` -- compiles to a chain of SFPMAD instructions

6. **File**: `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h`
   **Reason**: Macro definitions for SFPU dispatch (SFPU_THREE_PARAM_KERNEL_FP32_FIRST, SFPU_INIT_KERNEL_CALL)
   **Key Findings**: `SFPU_THREE_PARAM_KERNEL_FP32_FIRST` expands to `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::FN<APPROXIMATE, FP32, ITER>, DST_IDX, VECTOR_MODE)`

7. **File**: `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Parameters dispatch function managing per-face iteration and DEST addressing
   **Key Findings**: For VectorMode::RC, loops over 4 faces, calling the SFPU function once per face, with SETRWC between faces

8. **File**: `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: Base SFPU init and address mode configuration
   **Key Findings**: `ADDR_MOD_7` with all-zero increments for cosh (no special-cased SfpuType branch applies)

9. **File**: `runtime/sfpi/include/sfpi_lib.h`
   **Reason**: SFPI abstraction function definitions (vec_min_max, exexp, exman, setexp, etc.)
   **Key Findings**: `vec_min_max` -> SFPSWAP, `exexp` -> SFPEXEXP, `exman8`/`exman9` -> SFPEXMAN, `shft` -> SFPSHFT, `int32_to_float` -> SFPCAST, `setexp` -> SFPSETEXP, `float_to_fp16b` -> SFP_STOCH_RND

10. **File**: `tt_metal/jit_build/genfiles.cpp`
    **Reason**: Verify how `APPROX` and `DST_ACCUM_MODE` constexpr bools are generated
    **Key Findings**: `APPROX` is set from `hlk_math_approx_mode` (which comes from `math_approx_mode` in ComputeConfig); `DST_ACCUM_MODE` is set from `fp32_dest_acc_en`
