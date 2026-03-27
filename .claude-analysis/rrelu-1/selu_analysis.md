## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to for the SELU unary operation.

### Unary Dispatch Summary
- **UnaryOpType**: `SELU`
- **Compute kernel**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp` (default case in `get_compute_kernel_path()`)
- **SFPU_OP_CHAIN_0 expansion**: `selu_tile(0, 0x{scale_hex}u, 0x{alpha_hex}u)` where `scale` and `alpha` are the two required float parameters bit-cast to `uint32_t`

#### Approximation Mode Resolution
There are TWO independent controls for approximation. Both are reported below.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(SELU)` in `unary_op_utils.cpp` -- falls through to `default: return false` (no explicit SELU case) |
| Template parameter (SFPU_OP_CHAIN) | none (non-parameterized) | `get_op_init_and_func()` -- SELU case uses `selu_tile_init()` and `selu_tile(idst, param0_hex, param1_hex)` with no template arguments; `APPROX` is resolved from `ComputeConfig.math_approx_mode` at JIT compile time via `constexpr bool APPROX` in `genfiles.cpp` |
| Effective SFPU path | `APPROXIMATION_MODE=false` in `calculate_selu`, but the inner `_sfpu_exp_21f_bf16_` is always called with `is_fp32_dest_acc_en=true` (hardcoded at the call site) to avoid intermediate rounding. The `APPROXIMATION_MODE` template parameter is not used inside `calculate_selu` itself -- it is only passed through for potential future use. The exp helper `_sfpu_exp_21f_bf16_` always uses the same polynomial approximation regardless of `APPROXIMATION_MODE`. | `ckernel_sfpu_unary_selu.h` line 26: `_sfpu_exp_21f_bf16_<true>(v)` |

### SFPU Abstraction Layers
| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/selu.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_selu.h` (identical on Blackhole: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_selu.h`) |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_unary_selu.h` (identical on Blackhole: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_unary_selu.h`) |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (Blackhole: `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu_params.h`) |

### Call Chain
1. **Compute kernel** (`eltwise_sfpu.cpp`): The `SFPU_OP_CHAIN_0` macro expands to `selu_tile(0, scale_hex, alpha_hex)`.
2. **API header** (`selu.h`): `selu_tile(idst, param0, param1)` dispatches via `MATH((...))` to `llk_math_eltwise_unary_sfpu_selu<APPROX, DST_ACCUM_MODE>(idst, param0, param1)`.
3. **LLK dispatch** (`llk_math_eltwise_unary_sfpu_selu.h`): Calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>()` with the function pointer `ckernel::sfpu::calculate_selu<APPROXIMATE, is_fp32_dest_acc_en, ITERATIONS>` and the `scale`/`alpha` arguments.
4. **Parameters dispatch** (`llk_math_eltwise_unary_sfpu_params.h`): Sets DEST write address, stalls for SFPU readiness, then loops over 4 faces in `VectorMode::RC`, calling `calculate_selu(scale, alpha)` once per face, with `SETRWC` (or `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` on Blackhole) between faces to advance the DEST base address.
5. **Core SFPU** (`ckernel_sfpu_unary_selu.h`): `calculate_selu()` performs 8 iterations per face, computing SELU element-wise on 32 elements per iteration.

### Parameters Dispatch Summary
- **Vector mode**: `VectorMode::RC` (default). All 4 faces of the tile are processed (32x32 = 1024 elements total).
- **Operation invocation**: `calculate_selu(scale, alpha)` is called once per face (4 times total). Each invocation runs its internal loop of `ITERATIONS=8` iterations. The `scale` and `alpha` parameters are forwarded via variadic `Args&&...` from the params dispatch.
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, SETRWC between faces). On Wormhole, `ADDR_MOD_7` is configured with `.dest = {.incr = 0}` (the SFPI `dst_reg++` handles within-face advancement); between faces, two `TTI_SETRWC(CR_D, 8)` calls advance by 16 physical DEST rows (= 1 face). On Blackhole, `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` calls `math::inc_dst_addr<8>()` twice for the same effect.

### Annotated SFPU Kernel Source

The SELU kernel uses SFPI abstractions (`sfpi::vFloat`, `dst_reg`, `v_if`/`v_else`/`v_endif`), so Style A (inline-commented source) is used.

```cpp
// File: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_unary_selu.h

namespace ckernel {
namespace sfpu {

// SELU(x) = scale * ( max(0, x) + min(0, alpha * (exp(x)-1) ) )
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>
inline void calculate_selu(uint scale, uint alpha) { // APPROXIMATION_MODE=false, is_fp32_dest_acc_en=false, ITERATIONS=8
    sfpi::vFloat scale_value = Converter::as_float(scale);  // reinterpret uint32_t bits as float -> SFPLOADI
    sfpi::vFloat alpha_value = Converter::as_float(alpha);  // reinterpret uint32_t bits as float -> SFPLOADI
#pragma GCC unroll 8

    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];  // SFPLOAD: load 32 elements from current DEST row pair into LREG
        v_if(v >= 0.0f) { sfpi::dst_reg[0] = v * scale_value; }  // CC set by comparison; if x>=0: result = x * scale -> SFPMAD + SFPSTORE
        v_else {
            sfpi::vFloat exp_calc = _sfpu_exp_21f_bf16_<true>(
                v);  // is_fp32_dest_acc_en set to true to avoid rounding as it has to be done at the end of operation
            sfpi::vFloat minus_mul = exp_calc - sfpi::vConst1;  // exp(x) - 1.0 -> SFPMAD (a*1.0 + (-1.0))
            sfpi::vFloat result = minus_mul * alpha_value * scale_value;  // alpha * (exp(x)-1) * scale -> 2x SFPMAD

            if constexpr (!is_fp32_dest_acc_en) {  // true: DEST is bfloat16, so explicit rounding needed
                result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0));  // round-to-nearest-even to bf16
            }
            sfpi::dst_reg[0] = result;  // SFPSTORE: write result back to DEST
        }
        v_endif;
        sfpi::dst_reg++;  // advance 1 sfpi row = 2 physical DEST rows = 32 elements
    }
}

}  // namespace sfpu
}  // namespace ckernel
```

The helper function `_sfpu_exp_21f_bf16_<true>` (called with `is_fp32_dest_acc_en=true` to suppress intermediate rounding within the exp computation) is defined in the tt_llk shared SFPU library:

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_exp.h

// Helper: branch-free float-to-int32 for exp_21f (constraint: 0 <= val < 128.0f)
sfpi_inline sfpi::vInt _float_to_int32_for_exp_21f_(sfpi::vFloat val)
{
    sfpi::vInt exp = sfpi::exexp(val);    // SFPEXEXP: extract biased exponent
    sfpi::vInt man = sfpi::exman8(val);   // SFPEXMAN: extract mantissa with implicit bit
    man            = sfpi::reinterpret<sfpi::vInt>(sfpi::shft(sfpi::reinterpret<sfpi::vUInt>(man), exp));  // SFPSHFT: left-shift mantissa by exponent
    return man;
}

// Implementation notes, see the original file for more details
template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _sfpu_exp_21f_bf16_(sfpi::vFloat val) // is_fp32_dest_acc_en=true (as called from SELU)
{
    // exp(x) = 2^(x/ln2) = 2^(x_i) * 2^(x_f)
    constexpr float ONE_LN2 = 1.4426950216293334961f;
    sfpi::vFloat xlog2      = (val * ONE_LN2 + 127.f);  // SFPMAD: x * (1/ln2) + 127 (bias)

    // Clamp to [0, 255] to prevent overflow
    sfpi::vFloat threshold_low  = 0.f;                   // SFPLOADI
    sfpi::vFloat threshold_high = sfpi::vFloat(255.f);    // SFPLOADI
    sfpi::vec_min_max(threshold_low, xlog2);              // SFPSWAP: sorts the two vFloats (low gets min, xlog2 gets max)
    sfpi::vec_min_max(xlog2, threshold_high);             // SFPSWAP: xlog2 gets min of (xlog2, 255)

    sfpi::vInt z = _float_to_int32_for_exp_21f_(xlog2);  // Convert to integer representation (scaled by 2^23)

    sfpi::vInt exponential_part = exexp_nodebias(sfpi::reinterpret<sfpi::vFloat>(z)); // SFPEXEXP: extract exponent without debiasing
    sfpi::vInt fractional_part  = sfpi::exman9(sfpi::reinterpret<sfpi::vFloat>(z));   // SFPEXMAN: extract mantissa (9 zero-padded bits, no hidden bit)

    sfpi::vFloat frac = sfpi::int32_to_float(fractional_part, 0);  // SFPCAST: int32 -> float

    // 2nd degree polynomial approximation of 2^x on fractional part via Horner's method
    // eval(frac, c0, c1, c2) = c0 + frac * (c1 + frac * c2) -> chain of SFPMAD instructions
    frac = PolynomialEvaluator::eval(frac, 1.0017248f, 7.839635491371155e-08f, 4.791750143340323e-15f);

    // Recombine: 2^(x_i) * 2^(x_f) by setting exponent
    sfpi::vFloat y = sfpi::setexp(frac, exponential_part);  // SFPSETEXP: replace exponent field

    if constexpr (!is_fp32_dest_acc_en)  // false when called from SELU (template arg is true), so this block is SKIPPED
    {
        y = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(y, 0));
    }

    return y;
}
```

### SFPU Instructions Used

| Instruction / Intrinsic | Description |
|------------------------|-------------|
| `SFPLOAD` | Loads 32 elements from a DEST row pair into an LREG. Emitted by `dst_reg[0]` read. |
| `SFPSTORE` | Writes 32 elements from an LREG back to a DEST row pair. Emitted by `dst_reg[0] = ...` write. |
| `SFPLOADI` | Loads an immediate constant into an LREG. Emitted by `Converter::as_float()`, `vFloat(255.f)`, `0.f`, and polynomial coefficients. |
| `SFPMAD` | Multiply-accumulate: `a * b + c`. Emitted by all `vFloat` multiplications (`v * scale_value`, `minus_mul * alpha_value`, etc.) and additions/subtractions (`exp_calc - vConst1` becomes `exp_calc * 1.0 + (-1.0)`, `val * ONE_LN2 + 127.f`). The PolynomialEvaluator Horner chain produces 2 SFPMADs for the degree-2 polynomial. |
| `SFPSETCC` / `SFPENCC` | Set/enable condition codes for predicated execution. Emitted by `v_if(v >= 0.0f)` (comparison sets CC), `v_else` (inverts CC), `v_endif` (restores CC). |
| `SFPEXEXP` | Extract the 8-bit biased exponent from a float. Emitted by `sfpi::exexp(val)` and `exexp_nodebias()`. |
| `SFPEXMAN` | Extract the mantissa from a float. Emitted by `sfpi::exman8(val)` (with hidden bit, 8 zero-padded) and `sfpi::exman9()` (without hidden bit, 9 zero-padded). |
| `SFPSHFT` | Variable left/right shift on integer data. Emitted by `sfpi::shft()` in `_float_to_int32_for_exp_21f_`. |
| `SFPSETEXP` | Replace the exponent field of a float while preserving sign and mantissa. Emitted by `sfpi::setexp(frac, exponential_part)`. |
| `SFPSWAP` | Conditional swap of two registers (vector min/max). Emitted by `sfpi::vec_min_max()` for clamping. |
| `SFPCAST` | Convert between integer and float formats. Emitted by `sfpi::int32_to_float()` and `sfpi::float_to_fp16b()`. |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **DEST rows** | Source and destination for tile data. `dst_reg[0]` reads/writes the current DEST row pair (2 physical rows, 32 elements). `dst_reg++` advances to the next row pair. |
| **LREG0-LREG3** | General-purpose SFPU local registers (32-wide vectors). Used implicitly by the SFPI compiler to hold intermediate values: `v`, `scale_value`, `alpha_value`, `exp_calc`, `minus_mul`, `result`, and all intermediates within `_sfpu_exp_21f_bf16_` (xlog2, z, exponential_part, fractional_part, frac, y, threshold_low, threshold_high). The compiler allocates LREGs automatically; spills to DEST if needed. |
| **LREG4-LREG11** | Extended local registers. Available for compiler use but typically not needed for this kernel's complexity level. |
| **CC (Condition Code)** | Set by the comparison `v >= 0.0f` in `v_if`. Controls which SIMD lanes execute the positive branch (scale * x) vs the negative branch (scale * alpha * (exp(x) - 1)). Both branches execute as straight-line code; only the CC-enabled lanes write results. |

### Address Mode Configuration

SELU uses `SfpuType::selu` for initialization. In `eltwise_unary_sfpu_configure_addrmod<SfpuType::selu>()`, only `ADDR_MOD_7` is configured (the generic default for all SFPU ops that do not match special cases like topk, typecast, or max/min):

| Field | Value | Meaning |
|-------|-------|---------|
| `ADDR_MOD_7.srca.incr` | 0 | No auto-increment on SRC_A (not used by SFPU) |
| `ADDR_MOD_7.srcb.incr` | 0 | No auto-increment on SRC_B (not used by SFPU) |
| `ADDR_MOD_7.dest.incr` | 0 | No hardware auto-increment on DEST (SFPI `dst_reg++` handles advancement explicitly) |

This configuration is identical on both Wormhole B0 and Blackhole. The DEST address progression is managed entirely by SFPI's `dst_reg++` (within a face, each `++` advances 1 sfpi row = 2 physical rows) and the params dispatch layer's face-stride mechanism (`TTI_SETRWC` on WH, `math::inc_dst_addr<8>()` on BH) between faces.

## External Knowledge Sources
### DeepWiki Queries
1. **Query**: "What SFPU instructions are emitted by sfpi::vFloat operations like multiplication, subtraction, comparison, and conditional execution? How do dst_reg loads/stores, exexp, exman8, exman9, shft, setexp, float_to_fp16b, and int32_to_float intrinsics work?"
   **Reason**: Needed to map high-level SFPI abstractions used in `calculate_selu` and `_sfpu_exp_21f_bf16_` to their underlying SFPU instructions for the "SFPU Instructions Used" table.
   **Key Findings**: Confirmed that `vFloat * vFloat` emits `SFPMAD`, `dst_reg[0]` read/write emits `SFPLOAD`/`SFPSTORE`, `v_if`/`v_else`/`v_endif` use `SFPSETCC`/`SFPENCC` for predicated execution, `exexp` emits `SFPEXEXP`, `exman8`/`exman9` emit `SFPEXMAN`, `shft` emits `SFPSHFT`, `setexp` emits `SFPSETEXP`, `vec_min_max` emits `SFPSWAP`, `int32_to_float` and `float_to_fp16b` emit `SFPCAST`. The SFPI compiler translates all arithmetic into SFPMAD chains.

### Confluence References
No Confluence references were needed for this analysis. The SFPU instructions used by SELU (SFPMAD, SFPLOAD, SFPSTORE, SFPLOADI, SFPSETCC, SFPENCC, SFPEXEXP, SFPEXMAN, SFPSHFT, SFPSETEXP, SFPSWAP, SFPCAST) are standard SFPI intrinsics well-documented in the codebase and DeepWiki.

### Glean References
No Glean references were needed for this analysis.
