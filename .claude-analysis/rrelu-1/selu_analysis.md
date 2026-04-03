## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### Unary Dispatch Summary
- **UnaryOpType**: `SELU`
- **Compute kernel**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp` (default path)
- **SFPU_OP_CHAIN_0 expansion**: `selu_tile(idst, param0, param1)` where `param0` = bit-cast of `scale` float and `param1` = bit-cast of `alpha` float. Default values from Python binding: `scale = 1.0507`, `alpha = 1.67326`.

#### Approximation Mode Resolution
There are TWO independent controls for approximation. Both are reported below.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(SELU)` in `unary_op_utils.cpp` -- the switch has only `default: return false`, so all ops return false |
| Template parameter (SFPU_OP_CHAIN) | none (no template parameter in SFPU_OP_CHAIN_0) | `get_op_init_and_func()` -- SELU case emits `selu_tile_init()` (no template args) and `selu_tile(idst, param0, param1)` (no template args). The API header `selu.h` calls `llk_math_eltwise_unary_sfpu_selu<APPROX, DST_ACCUM_MODE>(...)` where `APPROX` is the global `APPROX` define derived from `math_approx_mode` |
| Effective SFPU path | `APPROXIMATION_MODE=false` is passed to `calculate_selu<false, false, 8>`. However, `calculate_selu` does NOT use `APPROXIMATION_MODE` directly -- it always calls `_sfpu_exp_21f_bf16_<true>(v)` with hardcoded `true` for `is_fp32_dest_acc_en` to avoid intermediate rounding inside the exp helper | The `_sfpu_exp_21f_bf16_<true>` call on line 26 of `ckernel_sfpu_unary_selu.h` |

### SFPU Abstraction Layers
| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/selu.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_selu.h` |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_unary_selu.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |

### Call Chain
1. **`selu_tile(idst, param0, param1)`** (API header `selu.h`): wraps `MATH(llk_math_eltwise_unary_sfpu_selu<APPROX, DST_ACCUM_MODE>(idst, param0, param1))`, routing the call to the math thread.
2. **`llk_math_eltwise_unary_sfpu_selu<APPROXIMATE, is_fp32_dest_acc_en, ITERATIONS=8>(dst_index, scale, alpha, VectorMode::RC)`** (LLK dispatch): calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>` passing a function pointer to `calculate_selu<APPROXIMATE, false, 8>` along with `scale` and `alpha` runtime parameters.
3. **`_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(...)`** (params dispatch): sets DEST write address, sets address mode base, stalls for SFPU availability, then iterates over 4 faces calling `calculate_selu(scale, alpha)` once per face with `SETRWC` between faces.
4. **`calculate_selu<false, false, 8>(scale, alpha)`** (core SFPU implementation): the actual SFPU kernel that processes 8 sfpi rows per face (256 elements per face).

### Parameters Dispatch Summary
- **Vector mode**: `VectorMode::RC` (default for SELU). All 4 faces of the tile are processed, covering 4 x 256 = 1024 elements.
- **Operation invocation**: The dispatch calls `calculate_selu(scale, alpha)` once per face in a 4-iteration loop. Each invocation of `calculate_selu` processes `ITERATIONS=8` sfpi rows (one full face).
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, `SETRWC` between faces). Between faces, two `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` calls advance the DEST read/write counter by 16 physical rows (1 face). The address mode is `ADDR_MOD_7` configured with `dest.incr = 0` (the SFPI `dst_reg++` abstraction manages per-iteration advancement internally).

### Annotated SFPU Kernel Source

The kernel uses SFPI abstractions (Style A). The Wormhole B0 and Blackhole implementations are **identical**.

```cpp
// File: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_unary_selu.h

namespace ckernel {
namespace sfpu {

// SELU(x) = scale * ( max(0, x) + min(0, alpha * (exp(x)-1) ) )
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>
inline void calculate_selu(uint scale, uint alpha) { // APPROXIMATION_MODE=false, is_fp32_dest_acc_en=false, ITERATIONS=8
    sfpi::vFloat scale_value = Converter::as_float(scale); // reinterpret uint32 bits as float, then load as vFloat via SFPLOADI
    sfpi::vFloat alpha_value = Converter::as_float(alpha); // same for alpha
#pragma GCC unroll 8

    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0]; // SFPLOAD from current DEST row pair
        v_if(v >= 0.0f) { sfpi::dst_reg[0] = v * scale_value; } // x >= 0: result = x * scale (SFPMAD + SFPSTORE)
        v_else {
            sfpi::vFloat exp_calc = _sfpu_exp_21f_bf16_<true>( // compute exp(x) using Moroz et al. 2022 algorithm
                v);  // is_fp32_dest_acc_en set to true to avoid rounding as it has to be done at the end of operation
            sfpi::vFloat minus_mul = exp_calc - sfpi::vConst1; // exp(x) - 1.0 (SFPMAD: exp_calc * 1.0 + (-1.0))
            sfpi::vFloat result = minus_mul * alpha_value * scale_value; // alpha * (exp(x)-1) * scale (two SFPMADs)

            if constexpr (!is_fp32_dest_acc_en) { // true when is_fp32_dest_acc_en=false (the default path)
                result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0)); // SFP_STOCH_RND: round FP32 to BF16 (round-to-nearest-even, rounding=0)
            }
            sfpi::dst_reg[0] = result; // SFPSTORE result back to current DEST row pair
        }
        v_endif;
        sfpi::dst_reg++; // advance to next sfpi row (next 32 elements)
    }
}

}  // namespace sfpu
}  // namespace ckernel
```

**The `_sfpu_exp_21f_bf16_<true>` helper** (called from the negative branch) is a shared exponential function from `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_exp.h`. It implements `exp(x)` using the algorithm from "Simple Multiple Precision Algorithms for Exponential Functions" by Moroz et al. 2022. The key steps are:

1. `xlog2 = val * (1/ln2) + 127.0` -- SFPMAD (multiply by 1/ln2 and add bias)
2. Clamp `xlog2` to `[0, 255]` via two `vec_min_max` calls -- two SFPSWAP instructions
3. `z = _float_to_int32_for_exp_21f_(xlog2)` -- SFPEXEXP (extract exponent) + SFPEXMAN (extract mantissa with implicit bit) + SFPSHFT (shift mantissa by exponent)
4. `exponential_part = exexp_nodebias(z)` -- SFPEXEXP (extract exponent without debiasing)
5. `fractional_part = exman9(z)` -- SFPEXMAN (extract 9-bit mantissa)
6. `frac = int32_to_float(fractional_part, 0)` -- SFPCAST (INT32 to FP32)
7. `frac = PolynomialEvaluator::eval(frac, c0, c1, c2)` -- chain of 2 SFPMADs (Horner's method: `c0 + frac*(c1 + frac*c2)`)
8. `y = setexp(frac, exponential_part)` -- SFPSETEXP (combine mantissa with exponent)
9. Since `is_fp32_dest_acc_en=true` for this call, the final BF16 rounding inside the exp helper is **skipped**. The rounding is instead done at the end of `calculate_selu` itself.

### SFPU Instructions Used

| Instruction | SFPI Abstraction | Usage in SELU |
|------------|-----------------|---------------|
| **SFPLOAD** | `dst_reg[0]` (read) | Load input element from DEST register into LREG |
| **SFPSTORE** | `dst_reg[0] = ...` (write) | Store computed result back to DEST register |
| **SFPLOADI** | `vFloat(scalar)`, `Converter::as_float()` | Load immediate scalar constants (scale, alpha, 1/ln2, 127.0, 0.0, 255.0, polynomial coefficients) into LREGs |
| **SFPMAD** | `vFloat * vFloat`, `vFloat + vFloat`, `vFloat - vFloat` | All float arithmetic: `v * scale_value`, `exp_calc - vConst1`, `minus_mul * alpha_value`, result `* scale_value`, and all operations inside `_sfpu_exp_21f_bf16_` (multiply, add, polynomial Horner chain) |
| **SFPSETCC** | `v_if(v >= 0.0f)` | Set condition code based on sign of input value (GTE0 test) to enable the positive branch |
| **SFPCOMPC** | `v_else` | Complement condition code to switch to the negative branch |
| **SFPENCC** | `v_if` entry / `v_endif` exit | Enable/disable condition code masking at the start and end of the conditional block |
| **SFPPUSHC** | `v_if` (implicit) | Push CC state onto stack for nested conditional support |
| **SFPPOPC** | `v_endif` (implicit) | Pop CC state from stack to restore prior CC state |
| **SFPSWAP** | `vec_min_max(a, b)` | Used in `_sfpu_exp_21f_bf16_` for clamping `xlog2` to `[0, 255]` (2 calls for min and max) |
| **SFPEXEXP** | `exexp_nodebias(z)`, `exexp(z)` | Extract exponent field from float (used twice in exp helper: once for float-to-int conversion, once for recombination) |
| **SFPEXMAN** | `exman8(v)`, `exman9(z)` | Extract mantissa field with implicit bit (used in exp helper for float-to-int and fractional part extraction) |
| **SFPSHFT** | `shft(man, exp)` | Shift mantissa by exponent value (used in `_float_to_int32_for_exp_21f_` to convert float to integer representation) |
| **SFPSETEXP** | `setexp(frac, exponential_part)` | Set exponent field of a float (recombine mantissa polynomial result with integer exponent in exp helper) |
| **SFPCAST** | `int32_to_float(fractional_part, 0)` | Convert INT32 to FP32 (used in exp helper to convert extracted mantissa to float for polynomial evaluation) |
| **SFP_STOCH_RND** | `float_to_fp16b(result, 0)` | Round FP32 to BF16 format using round-to-nearest-even (used at the end of negative branch, and also in the exp helper's `!is_fp32_dest_acc_en` path which is skipped in SELU since `true` is hardcoded) |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **DEST** | Source and destination for tile data. Input elements are loaded from DEST via `SFPLOAD`, results are stored back via `SFPSTORE`. |
| **LREG0-3** (LREGS1 bank) | Used implicitly by SFPI for temporary values: input `v`, `scale_value`, `alpha_value`, intermediate results (`exp_calc`, `minus_mul`, `result`). The SFPI compiler allocates these registers automatically. |
| **LREG4-7** (LREGS2 bank) | Available for additional temporaries needed by the `_sfpu_exp_21f_bf16_` helper (which uses multiple intermediate values: `xlog2`, `threshold_low`, `threshold_high`, `z`, `exponential_part`, `fractional_part`, `frac`, `y`). The compiler manages spilling between LREGs as needed. |
| **Constant registers** | `vConst1` (Fixed Const 2 = 1.0f, CREG index 10) is used for `exp(x) - 1.0`. The `vConst0p8373` (Fixed Const 0 = 0.8373) is NOT used by this kernel (it is used by the older `_sfpu_exp_bf16_` path which SELU does not call). |

### Address Mode Configuration

The address mode is set during `selu_tile_init()` which calls `llk_math_eltwise_unary_sfpu_init<SfpuType::selu, APPROXIMATE>()`, which in turn calls `eltwise_unary_sfpu_configure_addrmod<SfpuType::selu>()`.

Since `SfpuType::selu` does not match any of the special-cased `if constexpr` branches (`topk_local_sort`, `typecast`, `reciprocal`, etc.), only the default `ADDR_MOD_7` is configured:

**Wormhole B0 and Blackhole (identical)**:
```
ADDR_MOD_7: { .srca = { .incr = 0 }, .srcb = { .incr = 0 }, .dest = { .incr = 0 } }
```

- **`dest.incr = 0`**: The SFPU load/store address does NOT auto-increment after each SFPLOAD/SFPSTORE. Instead, per-iteration DEST advancement is handled by the SFPI `dst_reg++` abstraction (which increments the internal DEST read/write counter by `SFP_DESTREG_STRIDE = 2` physical rows per iteration).
- **Between faces**: `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` is called twice per face transition (in `_llk_math_eltwise_unary_sfpu_params_`), advancing the DEST counter by 8+8 = 16 physical rows = 1 face.
- **Both hardware generations use the same ADDR_MOD_7 configuration** with all increments set to 0.

## External Knowledge Sources
### DeepWiki Queries
1. **Query**: "How does the SFPU exp function _sfpu_exp_21f_bf16_ work in the LLK layer? What instructions does it emit?"
   **Reason**: Needed to understand the instruction-level details of the exponential helper called by SELU.
   **Key Findings**: DeepWiki returned "Repository not found" -- the tt-metal repository is not indexed. Analysis proceeded using source code alone.

2. **Query**: "What SFPU instructions does sfpi::PolynomialEvaluator::eval generate? How does the SFPI vFloat abstraction translate to SFPMAD instructions?"
   **Reason**: Needed to confirm instruction-level mapping of SFPI abstractions.
   **Key Findings**: DeepWiki returned "Repository not found". Instruction mappings were verified directly from `runtime/sfpi/include/sfpi.h` and `sfpi_lib.h` source code.

### Confluence References
No Confluence pages were consulted for this analysis. The SFPU hardware model reference document (`.claude/references/sfpu-hardware-model.md`) provided sufficient instruction semantics.

### Glean References
No Glean queries were made for this analysis.
