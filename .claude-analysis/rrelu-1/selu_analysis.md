## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### Unary Dispatch Summary
- **UnaryOpType**: `SELU`
- **Compute kernel**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp` (default path from `get_compute_kernel_path()`)
- **SFPU_OP_CHAIN_0 expansion**: `selu_tile(idst, param0, param1)` where `param0` = scale (bit-cast `uint32_t`), `param1` = alpha (bit-cast `uint32_t`)

**Default parameter values** (from `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp`):
- `scale` = 1.050700987f (standard SELU constant)
- `alpha` = 1.673263242f (standard SELU constant)

#### Approximation Mode Resolution
There are TWO independent controls for approximation. Both are reported below.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(SELU)` in `unary_op_utils.cpp` -- falls through to `default: return false` |
| Template parameter (SFPU_OP_CHAIN) | none | `get_op_init_and_func()` returns `selu_tile_init()` and `selu_tile(idst, param0, param1)` with no approximation template parameter; the API header `selu.h` forwards `APPROX` (which resolves to `false`) |
| Effective SFPU path | `APPROXIMATION_MODE=false` in `calculate_selu`; however, `_sfpu_exp_21f_bf16_` is always called with `is_fp32_dest_acc_en=true` (hardcoded), meaning the exp helper always operates in FP32 precision mode regardless of the APPROXIMATION_MODE template arg | The `APPROXIMATION_MODE` template parameter is declared on `calculate_selu` but is **not used** within the function body -- it is only forwarded as part of the function signature. The exp helper `_sfpu_exp_21f_bf16_<true>` is called with a hardcoded `true` for `is_fp32_dest_acc_en` to avoid premature rounding. |

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/selu.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_selu.h` |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_unary_selu.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |

### Call Chain

1. **Compute kernel** (`eltwise_sfpu.cpp`): `SFPU_OP_CHAIN_0` macro expands to `selu_tile(0, param0, param1)`.
2. **API Header** (`selu.h`): `selu_tile(idst, param0, param1)` calls `MATH((llk_math_eltwise_unary_sfpu_selu<APPROX, DST_ACCUM_MODE>(idst, param0, param1)))`.
3. **LLK Dispatch** (`llk_math_eltwise_unary_sfpu_selu.h`): `llk_math_eltwise_unary_sfpu_selu<APPROXIMATE, is_fp32_dest_acc_en, ITERATIONS>(dst_index, scale, alpha)` calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::calculate_selu<APPROXIMATE, is_fp32_dest_acc_en, ITERATIONS>, dst_index, VectorMode::RC, scale, alpha)`.
4. **Parameters Dispatch** (`llk_math_eltwise_unary_sfpu_params.h`): Sets up DEST addressing, stalls for SFPU availability, then in `VectorMode::RC` mode loops over 4 faces, calling `calculate_selu(scale, alpha)` once per face (8 iterations each), with `TTI_SETRWC` to advance between faces.
5. **Core SFPU Implementation** (`ckernel_sfpu_unary_selu.h`): `calculate_selu<APPROXIMATION_MODE, is_fp32_dest_acc_en, ITERATIONS>(scale, alpha)` -- the actual SFPU kernel function documented below.
6. **Exp helper** (`sfpu/ckernel_sfpu_exp.h` in `tt_llk`): `_sfpu_exp_21f_bf16_<true>(v)` computes exp(v) using the Moroz et al. 2022 polynomial approximation algorithm.

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` -- processes all 4 faces of the tile (full 32x32 = 1024 elements).
- **Operation invocation**: The params dispatch function loops `for (int face = 0; face < 4; face++)`, calling `calculate_selu(scale, alpha)` once per face. Each call processes `ITERATIONS=8` sfpi rows (= 1 face = 256 elements). Between faces, `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` is called twice (advancing 16 physical DEST rows = 1 face worth of rows).
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, SETRWC between faces). The address mode is `ADDR_MOD_7` on both Wormhole and Blackhole, configured with `{srca.incr=0, srcb.incr=0, dest.incr=0}` -- the SFPU kernel handles DEST address advancement explicitly via `dst_reg++` rather than relying on hardware auto-increment.

### Annotated SFPU Kernel Source

The kernel uses SFPI abstractions (`sfpi::vFloat`, `sfpi::dst_reg`, `v_if`/`v_else`/`v_endif`), so Style A is used. The Wormhole and Blackhole implementations are identical.

```cpp
// File: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_unary_selu.h

// SELU(x) = scale * ( max(0, x) + min(0, alpha * (exp(x)-1) ) )
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>
inline void calculate_selu(uint scale, uint alpha) { // APPROXIMATION_MODE=false, is_fp32_dest_acc_en=false, ITERATIONS=8
    sfpi::vFloat scale_value = Converter::as_float(scale); // Reinterpret uint32 bits as float -> SFPLOADI (2 instructions for 32-bit)
    sfpi::vFloat alpha_value = Converter::as_float(alpha); // Reinterpret uint32 bits as float -> SFPLOADI (2 instructions for 32-bit)
#pragma GCC unroll 8

    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0]; // SFPLOAD: load 32 elements from current DEST row into LREG
        v_if(v >= 0.0f) { sfpi::dst_reg[0] = v * scale_value; } // SFPSETCC(GTE0) + SFPMAD(v * scale + 0) + SFPSTORE
        v_else {
            sfpi::vFloat exp_calc = _sfpu_exp_21f_bf16_<true>( // Calls exp helper (detailed below); is_fp32_dest_acc_en=true to avoid premature rounding
                v);
            sfpi::vFloat minus_mul = exp_calc - sfpi::vConst1; // SFPMAD(exp_calc * 1.0 - 1.0); vConst1 = Fixed Const 2 (1.0f)
            sfpi::vFloat result = minus_mul * alpha_value * scale_value; // Two SFPMAD instructions: (minus_mul * alpha + 0) then (tmp * scale + 0)

            if constexpr (!is_fp32_dest_acc_en) { // true when is_fp32_dest_acc_en=false (the default)
                result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0)); // SFP_STOCH_RND: FP32->FP16B with round-to-nearest-even (mode 0)
            }
            sfpi::dst_reg[0] = result; // SFPSTORE: write result back to DEST
        }
        v_endif; // SFPENCC: restore all lanes to active
        sfpi::dst_reg++; // Advance DEST pointer by 1 sfpi row (= 2 physical rows = 32 elements)
    }
}
```

**Exp helper** (`_sfpu_exp_21f_bf16_`) called from the negative branch -- implements `exp(x)` using the Moroz et al. 2022 "exp_21f" algorithm:

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_exp.h

sfpi_inline sfpi::vInt _float_to_int32_for_exp_21f_(sfpi::vFloat val) // Helper: float to int for exp_21f
{
    sfpi::vInt exp = sfpi::exexp(val); // SFPEXEXP: extract biased exponent
    sfpi::vInt man = sfpi::exman8(val); // SFPEXMAN(PAD8): extract mantissa with implicit bit
    man            = sfpi::reinterpret<sfpi::vInt>(sfpi::shft(sfpi::reinterpret<sfpi::vUInt>(man), exp)); // SFPSHFT: shift mantissa left by exponent
    return man;
}

// Implementation notes, see the original file for more details
template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _sfpu_exp_21f_bf16_(sfpi::vFloat val) // is_fp32_dest_acc_en=true (hardcoded by selu)
{
    constexpr float ONE_LN2 = 1.4426950216293334961f;
    sfpi::vFloat xlog2      = (val * ONE_LN2 + 127.f); // SFPLOADI + SFPMAD + SFPLOADI + SFPMAD: x/ln2 + bias

    // Clamp xlog2 to [0, 255] to prevent overflow
    sfpi::vFloat threshold_low  = 0.f;               // SFPLOADI
    sfpi::vFloat threshold_high = sfpi::vFloat(255.f); // SFPLOADI
    sfpi::vec_min_max(threshold_low, xlog2);           // SFPSWAP(VEC_MIN_MAX): threshold_low=min, xlog2=max
    sfpi::vec_min_max(xlog2, threshold_high);          // SFPSWAP(VEC_MIN_MAX): xlog2=min(clamped), threshold_high=max

    sfpi::vInt z = _float_to_int32_for_exp_21f_(xlog2); // SFPEXEXP + SFPEXMAN + SFPSHFT

    sfpi::vInt exponential_part = exexp_nodebias(sfpi::reinterpret<sfpi::vFloat>(z)); // SFPEXEXP(NODEBIAS): extract integer part = 2^(int_part)
    sfpi::vInt fractional_part  = sfpi::exman9(sfpi::reinterpret<sfpi::vFloat>(z));   // SFPEXMAN(PAD9): extract fractional mantissa

    sfpi::vFloat frac = sfpi::int32_to_float(fractional_part, 0); // SFPCAST(INT32_TO_FP32_RNE): convert fractional int to float

    // 2nd-degree polynomial approximation of 2^(fractional) via Horner's method
    frac = PolynomialEvaluator::eval(frac, 1.0017248f, 7.839635491371155e-08f, 4.791750143340323e-15f);
    // Expands to: coeff0 + frac * (coeff1 + frac * coeff2) = chain of SFPMAD instructions (2 MADs + SFPLOADI for coefficients)

    // Recombine exponent and mantissa: 2^(int) * 2^(frac)
    sfpi::vFloat y = sfpi::setexp(frac, exponential_part); // SFPSETEXP: set exponent field of frac to exponential_part

    if constexpr (!is_fp32_dest_acc_en) // false for selu (is_fp32_dest_acc_en=true), so this branch is SKIPPED
    {
        y = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(y, 0));
    }

    return y;
}
```

### SFPU Instructions Used

The following SFPU instructions are emitted by the SFPI compiler from the `calculate_selu` kernel and its `_sfpu_exp_21f_bf16_` helper:

| Instruction | Emitted By | Description |
|-------------|-----------|-------------|
| `SFPLOAD` | `dst_reg[0]` reads | Load 32 elements from DEST row into LREG |
| `SFPSTORE` | `dst_reg[0] = ...` writes | Store LREG value back to DEST row |
| `SFPLOADI` | `Converter::as_float()`, float literal constants (`ONE_LN2`, `127.f`, `0.f`, `255.f`, polynomial coefficients) | Load 16-bit immediate into LREG (two instructions for a full 32-bit float) |
| `SFPMAD` | `v * scale_value`, `exp_calc - vConst1`, `minus_mul * alpha_value`, `tmp * scale_value`, `val * ONE_LN2 + 127.f`, polynomial Horner's evaluation | Fused multiply-add: `VD = VA * VB + VC`. Used for all float arithmetic (add = MAD with multiplier 1.0, subtract = MAD with sign inversion) |
| `SFPSETCC` | `v_if(v >= 0.0f)` | Set CC.Res based on `v >= 0` (mode `LREG_GTE0`, sign bit test) |
| `SFPENCC` | `v_endif` | Re-enable all lanes after conditional block |
| `SFPCOMPC` | `v_else` | Complement CC.Res for else-branch (invert which lanes are active) |
| `SFPPUSHC` | `v_if` (before `v_else`) | Push CC state onto stack to preserve it for the else branch |
| `SFPPOPC` | `v_endif` (restore) | Pop CC state from stack after conditional block completes |
| `SFPEXEXP` | `exexp(val)`, `exexp_nodebias(z)` | Extract exponent field from float (with/without bias subtraction) |
| `SFPEXMAN` | `exman8(val)`, `exman9(z)` | Extract mantissa field with implicit bit (8-bit or 9-bit padding mode) |
| `SFPSHFT` | `shft(man, exp)` | Logical shift of mantissa by exponent amount (within `_float_to_int32_for_exp_21f_`) |
| `SFPSETEXP` | `setexp(frac, exponential_part)` | Set exponent field of float to specified value (recombine 2^int * 2^frac) |
| `SFPSWAP` | `vec_min_max(a, b)` (2 calls) | Vector min/max operation for clamping xlog2 to [0, 255] |
| `SFPCAST` | `int32_to_float(fractional_part, 0)` | Convert INT32 to FP32 with round-to-nearest-even |
| `SFP_STOCH_RND` | `float_to_fp16b(result, 0)` | Round FP32 to FP16B (bfloat16) with round-to-nearest-even; used in the non-FP32 DEST accumulation path |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **LREG0-3** | General-purpose working registers. Used by the SFPI compiler for intermediate values: input `v`, `scale_value`, `alpha_value`, `exp_calc`, `minus_mul`, `result`, and temporaries within the `_sfpu_exp_21f_bf16_` helper (xlog2, thresholds, z, exponential_part, fractional_part, frac, polynomial intermediates, y). The exact register allocation is determined by the SFPI compiler backend. |
| **LREG4-7** | Additional GPRs available if the compiler needs more than 4 registers (SFPI has 8 LREGs per lane). LREG7 can serve as an indirect addressing register for SFPMAD, though this kernel does not use indirect mode. |
| **DEST rows** | Source and destination for tile data. `dst_reg[0]` accesses the current pair of physical DEST rows (32 elements). `dst_reg++` advances by 1 sfpi row = 2 physical DEST rows. |
| **Fixed Const 2 (index 10)** | Value 1.0f, accessed via `sfpi::vConst1`. Used in `exp_calc - vConst1` (i.e., exp(x) - 1). |

### Address Mode Configuration

The address mode is configured by `eltwise_unary_sfpu_configure_addrmod<SfpuType::selu>()` during initialization. Since `SfpuType::selu` does not match any of the special-cased `if constexpr` branches (topk_local_sort, typecast, unary_max/min, signbit), only the default `ADDR_MOD_7` is set.

**Wormhole B0 and Blackhole** (identical configuration):

| ADDR_MOD | srca.incr | srcb.incr | dest.incr | Purpose |
|----------|-----------|-----------|-----------|---------|
| `ADDR_MOD_7` | 0 | 0 | 0 | Default for SFPU operations -- no hardware auto-increment. DEST address advancement is handled explicitly by `dst_reg++` (which emits SFPU address pointer increment instructions) within the kernel loop. |

The `ADDR_MOD_7` slot is chosen to avoid conflicts with `ADDR_MOD_0` and `ADDR_MOD_2`, which are used by the A2D (Accumulate-to-DEST) copy operation that runs before the SFPU kernel.

## External Knowledge Sources
### DeepWiki Queries
1. [SFPU] **Query**: "How does the SFPU exp implementation work in tt-metal? Specifically, what SFPU instructions are emitted by _sfpu_exp_21f_bf16_ and its helpers?"
   **Reason**: Needed to understand the instruction-level mapping of SFPI intrinsics to hardware SFPU instructions.
   **Key Findings**: DeepWiki was unavailable (repository not indexed). Analysis was completed using source code inspection only. SFPI intrinsic-to-instruction mappings were traced through `runtime/sfpi/include/sfpi_lib.h` and `runtime/sfpi/include/sfpi.h`, which map C++ wrappers to `__builtin_rvtt_sfp*` compiler intrinsics that emit the corresponding SFPU instructions.

### Confluence References
No Confluence pages were consulted for this analysis. The SFPU ISA reference (page 1170505767) was not needed because all instruction semantics were available from the pre-compiled hardware model reference.

### Glean References
No Glean searches were performed for this analysis.
