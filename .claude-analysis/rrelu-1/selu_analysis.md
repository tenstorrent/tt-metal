## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### Unary Dispatch Summary
- **UnaryOpType**: `SELU`
- **Compute kernel**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp` (default path)
- **SFPU_OP_CHAIN_0 expansion**: `selu_tile(0, param0_hex, param1_hex)` where `param0` = scale (default 1.050700987f) and `param1` = alpha (default 1.673263242f), each bit-cast to `uint32_t`

#### Approximation Mode Resolution
There are TWO independent controls for approximation. Both are reported below.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(SELU)` in `unary_op_utils.cpp` -- switch has only a `default: return false` case |
| Template parameter (SFPU_OP_CHAIN) | none (no approximation template param in SELU chain) | `get_op_init_and_func()` -- `selu_tile_init()` and `selu_tile(idst, param0, param1)` have no approximation template parameter |
| Effective SFPU path | `APPROX` resolves to `false`. In `calculate_selu`, `APPROXIMATION_MODE=false`. The `_sfpu_exp_21f_bf16_` helper is called with `is_fp32_dest_acc_en=true` (hardcoded in the SELU kernel to preserve intermediate precision), so no BF16 rounding is applied inside the exp helper. A final `float_to_fp16b` rounding is applied in the SELU kernel when `is_fp32_dest_acc_en=false`. | `_sfpu_exp_21f_bf16_<true>(v)` at line 26 of `ckernel_sfpu_unary_selu.h` |

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/selu.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_selu.h` (WH) / `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_selu.h` (BH) |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_unary_selu.h` (WH) / `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_unary_selu.h` (BH) |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (WH) / `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (BH) |

### Call Chain

1. **`selu_tile(idst, param0, param1)`** (API Header `selu.h`): Wraps the call in `MATH(...)`, invoking `llk_math_eltwise_unary_sfpu_selu<APPROX, DST_ACCUM_MODE>(idst, param0, param1)`.
2. **`llk_math_eltwise_unary_sfpu_selu<APPROXIMATE, is_fp32_dest_acc_en, ITERATIONS>`** (LLK Dispatch): Delegates to `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(calculate_selu<APPROXIMATE, is_fp32_dest_acc_en, ITERATIONS>, dst_index, VectorMode::RC, scale, alpha)`.
3. **`_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>`** (Parameters Dispatch): Sets up DEST addressing, stalls for SFPU readiness, then iterates over 4 faces in `VectorMode::RC` mode, calling `calculate_selu(scale, alpha)` once per face (8 iterations each). Issues `TTI_SETRWC` (WH) or `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` (BH) between faces.
4. **`calculate_selu<APPROXIMATION_MODE, is_fp32_dest_acc_en, ITERATIONS>(scale, alpha)`** (Core SFPU Implementation): The actual SFPU compute function. Executes 8 iterations per face, reading/writing `dst_reg[0]` and advancing `dst_reg++` each iteration.

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` -- all 4 faces of the tile are processed (4 x 8 iterations = 32 sfpi rows = full tile).
- **Operation invocation**: The core function `calculate_selu(scale, alpha)` is called 4 times (once per face). Each invocation executes an internal loop of `ITERATIONS=8`, processing all 8 sfpi rows within that face.
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, SETRWC between faces). On Wormhole, the params dispatch uses `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` twice between faces (advancing by 16 physical DEST rows = 8 sfpi rows). On Blackhole, the equivalent `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` calls `math::inc_dst_addr<8>()` twice. The address mode is `ADDR_MOD_7` on both WH and BH, configured with `{srca=0, srcb=0, dest=0}` (no auto-increment -- the kernel manages DEST advancement explicitly via `dst_reg++`).

### Annotated SFPU Kernel Source

The kernel uses SFPI abstractions (`sfpi::vFloat`, `sfpi::vInt`, `dst_reg`, `v_if`, `v_else`, `v_endif`), so Style A annotation is used.

```cpp
// File: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_unary_selu.h

// SELU(x) = scale * ( max(0, x) + min(0, alpha * (exp(x)-1) ) )
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>
inline void calculate_selu(uint scale, uint alpha) { // APPROXIMATION_MODE=false, is_fp32_dest_acc_en=false, ITERATIONS=8
    sfpi::vFloat scale_value = Converter::as_float(scale);  // bit-cast uint32 param to float, loaded into LREG
    sfpi::vFloat alpha_value = Converter::as_float(alpha);   // bit-cast uint32 param to float, loaded into LREG
#pragma GCC unroll 8

    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0]; // SFPLOAD: load 32 elements from current DEST row pair into LREG
        v_if(v >= 0.0f) { sfpi::dst_reg[0] = v * scale_value; } // CC set for lanes where v >= 0; SFPMAD: v*scale+0; SFPSTORE guarded by CC
        v_else { // SFPCOMPC: invert CC to select lanes where v < 0
            sfpi::vFloat exp_calc = _sfpu_exp_21f_bf16_<true>( // is_fp32_dest_acc_en=true to avoid premature rounding
                v);
            sfpi::vFloat minus_mul = exp_calc - sfpi::vConst1; // SFPMAD: exp_calc * 1.0 + (-1.0)
            sfpi::vFloat result = minus_mul * alpha_value * scale_value; // Two SFPMADs: (minus_mul * alpha) then (intermediate * scale)

            if constexpr (!is_fp32_dest_acc_en) { // true when DST is BF16 (the common case)
                result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0)); // SFPSTOCHRND: round FP32 to BF16 with round-to-nearest-even
            }
            sfpi::dst_reg[0] = result; // SFPSTORE: write result back to DEST, guarded by CC
        }
        v_endif; // Restore CC to ALL_ENABLED (SFPPUSHC/SFPPOPC or equivalent CC stack ops)
        sfpi::dst_reg++; // Advance DEST pointer by 1 sfpi row (= 2 physical DEST rows = 32 elements)
    }
}
```

#### Helper: `_sfpu_exp_21f_bf16_` (from `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_exp.h`)

This function implements the `exp_21f` algorithm from Moroz et al. 2022 for fast exponential approximation.

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_exp.h

template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _sfpu_exp_21f_bf16_(sfpi::vFloat val) // is_fp32_dest_acc_en=true when called from SELU
{
    // Implementation notes, see the original file for more details
    constexpr float ONE_LN2 = 1.4426950216293334961f;
    sfpi::vFloat xlog2      = (val * ONE_LN2 + 127.f); // SFPMAD: val * (1/ln2) + 127.0; converts to base-2 biased representation

    // Clamp xlog2 to [0, 255] to prevent overflow in intermediate computation
    sfpi::vFloat threshold_low  = 0.f;       // SFPLOADI: load immediate 0.0
    sfpi::vFloat threshold_high = sfpi::vFloat(255.f); // SFPLOADI: load immediate 255.0
    sfpi::vec_min_max(threshold_low, xlog2);  // SFPSWAP with VEC_MIN_MAX mode: threshold_low=min, xlog2=max
    sfpi::vec_min_max(xlog2, threshold_high); // SFPSWAP with VEC_MIN_MAX mode: xlog2=min (clamped), threshold_high=max

    sfpi::vInt z = _float_to_int32_for_exp_21f_(xlog2); // See helper below

    sfpi::vInt exponential_part = exexp_nodebias(sfpi::reinterpret<sfpi::vFloat>(z)); // SFPEXEXP: extract exponent without debiasing
    sfpi::vInt fractional_part  = sfpi::exman9(sfpi::reinterpret<sfpi::vFloat>(z));   // SFPEXMAN: extract 9-bit mantissa with implicit bit

    sfpi::vFloat frac = sfpi::int32_to_float(fractional_part, 0); // SFPCAST: int32 to FP32 with round-to-nearest-even

    // 2nd degree polynomial approximation of 2^(fractional part) using Horner's method
    // eval(x, c0, c1, c2) = c0 + x*(c1 + x*c2) -> 2 SFPMADs
    frac = PolynomialEvaluator::eval(frac, 1.0017248f, 7.839635491371155e-08f, 4.791750143340323e-15f);

    // Recombine: result = 2^(integer_part) * 2^(fractional_part)
    sfpi::vFloat y = sfpi::setexp(frac, exponential_part); // SFPSETEXP: set exponent field of frac to exponential_part

    if constexpr (!is_fp32_dest_acc_en) // false when called from SELU (is_fp32_dest_acc_en=true), so this branch is SKIPPED
    {
        y = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(y, 0));
    }

    return y;
}
```

#### Helper: `_float_to_int32_for_exp_21f_` (from same file)

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_exp.h

sfpi_inline sfpi::vInt _float_to_int32_for_exp_21f_(sfpi::vFloat val)
{
    sfpi::vInt exp = sfpi::exexp(val);   // SFPEXEXP: extract biased exponent
    sfpi::vInt man = sfpi::exman8(val);  // SFPEXMAN: extract 8-bit mantissa with implicit bit (value in [1,2))
    man            = sfpi::reinterpret<sfpi::vInt>(sfpi::shft(sfpi::reinterpret<sfpi::vUInt>(man), exp)); // SFPSHFT: logical left shift mantissa by exponent
    return man;
}
```

### SFPU Instructions Used

| Instruction | SFPI Intrinsic / Operation | Description |
|---|---|---|
| **SFPLOAD** | `dst_reg[0]` (read) | Load 32 elements from current DEST row pair into an LREG |
| **SFPSTORE** | `dst_reg[0] = ...` (write) | Store 32 elements from LREG back to current DEST row pair |
| **SFPLOADI** | `vFloat(0.f)`, `vFloat(255.f)`, `Converter::as_float(...)` | Load immediate float constant into an LREG |
| **SFPMAD** | `*`, `+`, `-` on `vFloat`; `PolynomialEvaluator::eval` | Fused multiply-add: `a * b + c`. Used for all float arithmetic (multiply is `a * b + 0.0`, subtract is `a * 1.0 + (-b)`). The polynomial evaluator emits a chain of SFPMADs via Horner's method. |
| **SFPSETCC** | `v_if(v >= 0.0f)` | Set condition code based on comparison. Enables lanes where condition is true for subsequent CC-guarded instructions. |
| **SFPCOMPC** | `v_else` | Complement (invert) the condition code, enabling the lanes that were previously disabled. |
| **SFPPUSHC / SFPPOPC** | `v_if` / `v_endif` | Push/pop condition code state on the CC stack to support nested conditional regions. |
| **SFPSWAP** | `vec_min_max(a, b)` | Vector min/max operation: after execution, `a` contains element-wise min and `b` contains element-wise max. Used to clamp the base-2 representation to [0, 255]. |
| **SFPEXEXP** | `exexp(val)`, `exexp_nodebias(val)` | Extract the exponent field from a float. `exexp` debiases (subtracts 127), `exexp_nodebias` returns raw biased exponent. |
| **SFPEXMAN** | `exman8(val)`, `exman9(val)` | Extract mantissa from a float with implicit leading bit. `exman8` extracts 8-bit mantissa, `exman9` extracts 9-bit mantissa. |
| **SFPSHFT** | `shft(man, exp)` | Logical shift of an integer value by a variable amount. Used to convert float-as-integer representation by shifting mantissa left by the exponent. |
| **SFPCAST** | `int32_to_float(val, 0)` | Convert 32-bit integer to FP32 floating point, with round-to-nearest-even mode. |
| **SFPSETEXP** | `setexp(frac, exponential_part)` | Set the exponent field of a float to a specified value, used to recombine the integer and fractional parts of `2^x`. |
| **SFPSTOCHRND** | `float_to_fp16b(result, 0)` | Stochastic/deterministic rounding of FP32 to BF16. Mode 0 = round-to-nearest-even. Applied in two places: (1) final result rounding in `calculate_selu` when `!is_fp32_dest_acc_en`, and (2) skipped inside `_sfpu_exp_21f_bf16_` since SELU calls it with `is_fp32_dest_acc_en=true`. |

### SFPU Register Usage

| Register | Usage |
|---|---|
| **DEST rows** | Input tile data. Each iteration processes `dst_reg[0]` which maps to 2 physical DEST rows (32 elements). The pointer advances by `dst_reg++` (1 sfpi row = 2 physical rows) each iteration. |
| **LREGs (L0-L7)** | Temporary registers used by SFPI for intermediate values. The compiler allocates these automatically. Key allocations include: `scale_value` and `alpha_value` (loop-invariant, likely hoisted to persistent LREGs), `v` (loaded input), `exp_calc` (exponential result), `minus_mul` (exp-1), `result` (final value). The `_sfpu_exp_21f_bf16_` helper uses additional LREGs for `xlog2`, `threshold_low`, `threshold_high`, `z`, `exponential_part`, `fractional_part`, `frac`, `y`. |
| **CC (Condition Code)** | Used by `v_if(v >= 0.0f)` / `v_else` / `v_endif` to conditionally execute the positive branch (`v * scale`) vs negative branch (`scale * alpha * (exp(v) - 1)`). The CC stack is used to save/restore CC state across branches. |
| **vConst1** | Hardware constant register holding `1.0f`, used in the subtraction `exp_calc - vConst1` which compiles to `SFPMAD(exp_calc, 1.0, -1.0)`. |

### Address Mode Configuration

The address mode for SELU is configured by `eltwise_unary_sfpu_configure_addrmod<SfpuType::selu>()` during initialization. Since `SfpuType::selu` does not match any special-case `if constexpr` branch, only the default `ADDR_MOD_7` is configured:

**Wormhole B0 and Blackhole (identical configuration):**

| ADDR_MOD | srca.incr | srcb.incr | dest.incr | Purpose |
|---|---|---|---|---|
| `ADDR_MOD_7` | 0 | 0 | 0 | Default SFPU address mode. No auto-increment -- the kernel manages DEST address progression explicitly through SFPI's `dst_reg++` mechanism. |

The DEST address progression within the kernel is entirely managed by the SFPI `dst_reg++` abstraction, which internally advances the SFPU's DEST read/write pointer by `SFP_DESTREG_STRIDE=2` physical rows per iteration. Between faces, the params dispatch layer issues `TTI_SETRWC` (WH) or `inc_dst_addr<8>` (BH) to advance by one face (16 physical rows = 8 sfpi rows).

## External Knowledge Sources
### DeepWiki Queries
1. [SFPU] **Query**: "What SFPU instructions do these SFPI intrinsics map to: exexp_nodebias, exman9, setexp, vec_min_max, exexp, exman8, shft, float_to_fp16b, vFloat multiplication, vFloat addition/subtraction?"
   **Reason**: Needed to understand the mapping from high-level SFPI abstractions to low-level SFPU instructions for the instruction table.
   **Key Findings**: DeepWiki returned 429 Too Many Requests. Instruction mappings were derived from source code analysis of `runtime/sfpi/include/sfpi_lib.h` which maps each SFPI intrinsic to its `__builtin_rvtt_sfp*` compiler builtin, which directly corresponds to the SFPU instruction.

### Confluence References
No Confluence pages were consulted for this analysis.

### Glean References
No Glean queries were made for this analysis.
