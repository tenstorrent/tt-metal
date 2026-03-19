## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to for the xIELU (Expanded Integral of ELU) activation function.

### Unary Dispatch Summary
- **UnaryOpType**: `XIELU`
- **Compute kernel**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp` (default path from `get_compute_kernel_path()`)
- **SFPU_OP_CHAIN_0 expansion**: `xielu_tile_init(); xielu_tile({idst}, {alpha_p_bits}u, {alpha_n_bits}u);`

#### Approximation Mode Resolution
There are TWO independent controls for approximation. Both are reported below.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(XIELU)` in `unary_op_utils.cpp` -- falls through to `default: return false` |
| Template parameter (SFPU_OP_CHAIN) | `APPROX` (macro value from `math_approx_mode`, i.e., `false`) | `get_op_init_and_func()` returns `xielu_tile_init()` / `xielu_tile(idst, alpha_p, alpha_n)` -- the API header passes `APPROX` to the macro, which resolves to the compute config value |
| Effective SFPU path | `APPROXIMATION_MODE=false` in `calculate_xielu`. The template parameter is passed through but the kernel body does **not** branch on `APPROXIMATION_MODE` -- all code paths execute regardless of this value. The `APPROXIMATION_MODE` parameter is only forwarded to the `xielu_init<APPROXIMATION_MODE>()` function, which also does not branch on it. | `ckernel_sfpu_xielu.h` -- the `calculate_xielu` function has no `if constexpr (APPROXIMATION_MODE)` branching |

### SFPU Abstraction Layers
| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/xielu.h` |
| **LLK Dispatch** | This level of abstraction doesn't exist -- the API header uses the `SFPU_UNARY_TWO_PARAM_KERNEL_WITH_DST_ACCUM` macro (from `llk_math_eltwise_unary_sfpu_macros.h`) which directly calls `_llk_math_eltwise_unary_sfpu_params_` |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_xielu.h` (WH) / `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_xielu.h` (BH) -- implementations are identical |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (WH) / `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (BH) |

### Call Chain
1. The compute kernel calls the `SFPU_OP_CHAIN_0` macro, which expands to `xielu_tile_init()` followed by `xielu_tile(idst, alpha_p, alpha_n)`.
2. `xielu_tile_init()` (in `xielu.h`) expands via `SFPU_INIT_KERNEL_CALL(xielu, sfpu::xielu_init, APPROX)` to `llk_math_eltwise_unary_sfpu_init<SfpuType::xielu, APPROX>(xielu_init<APPROX>)`, which calls `_llk_math_eltwise_unary_sfpu_init_<SfpuType::xielu>()` (configures SFPU config reg and ADDR_MOD_7) then invokes `xielu_init<false>()` to load programmable constant registers.
3. `xielu_tile(idst, alpha_p, alpha_n)` (in `xielu.h`) expands via `SFPU_UNARY_TWO_PARAM_KERNEL_WITH_DST_ACCUM(calculate_xielu, RC, APPROX, DST_ACCUM_MODE, idst, alpha_p, alpha_n)` to `_llk_math_eltwise_unary_sfpu_params_<APPROX>(ckernel::sfpu::calculate_xielu<APPROX, DST_ACCUM_MODE>, idst, (int)VectorMode::RC, alpha_p, alpha_n)`.
4. `_llk_math_eltwise_unary_sfpu_params_` (in `llk_math_eltwise_unary_sfpu_params.h`) sets up the DEST write address, stalls until SFPU is ready, then loops over all 4 faces in `VectorMode::RC` mode, calling `calculate_xielu<false, DST_ACCUM_MODE>(alpha_p, alpha_n)` once per face (8 iterations each = 32 total iterations for the full tile), with `SETRWC` between faces to advance the DEST pointer.
5. `calculate_xielu` (in `ckernel_sfpu_xielu.h`) is the core SFPU implementation that performs the piecewise xIELU computation.

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` -- all 4 faces of the tile are processed (the standard mode for full-tile unary operations).
- **Operation invocation**: The params dispatch function calls `calculate_xielu<false, DST_ACCUM_MODE>(alpha_p, alpha_n)` once per face in a `for (int face = 0; face < 4; face++)` loop. Each invocation runs `ITERATIONS=8` (the template default), processing one full 16x16 face per call.
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, `SETRWC` / `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_` between faces). On Wormhole, `TTI_SETRWC` with `CR_D, 8` is called twice between faces (equivalent to advancing by 16 physical DEST rows = 1 face). On Blackhole, `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` calls `math::inc_dst_addr<8>()` twice for the same effect.

### Annotated SFPU Kernel Source

This kernel uses SFPI abstractions (`sfpi::vFloat`, `sfpi::vInt`, `dst_reg`, `v_if/v_elseif/v_else/v_endif`, etc.), so Style A is used.

```cpp
// File: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_xielu.h

// Helper: compute exp(val) for negative val using Cody-Waite range reduction + Taylor series
sfpi_inline sfpi::vFloat _sfpu_neg_exp_f32_(sfpi::vFloat val) {
    sfpi::vFloat result = sfpi::vConst0;

    constexpr float UNDERFLOW_THRESHOLD = -126.5f;

    // Step 1: Compute k = round(x / ln(2)); vConstFloatPrgm0 = 1/ln(2) loaded in xielu_init
    sfpi::vFloat z = val * sfpi::vConstFloatPrgm0; // SFPMAD: val * (1/ln2) + 0

    // Clamp z to -126.5: exp(x) underflows to 0 for large negative x
    sfpi::vFloat underflow_bound = UNDERFLOW_THRESHOLD; // SFPLOADI
    sfpi::vec_min_max(underflow_bound, z); // SFPSWAP: after call, z = max(z, -126.5)

    // Round z to nearest integer using round-to-nearest-even
    sfpi::vInt k_int;
    sfpi::vFloat k = _sfpu_round_to_nearest_int32_(z, k_int); // Uses Hacker's Delight magic constant 0x4B400000

    // Implementation notes, see the original file for more details
    constexpr float LN2_HI = -0.6931152343750000f;
    constexpr float LN2_LO = -3.19461832987e-05f;

    sfpi::vFloat r_hi = k * LN2_HI + val; // SFPMAD: k * (-LN2_HI) + val
    sfpi::vFloat r = k * LN2_LO + r_hi;   // SFPMAD: k * (-LN2_LO) + r_hi

    // 7th order Taylor series for exp(r) via Horner's method
    sfpi::vFloat p = PolynomialEvaluator::eval( // Chain of SFPMADs (Horner's method)
        r,
        sfpi::vConst1,  // c0 = 1
        sfpi::vConst1,  // c1 = 1
        0.5f,           // c2 = 1/2!
        1.0f / 6.0f,    // c3 = 1/3!
        1.0f / 24.0f,   // c4 = 1/4!
        1.0f / 120.0f,  // c5 = 1/5!
        1.0f / 720.0f,  // c6 = 1/6!
        1.0f / 5040.0f  // c7 = 1/7!
    );

    // Scale by 2^k using exponent manipulation: ldexp(p, k_int)
    sfpi::vInt p_exp = sfpi::exexp_nodebias(p); // SFPEXEXP: extract exponent without bias
    sfpi::vInt new_exp = p_exp + k_int;          // SFPIADD: integer add exponents

    result = sfpi::setexp(p, new_exp); // SFPSETEXP: set new exponent

    return result;
}

// Helper: multiply-accumulate with optional fp16b conversion for DEST accuracy
template <bool is_fp32_dest_acc_en>
sfpi_inline void _xielu_mad_(sfpi::vFloat mul_a, sfpi::vFloat mul_b, sfpi::vFloat addend) {
    sfpi::vFloat result = mul_a * mul_b + addend; // SFPMAD: mul_a * mul_b + addend
    if constexpr (!is_fp32_dest_acc_en) {
        // When DEST is bfloat16, explicitly round to fp16b to avoid truncation errors
        result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0)); // SFPSTOCHRND with FP32_TO_FP16B mode
    }
    sfpi::dst_reg[0] = result; // SFPSTORE: write result back to DEST
}

// Implementation notes, see the original file for more details
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>
inline void calculate_xielu(const uint32_t param0, const uint32_t param1) { // APPROXIMATION_MODE=false, is_fp32_dest_acc_en=DST_ACCUM_MODE, ITERATIONS=8
    sfpi::vFloat alpha_p = Converter::as_float(param0); // Reinterpret uint32 bits as float
    sfpi::vFloat alpha_n = Converter::as_float(param1);
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0]; // SFPLOAD: load input element from DEST
        sfpi::vFloat beta_mul_x = 0.5f * x; // SFPMAD: 0.5 * x + 0 (beta=0.5 is hardcoded)
        v_if(x > 0.0f) {  // SFPSETCC with GT condition; positive branch
            _xielu_mad_<is_fp32_dest_acc_en>(alpha_p * x, x, beta_mul_x);
            // Computes: alpha_p * x * x + 0.5 * x
        }
        v_elseif(x >= sfpi::vConstFloatPrgm1) {  // SFPSETCC/SFPCOMPC; vConstFloatPrgm1 = -1e-6 (eps)
            // Very small negative: x in [-1e-6, 0], use precomputed expm1(eps)
            sfpi::vFloat exp_term = sfpi::vConstFloatPrgm2 - x; // vConstFloatPrgm2 = expm1(-1e-6)
            _xielu_mad_<is_fp32_dest_acc_en>(alpha_n, exp_term, beta_mul_x);
            // Computes: alpha_n * (expm1(eps) - x) + 0.5 * x
        }
        v_elseif(x > -0.5f) {  // SFPSETCC/SFPCOMPC; moderate negative region
            // For x in [-0.5, -1e-6]: use Sollya-optimized polynomial for expm1(x)-x
            // expm1(x)-x = x^2 * P(x) where P is a degree-5 polynomial (Sollya minimax)
            sfpi::vFloat exp_term = x * x * // SFPMAD for x*x
                                    PolynomialEvaluator::eval( // Chain of SFPMADs (Horner's)
                                        x,
                                        0.500000059604644775390625f,          // c2 ~= 1/2!
                                        0.16666667163372039794921875f,        // c3 ~= 1/3!
                                        4.16650883853435516357421875e-2f,     // c4 ~= 1/4!
                                        8.333188481628894805908203125e-3f,    // c5 ~= 1/5!
                                        1.400390756316483020782470703125e-3f, // c6 ~= 1/6!
                                        1.99588379473425447940826416015625e-4f); // c7 ~= 1/7!
            _xielu_mad_<is_fp32_dest_acc_en>(alpha_n, exp_term, beta_mul_x);
            // Computes: alpha_n * (expm1(x) - x) + 0.5 * x
        }
        v_else {  // SFPCOMPC; large negative: x < -0.5
            // Use full exp computation via Cody-Waite + Taylor series
            sfpi::vFloat exp_term = _sfpu_neg_exp_f32_(x) - sfpi::vConst1 - x;
            // exp_term = exp(x) - 1 - x = expm1(x) - x
            _xielu_mad_<is_fp32_dest_acc_en>(alpha_n, exp_term, beta_mul_x);
            // Computes: alpha_n * (expm1(x) - x) + 0.5 * x
        }
        v_endif; // SFPENCC: re-enable all lanes
        sfpi::dst_reg++; // Advance DEST pointer by 1 sfpi row (= 2 physical rows = 32 elements)
    }
}

template <bool APPROXIMATION_MODE>
void xielu_init() {
    sfpi::vConstFloatPrgm0 = 1.4426950408889634f;   // 1/ln(2) -- used by _sfpu_neg_exp_f32_
    sfpi::vConstFloatPrgm1 = -1e-6f;                // eps value -- boundary for very-small-negative branch
    sfpi::vConstFloatPrgm2 = -0.0000009999995427f;  // expm1(eps) precomputed -- avoids computing exp for tiny negatives
}
```

### SFPU Instructions Used

| Instruction / Intrinsic | Description | Usage in xielu |
|------------------------|-------------|----------------|
| **SFPLOAD** | Load data from DEST register into LREG | `dst_reg[0]` read -- loads 32 elements (2 physical rows) from current DEST position into working register |
| **SFPSTORE** | Store data from LREG back to DEST register | `dst_reg[0] = result` -- writes computed result back to DEST |
| **SFPMAD** | Multiply-accumulate: `VD = VA * VB + VC` | All arithmetic: `0.5 * x`, `alpha_p * x`, `k * LN2_HI + val`, polynomial Horner steps, `mul_a * mul_b + addend`. This is the workhorse instruction -- there is no dedicated float add, so `a + b` compiles to `a * 1.0 + b` |
| **SFPLOADI** | Load immediate value into LREG | Loading float constants (0.5f, -0.5f, polynomial coefficients, LN2_HI, LN2_LO, etc.) |
| **SFPSETCC** | Set condition code from comparison result | `v_if(x > 0.0f)`, `v_elseif(x >= vConstFloatPrgm1)`, `v_elseif(x > -0.5f)` -- enables/disables lanes based on element-wise comparison |
| **SFPCOMPC** | Complement condition code | `v_elseif` / `v_else` -- inverts the active lane mask for the else-branch |
| **SFPENCC** | Enable all condition codes (disable predication) | `v_endif` -- restores all lanes to active |
| **SFPSWAP** | Vector min/max swap | `vec_min_max(underflow_bound, z)` -- clamps z to prevent underflow in exp computation |
| **SFPEXEXP** | Extract exponent from float | `exexp_nodebias(p)` -- extracts the unbiased exponent of the polynomial result for 2^k scaling |
| **SFPSETEXP** | Set exponent of float | `setexp(p, new_exp)` -- replaces the exponent to compute `p * 2^k` (ldexp operation) |
| **SFPIADD** | Integer addition on vector registers | `p_exp + k_int` -- adds the integer shift `k` to the extracted exponent |
| **SFPSTOCHRND** | Stochastic/deterministic rounding | `float_to_fp16b(result, 0)` -- converts fp32 result to bfloat16 with round-to-nearest-even (mode FP32_TO_FP16B) when DEST is not fp32 accumulation mode |

### SFPU Register Usage

| Register | Purpose |
|----------|---------|
| **DEST[dst_index]** | Source and destination for tile data. The kernel reads input `x` from DEST, computes the xIELU result, and writes it back to the same DEST position. |
| **LREG[0-3]** | General-purpose local registers used for intermediate values. The SFPI compiler allocates these for `vFloat`/`vInt` local variables (`x`, `beta_mul_x`, `alpha_p`, `alpha_n`, `exp_term`, polynomial intermediates, etc.). With 4 branches and multiple intermediates per branch, register pressure is significant. |
| **vConstFloatPrgm0** | Programmable constant register loaded with `1/ln(2) = 1.4426950408889634f` during `xielu_init()`. Used by `_sfpu_neg_exp_f32_` for base conversion. |
| **vConstFloatPrgm1** | Programmable constant register loaded with `-1e-6f` (eps). Used as the boundary between the "very small negative" and "moderate negative" branches. |
| **vConstFloatPrgm2** | Programmable constant register loaded with `expm1(-1e-6) = -0.0000009999995427f`. Used as the precomputed exp(eps)-1 value in the very-small-negative branch to avoid computing exp for near-zero inputs. |
| **vConst0** | Built-in constant `0.0f`. Used as the initial value of `result` in `_sfpu_neg_exp_f32_`. |
| **vConst1** | Built-in constant `1.0f`. Used in the Taylor series coefficients (c0=1, c1=1) and in `_sfpu_neg_exp_f32_(x) - vConst1 - x`. |

### Address Mode Configuration

The address mode is configured during `_llk_math_eltwise_unary_sfpu_init_<SfpuType::xielu>()` which calls `eltwise_unary_sfpu_configure_addrmod<SfpuType::xielu>()`.

Since `SfpuType::xielu` does not match any of the special-cased `SfpuType` values (topk_local_sort, typecast, unary_max, etc.), only the default `ADDR_MOD_7` is configured:

```
ADDR_MOD_7: { srca.incr = 0, srcb.incr = 0, dest.incr = 0 }
```

This is the same on both Wormhole and Blackhole. The zero-increment ADDR_MOD_7 means the hardware does not auto-increment DEST addresses between SFPU instructions within an iteration. Instead, DEST address progression is managed explicitly:
- **Within a face**: `dst_reg++` in the SFPI code advances the SFPU DEST pointer by 1 sfpi row (= 2 physical DEST rows = 32 elements) per iteration.
- **Between faces**: The params dispatch function uses `TTI_SETRWC` (Wormhole) or `math::inc_dst_addr<8>()` (Blackhole) to advance by 16 physical DEST rows (= 1 face width) between each of the 4 face invocations.

## External Knowledge Sources
### DeepWiki Queries
1. **Query**: "What SFPU instructions are emitted by sfpi::vFloat operations like dst_reg load/store, vFloat multiplication and addition (SFPMAD), v_if/v_elseif/v_else/v_endif conditional blocks, vec_min_max, exexp_nodebias, setexp, and float_to_fp16b?"
   **Reason**: Needed to map high-level SFPI abstractions used in the xielu kernel to their corresponding low-level SFPU instructions for accurate documentation of the instruction set usage.
   **Key Findings**: Confirmed that vFloat arithmetic maps to SFPMAD, dst_reg access maps to SFPLOAD/SFPSTORE, v_if/v_elseif/v_else/v_endif map to SFPSETCC/SFPCOMPC/SFPENCC, exexp_nodebias maps to SFPEXEXP, setexp maps to SFPSETEXP, float_to_fp16b maps to SFPSTOCHRND with FP32_TO_FP16B mode, and vec_min_max maps to SFPSWAP.

### Confluence References
Not consulted for this analysis.

### Glean References
Not consulted for this analysis.
