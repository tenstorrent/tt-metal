## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### Unary Dispatch Summary
- **UnaryOpType**: `XIELU`
- **Compute kernel**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp` (default path)
- **SFPU_OP_CHAIN_0 expansion**: `xielu_tile(0, {alpha_p_bits}u, {alpha_n_bits}u)` where `alpha_p_bits` and `alpha_n_bits` are the bit-cast `uint32_t` representations of the `float` parameters `alpha_p` and `alpha_n`

#### Approximation Mode Resolution

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(XIELU)` in `unary_op_utils.cpp` -- falls through to `default: return false` (no explicit XIELU case) |
| Template parameter (SFPU_OP_CHAIN) | `APPROX` (resolved from `math_approx_mode`, i.e., `false`) | `get_op_init_and_func()` returns `xielu_tile_init();` and `xielu_tile({}, {:#x}u, {:#x}u);` -- the API header uses the build-level `APPROX` define, which equals `math_approx_mode` |
| Effective SFPU path | `APPROXIMATION_MODE=false` throughout `calculate_xielu` and `_sfpu_neg_exp_f32_`. The `APPROXIMATION_MODE` template parameter is not checked in `calculate_xielu` (no `if constexpr` on it), so both paths produce the same code. In `_sfpu_neg_exp_f32_`, the parameter is also unused -- the full 7th-order Taylor expansion always runs. | The kernel has no approximation branching; `APPROXIMATION_MODE` is accepted but ignored. |

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/xielu.h` |
| **LLK Dispatch** | This level of abstraction doesn't exist (the API header uses `SFPU_UNARY_TWO_PARAM_KERNEL_WITH_DST_ACCUM` macro from `llk_math_eltwise_unary_sfpu_macros.h`, which directly calls `_llk_math_eltwise_unary_sfpu_params_`) |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_xielu.h` (identical on Blackhole: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_xielu.h`) |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (Blackhole: `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu_params.h`) |

### Call Chain

1. The compute kernel calls `SFPU_OP_CHAIN_0`, which expands to `xielu_tile(0, alpha_p_bits, alpha_n_bits)`.
2. `xielu_tile()` (in `xielu.h`) wraps the call in `MATH(SFPU_UNARY_TWO_PARAM_KERNEL_WITH_DST_ACCUM(calculate_xielu, RC, APPROX, DST_ACCUM_MODE, idst, alpha_p, alpha_n))`.
3. The `SFPU_UNARY_TWO_PARAM_KERNEL_WITH_DST_ACCUM` macro (in `llk_math_eltwise_unary_sfpu_macros.h`) expands to `_llk_math_eltwise_unary_sfpu_params_<APPROX>(ckernel::sfpu::calculate_xielu<APPROX, DST_ACCUM_MODE>, idst, (int)VectorMode::RC, alpha_p, alpha_n)`.
4. `_llk_math_eltwise_unary_sfpu_params_()` (in `llk_math_eltwise_unary_sfpu_params.h`) sets up DEST addressing, stalls for SFPU readiness, then calls `calculate_xielu(alpha_p, alpha_n)` once per face (4 faces for `VectorMode::RC`), with `SETRWC` advancing the DEST pointer between faces.
5. `calculate_xielu()` (in `ckernel_sfpu_xielu.h`) iterates 8 times per face, performing the piecewise xIELU computation on each 32-element SFPU vector.

The init path follows: `xielu_tile_init()` -> `SFPU_INIT_KERNEL_CALL(xielu, sfpu::xielu_init, APPROX)` -> `llk_math_eltwise_unary_sfpu_init<SfpuType::xielu, APPROX>(xielu_init<APPROX>)` -> `_llk_math_eltwise_unary_sfpu_init_<SfpuType::xielu>()` (configures SFPU config reg, address modes, resets counters) followed by `xielu_init<APPROX>()` (loads programmable constants).

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` -- all 4 faces of the tile are processed (full 32x32 tile).
- **Operation invocation**: `calculate_xielu<APPROX, DST_ACCUM_MODE>(alpha_p, alpha_n)` is called once per face in a loop of 4 iterations. Each call internally loops `ITERATIONS=8` times, covering 8 sfpi rows per face (8 x 32 elements = 256 elements = 1 face).
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, `SETRWC` between faces). On Wormhole, `SETRWC` with `CR_D, 8` is issued twice between faces (advancing 16 physical DEST rows = 1 face stride). On Blackhole, `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` calls `math::inc_dst_addr<8>()` twice for the same effect. The address mode configured is `ADDR_MOD_7` with all increments set to 0 (srca=0, srcb=0, dest=0). The `xielu` SfpuType does not match any special-case address mode configurations (those are for `topk_local_sort`, `typecast`, `unary_max/min`, etc.), so only `ADDR_MOD_7` with zero increments is set.

### Annotated SFPU Kernel Source

The kernel uses SFPI abstractions (`sfpi::vFloat`, `sfpi::vInt`, `dst_reg`, `v_if`/`v_elseif`/`v_else`/`v_endif`, `PolynomialEvaluator`), so Style A (inline-commented source) is used.

```cpp
// File: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_xielu.h

sfpi_inline sfpi::vFloat _sfpu_neg_exp_f32_(sfpi::vFloat val) {
    sfpi::vFloat result = sfpi::vConst0; // Initialize result to 0.0

    constexpr float UNDERFLOW_THRESHOLD = -126.5f;

    // z = x * (1/ln(2)), using programmable constant vConstFloatPrgm0 = 1/ln(2)
    sfpi::vFloat z = val * sfpi::vConstFloatPrgm0; // -> SFPMAD (val * 1/ln2 + 0)

    // Clamp z to -126.5 to prevent exponent underflow
    sfpi::vFloat underflow_bound = UNDERFLOW_THRESHOLD;
    sfpi::vec_min_max(underflow_bound, z); // After: underflow_bound=min, z=max; effectively clamps z >= -126.5

    // Round z to nearest integer
    sfpi::vInt k_int;
    sfpi::vFloat k = _sfpu_round_to_nearest_int32_(z, k_int); // Produces integer k and float k

    // Cody-Waite range reduction: r = val - k*ln(2) in extended precision
    constexpr float LN2_HI = -0.6931152343750000f;  // -(high bits of ln(2))
    constexpr float LN2_LO = -3.19461832987e-05f;   // -(low bits of ln(2))

    sfpi::vFloat r_hi = k * LN2_HI + val; // -> SFPMAD (k * LN2_HI + val)
    sfpi::vFloat r = k * LN2_LO + r_hi;   // -> SFPMAD (k * LN2_LO + r_hi)

    // 7th-order Taylor polynomial for exp(r): each coeff -> SFPMAD via Horner's method
    sfpi::vFloat p = PolynomialEvaluator::eval(
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

    // Scale by 2^k via exponent manipulation (no multiplication needed)
    sfpi::vInt p_exp = sfpi::exexp_nodebias(p); // -> SFPEXEXP: extract exponent of p
    sfpi::vInt new_exp = p_exp + k_int;          // -> SFPIADD (integer addition)
    result = sfpi::setexp(p, new_exp);           // -> SFPSETEXP: write new exponent into p

    return result;
}

// MAD helper: writes result to dst_reg[0], with optional fp16b conversion
template <bool is_fp32_dest_acc_en> // is_fp32_dest_acc_en=false (DST_ACCUM_MODE)
sfpi_inline void _xielu_mad_(sfpi::vFloat mul_a, sfpi::vFloat mul_b, sfpi::vFloat addend) {
    sfpi::vFloat result = mul_a * mul_b + addend; // -> SFPMAD (mul_a * mul_b + addend)
    if constexpr (!is_fp32_dest_acc_en) {
        // Convert to fp16b to match DEST format when not in fp32 accumulation mode
        result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0)); // -> SFPSTOCHRND (truncation mode 0)
    }
    sfpi::dst_reg[0] = result; // -> SFPSTORE to current DEST row
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>
inline void calculate_xielu(const uint32_t param0, const uint32_t param1) { // APPROXIMATION_MODE=false, is_fp32_dest_acc_en=false, ITERATIONS=8
    sfpi::vFloat alpha_p = Converter::as_float(param0); // Reinterpret uint32 bits as float
    sfpi::vFloat alpha_n = Converter::as_float(param1);
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];     // -> SFPLOAD from current DEST row
        sfpi::vFloat beta_mul_x = 0.5f * x;    // -> SFPMAD (0.5 * x + 0.0); beta is fixed at 0.5
        v_if(x > 0.0f) {  // CC set by comparison; positive branch
            // alpha_p * x * x + 0.5 * x (quadratic for positive inputs)
            _xielu_mad_<is_fp32_dest_acc_en>(alpha_p * x, x, beta_mul_x); // alpha_p*x*x + beta*x
        }
        v_elseif(x >= sfpi::vConstFloatPrgm1) {  // vConstFloatPrgm1 = -1e-6 (eps); very small negative
            // For x in [-1e-6, 0]: use precomputed expm1(eps) constant
            sfpi::vFloat exp_term = sfpi::vConstFloatPrgm2 - x; // vConstFloatPrgm2 = expm1(-1e-6)
            _xielu_mad_<is_fp32_dest_acc_en>(alpha_n, exp_term, beta_mul_x); // alpha_n*(expm1(eps)-x) + beta*x
        }
        v_elseif(x > -0.5f) {  // moderate negative: x in (-0.5, -1e-6)
            // Implementation notes, see the original file for more details
            // expm1(x)-x = x^2 * P(x) where P is a 5th-degree Sollya-optimized polynomial
            sfpi::vFloat exp_term = x * x *
                                    PolynomialEvaluator::eval(
                                        x,
                                        0.500000059604644775390625f,              // ~1/2!
                                        0.16666667163372039794921875f,             // ~1/3!
                                        4.16650883853435516357421875e-2f,          // ~1/4!
                                        8.333188481628894805908203125e-3f,         // ~1/5!
                                        1.400390756316483020782470703125e-3f,      // ~1/6!
                                        1.99588379473425447940826416015625e-4f);   // ~1/7!
            _xielu_mad_<is_fp32_dest_acc_en>(alpha_n, exp_term, beta_mul_x);
        }
        v_else {  // large negative: x <= -0.5
            // Full exp computation via _sfpu_neg_exp_f32_ (Cody-Waite + 7th-order Taylor)
            sfpi::vFloat exp_term = _sfpu_neg_exp_f32_(x) - sfpi::vConst1 - x; // exp(x)-1-x = expm1(x)-x
            _xielu_mad_<is_fp32_dest_acc_en>(alpha_n, exp_term, beta_mul_x);
        }
        v_endif;
        sfpi::dst_reg++; // Advance to next sfpi row (2 physical DEST rows, 32 elements)
    }
}

template <bool APPROXIMATION_MODE>
void xielu_init() {
    sfpi::vConstFloatPrgm0 = 1.4426950408889634f;   // 1/ln(2) -- used by _sfpu_neg_exp_f32_
    sfpi::vConstFloatPrgm1 = -1e-6f;                // eps threshold for very-small-negative branch
    sfpi::vConstFloatPrgm2 = -0.0000009999995427f;  // precomputed expm1(-1e-6)
}
```

### SFPU Instructions Used

| Instruction/Intrinsic | Description |
|---|---|
| `SFPLOAD` | Loads 32 elements from the current DEST row pair into an LREG. Emitted by `dst_reg[0]` reads. |
| `SFPSTORE` | Stores 32 elements from an LREG back to the current DEST row pair. Emitted by `dst_reg[0] = result` writes. |
| `SFPMAD` | Fused multiply-add: `VD = VA * VB + VC`. Emitted by every `vFloat * vFloat + vFloat` expression, including all `PolynomialEvaluator::eval` Horner steps, the `0.5f * x` computation, `alpha_p * x`, and arithmetic in `_sfpu_neg_exp_f32_`. This is the dominant instruction in the kernel. |
| `SFPLOADI` | Loads an immediate scalar constant into an LREG. Emitted for float literals like `0.5f`, `-0.5f`, `0.0f`, polynomial coefficients, and the Cody-Waite constants (`LN2_HI`, `LN2_LO`). |
| `SFPSETCC` | Sets the condition code register based on a comparison result. Emitted by `v_if(x > 0.0f)`, `v_elseif(x >= ...)`, and `v_elseif(x > -0.5f)` to enable per-lane predication. |
| `SFPENCC` | Enables/complements the condition code for else-branch processing. Emitted by `v_elseif` and `v_else` constructs. |
| `SFPCOMPC` | Complements the condition code. Used internally by `v_elseif`/`v_else`/`v_endif` to manage nested conditional regions. |
| `SFPEXEXP` | Extracts the exponent field from a float without debiasing. Emitted by `sfpi::exexp_nodebias(p)` in `_sfpu_neg_exp_f32_`. |
| `SFPSETEXP` | Sets (replaces) the exponent field of a float. Emitted by `sfpi::setexp(p, new_exp)` for the 2^k scaling step. |
| `SFPIADD` | Integer addition on LREG values. Emitted by `p_exp + k_int` in `_sfpu_neg_exp_f32_` (adds integer k to the exponent). |
| `SFPSTOCHRND` | Stochastic/truncation rounding for format conversion. Emitted by `sfpi::float_to_fp16b(result, 0)` in `_xielu_mad_` when `is_fp32_dest_acc_en=false` (truncation mode 0, converting to fp16b). |
| `SFPMOV` | Moves data between LREGs. Emitted by the compiler for register allocation and data routing between intermediate results. |
| `SFPSWAP` | Swaps two vector values (used by `vec_min_max` to produce min/max pair). Emitted by `sfpi::vec_min_max(underflow_bound, z)` in `_sfpu_neg_exp_f32_`. |

### SFPU Register Usage

| Register | Usage |
|---|---|
| **DEST register (tile)** | The input tile resides in DEST. Each SFPU iteration reads from and writes back to `dst_reg[0]` (current DEST row pair). The write-back contains the final xIELU-transformed value. |
| **LREGs (L0-L7)** | The SFPU has 8 local vector registers (LREGs). `dst_reg[0]` implicitly maps to LREG[0] for loads/stores. Intermediate values (`x`, `beta_mul_x`, `alpha_p`, `alpha_n`, polynomial intermediates, `exp_term`, `result`) are held in LREGs during computation. The compiler allocates these; with the 4-branch structure and `_sfpu_neg_exp_f32_` helper, register pressure is moderate since branches are mutually exclusive (only one branch's intermediates are live at a time). |
| **vConstFloatPrgm0** | Programmable constant register, loaded with `1/ln(2) = 1.4426950408889634` by `xielu_init()`. Used by `_sfpu_neg_exp_f32_` for the base-2 conversion `z = val * (1/ln2)`. |
| **vConstFloatPrgm1** | Programmable constant register, loaded with `-1e-6` (epsilon threshold). Used as the boundary for the very-small-negative branch (`x >= -1e-6`). |
| **vConstFloatPrgm2** | Programmable constant register, loaded with `expm1(-1e-6) = -0.0000009999995427`. Used as the precomputed `expm1(eps)` value in the very-small-negative branch. |
| **vConst0** | Hardware constant `0.0f`. Used to initialize `result` in `_sfpu_neg_exp_f32_`. |
| **vConst1** | Hardware constant `1.0f`. Used as polynomial coefficients `c0` and `c1` in the Taylor expansion, and subtracted in the large-negative branch (`exp(x) - 1 - x`). |
| **Condition Code (CC)** | The per-lane predication register. Managed by `v_if`/`v_elseif`/`v_else`/`v_endif` to select which of the 4 branches writes to each lane. The SFPI abstraction handles CC state transitions automatically. |

### Address Mode Configuration

The address mode for the `xielu` SfpuType is configured in `eltwise_unary_sfpu_configure_addrmod<SfpuType::xielu>()` during `_llk_math_eltwise_unary_sfpu_init_()`. Since `xielu` does not match any special-case `if constexpr` branches, only the default address mode is set:

**Both Wormhole and Blackhole (identical configuration):**

| ADDR_MOD | srca.incr | srcb.incr | dest.incr | Purpose |
|---|---|---|---|---|
| `ADDR_MOD_7` | 0 | 0 | 0 | Default SFPU address mode. All increments are zero because SFPU kernels manage DEST addressing internally via `dst_reg++` (which uses the stride-2 mechanism) and `SETRWC` between faces. No auto-increment from the address mode is needed. |

No other ADDR_MODs are configured for this operation. The `ADDR_MOD_6` variants (for `topk_local_sort`, `typecast`, `unary_max/min`) are not activated for `xielu`.

## External Knowledge Sources
### DeepWiki Queries
No DeepWiki queries were needed for this analysis. The SFPU kernel implementation was fully documented from source code inspection.

### Confluence References
No Confluence references were needed. The kernel uses standard SFPI abstractions whose instruction mappings are well-established from the hardware model reference.

### Glean References
No Glean references were needed for this analysis.
