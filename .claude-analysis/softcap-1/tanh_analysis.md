## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### Unary Dispatch Summary
- **UnaryOpType**: `TANH`
- **Compute kernel**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`
- **SFPU_OP_CHAIN_0 expansion**: `tanh_tile_init(); tanh_tile(0);`

**Critical Note**: The SFPU implementation files for `tanh` (`ckernel_sfpu_tanh.h` and `llk_math_eltwise_unary_sfpu_tanh.h`) have been **removed from this worktree** as part of the generator evaluation environment (see comment in `sfpu_operations.h`: "Metal SFPU includes removed -- primitives nuked for generator evaluation"). The API-level header `compute_kernel_api.h` still defines the `tanh_tile()` / `tanh_tile_init()` entry points and references the undefined `llk_math_eltwise_unary_sfpu_tanh()` function. The `get_op_init_and_func_default()` switch in `unary_op_utils.cpp` does not have a `case UnaryOpType::TANH` entry, meaning the program factory would `TT_THROW` at runtime if tanh dispatch were attempted in this worktree.

The analysis below reconstructs the tanh SFPU architecture based on:
1. The existing API-level signatures in `compute_kernel_api.h`
2. The sibling hyperbolic SFPU implementations that survived the stripping (`sinh`, `atanh`)
3. The shared `exp_21f` helper function used by `sinh`
4. The LLK dispatch infrastructure (`llk_math_eltwise_unary_sfpu_params.h`)
5. The SFPU hardware model reference (`.claude/references/sfpu-hardware-model.md`)

#### Approximation Mode Resolution
There are TWO independent controls for approximation. Both are reported below.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(TANH)` in `unary_op_utils.cpp` -- falls through to `default: return false` |
| Template parameter (SFPU_OP_CHAIN) | `fast_and_approx=false` (default) | `tanh_tile<>()` and `tanh_tile_init<>()` in `compute_kernel_api.h` both default `fast_and_approx=false`; the op chain string `"tanh_tile_init(); tanh_tile(0);"` uses default template args |
| Effective SFPU path | Non-approximate path: `llk_math_eltwise_unary_sfpu_tanh<false, DST_ACCUM_MODE>` | The `fast_and_approx` template parameter controls which code branch executes in the SFPU kernel. LLK tests explicitly skip tanh with `ApproximationMode.Yes` ("Metal tanh does not support approximation mode"). |

### SFPU Abstraction Layers
The file paths for each abstraction layer are listed below. Paths marked as **[MISSING]** indicate files that existed in the upstream codebase but were removed from this worktree.

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/compute_kernel_api.h` (lines 154-180, defines `tanh_tile_init<>()` and `tanh_tile<>()`) |
| **LLK Dispatch** | **[MISSING]** `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_tanh.h` (would define `llk_math_eltwise_unary_sfpu_tanh()` and `llk_math_eltwise_unary_sfpu_tanh_init()`) |
| **Core SFPU Implementation** | **[MISSING]** `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_tanh.h` (would define `calculate_tanh()` and `tanh_init()`) |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (shared generic dispatch: `_llk_math_eltwise_unary_sfpu_params_()`) |

### Call Chain
The SFPU kernel invocation follows the standard metal unary SFPU call chain pattern (reconstructed from the API header and sibling operations):

1. **Compute kernel** (`eltwise_sfpu.cpp`): The `SFPU_OP_CHAIN_0` macro expands to `tanh_tile_init(); tanh_tile(0);`, which calls the API functions.
2. **API Header** (`compute_kernel_api.h`): `tanh_tile_init<false>()` calls `MATH((llk_math_eltwise_unary_sfpu_tanh_init<false, DST_ACCUM_MODE>()))`, and `tanh_tile<false>(0)` calls `MATH((llk_math_eltwise_unary_sfpu_tanh<false, DST_ACCUM_MODE>(0)))`.
3. **LLK Dispatch** [MISSING]: Based on the sinh/atanh pattern, `llk_math_eltwise_unary_sfpu_tanh_init()` would call `llk_math_eltwise_unary_sfpu_init<SfpuType::tanh, APPROXIMATE>()` plus optionally a `tanh_init<APPROXIMATE>()` function to load programmable constants. `llk_math_eltwise_unary_sfpu_tanh()` would call `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::calculate_tanh<APPROXIMATE, ITERATIONS>, dst_index, vector_mode)`.
4. **Parameters Dispatch** (`llk_math_eltwise_unary_sfpu_params.h`): Iterates over 4 faces in `VectorMode::RC`, calling the SFPU function once per face, with `TTI_SETRWC` between faces to advance the DEST address.
5. **Core SFPU** [MISSING]: The `calculate_tanh<APPROXIMATE, ITERATIONS>()` function would iterate `ITERATIONS=8` times per face, processing 32 elements per iteration (2 physical DEST rows x 16 elements).

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` (all 4 faces processed, covering the full 32x32 tile).
- **Operation invocation**: The `_llk_math_eltwise_unary_sfpu_params_` function loops over 4 faces, calling `calculate_tanh()` once per face. Each call processes 8 iterations (ITERATIONS=8 default), covering one 16x16 face.
- **DEST address progression**: Standard DEST progression. On Wormhole, `ADDR_MOD_7` is configured with `{.srca={.incr=0}, .srcb={.incr=0}, .dest={.incr=0}}` (no auto-increment from the addr_mod; the SFPU kernel manages its own progression via `dst_reg++`). Between faces, `TTI_SETRWC` with `CR_D, 8, SET_D` is called twice (advancing 16 physical DEST rows = 1 face stride). Within the SFPU function, `dst_reg++` advances 1 sfpi row per iteration (= 2 physical DEST rows = 32 elements).

### Annotated SFPU Kernel Source

**Note**: The core SFPU implementation file `ckernel_sfpu_tanh.h` is **missing** from this worktree. Below, the closely related `ckernel_sfpu_sinh.h` is included as the authoritative reference for how hyperbolic SFPU functions are implemented on Wormhole/Blackhole. The tanh operation follows an almost identical structure: where sinh computes `(exp(x) - exp(-x)) / 2`, tanh computes `(exp(x) - exp(-x)) / (exp(x) + exp(-x))`, using the same `exp_21f` helper.

The `sinh` SFPU kernel is included here because:
1. It shares the same `exp_21f` helper function that tanh would use
2. It demonstrates the exact SFPI coding patterns (conditional execution, iteration structure, bfloat16 rounding)
3. It handles the same numerical edge cases (underflow clamping, small-x Taylor approximation)

```cpp
// File: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_sinh.h

namespace ckernel::sfpu {

// Helper: compute 2^z using exp_21f algorithm (Moroz et al. 2022)
// Input z must be clamped to avoid overflow/underflow before calling.
// Returns 2^z as a vFloat.
template <bool APPROXIMATION_MODE>
inline sfpi::vFloat exp_21f(sfpi::vFloat z) {
    // Step 1: Scale by 2^23 to shift fractional bits into integer position
    z = sfpi::addexp(z, 23);

    // Step 2: Add IEEE 754 bias (0x3F800000 = 1.0f) and convert to int
    const sfpi::vFloat bias = sfpi::vFloat(0x3f800000);
    sfpi::vInt z_int = _float_to_int32_positive_(z + bias);

    // Step 3: Decompose into exponent and mantissa parts
    sfpi::vInt exp_part = sfpi::exexp(sfpi::reinterpret<sfpi::vFloat>(z_int));
    sfpi::vInt man_part = sfpi::exman9(sfpi::reinterpret<sfpi::vFloat>(z_int));

    // Step 4: Polynomial refinement for 2^frac(z)
    sfpi::vFloat d1 = sfpi::vFloat(0.40196114e-7f);
    sfpi::vFloat d2 = sfpi::int32_to_float(sfpi::vInt(0xf94ee7) + man_part, 0);
    sfpi::vFloat d3 = sfpi::int32_to_float(sfpi::vInt(0x560e) + man_part, 0);

    d2 = d1 * d2;
    sfpi::vInt frac_int = _float_to_int32_positive_(d2 * d3);

    // Step 5: Reconstruct result = mantissa_frac * 2^exponent
    sfpi::vInt result_int =
        sfpi::reinterpret<sfpi::vInt>(sfpi::setexp(sfpi::reinterpret<sfpi::vFloat>(frac_int), 127U + exp_part));

    return sfpi::reinterpret<sfpi::vFloat>(result_int);
}

// sinh(x) = (exp(x) - exp(-x)) / 2
//         = (2^(x * log2(e)) - 2^(-x * log2(e))) / 2
//
// For small |x| (< 0.5), the exp subtraction suffers catastrophic cancellation
// because exp(x) and exp(-x) are both close to 1.0. In that regime we use the
// Taylor approximation sinh(x) ~ x + x^3/6, which is accurate to < 1 ULP in
// bfloat16 for |x| < 0.5.
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_sinh() { // APPROXIMATION_MODE=false, ITERATIONS=8
    constexpr float log2e = 1.4426950408889634f;
    const sfpi::vFloat v_log2e = log2e;
    const sfpi::vFloat v_half = 0.5f;
    const sfpi::vFloat v_low_threshold = -127.0f;
    const sfpi::vFloat v_sixth = 0.16666667f;

#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0]; // SFPLOAD: load 32 elements from current DEST position

        // Compute z_pos = x * log2(e) for exp(x) = 2^z_pos
        sfpi::vFloat z_pos = x * v_log2e; // SFPMAD

        // Clamp to prevent underflow
        v_if(z_pos < v_low_threshold) { z_pos = v_low_threshold; } // CC: SFPSETCC + SFPMOV
        v_endif;

        sfpi::vFloat exp_pos = exp_21f<APPROXIMATION_MODE>(z_pos); // ~10 instructions

        // Compute z_neg = -x * log2(e) for exp(-x) = 2^z_neg
        sfpi::vFloat z_neg = -z_pos; // SFPMAD (negate via multiply by -1 or sign manipulation)

        // Clamp to prevent underflow (z_neg could be very negative for large positive x)
        v_if(z_neg < v_low_threshold) { z_neg = v_low_threshold; }
        v_endif;

        sfpi::vFloat exp_neg = exp_21f<APPROXIMATION_MODE>(z_neg);

        // sinh(x) = (exp(x) - exp(-x)) / 2
        sfpi::vFloat y = (exp_pos - exp_neg) * v_half; // SFPMAD (sub) + SFPMAD (mul)

        // For small |x|, override with Taylor: sinh(x) ~ x + x^3/6
        sfpi::vFloat abs_x = sfpi::setsgn(x, 0); // SFPSETSGN: clear sign bit
        v_if(abs_x < v_half) {
            sfpi::vFloat x_sq = x * x; // SFPMAD
            y = x + x_sq * x * v_sixth; // SFPMAD chain
        }
        v_endif;

        // Convert to bfloat16 for deterministic rounding
        y = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(y, 0)); // SFP_STOCH_RND or SFPCAST

        sfpi::dst_reg[0] = y; // SFPSTORE: write 32 elements back to DEST
        sfpi::dst_reg++; // advance sfpi address by 1 (= 2 physical DEST rows)
    }
}

template <bool APPROXIMATION_MODE>
inline void sinh_init() {
    // No programmable constants needed
}

}  // namespace ckernel::sfpu
```

### Reconstructed tanh Kernel Structure

Based on the sinh implementation pattern and the mathematical relationship `tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))`, the missing `calculate_tanh()` function would follow this structure:

```
tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
         = (2^(x*log2e) - 2^(-x*log2e)) / (2^(x*log2e) + 2^(-x*log2e))
```

The key differences from sinh would be:
1. **Division instead of halving**: Instead of `(exp_pos - exp_neg) * 0.5`, tanh computes `(exp_pos - exp_neg) / (exp_pos + exp_neg)`, requiring a reciprocal operation (SFPNONLINEAR with RECIP_MODE, or an SFPMAD-based Newton-Raphson iteration).
2. **Small-x approximation**: For `|x| < threshold`, tanh(x) is approximated by `x - x^3/3` (first two terms of the Taylor series), compared to sinh's `x + x^3/6`.
3. **Saturation clamping**: For large `|x|` (e.g., `|x| > 9`), tanh saturates to +/-1.0, which can be handled with a simple sign-preserving clamp rather than computing the exponentials.

An alternative approach (used in some implementations) computes `tanh(x) = 1 - 2/(exp(2x) + 1)`, which requires only a single exponential evaluation.

### SFPU Instructions Used

The following SFPU instructions are used by the `exp_21f` helper and the hyperbolic function pattern. These are the instructions that the missing tanh kernel would use (based on the sinh reference implementation):

| Instruction | Description | Used For |
|-------------|-------------|----------|
| `SFPLOAD` | Load DEST row to LREG with format conversion | Loading tile data from DEST into SFPU registers (`dst_reg[0]`) |
| `SFPSTORE` | Store LREG to DEST row with format conversion | Writing computed results back to DEST (`dst_reg[0] = y`) |
| `SFPMAD` | Fused multiply-add: `VD = VA * VB + VC` | All floating-point arithmetic: multiplication, addition, subtraction (via sign-inverted addend). Core of Horner polynomial evaluation in `exp_21f`. |
| `SFPSETSGN` | Set sign field of FP32 value | Computing absolute value (`setsgn(x, 0)`) for threshold comparison |
| `SFPEXEXP` | Extract exponent field from FP32 | IEEE 754 decomposition in `exp_21f`: extracting exponent of intermediate result |
| `SFPSETEXP` | Set exponent field of FP32 value | Reconstructing `2^exp` in `exp_21f`: setting exponent to `127 + exp_part` |
| `SFPDIVP2` | Divide by power of 2 (exponent subtract) | `addexp(z, 23)` scales the input in `exp_21f` |
| `SFPIADD` | Integer add (2's complement) | Integer arithmetic in `exp_21f`: adding mantissa corrections (`0xf94ee7 + man_part`) |
| `SFPSETCC` | Set condition code based on comparison | Conditional execution for underflow clamping (`z_pos < v_low_threshold`) and small-x branch |
| `SFPENCC` | Enable/disable condition code | Entry/exit of `v_if`/`v_endif` conditional blocks |
| `SFPCOMPC` | Complement condition code | Implicit in `v_if`/`v_endif` expansion for else-branches |
| `SFPPUSHC` | Push CC state onto stack | Nested conditional management in `v_if` blocks |
| `SFPPOPC` | Pop CC state from stack | Restoring CC state at `v_endif` |
| `SFP_STOCH_RND` or `SFPCAST` | Format conversion with rounding | Converting FP32 result to bfloat16 (`float_to_fp16b`) for deterministic rounding |

Additionally, if the tanh implementation uses hardware-accelerated reciprocal:
| `SFPNONLINEAR` (mode 0) | Hardware reciprocal approximation | Computing `1/(exp_pos + exp_neg)` for the division step |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **DEST rows** | Input/output tile data. Each iteration processes 2 physical rows (32 elements) via stride-2 addressing. |
| **LREG0-LREG3** (LREGS1 bank) | General-purpose: hold intermediate values (`x`, `z_pos`, `z_neg`, `exp_pos`, `exp_neg`, `y`). The SFPI compiler maps `vFloat` temporaries to these registers. |
| **LREG4-LREG7** (LREGS2 bank) | Additional temporaries for the `exp_21f` helper computation (mantissa corrections, polynomial coefficients). LREG7 may be used for indirect addressing by SFPMAD if the compiler emits indirect mode. |
| **Programmable Constants** (`vConstFloatPrgm0-2`) | If `tanh_init()` loads constants (like `atanh_init()` does), these would hold polynomial coefficients or thresholds. If no init function sets them (like `sinh_init()`), they retain their default/previous values. |
| **Fixed Constants** | `vConst1` (1.0f) for computing `exp(x) + 1` or similar. `vConst0` (0.0f) for sign manipulation. |

### Address Mode Configuration

The address mode for tanh follows the standard unary SFPU pattern:

**Wormhole B0:**
- `ADDR_MOD_7` is configured in `eltwise_unary_sfpu_configure_addrmod()` with:
  ```
  .srca = {.incr = 0}
  .srcb = {.incr = 0}
  .dest = {.incr = 0}
  ```
  This is a zero-increment mode because the SFPU manages its own DEST progression internally via `dst_reg++` (which increments the SFPU's DEST read/write counter by `SFP_DESTREG_STRIDE=2` physical rows per step).

**Blackhole:**
- Same `ADDR_MOD_7` configuration. The Blackhole LLK `eltwise_unary_sfpu_configure_addrmod()` uses `ADDR_MOD_6` for certain ops (typecast, unary_max/min, signbit) with `.dest={.incr=2}`, but the default unary path uses `ADDR_MOD_7` with `.dest={.incr=0}`, identical to Wormhole.

The `_llk_math_eltwise_unary_sfpu_init_()` function configures the SFPU config register, sets up the address mode, and resets counters via `math::reset_counters(p_setrwc::SET_ABD_F)`.

## Local Knowledge Sources
### Local References
1. **File**: `tt_metal/hw/inc/api/compute/compute_kernel_api.h` (lines 154-180)
   **Reason**: Contains the API-level `tanh_tile<>()` and `tanh_tile_init<>()` function definitions
   **Key Findings**: `tanh_tile<fast_and_approx=false>()` calls `llk_math_eltwise_unary_sfpu_tanh<false, DST_ACCUM_MODE>(idst)`. `tanh_tile_init<false>()` calls `llk_math_eltwise_unary_sfpu_tanh_init<false, DST_ACCUM_MODE>()`.

2. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Checked dispatch configuration for TANH
   **Key Findings**: `TANH` is not in `get_op_init_and_func_default()` (would TT_THROW). `get_op_approx_mode()` returns `false` for all ops (default case). `get_compute_kernel_path()` returns `"eltwise_sfpu.cpp"` for all ops (default case). `get_macro_definition()` returns `"SFPU_OP_COMPUTE_KERNEL_API_INCLUDE"` for TANH (default case).

3. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_sinh.h`
   **Reason**: Sibling hyperbolic SFPU implementation using the same exp_21f helper
   **Key Findings**: sinh uses `exp_21f()` for 2^z approximation via Moroz et al. 2022 algorithm, with underflow clamping and small-x Taylor fallback. Same pattern would apply to tanh.

4. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_atanh.h`
   **Reason**: Sibling hyperbolic SFPU implementation showing polynomial approximation patterns
   **Key Findings**: Uses IEEE 754 decomposition with `exexp()`/`setexp()` and cubic minimax polynomial for ln(m). Loads programmable constants in `atanh_init()`.

5. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_sinh.h`
   **Reason**: LLK dispatch pattern reference for tanh
   **Key Findings**: `llk_math_eltwise_unary_sfpu_sinh()` calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(calculate_sinh<APPROXIMATE, ITERATIONS>, dst_index, vector_mode)`. Init calls `llk_math_eltwise_unary_sfpu_init<SfpuType::sinh, APPROXIMATE>()`.

6. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Generic parameters dispatch that all unary SFPU ops use
   **Key Findings**: Iterates 4 faces in `VectorMode::RC`, calling the SFPU function once per face with `TTI_SETRWC` between faces.

7. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: Core SFPU infrastructure (init, addrmod, start/done)
   **Key Findings**: `ADDR_MOD_7` configured with zero increments. `_llk_math_eltwise_unary_sfpu_init_()` sets up SFPU config, addrmod, and resets counters.

8. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h`
   **Reason**: Checked whether tanh has a dedicated SfpuType enum
   **Key Findings**: Only `frac`, `swish`, `atanh`, `sinh` survive in the metal SfpuType enum. No `tanh` entry.

9. **File**: `tt_metal/third_party/tt_llk/tt_llk_quasar/common/inc/ckernel_instr_params.h`
   **Reason**: Checked hardware-level tanh support
   **Key Findings**: Quasar has `p_sfpnonlinear::TANH_MODE = 0x5` for hardware-accelerated tanh. Wormhole and Blackhole do NOT have this mode -- tanh must be computed in software.

10. **File**: `tests/tt_metal/tt_metal/integration/test_sfpu_compute.cpp`
    **Reason**: Confirmed SFPU_OP_CHAIN_0 expansion for tanh
    **Key Findings**: Line 58: `{"tanh", {{"SFPU_OP_CHAIN_0", "tanh_tile_init(); tanh_tile(0);"}}}`.

11. **File**: `tt_metal/third_party/tt_llk/tests/python_tests/test_zzz_eltwise_unary_sfpu.py`
    **Reason**: Verified approximation mode constraints
    **Key Findings**: Line 199-200: `if mathop == MathOperation.Tanh and approx_mode == ApproximationMode.Yes: pytest.skip(reason="Metal tanh does not support approximation mode")`.

12. **File**: `.claude/references/sfpu-hardware-model.md`
    **Reason**: Authoritative reference for SFPU instruction semantics and hardware model
    **Key Findings**: SFPNONLINEAR mode 5 = tanh (1 ULP max error on FP16_B). SFPMAD is used for all float add/multiply. SFPIADD is integer-only. stride-2 addressing model confirmed.
