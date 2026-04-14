## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### Unary Dispatch Summary
- **UnaryOpType**: `SINH`
- **Compute kernel**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`
- **SFPU_OP_CHAIN_0 expansion**: `sinh_tile_init(); sinh_tile(0);`

#### Approximation Mode Resolution
There are TWO independent controls for approximation. Both are reported below.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(SINH)` in `unary_op_utils.cpp` -- falls through to `default: return false` |
| Template parameter (SFPU_OP_CHAIN) | `false` (no parameterization) | `get_op_init_and_func_default()` returns `sinh_tile_init()` / `sinh_tile(0)` with no template arguments; the API header `sinh.h` passes `APPROX` which is `false` |
| Effective SFPU path | `APPROXIMATION_MODE=false` throughout `calculate_sinh` and `exp_21f` | Since `sinh_tile_init<APPROX>()` and `sinh_tile<APPROX>()` receive `APPROX=false`, `calculate_sinh<false, 8>` is instantiated. However, the `exp_21f` helper does not branch on `APPROXIMATION_MODE` (no `if constexpr` on it), so both paths execute the same code. |

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/sinh.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_sinh.h` (identical on Blackhole) |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_sinh.h` (identical on Blackhole) |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (Blackhole variant at `tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu_params.h`) |

### Call Chain

1. **`sinh_tile(0)`** (in `sinh.h`) calls `MATH((llk_math_eltwise_unary_sfpu_sinh<APPROX>(0)))`.
2. **`llk_math_eltwise_unary_sfpu_sinh<false, 8>`** (in `llk_math_eltwise_unary_sfpu_sinh.h`) calls `_llk_math_eltwise_unary_sfpu_params_<false>(ckernel::sfpu::calculate_sinh<false, 8>, dst_index, (int)VectorMode::RC)`.
3. **`_llk_math_eltwise_unary_sfpu_params_`** (in `llk_math_eltwise_unary_sfpu_params.h`) sets the DEST write address, activates ADDR_MOD base, stalls for SFPU readiness, then loops over 4 faces calling `calculate_sinh<false, 8>()` once per face with `SETRWC` between faces.
4. **`calculate_sinh<false, 8>()`** (in `ckernel_sfpu_sinh.h`) iterates 8 times per face, loading from `dst_reg[0]`, computing sinh via the exp_21f helper and Taylor fallback, and writing back to `dst_reg[0]` with `dst_reg++` after each iteration.

For init: **`sinh_tile_init()`** calls `llk_math_eltwise_unary_sfpu_init<SfpuType::sinh, false>()`, which invokes `_llk_math_eltwise_unary_sfpu_init_<SfpuType::sinh>()`. This configures ADDR_MOD_7 (all increments = 0), initializes the SFPU config register, and resets RWC counters. No custom init callback is called since `sinh_init<false>()` is a no-op (no programmable constants needed).

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` (default) -- all 4 faces of the tile are processed.
- **Operation invocation**: The params dispatch loops `for (int face = 0; face < 4; face++)`, calling `calculate_sinh<false, 8>()` once per face. Each invocation runs 8 SFPU iterations (ITERATIONS=8), processing one face of 256 elements (8 iterations x 32 elements/iteration).
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, `SETRWC` between faces). On Wormhole, the params dispatch uses `TTI_SETRWC(CLR_NONE, CR_D, 8, ..., SET_D)` twice between faces (advancing by 8+8=16 physical DEST rows = 1 face). On Blackhole, it calls `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` which also does `inc_dst_addr<8>()` twice. The ADDR_MOD base is set (using ADDR_MOD slots 4-7), but only ADDR_MOD_7 is configured for `SfpuType::sinh`, with all increments set to 0.

### Annotated SFPU Kernel Source

The kernel uses SFPI abstractions (`sfpi::vFloat`, `sfpi::vInt`, `sfpi::dst_reg`, `v_if`/`v_endif`), so Style A (inline-commented source code) is used.

```cpp
// File: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_sinh.h

// Helper: compute 2^z using exp_21f algorithm (Moroz et al. 2022)
// Input z must be clamped to avoid overflow/underflow before calling.
// Returns 2^z as a vFloat.
template <bool APPROXIMATION_MODE>
inline sfpi::vFloat exp_21f(sfpi::vFloat z) { // APPROXIMATION_MODE=false (not branched on)
    // Step 1: Scale by 2^23 to shift fractional bits into integer position
    z = sfpi::addexp(z, 23); // SFPDIVP2: adds 23 to exponent field, effectively z *= 2^23

    // Step 2: Add IEEE 754 bias (0x3F800000 = 1.0f) and convert to int
    const sfpi::vFloat bias = sfpi::vFloat(0x3f800000); // raw hex constructor: 1.0f as IEEE 754
    sfpi::vInt z_int = _float_to_int32_positive_(z + bias); // [UNVERIFIED] FP32->INT32 conversion for positive values

    // Step 3: Decompose into exponent and mantissa parts
    sfpi::vInt exp_part = sfpi::exexp(sfpi::reinterpret<sfpi::vFloat>(z_int)); // SFPEXEXP: extract debiased exponent
    sfpi::vInt man_part = sfpi::exman9(sfpi::reinterpret<sfpi::vFloat>(z_int)); // SFPEXMAN: extract 9-bit mantissa

    // Step 4: Polynomial refinement for 2^frac(z)
    sfpi::vFloat d1 = sfpi::vFloat(0.40196114e-7f);
    sfpi::vFloat d2 = sfpi::int32_to_float(sfpi::vInt(0xf94ee7) + man_part, 0); // SFPIADD (int add), then SFPCAST (int->float, RTNE)
    sfpi::vFloat d3 = sfpi::int32_to_float(sfpi::vInt(0x560e) + man_part, 0); // SFPIADD (int add), then SFPCAST (int->float, RTNE)

    d2 = d1 * d2; // SFPMAD: d1 * d2 + 0.0
    sfpi::vInt frac_int = _float_to_int32_positive_(d2 * d3); // [UNVERIFIED] SFPMAD then FP32->INT32

    // Step 5: Reconstruct result = mantissa_frac * 2^exponent
    sfpi::vInt result_int =
        sfpi::reinterpret<sfpi::vInt>(sfpi::setexp(sfpi::reinterpret<sfpi::vFloat>(frac_int), 127U + exp_part));
        // SFPIADD: 127 + exp_part, then SFPSETEXP: sets exponent field of frac_int

    return sfpi::reinterpret<sfpi::vFloat>(result_int);
}

// sinh(x) = (exp(x) - exp(-x)) / 2
//         = (2^(x * log2(e)) - 2^(-x * log2(e))) / 2
//
// Implementation notes, see the original file for more details
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_sinh() { // APPROXIMATION_MODE=false, ITERATIONS=8
    constexpr float log2e = 1.4426950408889634f;
    const sfpi::vFloat v_log2e = log2e;
    const sfpi::vFloat v_half = 0.5f;
    const sfpi::vFloat v_low_threshold = -127.0f; // clamp threshold to prevent 2^z underflow
    const sfpi::vFloat v_sixth = 0.16666667f; // 1/6 for Taylor term

#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0]; // SFPLOAD: load 32 elements from current DEST row

        // Compute z_pos = x * log2(e) for exp(x) = 2^z_pos
        sfpi::vFloat z_pos = x * v_log2e; // SFPMAD: x * log2e + 0.0

        // Clamp to prevent underflow
        v_if(z_pos < v_low_threshold) { z_pos = v_low_threshold; } // CC: SFPSETCC + guarded SFPMOV
        v_endif;

        sfpi::vFloat exp_pos = exp_21f<APPROXIMATION_MODE>(z_pos); // inline call to 2^z

        // Compute z_neg = -x * log2(e) for exp(-x) = 2^z_neg
        sfpi::vFloat z_neg = -z_pos; // SFPMAD with sign inversion (InstrMod[0]=1)

        // Clamp to prevent underflow (z_neg could be very negative for large positive x)
        v_if(z_neg < v_low_threshold) { z_neg = v_low_threshold; } // CC: SFPSETCC + guarded SFPMOV
        v_endif;

        sfpi::vFloat exp_neg = exp_21f<APPROXIMATION_MODE>(z_neg); // inline call to 2^(-z)

        // sinh(x) = (exp(x) - exp(-x)) / 2
        sfpi::vFloat y = (exp_pos - exp_neg) * v_half; // SFPMAD (subtract), then SFPMAD (multiply by 0.5)

        // For small |x|, override with Taylor: sinh(x) ~ x + x^3/6
        sfpi::vFloat abs_x = sfpi::setsgn(x, 0); // SFPSETSGN: clear sign bit = absolute value
        v_if(abs_x < v_half) { // CC: compare |x| < 0.5
            sfpi::vFloat x_sq = x * x; // SFPMAD: x * x + 0.0
            y = x + x_sq * x * v_sixth; // SFPMAD chain: x_sq*x, then *v_sixth, then +x
        }
        v_endif;

        // Convert to bfloat16 for deterministic rounding
        y = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(y, 0)); // SFP_STOCH_RND: FP32->FP16B with RTNE rounding (mode 0 = NearestEven)

        sfpi::dst_reg[0] = y; // SFPSTORE: write 32 elements back to current DEST row
        sfpi::dst_reg++; // advance DEST pointer by 1 sfpi row (= 2 physical rows = 32 elements)
    }
}
```

### SFPU Instructions Used

| Instruction | SFPI Abstraction | Usage in Kernel |
|-------------|-----------------|-----------------|
| **SFPLOAD** | `dst_reg[0]` (read) | Load 32 elements from the current DEST row into an LREG at the start of each iteration |
| **SFPSTORE** | `dst_reg[0] = y` (write) | Store the computed sinh result back to the current DEST row |
| **SFPMAD** | `vFloat * vFloat`, `vFloat + vFloat`, `vFloat - vFloat`, `-vFloat` | Core arithmetic: multiplication (a*b+0), addition (a*1+b), subtraction (via InstrMod sign inversion), negation. Used extensively in exp_21f polynomial and in the main sinh formula |
| **SFPDIVP2** | `sfpi::addexp(z, 23)` | Add 23 to the exponent field of z, effectively multiplying by 2^23 to prepare for integer truncation in exp_21f |
| **SFPEXEXP** | `sfpi::exexp(...)` | Extract the debiased exponent from a float, used in exp_21f to decompose the integer/fractional parts |
| **SFPEXMAN** | `sfpi::exman9(...)` | Extract the 9-bit mantissa from a float, used in exp_21f for polynomial refinement |
| **SFPCAST** | `sfpi::int32_to_float(..., 0)` | Convert INT32 to FP32 with round-to-nearest-even, used in exp_21f to convert integer polynomial terms to float |
| **SFPIADD** | `sfpi::vInt(constant) + man_part`, `127U + exp_part` | Integer addition of mantissa offset constants and exponent bias in exp_21f |
| **SFPSETEXP** | `sfpi::setexp(...)` | Set the exponent field of the reconstructed result in exp_21f Step 5 |
| **SFPSETSGN** | `sfpi::setsgn(x, 0)` | Clear the sign bit to compute absolute value for the small-x Taylor branch check |
| **SFP_STOCH_RND** | `sfpi::float_to_fp16b(y, 0)` | Convert FP32 result to bfloat16 (FP16B) using round-to-nearest-even for deterministic rounding |
| **SFPLOADI** | `vFloat(constant)`, `vInt(constant)` | Load immediate constants (log2e, 0.5, -127.0, 1/6, polynomial coefficients) into LREGs |
| **SFPSETCC** | `v_if(condition)` | Set per-lane condition codes for the underflow clamp and small-x Taylor branch |
| **SFPENCC** | `v_if`/`v_endif` boundaries | Enable/disable condition code masking at the start and end of conditional blocks |
| **SFPCOMPC** | implicit in `v_endif` | Complement condition codes as part of the conditional execution mechanism |
| **SFPPUSHC / SFPPOPC** | nested `v_if` support | Push/pop CC state for nested conditional blocks |
| **_float_to_int32_positive_** | `_float_to_int32_positive_(...)` | **[UNVERIFIED]** -- this function is called in exp_21f but has no definition in the current codebase. It is expected to convert a positive FP32 value to INT32, likely mapping to SFPCAST with FP32_TO_INT32 mode |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **DEST registers** | Input tile data is read from and written back to DEST via `dst_reg[0]`. The stride-2 addressing means each `dst_reg[0]` access covers 2 physical DEST rows (32 elements). `dst_reg++` advances the SFPU pointer by 1 sfpi row. |
| **LREGs (general purpose)** | Used implicitly by the SFPI compiler for all intermediate `vFloat` and `vInt` values. The kernel has many live variables (x, z_pos, z_neg, exp_pos, exp_neg, y, abs_x, x_sq, plus all exp_21f temporaries), so register pressure is high. The compiler allocates from LREG0-LREG7 (8 registers per lane). |
| **LREG for constants** | Immediate constants (log2e, 0.5, -127.0, 1/6, 0x3f800000, 0.40196114e-7, 0xf94ee7, 0x560e) are loaded into LREGs via SFPLOADI. Since these are declared inside the loop body (or as `const` outside the loop with `#pragma GCC unroll 0`), the compiler may reload them each iteration or hoist them depending on register availability. |
| **CC (Condition Code) bits** | Per-lane CC bits are used for three conditional branches: (1) clamp z_pos when < -127, (2) clamp z_neg when < -127, (3) override with Taylor approximation when |x| < 0.5. Each `v_if`/`v_endif` activates and deactivates CC masking. |

### Address Mode Configuration

The address mode is configured in `eltwise_unary_sfpu_configure_addrmod<SfpuType::sinh>()` (in `llk_math_eltwise_unary_sfpu.h`). Since `SfpuType::sinh` does not match any of the special-cased types (topk_local_sort, typecast, unary_max, etc.), only the default ADDR_MOD_7 is configured:

**Wormhole B0:**
```
ADDR_MOD_7: { .srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0} }
```

This is the standard configuration for most unary SFPU operations. The zero-increment address mode means DEST addressing does not auto-increment between SFPU instructions within a single iteration -- advancement is handled explicitly by `dst_reg++` (which the compiler emits as pointer arithmetic) and by `SETRWC` between faces.

The `set_addr_mod_base()` call in `_llk_math_eltwise_unary_sfpu_params_` sets the ADDR_MOD base register to 1, which means ADDR_MOD slots 4-7 are active during SFPU execution. ADDR_MOD_7 (slot 7) is the one configured for this operation.

**Blackhole:**
The same ADDR_MOD_7 configuration applies. The Blackhole params dispatch uses `_llk_math_eltwise_unary_sfpu_start_`/`_llk_math_eltwise_unary_sfpu_done_` which internally call `set_addr_mod_base()`/`clear_addr_mod_base()`, and `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` for face advancement.

Both hardware generations use identical ADDR_MOD configuration for this operation.

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Determine compute kernel path, SFPU_OP_CHAIN_0 expansion, and approximation mode for SINH
   **Key Findings**: SINH uses `eltwise_sfpu.cpp`, expands to `sinh_tile_init()/sinh_tile(i)`, `get_op_approx_mode` returns false (default case), `get_macro_definition` returns `SFPU_OP_SINH_INCLUDE`

2. **File**: `tt_metal/hw/inc/api/compute/eltwise_unary/sinh.h`
   **Reason**: API header exposing `sinh_tile()` and `sinh_tile_init()` to the compute kernel
   **Key Findings**: Passes `APPROX` template parameter to `llk_math_eltwise_unary_sfpu_sinh<APPROX>(idst)` and `llk_math_eltwise_unary_sfpu_sinh_init<APPROX>()`

3. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_sinh.h`
   **Reason**: LLK dispatch layer connecting API to core SFPU implementation
   **Key Findings**: Calls `_llk_math_eltwise_unary_sfpu_params_` with `calculate_sinh<APPROXIMATE, 8>` as the functor and `VectorMode::RC` (default). Init calls `llk_math_eltwise_unary_sfpu_init<SfpuType::sinh, APPROXIMATE>()`

4. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_sinh.h`
   **Reason**: Core SFPU implementation containing `calculate_sinh` and `exp_21f` helper
   **Key Findings**: Implements sinh via dual exp_21f calls (Moroz et al. 2022 algorithm for 2^z) with Taylor fallback for small |x|. Uses `_float_to_int32_positive_` which is undefined in the current codebase. Includes bfloat16 rounding via `float_to_fp16b(y, 0)` for deterministic output.

5. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Parameters dispatch function controlling face iteration and DEST address progression
   **Key Findings**: VectorMode::RC iterates 4 faces, calling the SFPU functor once per face, with SETRWC advancing 8+8=16 physical rows between faces

6. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: ADDR_MOD configuration and SFPU init/start/done functions
   **Key Findings**: `eltwise_unary_sfpu_configure_addrmod<SfpuType::sinh>()` only configures ADDR_MOD_7 with all-zero increments (no special cases match). Init calls `_init_sfpu_config_reg()` and `reset_counters(SET_ABD_F)`

7. **File**: `.claude/references/sfpu-hardware-model.md`
   **Reason**: Authoritative SFPU hardware model reference for instruction semantics, addressing, and tile geometry
   **Key Findings**: Confirmed stride-2 addressing model, ITERATIONS=8 per face, SFPMAD for all float arithmetic, SFPIADD for integer-only operations

8. **File**: `runtime/sfpi/include/sfpi_lib.h`
   **Reason**: SFPI intrinsic-to-instruction mapping for all SFPI library functions used in the kernel
   **Key Findings**: `addexp` -> SFPDIVP2, `exexp` -> SFPEXEXP, `exman9` -> SFPEXMAN, `int32_to_float` -> SFPCAST, `setexp` -> SFPSETEXP, `setsgn` -> SFPSETSGN, `float_to_fp16b` -> SFP_STOCH_RND (with RTNE rounding when mode=0)

9. **File**: `tt_metal/jit_build/genfiles.cpp` (line 394)
   **Reason**: Confirm how `math_approx_mode` maps to the `APPROX` compile-time constant in compute kernels
   **Key Findings**: `constexpr bool APPROX = {};` is generated from `desc.get_hlk_math_approx_mode()`, which is set from `math_approx_mode` in ComputeConfig

10. **File**: `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h`
    **Reason**: Verify the conditional include mechanism for SFPU_OP_SINH_INCLUDE
    **Key Findings**: When `SFPU_OP_SINH_INCLUDE` is defined to 1, it includes `api/compute/eltwise_unary/sinh.h`
