## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### Unary Dispatch Summary
- **UnaryOpType**: `RPOW`
- **Compute kernel**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`
- **SFPU_OP_CHAIN_0 expansion**: `rpow_tile_init(); rpow_tile(idst, base_val);` where `base_val` is the IEEE 754 bit representation of the `base` float parameter passed via `param0`

**Note on dispatch wiring**: In the current codebase state, `UnaryOpType::RPOW` is registered in `unary_op_types.hpp` and the compute kernel `eltwise_sfpu.cpp` includes `rpow.h`, but the `get_op_init_and_func_default` / `get_op_init_and_func_parameterized` switch in `unary_op_utils.cpp` does NOT have a case for `RPOW`. This means the host-side dispatch is incomplete -- the SFPU chain definition would need to be wired to generate `rpow_tile_init()` and `rpow_tile(idst, param0_bits)`. Additionally, `SfpuType::rpow` is referenced in the LLK dispatch header but is NOT present in `llk_sfpu_types.h`. These are build-breaking issues.

#### Approximation Mode Resolution

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(RPOW)` in `unary_op_utils.cpp` -- falls through to `default: return false` |
| Template parameter (SFPU_OP_CHAIN) | `APPROX` (compile-time define from ComputeConfig) | `rpow_tile_init()` calls `llk_math_eltwise_unary_sfpu_rpow_init<APPROX>()` which forwards to `rpow_init<APPROX>()`. The `APPROX` define resolves to the `math_approx_mode` from ComputeConfig, which is `false`. |
| Effective SFPU path | The `APPROXIMATION_MODE` template parameter is `false`. However, `calculate_rpow` does not contain any `if constexpr (APPROXIMATION_MODE)` branching -- the algorithm is identical regardless of the approximation mode setting. The template parameter is accepted but unused. | `ckernel_sfpu_rpow.h` line 38: `template <bool APPROXIMATION_MODE, int ITERATIONS = 8>` -- no branch on `APPROXIMATION_MODE` in the function body |

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/rpow.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_rpow.h` (identical for WH and BH) |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_rpow.h` (identical for WH and BH) |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |

### Call Chain

1. **`rpow_tile(idst, base_val)`** (API header `rpow.h`): Calls `MATH((llk_math_eltwise_unary_sfpu_rpow<APPROX>(idst, base_val)))`, gating execution to the math thread via the `MATH()` macro.

2. **`llk_math_eltwise_unary_sfpu_rpow<APPROXIMATE, ITERATIONS=8>(dst_index, base_val, VectorMode::RC)`** (LLK dispatch): Calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::calculate_rpow<APPROXIMATE, ITERATIONS>, dst_index, vector_mode, base_val)`, passing the core SFPU function as a callable with `base_val` forwarded as a runtime argument.

3. **`_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(...)`** (params dispatch in `llk_math_eltwise_unary_sfpu_params.h`): Sets the DEST write address for the tile, configures ADDR_MOD base, stalls until SFPU is ready, then loops over faces (4 faces for `VectorMode::RC`). For each face, it invokes the callable (`calculate_rpow(base_val)`) and then executes `TTI_SETRWC` twice (advancing DEST address by 16 rows) to move to the next face.

4. **`calculate_rpow<APPROXIMATION_MODE, ITERATIONS=8>(base_val)`** (core SFPU implementation in `ckernel_sfpu_rpow.h`): Decodes the base parameter from IEEE 754 bits, precomputes `log2(|base|)` as a scalar on the RISC-V core, then loops 8 iterations per face, computing `base^x = 2^(x * log2(base))` for each SFPU row using the exp_21f algorithm.

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` -- all 4 faces of the tile are processed (face 0 through face 3), covering all 1024 elements.
- **Operation invocation**: The core SFPU function `calculate_rpow(base_val)` is called once per face (4 total calls per tile). Each call processes `ITERATIONS=8` sfpi rows (256 elements per face). The `base_val` parameter (IEEE 754 bits of the base scalar) is forwarded to each call.
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, `SETRWC` between faces). The address mode is `ADDR_MOD_7` on both Wormhole and Blackhole, configured with `{srca.incr=0, srcb.incr=0, dest.incr=0}`. The `SfpuType::rpow` does not match any of the special-case `if constexpr` branches (topk_local_sort, typecast, etc.) so only the default `ADDR_MOD_7` is configured. Since `rpow` is not in the `SfpuType` enum, the actual addr_mod configuration at compile time would depend on what `SfpuType` value is used -- this is another manifestation of the incomplete wiring.

### Annotated SFPU Kernel Source

The kernel uses SFPI abstractions (`sfpi::vFloat`, `sfpi::vInt`, `dst_reg`, `v_if`/`v_endif`), so Style A applies.

**Critical issue**: The kernel calls `_float_to_int32_positive_()` which is NOT defined anywhere in the codebase. The equivalent function in the upstream tt_llk `ckernel_sfpu_exp.h` is `_float_to_int32_for_exp_21f_()`. This is a compilation-blocking bug. The analysis below documents the intended behavior.

```cpp
// File: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_rpow.h

// Implementation notes, see the original file for more details

namespace {
inline uint32_t float_to_bits(float f) { // scalar helper: reinterpret float as uint32
    union {
        float fval;
        uint32_t uval;
    } conv;
    conv.fval = f;
    return conv.uval;
}
}  // namespace

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_rpow(const uint32_t base_val) { // APPROXIMATION_MODE=false (unused), ITERATIONS=8
    // Decode base parameter from IEEE 754 bits
    const float base_scalar = Converter::as_float(base_val); // scalar reinterpret uint32 -> float
    const float abs_base = base_scalar < 0.0f ? -base_scalar : base_scalar;

    // Precompute log2(|base|) on RISC-V scalar unit (not SFPU)
    // IEEE 754: float = 2^exp * mantissa, so log2(float) = exp + log2(mantissa)
    uint32_t base_bits = float_to_bits(abs_base);
    int32_t base_exp = static_cast<int32_t>(((base_bits >> 23) & 0xFF)) - 127; // extract biased exponent, debias
    // Normalize mantissa to [1,2) by setting exponent field to 127 (bias)
    uint32_t mantissa_bits = (base_bits & 0x007FFFFF) | 0x3F800000;
    float mantissa_norm = Converter::as_float(mantissa_bits);

    // 3rd order polynomial approximation for log2(x) over [1,2)
    const float c3 = 0x2.44734p-4f;       // ~0.1416
    const float c2 = -0xd.e712ap-4f;      // ~-0.8691
    const float c1 = 0x2.4f5388p+0f;      // ~2.3110
    const float c0 = -0x1.952992p+0f;     // ~-1.5828
    const float inv_ln2 = 1.4426950408889634f; // 1/ln(2)

    float series = c0 + mantissa_norm * (c1 + mantissa_norm * (c2 + mantissa_norm * c3)); // Horner eval
    float log2_base = static_cast<float>(base_exp) + series * inv_ln2;

    // Load precomputed log2(base) into a vector register
    const sfpi::vFloat v_log2_base = log2_base; // SFPLOADI: broadcast scalar to all SFPU lanes
    const sfpi::vFloat v_low_threshold = -127.0f; // SFPLOADI: load -127.0 as clamp threshold

#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) { // 8 iterations per face
        sfpi::vFloat x = sfpi::dst_reg[0]; // SFPLOAD: load current element from DEST

        // z = x * log2(base)
        sfpi::vFloat z_f32 = x * v_log2_base; // SFPMAD: z = x * log2_base + 0.0

        // Clamp to prevent overflow: if z < -127, set to -127
        v_if(z_f32 < v_low_threshold) { z_f32 = v_low_threshold; } // SFPSETCC(LT0) after subtract, SFPMOV guarded
        v_endif; // SFPENCC to restore all-lanes-enabled

        // Compute 2^z using exp_21f algorithm (Moroz et al. 2022, Section 5)
        z_f32 = sfpi::addexp(z_f32, 23);  // SFPDIVP2 with ADD mode: multiply by 2^23
        const sfpi::vFloat bias = sfpi::vFloat(0x3f800000); // SFPLOADI: 127.0 * 2^23 = 0x3F800000
        sfpi::vInt z = _float_to_int32_positive_(z_f32 + bias); // [UNDEFINED] intended: convert positive float to int32

        sfpi::vInt zii = sfpi::exexp(sfpi::reinterpret<sfpi::vFloat>(z));   // SFPEXEXP: extract exponent (debiased)
        sfpi::vInt zif = sfpi::exman9(sfpi::reinterpret<sfpi::vFloat>(z));  // SFPEXMAN: extract 9-bit mantissa with padding

        // Compute 2^frac(z) using Horner form polynomial
        sfpi::vFloat d1 = sfpi::vFloat(0.40196114e-7);  // SFPLOADI: small coefficient
        sfpi::vFloat d2 = sfpi::int32_to_float(sfpi::vInt(0xf94ee7) + zif, 0); // SFPIADD + SFPCAST: integer add, then int->float
        sfpi::vFloat d3 = sfpi::int32_to_float(sfpi::vInt(0x560e) + zif, 0);   // SFPIADD + SFPCAST: integer add, then int->float

        d2 = d1 * d2; // SFPMAD: multiply
        zif = _float_to_int32_positive_(d2 * d3); // [UNDEFINED] SFPMAD then int conversion

        // Restore exponent: result = mantissa * 2^exponent
        zii = sfpi::reinterpret<sfpi::vInt>(sfpi::setexp(sfpi::reinterpret<sfpi::vFloat>(zif), 127U + zii));
        // SFPIADD (127 + zii) then SFPSETEXP: set exponent field of mantissa result

        sfpi::vFloat y = sfpi::reinterpret<sfpi::vFloat>(zii); // no instruction, type reinterpret

        // Handle special cases (scalar branches, evaluated at compile-time per invocation)
        if (abs_base == 0.0f) {
            // base == 0: 0^x = 0 for x > 0, 1 for x == 0, inf for x < 0
            v_if(x > 0.0f) { y = 0.0f; }  // SFPSETCC(GTE0 after negate) + guarded SFPMOV
            v_endif;
            v_if(x == 0.0f) { y = sfpi::vConst1; }  // SFPSETCC(EQ0) + guarded SFPMOV
            v_endif;
            v_if(x < 0.0f) {
                y = sfpi::vFloat(std::numeric_limits<float>::infinity()); // SFPLOADI + guarded SFPMOV
            }
            v_endif;
        } else if (base_scalar < 0.0f) {
            // Negative base: result is real only for integer exponents
            sfpi::vInt x_int = sfpi::float_to_int16(x, 0); // SFP_STOCH_RND: FP32 -> INT16 (nearest even)
            sfpi::vFloat x_rounded = sfpi::int32_to_float(x_int, 0); // SFPCAST: INT32 -> FP32

            // If x is odd integer, negate the result
            y = sfpi::setsgn(y, x_int << 31); // SFPSHFT (left shift 31) + SFPSETSGN: sign = LSB of x_int

            // If x is not an integer, set result to NaN
            v_if(x_rounded != x) { y = sfpi::vFloat(std::numeric_limits<float>::quiet_NaN()); }
            v_endif;
        }

        // Convert to bfloat16 with round-to-nearest-even for accuracy
        y = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(y, 0)); // SFP_STOCH_RND: FP32 -> FP16B (nearest even)

        sfpi::dst_reg[0] = y; // SFPSTORE: write result back to DEST
        sfpi::dst_reg++;      // advance DEST pointer by 1 sfpi row (= 2 physical rows = 32 elements)
    }
}

template <bool APPROXIMATION_MODE>
inline void rpow_init() {
    // No programmable constants needed - log2(base) is computed from the parameter
}
```

### SFPU Instructions Used

| Instruction | SFPI Abstraction | Purpose in rpow |
|-------------|-----------------|-----------------|
| **SFPLOAD** | `dst_reg[0]` (read) | Load 32 elements from current DEST row into LREG for processing |
| **SFPSTORE** | `dst_reg[0] = y` (write) | Write computed result back to DEST row |
| **SFPLOADI** | `vFloat(scalar)`, `vFloat(0x3f800000)` | Load 16-bit immediate values into LREGs (used for constants: `log2_base`, `-127.0`, `0.40196114e-7`, `0x3f800000`, `infinity`, `NaN`) |
| **SFPMAD** | `x * v_log2_base`, `d1 * d2`, `d2 * d3`, `z_f32 + bias` | Fused multiply-add: used for multiplication (a * b + 0.0) and addition (a * 1.0 + b) |
| **SFPDIVP2** | `sfpi::addexp(z_f32, 23)` | Add to exponent field: effectively multiplies by 2^23 without touching the mantissa |
| **SFPEXEXP** | `sfpi::exexp(...)` | Extract debiased exponent from float, yielding the integer part of the 2^z decomposition |
| **SFPEXMAN** | `sfpi::exman9(...)` | Extract 9-bit mantissa with implicit bit padding, yielding the fractional part for polynomial evaluation |
| **SFPCAST** | `sfpi::int32_to_float(...)` | Convert INT32 to FP32 (used after integer arithmetic on mantissa coefficients) |
| **SFPSETEXP** | `sfpi::setexp(...)` | Set the exponent field of a float (used to reconstruct 2^integer_part * mantissa_result) |
| **SFPSETSGN** | `sfpi::setsgn(y, x_int << 31)` | Set sign bit of result based on parity of exponent (negative base case) |
| **SFPIADD** | `sfpi::vInt(0xf94ee7) + zif`, `127U + zii` | Integer addition: add polynomial coefficients to extracted mantissa; add 127 bias to exponent |
| **SFPSHFT** | `x_int << 31` | Left shift: move LSB (odd/even indicator) to sign position |
| **SFP_STOCH_RND** | `sfpi::float_to_int16(x, 0)`, `sfpi::float_to_fp16b(y, 0)` | Format conversion: FP32 to INT16 (for integer check), FP32 to BF16 (final output rounding) |
| **SFPSETCC** | `v_if(z_f32 < ...)`, `v_if(x > 0.0f)`, etc. | Set per-lane condition codes for conditional execution |
| **SFPENCC** | `v_endif` | Enable/disable condition code masking (reset to all-lanes-active) |
| **SFPCOMPC** | implicit in `v_if`/`v_endif` pairs | Complement CC for else-branch logic (used internally by SFPI CC stack) |
| **SFPPUSHC** | implicit in nested `v_if` blocks | Push CC state for nested conditional blocks |
| **SFPPOPC** | implicit in nested `v_if` blocks | Pop CC state after nested conditional blocks |

**Note**: `_float_to_int32_positive_` is referenced but **never defined** in the codebase. The analogous function in `ckernel_sfpu_exp.h` is `_float_to_int32_for_exp_21f_`, which uses `SFPEXEXP` (extract exponent), `SFPEXMAN` (extract 8-bit mantissa), and `SFPSHFT2` (barrel shift by exponent amount) to convert a positive float to integer. If the rpow kernel intended to use that function, it would additionally emit SFPEXEXP, SFPEXMAN (PAD8 mode), and SFPSHFT2.

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **DEST rows** | Input tile elements are read from DEST via `SFPLOAD` into LREGs, processed, and written back via `SFPSTORE`. The stride-2 addressing means each iteration touches 2 physical DEST rows (32 elements). |
| **LREG0-LREG3** | General purpose registers used by the SFPI compiler for intermediate values (`x`, `z_f32`, `z`, `zii`, `zif`, `d1`, `d2`, `d3`, `y`, etc.). The SFPI compiler allocates these automatically. |
| **LREG4-LREG7** | Potentially used by the compiler for additional intermediates. LREG7 may be used for indirect addressing in `SFPMAD` instructions if the compiler chooses that encoding. |
| **Programmable Constants** | Not used -- `rpow_init()` is empty. The `log2(base)` value is computed at runtime on the RISC-V scalar core and loaded into a vector register via `SFPLOADI`. |
| **Fixed Const 2** | Value 1.0 -- used implicitly by `sfpi::vConst1` in the `base == 0` special case. |

### Address Mode Configuration

The address mode is configured during the `llk_math_eltwise_unary_sfpu_init<SfpuType::rpow, APPROXIMATE>()` call, which invokes `eltwise_unary_sfpu_configure_addrmod<SfpuType::rpow>()`.

**Note**: Because `SfpuType::rpow` is not currently defined in the `SfpuType` enum, this analysis documents what WOULD happen if it were properly added.

| Property | Value (Wormhole B0) | Value (Blackhole) |
|----------|---------------------|-------------------|
| **ADDR_MOD_7** | `{srca.incr=0, srcb.incr=0, dest.incr=0}` | `{srca.incr=0, srcb.incr=0, dest.incr=0}` |
| **Additional ADDR_MODs** | None (rpow would not match any `if constexpr` special cases) | None |

The `ADDR_MOD_7` with `dest.incr=0` means that DEST addressing does NOT auto-increment between SFPU instructions. Instead, the kernel manages DEST addressing manually:
- **Within a face**: `dst_reg++` in the SFPI loop advances the DEST pointer by 1 sfpi row (2 physical rows) per iteration.
- **Between faces**: `TTI_SETRWC` in the params dispatch function advances by 8+8=16 physical rows to jump to the next face.

This is the standard unary SFPU address progression pattern.

## Local Knowledge Sources
### Local References
1. **File**: `tt_metal/hw/inc/api/compute/eltwise_unary/rpow.h`
   **Reason**: API header exposing `rpow_tile_init()` and `rpow_tile(idst, base_val)` -- entry point for the SFPU chain.
   **Key Findings**: Takes two parameters (tile index and base value as IEEE 754 bits). Calls `llk_math_eltwise_unary_sfpu_rpow<APPROX>()`.

2. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_rpow.h`
   **Reason**: LLK dispatch layer bridging API to core SFPU function.
   **Key Findings**: Uses `_llk_math_eltwise_unary_sfpu_params_` with `VectorMode::RC`, `ITERATIONS=8`. Init calls `rpow_init<APPROXIMATE>()`.

3. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_rpow.h`
   **Reason**: Core SFPU kernel implementation -- primary analysis target.
   **Key Findings**: Implements `base^x = 2^(x * log2(base))` using scalar precomputation of log2(base) and the Moroz exp_21f algorithm. Handles special cases (base=0, negative base). References undefined function `_float_to_int32_positive_`.

4. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Parameters dispatch template -- controls per-face iteration and DEST advancement.
   **Key Findings**: For `VectorMode::RC`, loops 4 faces, calling the SFPU function once per face, with `TTI_SETRWC` to advance DEST by 16 rows between faces.

5. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: Init function and addr_mod configuration for all unary SFPU operations.
   **Key Findings**: `ADDR_MOD_7` configured with `{srca.incr=0, srcb.incr=0, dest.incr=0}` for standard unary ops. The rpow SfpuType would not match any special-case branches.

6. **File**: `runtime/sfpi/include/sfpi_lib.h`
   **Reason**: SFPI library providing builtin wrappers for SFPU instructions.
   **Key Findings**: Mapped SFPI functions to hardware instructions: `addexp` -> SFPDIVP2, `exexp` -> SFPEXEXP, `exman9` -> SFPEXMAN, `setexp` -> SFPSETEXP, `setsgn` -> SFPSETSGN, `int32_to_float` -> SFPCAST, `float_to_fp16b` -> SFP_STOCH_RND, `float_to_int16` -> SFP_STOCH_RND.

7. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_exp.h`
   **Reason**: Reference for the exp_21f algorithm and the `_float_to_int32_for_exp_21f_` function.
   **Key Findings**: The upstream exp kernel uses `_float_to_int32_for_exp_21f_` (not `_float_to_int32_positive_`). This function extracts exponent and mantissa, then barrel-shifts the mantissa by the exponent to produce an integer. The rpow kernel appears to have been written referencing a different (non-existent) function name.

8. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Host-side dispatch for approximation mode and SFPU chain defines.
   **Key Findings**: `get_op_approx_mode()` returns `false` for all ops (default case). `get_compute_kernel_path()` returns `eltwise_sfpu.cpp` for all ops (default case). The `RPOW` case is missing from `get_op_init_and_func_default` and `get_op_init_and_func_parameterized`.

9. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp`
   **Reason**: Verified `RPOW` exists in the `UnaryOpType` enum.
   **Key Findings**: `RPOW` is defined at enum value position 128 (sequential).

10. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h`
    **Reason**: Checked whether `SfpuType::rpow` is defined.
    **Key Findings**: The `SfpuType` enum does NOT include `rpow`. Only `hardsigmoid`, `hardtanh`, `hardswish`, `softshrink` are present. This is a build-breaking issue for the rpow LLK dispatch.

11. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_converter.h`
    **Reason**: Provides the `Converter::as_float()` utility used to decode the base parameter.
    **Key Findings**: Simple union-based `uint32_t` to `float` reinterpretation.

12. **File**: `.claude/references/sfpu-hardware-model.md`
    **Reason**: Authoritative reference for SFPU instruction semantics, register layout, and addressing model.
    **Key Findings**: Confirmed stride-2 addressing model, SFPMAD as the universal float add/multiply instruction, SFPIADD as integer-only, and SFP_STOCH_RND for format conversions.
