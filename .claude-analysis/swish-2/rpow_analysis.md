## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### Unary Dispatch Summary
- **UnaryOpType**: `RPOW`
- **Compute kernel**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp` (default `eltwise_sfpu.cpp`)
- **SFPU_OP_CHAIN_0 expansion**: `rpow_tile_init()` / `rpow_tile(0, base_val)` where `base_val` is the IEEE 754 bit-representation of the base parameter passed as `uint32_t`

**Note**: The `rpow` operation is a work-in-progress addition. The `SfpuType` enum in `llk_sfpu_types.h` does not yet contain `rpow`, and the `is_parametrized_type()` switch in `unary_op_utils.hpp` does not include `RPOW`. Additionally, the helper function `_float_to_int32_positive_()` used in the SFPU kernel is not defined anywhere in the current source tree. These are integration gaps that need to be resolved before the operation can compile and run.

#### Approximation Mode Resolution
There are TWO independent controls for approximation. You MUST report both.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(RPOW)` in `unary_op_utils.cpp` — currently returns `false` for all ops (switch has only a `default: return false` case) |
| Template parameter (SFPU_OP_CHAIN) | none (no parameterized template arg for approx mode) | `rpow_tile_init()` calls `llk_math_eltwise_unary_sfpu_rpow_init<APPROX>()` where `APPROX` is the `math_approx_mode` value; `rpow_tile(idst, base_val)` calls `llk_math_eltwise_unary_sfpu_rpow<APPROX>(idst, base_val)` |
| Effective SFPU path | `APPROXIMATION_MODE=false` in `calculate_rpow<false, 8>()`. The kernel does NOT use `APPROXIMATION_MODE` anywhere in its body — there are no `if constexpr (APPROXIMATION_MODE)` branches. The template parameter is declared but unused, meaning the kernel always follows the same code path regardless of approximation mode. | `ckernel_sfpu_rpow.h` line 38: `template <bool APPROXIMATION_MODE, int ITERATIONS = 8>` — parameter is never referenced in the function body |

### SFPU Abstraction Layers
List the file path for each abstraction layer.

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/rpow.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_rpow.h` (WH) / `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_rpow.h` (BH) — both identical |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_rpow.h` (WH) / `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_rpow.h` (BH) — both identical |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (WH) / `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (BH) |

### Call Chain
1. **Compute kernel** (`eltwise_sfpu.cpp`): The `SFPU_OP_CHAIN_0` macro expands to `rpow_tile_init(); rpow_tile(0, base_val);`. The init is called once before the tile loop; `rpow_tile(0, base_val)` is called per-tile.
2. **API Header** (`rpow.h`): `rpow_tile(idst, base_val)` calls `MATH((llk_math_eltwise_unary_sfpu_rpow<APPROX>(idst, base_val)))`, dispatching to the math thread.
3. **LLK Dispatch** (`llk_math_eltwise_unary_sfpu_rpow.h`): `llk_math_eltwise_unary_sfpu_rpow<APPROX>(dst_index, base_val)` calls `_llk_math_eltwise_unary_sfpu_params_<APPROX>(ckernel::sfpu::calculate_rpow<APPROX, 8>, dst_index, VectorMode::RC, base_val)`.
4. **Parameters Dispatch** (`llk_math_eltwise_unary_sfpu_params.h`): Sets the DEST write address, stalls until SFPU is ready, then iterates over 4 faces calling `calculate_rpow<false, 8>(base_val)` once per face with `TTI_SETRWC` between faces to advance the DEST address.
5. **Core SFPU Implementation** (`ckernel_sfpu_rpow.h`): `calculate_rpow<false, 8>(base_val)` decodes the base parameter, precomputes `log2(base)` as a scalar, then loops 8 iterations (one face) computing `base^x = 2^(x * log2(base))` for each sfpi row of 32 elements.

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` (default) — all 4 faces of the tile are processed. The params dispatch calls `calculate_rpow` once per face (4 times total), each call internally loops `ITERATIONS=8` times, processing all 256 elements of the face.
- **Operation invocation**: The params dispatch function `_llk_math_eltwise_unary_sfpu_params_` uses a `for (int face = 0; face < 4; face++)` loop. Each iteration calls `calculate_rpow(base_val)` (forwarding the `base_val` argument), then issues two `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` instructions to advance the DEST write pointer by 16 physical rows (= 1 face) to the next face.
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, `SETRWC` between faces). On both Wormhole and Blackhole, the default address mode `ADDR_MOD_7` is configured with `{srca.incr=0, srcb.incr=0, dest.incr=0}` — the SFPU kernel itself manages DEST addressing via `dst_reg++` (SFPI abstraction auto-increments by stride-2). The `rpow` SfpuType does not match any special-case in `eltwise_unary_sfpu_configure_addrmod`, so only `ADDR_MOD_7` is set.

### Annotated SFPU Kernel Source

The kernel uses **SFPI abstractions** (`vFloat`, `vInt`, `dst_reg`, `v_if`/`v_endif`) — Style A.

```cpp
// File: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_rpow.h

// rpow(x) = base^x  where base is a scalar float parameter
// Algorithm: base^x = 2^(x * log2(base))
// Uses the exp_21f algorithm from Moroz et al. 2022
// "Simple Multiple Precision Algorithms for Exponential Functions"
// Since base is a constant scalar, we precompute log2(base) once
// and then compute 2^(x * log2_base) for each element.

namespace {
inline uint32_t float_to_bits(float f) {
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
    const float base_scalar = Converter::as_float(base_val); // bit_cast uint32_t -> float
    const float abs_base = base_scalar < 0.0f ? -base_scalar : base_scalar;

    // Precompute log2(|base|) as a scalar float (runs on RISC-V, not SFPU)
    // IEEE 754: float = 2^exp * mantissa, so log2(float) = exp + log2(mantissa)
    uint32_t base_bits = float_to_bits(abs_base);
    int32_t base_exp = static_cast<int32_t>(((base_bits >> 23) & 0xFF)) - 127; // extract biased exponent, debias
    // Normalize mantissa to [1,2) by setting exponent to 127
    uint32_t mantissa_bits = (base_bits & 0x007FFFFF) | 0x3F800000;
    float mantissa_norm = Converter::as_float(mantissa_bits);

    // 3rd order polynomial approximation for log2(x) over [1,2)
    const float c3 = 0x2.44734p-4f;       // ≈ 0.14172
    const float c2 = -0xd.e712ap-4f;      // ≈ -0.86904
    const float c1 = 0x2.4f5388p+0f;      // ≈ 2.30964
    const float c0 = -0x1.952992p+0f;     // ≈ -1.58237
    const float inv_ln2 = 1.4426950408889634f; // 1/ln(2)

    float series = c0 + mantissa_norm * (c1 + mantissa_norm * (c2 + mantissa_norm * c3)); // Horner evaluation
    float log2_base = static_cast<float>(base_exp) + series * inv_ln2;

    // Load precomputed log2(base) into a vector register → SFPLOADI
    const sfpi::vFloat v_log2_base = log2_base;
    const sfpi::vFloat v_low_threshold = -127.0f;

#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) { // 8 iterations per face
        sfpi::vFloat x = sfpi::dst_reg[0]; // SFPLOAD: load 32 elements from current DEST row

        // z = x * log2(base) → SFPMAD
        sfpi::vFloat z_f32 = x * v_log2_base;

        // Clamp to prevent overflow: if z < -127, set to -127
        v_if(z_f32 < v_low_threshold) { z_f32 = v_low_threshold; } // CC: SFPENCC + SFPSETCC + guarded SFPLOADI + SFPCOMPC/SFPENCC
        v_endif;

        // Compute 2^z using exp_21f algorithm (Moroz et al. 2022, Section 5)
        z_f32 = sfpi::addexp(z_f32, 23);  // SFPDIVP2 with ADD mode: multiply by 2^23
        const sfpi::vFloat bias = sfpi::vFloat(0x3f800000); // SFPLOADI: IEEE 754 bits for 1.0f
        sfpi::vInt z = _float_to_int32_positive_(z_f32 + bias); // SFPMAD (add) + [undefined fn: float→int32 conversion]

        sfpi::vInt zii = sfpi::exexp(sfpi::reinterpret<sfpi::vFloat>(z));   // SFPEXEXP: extract debiased exponent
        sfpi::vInt zif = sfpi::exman9(sfpi::reinterpret<sfpi::vFloat>(z));  // SFPEXMAN: extract 9-bit mantissa (padded)

        // Compute 2^frac(z) using Horner form polynomial
        sfpi::vFloat d1 = sfpi::vFloat(0.40196114e-7); // SFPLOADI: small coefficient
        sfpi::vFloat d2 = sfpi::int32_to_float(sfpi::vInt(0xf94ee7) + zif, 0); // SFPIADD + SFPCAST(INT32→FP32, RNE)
        sfpi::vFloat d3 = sfpi::int32_to_float(sfpi::vInt(0x560e) + zif, 0);   // SFPIADD + SFPCAST(INT32→FP32, RNE)

        d2 = d1 * d2; // SFPMAD
        zif = _float_to_int32_positive_(d2 * d3); // SFPMAD + [undefined fn]

        // Restore exponent: result = mantissa * 2^exponent
        zii = sfpi::reinterpret<sfpi::vInt>(sfpi::setexp(sfpi::reinterpret<sfpi::vFloat>(zif), 127U + zii)); // SFPSETEXP

        sfpi::vFloat y = sfpi::reinterpret<sfpi::vFloat>(zii);

        // Handle special cases (compile-time branches on scalar base value)
        if (abs_base == 0.0f) {
            // base == 0: 0^x = 0 for x > 0, 1 for x == 0, inf for x < 0
            v_if(x > 0.0f) { y = 0.0f; }      // CC guarded
            v_endif;
            v_if(x == 0.0f) { y = sfpi::vConst1; } // vConst1 = 1.0f (fixed constant register)
            v_endif;
            v_if(x < 0.0f) {
                y = sfpi::vFloat(std::numeric_limits<float>::infinity()); // SFPLOADI: +inf
            }
            v_endif;
        } else if (base_scalar < 0.0f) {
            // Negative base: result is real only for integer exponents
            sfpi::vInt x_int = sfpi::float_to_int16(x, 0); // SFP_STOCH_RND: FP32→INT16 with RNE rounding
            sfpi::vFloat x_rounded = sfpi::int32_to_float(x_int, 0); // SFPCAST: INT32→FP32 with RNE

            // If x is odd integer, negate the result
            y = sfpi::setsgn(y, x_int << 31); // SFPSHFT (shift left 31) + SFPSETSGN: copy bit 31 (odd→sign) to y's sign

            // If x is not an integer, set result to NaN
            v_if(x_rounded != x) { y = sfpi::vFloat(std::numeric_limits<float>::quiet_NaN()); } // SFPLOADI + CC guarded
            v_endif;
        }

        // Convert to bfloat16 with round-to-nearest-even for accuracy
        y = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(y, 0)); // SFP_STOCH_RND: FP32→BF16 with RNE

        sfpi::dst_reg[0] = y; // SFPSTORE: write 32 elements back to current DEST row
        sfpi::dst_reg++;      // advance to next sfpi row (2 physical DEST rows)
    }
}

template <bool APPROXIMATION_MODE>
inline void rpow_init() {
    // No programmable constants needed - log2(base) is computed from the parameter
}
```

### SFPU Instructions Used

| Instruction | SFPI Abstraction | Usage in rpow |
|-------------|-----------------|---------------|
| **SFPLOAD** | `dst_reg[0]` (read) | Load 32 elements from current DEST row into LREG for processing |
| **SFPSTORE** | `dst_reg[0] =` (write) | Store 32 processed elements back to current DEST row |
| **SFPLOADI** | `vFloat(constant)`, `vFloat v_log2_base = scalar` | Load immediate float constants: `log2(base)`, `-127.0f`, `0x3f800000` (bias), `0.40196114e-7`, `0.0f`, `1.0f`, `+inf`, `NaN` |
| **SFPMAD** | `*` (multiply), `+` (float add) | Multiply `x * v_log2_base`, float additions `z_f32 + bias`, multiplications `d1 * d2`, `d2 * d3` |
| **SFPDIVP2** | `addexp(z_f32, 23)` | Add 23 to the exponent field of z_f32, effectively multiplying by 2^23 |
| **SFPEXEXP** | `exexp(z)` | Extract the debiased exponent field from the reinterpreted float value of z |
| **SFPEXMAN** | `exman9(z)` | Extract the 9-bit mantissa (with padding) from the reinterpreted float value of z |
| **SFPSETEXP** | `setexp(zif, 127U + zii)` | Set the exponent field of the mantissa result to reconstruct the final 2^z value |
| **SFPSETSGN** | `setsgn(y, x_int << 31)` | Copy the LSB of x_int (shifted to bit 31) as the sign of y — negates result for odd integer exponents with negative base |
| **SFPCAST** | `int32_to_float(vInt, 0)` | Convert integer values to FP32 with round-to-nearest-even (used 3 times: for d2, d3, and x_rounded) |
| **SFP_STOCH_RND** | `float_to_fp16b(y, 0)`, `float_to_int16(x, 0)` | Format conversion: FP32→BF16 with RNE (output), FP32→INT16 with RNE (negative base integer check) |
| **SFPIADD** | `vInt(constant) + zif` | Integer addition of mantissa fragment offsets (`0xf94ee7 + zif`, `0x560e + zif`) used in the Horner polynomial evaluation |
| **SFPSHFT** | `x_int << 31` | Logical left shift of x_int by 31 bits — isolates the LSB (odd/even) into the sign bit position |
| **SFPSETCC/SFPENCC/SFPCOMPC/SFPPUSHC/SFPPOPC** | `v_if`/`v_endif` | Predicated execution for: overflow clamping (`z < -127`), special-case handling for base=0 (`x > 0`, `x == 0`, `x < 0`), and non-integer exponent NaN check (`x_rounded != x`) |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **DEST rows** | Source and destination for tile data. `dst_reg[0]` reads from the current DEST address (2 physical rows = 32 elements); `dst_reg[0] =` writes back. `dst_reg++` advances by 1 sfpi row (stride-2). |
| **LREGs (general purpose)** | Used implicitly by SFPI abstractions. The compiler allocates LREGs for intermediate values: `x`, `z_f32`, `z`, `zii`, `zif`, `d1`, `d2`, `d3`, `y`, `bias`, `v_log2_base`, `v_low_threshold`, `x_int`, `x_rounded`. The kernel uses many simultaneous live values, which may require LREG spills. |
| **LREG0-LREG7** | Mapped by the SFPI compiler backend. No explicit LREG usage in the SFPI source code — all register allocation is automatic. |
| **Programmable Constants** | Not used. The `rpow_init()` function is empty — no programmable constant registers are configured. The `log2(base)` value is computed from the `base_val` parameter at the start of `calculate_rpow` using RISC-V scalar math, then loaded into a vector register via `vFloat v_log2_base = log2_base`. |
| **Fixed Constants** | `sfpi::vConst1` (Fixed Const 2 = 1.0f) is used for the `base==0, x==0` case. |

### Address Mode Configuration

The `rpow` operation uses the default unary SFPU address mode configuration. Since `SfpuType::rpow` does not match any `if constexpr` special case in `eltwise_unary_sfpu_configure_addrmod`, only `ADDR_MOD_7` is configured.

**Wormhole B0 and Blackhole** (identical for rpow):
```
ADDR_MOD_7: { srca.incr = 0, srcb.incr = 0, dest.incr = 0 }
```

This is the standard configuration for most unary SFPU operations. The DEST address is not auto-incremented by the hardware address mode — instead, the SFPI `dst_reg++` abstraction handles per-iteration DEST advancement (stride-2, i.e., 2 physical rows per increment), and the params dispatch issues `TTI_SETRWC` instructions between faces to advance by 16 physical rows.

No additional `ADDR_MOD_6` is configured for rpow (unlike `typecast`, `topk_local_sort`, or `unary_max`/`unary_min` ops which set `ADDR_MOD_6` with special dest increments).

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Determine `get_op_approx_mode()`, `get_compute_kernel_path()`, and `get_block_defines()` behavior for RPOW
   **Key Findings**: `get_op_approx_mode` returns `false` for all ops (default case); `get_compute_kernel_path` returns `eltwise_sfpu.cpp` (default); RPOW is not in `is_parametrized_type()` or `get_op_init_and_func_parameterized()`

2. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp`
   **Reason**: Check `is_parametrized_type()` for RPOW
   **Key Findings**: RPOW is NOT listed in `is_parametrized_type()` — only HARDTANH and SOFTSHRINK are

3. **File**: `tt_metal/hw/inc/api/compute/eltwise_unary/rpow.h`
   **Reason**: API header defining `rpow_tile_init()` and `rpow_tile(idst, base_val)`
   **Key Findings**: Calls `llk_math_eltwise_unary_sfpu_rpow_init<APPROX>()` and `llk_math_eltwise_unary_sfpu_rpow<APPROX>(idst, base_val)`

4. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_rpow.h`
   **Reason**: LLK dispatch layer for rpow
   **Key Findings**: Init calls `llk_math_eltwise_unary_sfpu_init<SfpuType::rpow, APPROXIMATE>(rpow_init<APPROXIMATE>)`. Compute calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(calculate_rpow<APPROXIMATE, ITERATIONS>, dst_index, VectorMode::RC, base_val)`.

5. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_rpow.h`
   **Reason**: Core SFPU kernel implementation
   **Key Findings**: Implements `base^x = 2^(x * log2(base))` using the exp_21f algorithm. Precomputes log2(base) on RISC-V scalar math, then computes 2^z per-element using SFPU vector instructions. Handles special cases (base=0, negative base) with compile-time scalar branches and runtime CC-guarded per-lane operations.

6. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Parameters dispatch function `_llk_math_eltwise_unary_sfpu_params_`
   **Key Findings**: Sets DEST write address, stalls for SFPU, then for VectorMode::RC iterates 4 faces calling the sfpu_func with forwarded args, issuing 2x `TTI_SETRWC(CR_D, 8)` between faces

7. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: SFPU init and address mode configuration
   **Key Findings**: `eltwise_unary_sfpu_configure_addrmod<SfpuType::rpow>()` sets only `ADDR_MOD_7` with all-zero increments (rpow doesn't match any special-case constexpr branch)

8. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h`
   **Reason**: Verify SfpuType enum
   **Key Findings**: `SfpuType` does NOT contain `rpow` — only `unused`, `hardsigmoid`, `hardtanh`, `hardswish`, `softshrink`. This is an integration gap.

9. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_converter.h`
   **Reason**: `Converter::as_float()` utility used in rpow kernel
   **Key Findings**: Simple union-based uint32_t → float reinterpret

10. **File**: `runtime/sfpi/include/sfpi_lib.h`
    **Reason**: SFPI intrinsic function definitions (addexp, exexp, exman9, setexp, setsgn, int32_to_float, float_to_fp16b, float_to_int16)
    **Key Findings**: `addexp` → `SFPDIVP2` with ADD mode; `exexp` → `SFPEXEXP` with debias; `exman9` → `SFPEXMAN` with 9-bit pad; `setexp` → `SFPSETEXP`; `setsgn` → `SFPSETSGN`; `int32_to_float` → `SFPCAST`; `float_to_fp16b` → `SFP_STOCH_RND`; `float_to_int16` → `SFP_STOCH_RND`

11. **File**: `.claude/references/sfpu-hardware-model.md`
    **Reason**: SFPU hardware architecture reference
    **Key Findings**: Confirmed stride-2 addressing model, ITERATIONS=8 per face, tile geometry, instruction semantics
