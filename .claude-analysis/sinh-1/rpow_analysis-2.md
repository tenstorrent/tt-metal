## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### Unary Dispatch Summary
- **UnaryOpType**: `RPOW`
- **Compute kernel**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`
- **SFPU_OP_CHAIN_0 expansion**: `rpow_tile(0, <base_val_as_uint32>u)`

#### Approximation Mode Resolution
There are TWO independent controls for approximation. Both are reported below.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(RPOW)` in `unary_op_utils.cpp` -- falls through to `default: return false` |
| Template parameter (SFPU_OP_CHAIN) | none (uses default `APPROX` from compute config) | `get_op_init_and_func_parameterized()` -- returns `rpow_tile_init()` / `rpow_tile(idst, base_val)` with no explicit template parameter; `rpow_tile_init<APPROX>()` and `rpow_tile<APPROX>(idst, base_val)` use `APPROX` which resolves from `math_approx_mode` = `false` |
| Effective SFPU path | `APPROXIMATION_MODE=false` in `calculate_rpow` | The kernel does not branch on `APPROXIMATION_MODE` -- the template parameter is accepted but unused in the current implementation |

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/rpow.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_rpow.h` (identical for both architectures) |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_rpow.h` (identical for both architectures) |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |

### Call Chain
1. **Compute kernel** (`eltwise_sfpu.cpp`): The `SFPU_OP_CHAIN_0` macro expands to `rpow_tile_init(); rpow_tile(0, <base_val>u);`, where `rpow_tile_init()` is called once during init and `rpow_tile(0, base_val)` is called per tile.
2. **API Header** (`rpow.h`): `rpow_tile(idst, base_val)` calls `MATH((llk_math_eltwise_unary_sfpu_rpow<APPROX>(idst, base_val)))`, which is only compiled on the TRISC_MATH processor.
3. **LLK Dispatch** (`llk_math_eltwise_unary_sfpu_rpow.h`): `llk_math_eltwise_unary_sfpu_rpow<APPROX>(dst_index, base_val)` calls `_llk_math_eltwise_unary_sfpu_params_<APPROX>(ckernel::sfpu::calculate_rpow<APPROX, 8>, dst_index, VectorMode::RC, base_val)`.
4. **Parameters Dispatch** (`llk_math_eltwise_unary_sfpu_params.h`): `_llk_math_eltwise_unary_sfpu_params_` sets up DEST addressing, stalls for SFPU readiness, then iterates over 4 faces calling `calculate_rpow(base_val)` per face, with `TTI_SETRWC` advancing the DEST address between faces.
5. **Core SFPU Implementation** (`ckernel_sfpu_rpow.h`): `calculate_rpow<false, 8>(base_val)` decodes the base parameter, precomputes `log2(|base|)` as a scalar, then loops 8 iterations per face computing `base^x = 2^(x * log2(base))` for each DEST row pair.

### Parameters Dispatch Summary
- **Vector mode**: `VectorMode::RC` (all 4 faces are processed). This processes all 1024 elements of the tile.
- **Operation invocation**: The parameters dispatch calls `calculate_rpow(base_val)` once per face (4 times total for RC mode). Each call runs the inner loop of ITERATIONS=8, processing 8 sfpi rows (= 256 elements per face).
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, `SETRWC` between faces). The address mode is `ADDR_MOD_7` on both Wormhole and Blackhole, configured with all-zero increments (`srca.incr=0, srcb.incr=0, dest.incr=0`). The `rpow` operation type does not match any special-case branches in `eltwise_unary_sfpu_configure_addrmod`, so only the default `ADDR_MOD_7` is set.

### Annotated SFPU Kernel Source

The kernel uses SFPI abstractions (`sfpi::vFloat`, `sfpi::vInt`, `dst_reg`, `v_if`/`v_endif`), so Style A (inline-commented source) is used.

**CRITICAL BUILD ISSUE**: The function `_float_to_int32_positive_` is called on lines 85 and 96 but is **never defined** in any header file in the current codebase. This means the rpow kernel **will not compile**. The function appears to be modeled after `_float_to_int32_for_exp_21f_` from `ckernel_sfpu_exp.h` (which converts a positive float to int32 using exponent extraction and mantissa shifting), but the rpow-specific version was never implemented. See the "SFPU Instructions Used" section for the expected instruction mapping if this function were to be defined similarly to `_float_to_int32_for_exp_21f_`.

```cpp
// File: tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_rpow.h

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
inline void calculate_rpow(const uint32_t base_val) { // APPROXIMATION_MODE=false, ITERATIONS=8
    // Decode base parameter from IEEE 754 bits
    const float base_scalar = Converter::as_float(base_val); // scalar host-side decode, no SFPU instruction
    const float abs_base = base_scalar < 0.0f ? -base_scalar : base_scalar;

    // Implementation notes, see the original file for more details
    uint32_t base_bits = float_to_bits(abs_base);
    int32_t base_exp = static_cast<int32_t>(((base_bits >> 23) & 0xFF)) - 127; // scalar host-side computation
    uint32_t mantissa_bits = (base_bits & 0x007FFFFF) | 0x3F800000; // normalize mantissa to [1,2)
    float mantissa_norm = Converter::as_float(mantissa_bits);

    // 3rd order polynomial approximation for log2(x) over [1,2)
    const float c3 = 0x2.44734p-4f;   // ~0.1418
    const float c2 = -0xd.e712ap-4f;  // ~-0.8686
    const float c1 = 0x2.4f5388p+0f;  // ~2.3103
    const float c0 = -0x1.952992p+0f; // ~-1.5828
    const float inv_ln2 = 1.4426950408889634f;

    float series = c0 + mantissa_norm * (c1 + mantissa_norm * (c2 + mantissa_norm * c3)); // scalar Horner eval
    float log2_base = static_cast<float>(base_exp) + series * inv_ln2; // scalar log2(|base|)

    const sfpi::vFloat v_log2_base = log2_base;   // SFPLOADI: broadcast scalar into vector register
    const sfpi::vFloat v_low_threshold = -127.0f;  // SFPLOADI: load clamp threshold

#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0]; // SFPLOAD: read 32 elements from DEST

        sfpi::vFloat z_f32 = x * v_log2_base; // SFPMAD: z = x * log2(base) + 0.0

        // Clamp to prevent underflow/overflow
        v_if(z_f32 < v_low_threshold) { z_f32 = v_low_threshold; } // SFPSETCC(LT0) + CC-guarded SFPMOV
        v_endif; // SFPENCC to restore all-enabled

        // Compute 2^z using exp_21f algorithm (Moroz et al. 2022)
        z_f32 = sfpi::addexp(z_f32, 23);  // SFPDIVP2: multiply by 2^23 (add 23 to exponent)
        const sfpi::vFloat bias = sfpi::vFloat(0x3f800000); // SFPLOADI: IEEE 754 encoding of 1.0 used as integer bias
        sfpi::vInt z = _float_to_int32_positive_(z_f32 + bias); // [UNDEFINED FUNCTION] SFPMAD(add) then float-to-int conversion

        sfpi::vInt zii = sfpi::exexp(sfpi::reinterpret<sfpi::vFloat>(z));   // SFPEXEXP: extract debiased exponent (integer part)
        sfpi::vInt zif = sfpi::exman9(sfpi::reinterpret<sfpi::vFloat>(z));  // SFPEXMAN: extract 9-bit mantissa (fractional part)

        // Compute 2^frac(z) using Horner-form polynomial with integer arithmetic
        sfpi::vFloat d1 = sfpi::vFloat(0.40196114e-7); // SFPLOADI: polynomial coefficient
        sfpi::vFloat d2 = sfpi::int32_to_float(sfpi::vInt(0xf94ee7) + zif, 0); // SFPIADD + SFPCAST: integer add then convert to float (RNE)
        sfpi::vFloat d3 = sfpi::int32_to_float(sfpi::vInt(0x560e) + zif, 0);   // SFPIADD + SFPCAST: integer add then convert to float (RNE)

        d2 = d1 * d2; // SFPMAD: multiply polynomial terms
        zif = _float_to_int32_positive_(d2 * d3); // [UNDEFINED FUNCTION] SFPMAD(mul) then float-to-int conversion

        // Restore exponent: result = mantissa * 2^exponent
        zii = sfpi::reinterpret<sfpi::vInt>(sfpi::setexp(sfpi::reinterpret<sfpi::vFloat>(zif), 127U + zii)); // SFPIADD(127+zii) + SFPSETEXP

        sfpi::vFloat y = sfpi::reinterpret<sfpi::vFloat>(zii); // reinterpret, no instruction emitted

        // Handle special cases -- these are compile-time branches on the scalar base value
        if (abs_base == 0.0f) {
            // base == 0: 0^x = 0 for x > 0, 1 for x == 0, inf for x < 0
            v_if(x > 0.0f) { y = 0.0f; }    // SFPSETCC(GTE0) + CC-guarded SFPLOADI/SFPMOV
            v_endif;
            v_if(x == 0.0f) { y = sfpi::vConst1; } // SFPSETCC(EQ0) + CC-guarded SFPMOV
            v_endif;
            v_if(x < 0.0f) {
                y = sfpi::vFloat(std::numeric_limits<float>::infinity()); // SFPSETCC(LT0) + CC-guarded SFPLOADI
            }
            v_endif;
        } else if (base_scalar < 0.0f) {
            // Negative base: result is real only for integer exponents
            sfpi::vInt x_int = sfpi::float_to_int16(x, 0); // SFP_STOCH_RND: convert float to int16 (nearest-even)
            sfpi::vFloat x_rounded = sfpi::int32_to_float(x_int, 0); // SFPCAST: convert back to float for comparison

            y = sfpi::setsgn(y, x_int << 31); // SFPSHFT(left 31) + SFPSETSGN: set sign from parity of integer exponent

            v_if(x_rounded != x) { y = sfpi::vFloat(std::numeric_limits<float>::quiet_NaN()); } // SFPSETCC(NE0) on (x_rounded - x) + CC-guarded SFPLOADI
            v_endif;
        }

        // Convert to bfloat16 with round-to-nearest-even for accuracy
        y = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(y, 0)); // SFP_STOCH_RND: round FP32 to FP16_B (nearest-even)

        sfpi::dst_reg[0] = y; // SFPSTORE: write 32 elements back to DEST
        sfpi::dst_reg++;       // advance DEST pointer by 1 sfpi row (= 2 physical rows = 32 elements)
    }
}

template <bool APPROXIMATION_MODE>
inline void rpow_init() {
    // No programmable constants needed - log2(base) is computed from the parameter
}
```

### SFPU Instructions Used

| Instruction | SFPI Abstraction | Description | Usage in rpow |
|-------------|-----------------|-------------|---------------|
| `SFPLOAD` | `dst_reg[0]` (read) | Load 32 elements from DEST row pair into LREG | Load input element `x` at start of each iteration |
| `SFPSTORE` | `dst_reg[0] = y` (write) | Store LREG value back to DEST row pair | Write computed result `y` at end of each iteration |
| `SFPLOADI` | `vFloat(constant)`, scalar-to-vector broadcast | Load 16-bit immediate into LREG | Load polynomial coefficients, bias constant, threshold, infinity, NaN, and zero constants |
| `SFPMAD` | `vFloat * vFloat`, `vFloat + vFloat` | Fused multiply-add: `VD = VA * VB + VC` | Core arithmetic: `x * v_log2_base`, `z_f32 + bias`, `d1 * d2`, `d2 * d3`, and all vFloat additions |
| `SFPDIVP2` | `sfpi::addexp(v, 23)` | Add immediate to exponent field (multiply by power of 2) | Multiply `z_f32` by `2^23` to prepare for integer conversion |
| `SFPEXEXP` | `sfpi::exexp(v)` | Extract and debias exponent field to signed integer | Extract integer part of `z` for the `2^z_i` computation |
| `SFPEXMAN` | `sfpi::exman9(v)` | Extract mantissa with 9-bit padding | Extract fractional part of `z` for the `2^z_f` polynomial |
| `SFPSETEXP` | `sfpi::setexp(v, exp)` | Set exponent field of float from integer | Reassemble the final result by setting the exponent from the integer part |
| `SFPSETSGN` | `sfpi::setsgn(y, x_int << 31)` | Set sign bit of float from integer value | Apply sign correction for negative base with odd integer exponent |
| `SFPCAST` | `sfpi::int32_to_float(v, 0)` | Convert integer to float (round-to-nearest-even) | Convert integer mantissa offsets to float for polynomial evaluation; convert `x_int` back to float for comparison |
| `SFP_STOCH_RND` | `sfpi::float_to_fp16b(v, 0)`, `sfpi::float_to_int16(v, 0)` | Stochastic/nearest-even rounding for format conversion | Convert result to BF16 format; convert exponent `x` to int16 for parity check |
| `SFPIADD` | `vInt(imm) + vInt` | Integer addition | Add constant offsets to mantissa (`0xf94ee7 + zif`, `0x560e + zif`) and add bias to exponent (`127 + zii`) |
| `SFPSHFT` | `x_int << 31` | Bit shift left | Shift LSB of integer exponent to sign position for sign correction |
| `SFPSETCC` | `v_if(cond)` | Set per-lane condition code based on comparison | Clamp guard (`z_f32 < -127`), special case handling (`x > 0`, `x == 0`, `x < 0`, `x_rounded != x`) |
| `SFPENCC` | `v_endif` (partial), `v_if` (partial) | Enable/disable condition code masking | Begin and end conditional blocks |
| `SFPCOMPC` | `v_endif` (partial) | Complement condition code for else-branch | Part of v_if/v_endif expansion |
| `SFPPUSHC` | `v_if` (partial) | Push CC state onto stack | Part of v_if expansion for nested conditionals |
| `SFPPOPC` | `v_endif` (partial) | Pop CC state from stack | Part of v_endif expansion for nested conditionals |

**Note on `_float_to_int32_positive_`**: This function is called twice (lines 85 and 96) but is **undefined** in the current codebase. If it were implemented similarly to the analogous `_float_to_int32_for_exp_21f_` in `ckernel_sfpu_exp.h`, it would emit: `SFPEXEXP` (extract exponent), `SFPEXMAN` (extract mantissa with implicit bit), and `SFPSHFT` (shift mantissa by exponent amount). This is conjectural based on the pattern in the reference implementation.

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **LREG0-3** | General-purpose registers used by the SFPI compiler for intermediate values (`x`, `z_f32`, `z`, `zii`, `zif`, `d1`, `d2`, `d3`, `y`, `x_int`, `x_rounded`). The SFPI compiler manages register allocation automatically. |
| **DEST rows** | Input tile data (32x32 = 1024 elements). Each iteration reads and writes a pair of physical DEST rows (32 elements) via `dst_reg[0]`. |
| **Programmable constants** | Not used -- `rpow_init()` is empty. The `log2(base)` value is computed at runtime from the parameter and loaded via `SFPLOADI`. |

**Scalar vs Vector computation**: A significant portion of the kernel (lines 41-63) runs as scalar host-side C++ computation to precompute `log2(|base|)`. This scalar work (IEEE 754 bit extraction, polynomial evaluation for log2) runs on the RISC-V processor once per face invocation, NOT on the SFPU. The SFPU vector computation begins at the `for (int d = 0; d < ITERATIONS; d++)` loop.

### Address Mode Configuration

The address mode for rpow is configured by `eltwise_unary_sfpu_configure_addrmod<SfpuType::rpow>()` in `llk_math_eltwise_unary_sfpu.h`. Since `SfpuType::rpow` does not match any special-case `if constexpr` branches, only the default `ADDR_MOD_7` is configured:

**Both Wormhole and Blackhole** (identical):
```
ADDR_MOD_7:
  .srca = {.incr = 0}
  .srcb = {.incr = 0}
  .dest = {.incr = 0}
```

This all-zero configuration means the hardware auto-increment mechanism does not advance the DEST address between SFPU instructions. Instead, the DEST address progression is handled by:
1. **Within a face**: `dst_reg++` in the SFPI loop (SFPU compiler emits explicit address increments)
2. **Between faces**: `TTI_SETRWC` instructions emitted by `_llk_math_eltwise_unary_sfpu_params_` (advances DEST pointer by 8 sfpi rows = 16 physical DEST rows per SETRWC pair)

## Local Knowledge Sources
### Local References
1. **File**: `tt_metal/hw/inc/api/compute/eltwise_unary/rpow.h`
   **Reason**: API header exposing `rpow_tile_init()` and `rpow_tile()` to the compute kernel
   **Key Findings**: Calls `llk_math_eltwise_unary_sfpu_rpow<APPROX>()` with `APPROX` from compute config

2. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_rpow.h`
   **Reason**: LLK dispatch layer bridging API to core SFPU implementation
   **Key Findings**: Uses `_llk_math_eltwise_unary_sfpu_params_` with `VectorMode::RC`, passes `base_val` as runtime argument

3. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_rpow.h`
   **Reason**: Core SFPU kernel implementation -- primary target of this analysis
   **Key Findings**: Implements `base^x = 2^(x * log2(base))` using exp_21f algorithm. Precomputes `log2(base)` on scalar RISC-V. **Critical issue**: references undefined function `_float_to_int32_positive_` -- kernel will not compile.

4. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Parameters dispatch layer -- controls face iteration and DEST address management
   **Key Findings**: For VectorMode::RC, iterates 4 faces with `TTI_SETRWC` advancing DEST between faces

5. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: Address mode configuration via `eltwise_unary_sfpu_configure_addrmod`
   **Key Findings**: `SfpuType::rpow` takes default path, only `ADDR_MOD_7` set with all-zero increments

6. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Determine SFPU_OP_CHAIN_0 expansion, approx mode, and compute kernel path
   **Key Findings**: `get_op_approx_mode()` returns false (default case); `get_op_init_and_func_parameterized()` returns `rpow_tile_init()` / `rpow_tile(idst, base_val_hex_u)`; compute kernel is `eltwise_sfpu.cpp`

7. **File**: `runtime/sfpi/include/sfpi_lib.h`
   **Reason**: Understand SFPI intrinsic-to-hardware-instruction mappings
   **Key Findings**: `addexp` -> `SFPDIVP2`, `exexp` -> `SFPEXEXP`, `exman9` -> `SFPEXMAN`, `setexp` -> `SFPSETEXP`, `setsgn` -> `SFPSETSGN`, `int32_to_float` -> `SFPCAST`, `float_to_fp16b` -> `SFP_STOCH_RND`, `float_to_int16` -> `SFP_STOCH_RND`

8. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_exp.h`
   **Reason**: Reference implementation of `_float_to_int32_for_exp_21f_` which is the likely model for the undefined `_float_to_int32_positive_`
   **Key Findings**: `_float_to_int32_for_exp_21f_` uses `exexp` + `exman8` + `shft` to convert positive float to int32

9. **File**: `.claude/references/sfpu-hardware-model.md`
   **Reason**: Authoritative reference for SFPU tile geometry, addressing modes, instruction semantics
   **Key Findings**: Confirmed stride-2 model, ITERATIONS=8 per face, 32 elements per sfpi row, instruction latencies and mappings
