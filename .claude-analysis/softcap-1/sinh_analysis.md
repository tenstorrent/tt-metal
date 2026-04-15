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
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(UnaryOpType::SINH)` in `unary_op_utils.cpp` -- falls through to `default: return false` |
| Template parameter (SFPU_OP_CHAIN) | none (uses default template args) | `get_op_init_and_func_default()` returns `{"sinh_tile_init();", "sinh_tile({idst});"}` -- no parameterized version exists for SINH |
| Effective SFPU path | `APPROXIMATION_MODE=false` throughout `calculate_sinh` and `exp_21f` | `APPROX` is JIT-generated as `constexpr bool APPROX = false` from `hlk_math_approx_mode`, which comes from `get_op_approx_mode()`. `sinh_tile()` passes `<APPROX>` to `llk_math_eltwise_unary_sfpu_sinh<APPROX>()`. Since the kernel currently has no `if constexpr (APPROXIMATION_MODE)` branches, the value does not affect execution. |

### SFPU Abstraction Layers
| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/sinh.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_sinh.h` |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_sinh.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |

### Call Chain
1. **`sinh_tile(idst)`** (API header `sinh.h`): Calls `MATH((llk_math_eltwise_unary_sfpu_sinh<APPROX>(idst)))`.
2. **`llk_math_eltwise_unary_sfpu_sinh<APPROXIMATE>(dst_index, vector_mode=RC)`** (LLK dispatch `llk_math_eltwise_unary_sfpu_sinh.h`): Calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::calculate_sinh<APPROXIMATE, 8>, dst_index, VectorMode::RC)`.
3. **`_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(...)`** (Parameters dispatch `llk_math_eltwise_unary_sfpu_params.h`): Sets up DEST addressing, stalls for SFPU availability, loops over 4 faces calling `calculate_sinh<false, 8>()` per face with `SETRWC`/`inc_dst_addr` between faces.
4. **`calculate_sinh<false, 8>()`** (Core SFPU `ckernel_sfpu_sinh.h`): Executes 8 SFPU iterations per face, computing `sinh(x) = (exp(x) - exp(-x)) / 2` with a Taylor fallback for small `|x|`.

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` (default). All 4 faces of the tile are processed (faces 0-3), covering the full 32x32 = 1024-element tile.
- **Operation invocation**: The core function `calculate_sinh<false, 8>()` is called once per face (4 times total). Each invocation executes an internal loop of `ITERATIONS=8`, processing 8 sfpi rows per face.
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, `SETRWC` between faces on Wormhole / `inc_dst_addr<8>` twice between faces on Blackhole).

### Annotated SFPU Kernel Source

The kernel uses SFPI abstractions (`sfpi::vFloat`, `sfpi::vInt`, `sfpi::dst_reg`, `v_if`/`v_endif`), so Style A (inline-commented source) is used.

```cpp
// File: tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_sinh.h

// Helper: compute 2^z using exp_21f algorithm (Moroz et al. 2022)
// Input z must be clamped to avoid overflow/underflow before calling.
// Returns 2^z as a vFloat.
template <bool APPROXIMATION_MODE>
inline sfpi::vFloat exp_21f(sfpi::vFloat z) { // APPROXIMATION_MODE=false (not currently branched on)
    // Step 1: Scale by 2^23 to shift fractional bits into integer position
    z = sfpi::addexp(z, 23); // SFPDIVP2: adds 23 to exponent field, effectively z *= 2^23

    // Step 2: Add IEEE 754 bias (0x3F800000 = 1.0f) and convert to int
    const sfpi::vFloat bias = sfpi::vFloat(0x3f800000); // SFPLOADI: load 1.0f as raw hex
    sfpi::vInt z_int = _float_to_int32_positive_(z + bias); // SFPMAD (add) then float-to-int conversion

    // Step 3: Decompose into exponent and mantissa parts
    sfpi::vInt exp_part = sfpi::exexp(sfpi::reinterpret<sfpi::vFloat>(z_int)); // SFPEXEXP: extract debiased exponent
    sfpi::vInt man_part = sfpi::exman9(sfpi::reinterpret<sfpi::vFloat>(z_int)); // SFPEXMAN: extract 9-bit mantissa

    // Step 4: Polynomial refinement for 2^frac(z)
    sfpi::vFloat d1 = sfpi::vFloat(0.40196114e-7f); // SFPLOADI
    sfpi::vFloat d2 = sfpi::int32_to_float(sfpi::vInt(0xf94ee7) + man_part, 0); // SFPIADD + SFPCAST
    sfpi::vFloat d3 = sfpi::int32_to_float(sfpi::vInt(0x560e) + man_part, 0);   // SFPIADD + SFPCAST

    d2 = d1 * d2; // SFPMAD
    sfpi::vInt frac_int = _float_to_int32_positive_(d2 * d3); // SFPMAD then float-to-int conversion

    // Step 5: Reconstruct result = mantissa_frac * 2^exponent
    sfpi::vInt result_int =
        sfpi::reinterpret<sfpi::vInt>(sfpi::setexp(sfpi::reinterpret<sfpi::vFloat>(frac_int), 127U + exp_part));
        // SFPIADD (127+exp_part) then SFPSETEXP: set exponent field to 127+exp_part

    return sfpi::reinterpret<sfpi::vFloat>(result_int);
}

// sinh(x) = (exp(x) - exp(-x)) / 2
//         = (2^(x * log2(e)) - 2^(-x * log2(e))) / 2
//
// Implementation notes, see the original file for more details
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_sinh() { // APPROXIMATION_MODE=false, ITERATIONS=8
    constexpr float log2e = 1.4426950408889634f;
    const sfpi::vFloat v_log2e = log2e; // SFPLOADI
    const sfpi::vFloat v_half = 0.5f;   // SFPLOADI
    const sfpi::vFloat v_low_threshold = -127.0f; // SFPLOADI: clamp floor to prevent 2^z underflow
    const sfpi::vFloat v_sixth = 0.16666667f;     // SFPLOADI: 1/6 for Taylor approximation

#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) { // 8 iterations per face
        sfpi::vFloat x = sfpi::dst_reg[0]; // SFPLOAD: read 32 elements from current DEST row pair

        // Compute z_pos = x * log2(e) for exp(x) = 2^z_pos
        sfpi::vFloat z_pos = x * v_log2e; // SFPMAD

        // Clamp to prevent underflow
        v_if(z_pos < v_low_threshold) { z_pos = v_low_threshold; } // SFPSETCC + conditional SFPLOADI
        v_endif;

        sfpi::vFloat exp_pos = exp_21f<APPROXIMATION_MODE>(z_pos); // inline helper: computes 2^z_pos

        // Compute z_neg = -x * log2(e) for exp(-x) = 2^z_neg
        sfpi::vFloat z_neg = -z_pos; // SFPMAD (negate via multiply by -1 or sign flip)

        // Clamp to prevent underflow (z_neg could be very negative for large positive x)
        v_if(z_neg < v_low_threshold) { z_neg = v_low_threshold; } // SFPSETCC + conditional SFPLOADI
        v_endif;

        sfpi::vFloat exp_neg = exp_21f<APPROXIMATION_MODE>(z_neg); // inline helper: computes 2^z_neg

        // sinh(x) = (exp(x) - exp(-x)) / 2
        sfpi::vFloat y = (exp_pos - exp_neg) * v_half; // SFPMAD (sub) then SFPMAD (mul by 0.5)

        // For small |x|, override with Taylor: sinh(x) ~ x + x^3/6
        sfpi::vFloat abs_x = sfpi::setsgn(x, 0); // SFPSETSGN: clear sign bit to get |x|
        v_if(abs_x < v_half) { // SFPSETCC: CC set for lanes where |x| < 0.5
            sfpi::vFloat x_sq = x * x;         // SFPMAD
            y = x + x_sq * x * v_sixth;        // SFPMAD chain: x^2*x*1/6 then add x
        }
        v_endif;

        // Convert to bfloat16 for deterministic rounding
        y = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(y, 0)); // SFPSTOCHRND: FP32->FP16B nearest even

        sfpi::dst_reg[0] = y; // SFPSTORE: write result back to current DEST row pair
        sfpi::dst_reg++;      // advance to next sfpi row (2 physical DEST rows)
    }
}

template <bool APPROXIMATION_MODE>
inline void sinh_init() {
    // No programmable constants needed
}
```

### SFPU Instructions Used

| SFPU Instruction | SFPI Abstraction | Usage in Kernel |
|-----------------|-----------------|-----------------|
| `SFPLOAD` | `dst_reg[0]` (read) | Load 32 elements from current DEST row pair into LREG for processing |
| `SFPSTORE` | `dst_reg[0] = y` (write) | Store computed result back to current DEST row pair |
| `SFPMAD` | `vFloat * vFloat`, `vFloat + vFloat`, `vFloat - vFloat` | All floating-point arithmetic: multiply by log2(e), add bias, subtract exp_neg from exp_pos, multiply by 0.5, Taylor polynomial terms. Most heavily used instruction. |
| `SFPLOADI` | `vFloat(constant)` | Load immediate float constants: log2(e), 0.5, -127.0, 1/6, 0x3F800000 (1.0f), polynomial coefficients |
| `SFPDIVP2` | `sfpi::addexp(z, 23)` | Add 23 to exponent field of z, effectively multiplying by 2^23 for the exp_21f algorithm |
| `SFPEXEXP` | `sfpi::exexp(v)` | Extract debiased exponent from IEEE 754 float (used in exp_21f to decompose the integer part) |
| `SFPEXMAN` | `sfpi::exman9(v)` | Extract 9-bit mantissa from IEEE 754 float (used in exp_21f for fractional part refinement) |
| `SFPSETSGN` | `sfpi::setsgn(x, 0)` | Clear sign bit to compute absolute value `|x|` for the small-value threshold test |
| `SFPSETEXP` | `sfpi::setexp(v, exp)` | Set exponent field to reconstruct `2^exp * mantissa` in the final step of exp_21f |
| `SFPCAST` | `sfpi::int32_to_float(v, 0)` | Convert integer to float (RNE mode) -- used in exp_21f polynomial refinement |
| `SFPSTOCHRND` | `sfpi::float_to_fp16b(y, 0)` | Convert FP32 to BF16 with nearest-even rounding for deterministic output |
| `SFPSETCC` | `v_if(cond)` | Set per-lane condition codes for conditional execution: clamping z to -127, Taylor override for small `|x|` |
| `SFPIADD` | `vInt(const) + man_part` | Integer addition of mantissa offset constants in exp_21f polynomial refinement |

**Note**: `_float_to_int32_positive_` is called but not defined in any header reachable from the include chain. This function is used to convert a known-positive vFloat to vInt (truncation). This may be an undefined symbol that would cause a compilation failure, or it may be resolved through a mechanism not visible in the source tree (e.g., a compiler built-in or a header generated at build time). The semantic intent is a float-to-int32 conversion for positive values.

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **DEST rows** | Input tile data is read from DEST via `dst_reg[0]` (SFPLOAD); results are written back via `dst_reg[0] = y` (SFPSTORE). The stride-2 addressing means each `dst_reg[0]` access touches 2 physical DEST rows (32 elements). |
| **LREGs (L0-L3)** | Used implicitly by the SFPI compiler for all intermediate vFloat/vInt values. The compiler allocates L0-L3 for the numerous temporaries: `x`, `z_pos`, `z_neg`, `exp_pos`, `exp_neg`, `y`, `abs_x`, `x_sq`, and all intermediates within `exp_21f()`. Due to the high register pressure from two `exp_21f` calls per iteration, the compiler must spill to DEST stack or recompute values. |
| **SFPU Condition Codes (CC)** | Used by `v_if`/`v_endif` for three conditional blocks: (1) clamp z_pos to -127, (2) clamp z_neg to -127, (3) Taylor override when `|x| < 0.5`. Each `v_if` saves/sets CC, and `v_endif` restores it. |
| **Programmable constants** | None used (`sinh_init()` is empty). Constants are loaded inline via `SFPLOADI`. |

### Address Mode Configuration

The `eltwise_unary_sfpu_configure_addrmod<SfpuType::sinh>()` function configures address modes during initialization. Since `SfpuType::sinh` does not match any of the special-cased `if constexpr` conditions (topk_local_sort, typecast, unary_max, unary_min, etc.), only the default `ADDR_MOD_7` is configured:

**Wormhole B0 and Blackhole (identical)**:
```
ADDR_MOD_7: { srca.incr = 0, srcb.incr = 0, dest.incr = 0 }
```

This is a zero-increment address mode. The SFPU kernel manages DEST address progression explicitly through `dst_reg++` (which increments by 1 sfpi row = 2 physical DEST rows per iteration) and `SETRWC`/`inc_dst_addr` between faces (handled by the params dispatch layer). The address mode hardware auto-increment is not used.

Both Wormhole B0 and Blackhole use identical address mode configuration for this operation.

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Determine compute kernel path, SFPU_OP_CHAIN_0 expansion, and approximation mode for SINH
   **Key Findings**: SINH uses `eltwise_sfpu.cpp`, expands to `sinh_tile_init(); sinh_tile({idst});`, `get_op_approx_mode()` returns false (default case)

2. **File**: `tt_metal/hw/inc/api/compute/eltwise_unary/sinh.h`
   **Reason**: API header exposing `sinh_tile()` and `sinh_tile_init()` to compute kernels
   **Key Findings**: `sinh_tile(idst)` calls `llk_math_eltwise_unary_sfpu_sinh<APPROX>(idst)`, `sinh_tile_init()` calls `llk_math_eltwise_unary_sfpu_sinh_init<APPROX>()`

3. **File**: `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_sinh.h`
   **Reason**: LLK dispatch layer bridging API to core SFPU function
   **Key Findings**: Calls `_llk_math_eltwise_unary_sfpu_params_` with `calculate_sinh<APPROXIMATE, 8>` as the SFPU functor, default `VectorMode::RC`, init uses `SfpuType::sinh`

4. **File**: `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_sinh.h`
   **Reason**: Core SFPU implementation -- the primary target of this analysis
   **Key Findings**: Implements sinh via `exp_21f` helper (Moroz 2022 fast 2^z algorithm) with Taylor fallback for small |x|. Uses SFPI abstractions. Identical across Wormhole and Blackhole. Contains undefined reference to `_float_to_int32_positive_`.

5. **File**: `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Parameters dispatch -- understand face iteration, vector mode, DEST address progression
   **Key Findings**: VectorMode::RC loops over 4 faces, calls SFPU functor once per face, advances DEST address by SETRWC (WH) or inc_dst_addr (BH) between faces

6. **File**: `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: Init and address mode configuration
   **Key Findings**: `eltwise_unary_sfpu_configure_addrmod<SfpuType::sinh>()` only sets `ADDR_MOD_7` with all increments=0. SfpuType::sinh is not in any special-case branch.

7. **File**: `tt_metal/jit_build/genfiles.cpp`
   **Reason**: Determine how `APPROX` compile-time constant is generated
   **Key Findings**: `APPROX` is emitted as `constexpr bool APPROX = {hlk_math_approx_mode}` in the JIT-generated `chlkc_descriptors.h`

8. **File**: `runtime/sfpi/include/sfpi_lib.h`
   **Reason**: Map SFPI abstraction functions to underlying SFPU hardware instructions
   **Key Findings**: `addexp` -> `SFPDIVP2`, `exexp` -> `SFPEXEXP`, `exman9` -> `SFPEXMAN`, `setsgn` -> `SFPSETSGN`, `setexp` -> `SFPSETEXP`, `float_to_fp16b` -> `SFPSTOCHRND`, `int32_to_float` -> `SFPCAST`

9. **File**: `.claude/references/sfpu-hardware-model.md`
   **Reason**: SFPU hardware model reference for tile geometry, DEST addressing, and instruction semantics
   **Key Findings**: Confirmed stride-2 addressing (32 elements per dst_reg access), 8 iterations per face, 4 faces per tile = 32 total iterations covering 1024 elements

10. **File**: `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h`
    **Reason**: Verify how `SFPU_OP_SINH_INCLUDE` triggers the include of `sinh.h`
    **Key Findings**: `#if SFPU_OP_SINH_INCLUDE` includes `api/compute/eltwise_unary/sinh.h`

11. **File**: `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu_types.h`
    **Reason**: Verify `SfpuType::sinh` enum exists
    **Key Findings**: `sinh` is defined in the `SfpuType` enum on both architectures
