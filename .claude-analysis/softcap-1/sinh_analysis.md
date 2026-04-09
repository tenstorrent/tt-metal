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
| Template parameter (SFPU_OP_CHAIN) | none (uses default template arg = `APPROX`) | `get_op_init_and_func_default()` returns `sinh_tile_init()` / `sinh_tile({idst})` with no parameterized variant |
| Effective SFPU path | `APPROXIMATION_MODE=false` in both `calculate_sinh` and `exp_21f` | `APPROX` is generated as `constexpr bool APPROX = false;` by `jit_build/genfiles.cpp:394` from `math_approx_mode`. The API header `sinh.h` passes `APPROX` as template arg to `llk_math_eltwise_unary_sfpu_sinh<APPROX>()`. Since `APPROXIMATION_MODE` is not used in any `if constexpr` branch in this kernel, both paths (true/false) produce identical code. |

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/sinh.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_sinh.h` (identical for blackhole) |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_sinh.h` (identical for blackhole) |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (blackhole version at `tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu_params.h`) |

### Call Chain

1. **Compute kernel** (`eltwise_sfpu.cpp`): The `SFPU_OP_CHAIN_0` macro expands to `sinh_tile_init(); sinh_tile(0);`, which invokes `sinh_tile(uint32_t idst)` for each tile.
2. **API header** (`sinh.h`): `sinh_tile(idst)` calls `llk_math_eltwise_unary_sfpu_sinh<APPROX>(idst)` inside the `MATH(...)` macro (executes only on the TRISC_MATH thread).
3. **LLK dispatch** (`llk_math_eltwise_unary_sfpu_sinh.h`): `llk_math_eltwise_unary_sfpu_sinh<APPROXIMATE, ITERATIONS=8>(dst_index, VectorMode::RC)` calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::calculate_sinh<APPROXIMATE, 8>, dst_index, VectorMode::RC)`.
4. **Parameters dispatch** (`llk_math_eltwise_unary_sfpu_params.h`): Sets up DEST addressing, stalls until SFPU is ready, then in VectorMode::RC mode loops over 4 faces calling `calculate_sinh<false, 8>()` per face, advancing DEST address between faces via `SETRWC` (Wormhole) or `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` (Blackhole).
5. **Core SFPU** (`ckernel_sfpu_sinh.h`): `calculate_sinh<false, 8>()` iterates 8 times per face, computing sinh(x) = (exp(x) - exp(-x)) / 2 using the `exp_21f` helper for fast 2^z computation, with a Taylor series fallback for small |x|.

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` (default) -- processes all 4 faces of the tile (Face 0, 1, 2, 3), covering the full 32x32 = 1024 elements.
- **Operation invocation**: The params dispatch calls `calculate_sinh<false, 8>()` once per face (4 calls total). Each call runs the inner loop 8 times (ITERATIONS=8), processing 8 sfpi rows x 32 elements/row = 256 elements = one full face.
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, dst_reg++ per iteration, SETRWC between faces). On Wormhole, the params dispatch uses `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` twice between faces (advancing 16 physical rows = 1 face). On Blackhole, `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` calls `math::inc_dst_addr<8>()` twice for the same effect.

### Annotated SFPU Kernel Source

This kernel uses SFPI abstractions (`sfpi::vFloat`, `sfpi::vInt`, `dst_reg`, `v_if`/`v_endif`) -- Style A.

```cpp
// File: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_sinh.h

namespace ckernel::sfpu {

// Helper: compute 2^z using exp_21f algorithm (Moroz et al. 2022)
// Input z must be clamped to avoid overflow/underflow before calling.
// Returns 2^z as a vFloat.
template <bool APPROXIMATION_MODE> // APPROXIMATION_MODE=false (not used in any branch)
inline sfpi::vFloat exp_21f(sfpi::vFloat z) {
    // Scale by 2^23 to shift fractional bits into integer position
    z = sfpi::addexp(z, 23); // SFPDIVP2 with ADD mode: adds 23 to the exponent field of z

    // Add IEEE 754 bias (0x3F800000 = 1.0f) and convert to int
    const sfpi::vFloat bias = sfpi::vFloat(0x3f800000); // SFPLOADI to load 1.0f
    sfpi::vInt z_int = _float_to_int32_positive_(z + bias); // [UNVERIFIED] undefined function -- intended as float-to-int cast

    // Decompose into exponent and mantissa parts
    sfpi::vInt exp_part = sfpi::exexp(sfpi::reinterpret<sfpi::vFloat>(z_int)); // SFPEXEXP: extract biased exponent, debias by 127
    sfpi::vInt man_part = sfpi::exman9(sfpi::reinterpret<sfpi::vFloat>(z_int)); // SFPEXMAN with PAD9: extract 9-bit mantissa

    // Polynomial refinement for 2^frac(z)
    sfpi::vFloat d1 = sfpi::vFloat(0.40196114e-7f); // SFPLOADI: small polynomial coefficient
    sfpi::vFloat d2 = sfpi::int32_to_float(sfpi::vInt(0xf94ee7) + man_part, 0); // SFPIADD + SFPCAST: integer add then convert to float
    sfpi::vFloat d3 = sfpi::int32_to_float(sfpi::vInt(0x560e) + man_part, 0); // SFPIADD + SFPCAST: integer add then convert to float

    d2 = d1 * d2; // SFPMAD: d1 * d2 + 0.0
    sfpi::vInt frac_int = _float_to_int32_positive_(d2 * d3); // [UNVERIFIED] undefined function -- intended as float-to-int cast

    // Reconstruct result = mantissa_frac * 2^exponent
    sfpi::vInt result_int =
        sfpi::reinterpret<sfpi::vInt>(sfpi::setexp(sfpi::reinterpret<sfpi::vFloat>(frac_int), 127U + exp_part)); // SFPSETEXP: set exponent to (127 + exp_part)

    return sfpi::reinterpret<sfpi::vFloat>(result_int);
}

// sinh(x) = (exp(x) - exp(-x)) / 2
// Implementation notes, see the original file for more details
template <bool APPROXIMATION_MODE, int ITERATIONS = 8> // APPROXIMATION_MODE=false, ITERATIONS=8
inline void calculate_sinh() {
    constexpr float log2e = 1.4426950408889634f;
    const sfpi::vFloat v_log2e = log2e; // SFPLOADI: log2(e) constant
    const sfpi::vFloat v_half = 0.5f; // SFPLOADI: 0.5
    const sfpi::vFloat v_low_threshold = -127.0f; // SFPLOADI: clamp threshold to prevent underflow in 2^z
    const sfpi::vFloat v_sixth = 0.16666667f; // SFPLOADI: 1/6 for Taylor series

#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0]; // SFPLOAD: load 32 elements from current DEST position

        // Compute z_pos = x * log2(e) for exp(x) = 2^z_pos
        sfpi::vFloat z_pos = x * v_log2e; // SFPMAD: x * log2e + 0.0

        // Clamp to prevent underflow
        v_if(z_pos < v_low_threshold) { z_pos = v_low_threshold; } // CC: SFPSETCC + guard, clamp z_pos >= -127
        v_endif;

        sfpi::vFloat exp_pos = exp_21f<APPROXIMATION_MODE>(z_pos); // inline: compute 2^z_pos

        // Compute z_neg = -x * log2(e) for exp(-x) = 2^z_neg
        sfpi::vFloat z_neg = -z_pos; // SFPMAD: z_pos * (-1.0) + 0.0 (sign inversion)

        // Clamp to prevent underflow (z_neg could be very negative for large positive x)
        v_if(z_neg < v_low_threshold) { z_neg = v_low_threshold; } // CC: SFPSETCC + guard, clamp z_neg >= -127
        v_endif;

        sfpi::vFloat exp_neg = exp_21f<APPROXIMATION_MODE>(z_neg); // inline: compute 2^z_neg

        // sinh(x) = (exp(x) - exp(-x)) / 2
        sfpi::vFloat y = (exp_pos - exp_neg) * v_half; // SFPMAD (subtract) then SFPMAD (multiply by 0.5)

        // For small |x|, override with Taylor: sinh(x) ~ x + x^3/6
        sfpi::vFloat abs_x = sfpi::setsgn(x, 0); // SFPSETSGN: clear sign bit to get |x|
        v_if(abs_x < v_half) { // CC: compare |x| < 0.5
            sfpi::vFloat x_sq = x * x; // SFPMAD: x * x + 0.0
            y = x + x_sq * x * v_sixth; // SFPMAD chain: x_sq * x + 0.0, then result * v_sixth + x
        }
        v_endif;

        // Convert to bfloat16 for deterministic rounding
        y = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(y, 0)); // SFPSTOCHRND: FP32 -> FP16B with round-to-nearest-even

        sfpi::dst_reg[0] = y; // SFPSTORE: write 32 elements back to current DEST position
        sfpi::dst_reg++; // advance DEST pointer by 1 sfpi row (2 physical rows, 32 elements)
    }
}

template <bool APPROXIMATION_MODE>
inline void sinh_init() {
    // No programmable constants needed
}

}  // namespace ckernel::sfpu
```

**Critical Note: Undefined Symbol**

The function `_float_to_int32_positive_` called at lines 23 and 35 of `ckernel_sfpu_sinh.h` is **not defined anywhere in this codebase**. A comprehensive search across all SFPI headers (`runtime/sfpi/include/`), all ckernel headers (`tt_metal/hw/ckernels/`, `tt_metal/third_party/tt_llk/`), and the entire repository produced zero definitions. This function is used as if it were a standard SFPI utility for converting a positive float to int32, but no such function exists. The SFPI library provides `int32_to_float()` (via `SFPCAST`) but no inverse `float_to_int32()` for FP32->INT32 conversion. This is a **compilation-blocking defect** in the generated kernel.

### SFPU Instructions Used

| Instruction | SFPI Abstraction | Description |
|------------|-----------------|-------------|
| `SFPLOAD` | `dst_reg[0]` read | Load 32 elements from DEST register into LREG for processing |
| `SFPSTORE` | `dst_reg[0] = y` write | Store 32 processed elements from LREG back to DEST register |
| `SFPLOADI` | `vFloat(constant)` | Load 16-bit immediate constant into LREG (used for log2e, 0.5, -127.0, 1/6, polynomial coefficients) |
| `SFPMAD` | `*`, `+`, `-` on vFloat | Fused multiply-add for all float arithmetic: multiplication (a*b+0), addition (a*1+b), subtraction (a*1-b via sign invert), and sign negation |
| `SFPDIVP2` | `sfpi::addexp(z, 23)` | Add integer to exponent field -- scales z by 2^23 to shift fractional bits to integer position |
| `SFPEXEXP` | `sfpi::exexp()` | Extract and debias the exponent field of a float (subtract 127 bias) |
| `SFPEXMAN` | `sfpi::exman9()` | Extract 9-bit mantissa from a float, zero-padded to integer |
| `SFPSETEXP` | `sfpi::setexp()` | Set the exponent field of a float to a specified value (used to reconstruct 2^exp * mantissa) |
| `SFPSETSGN` | `sfpi::setsgn(x, 0)` | Set sign bit to 0 (compute absolute value) |
| `SFPCAST` | `sfpi::int32_to_float()` | Convert INT32 to FP32 with rounding (used twice in exp_21f for polynomial evaluation) |
| `SFPSTOCHRND` | `sfpi::float_to_fp16b(y, 0)` | Convert FP32 to BF16 with round-to-nearest-even (deterministic rounding before store) |
| `SFPIADD` | `vInt(constant) + man_part` | Integer addition of mantissa parts in exp_21f polynomial refinement |
| `SFPSETCC` | `v_if(condition)` | Set per-lane condition code based on comparison (less-than for clamping and Taylor branch) |
| `SFPENCC` | `v_if` / `v_endif` preamble/postamble | Enable/disable condition code masking at start and end of conditional blocks |
| `SFPPUSHC` | nested `v_if` support | Push condition code state onto CC stack for nested conditionals |
| `SFPPOPC` | `v_endif` | Pop condition code state from CC stack to restore prior state |
| `SFPCOMPC` | implicit in `v_if`/`v_endif` | Complement CC for else-branch handling within SFPI abstractions |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **DEST** | Source and destination for tile data. Each iteration loads 32 elements (2 physical rows x 16 cols) from DEST via `dst_reg[0]`, processes them, and writes results back. 8 iterations per face, 4 faces per tile. |
| **LREGs (0-7)** | Used as temporary storage for intermediate values. The SFPI compiler maps `vFloat`/`vInt` local variables to LREGs. Key temporaries include: `x` (input), `z_pos`/`z_neg` (scaled arguments), `exp_pos`/`exp_neg` (exponential results), `y` (output), and various polynomial coefficients in `exp_21f`. The compiler handles register allocation; no explicit LREG indexing is used. |
| **Programmable Constants** | Not used by this kernel (`sinh_init()` is empty). |

### Address Mode Configuration

The address mode for the `sinh` SFPU operation is configured by `eltwise_unary_sfpu_configure_addrmod<SfpuType::sinh>()` during init. Since `SfpuType::sinh` does not match any special-cased `if constexpr` branch (it is not `topk_local_sort`, `typecast`, `unary_max`, `unary_min`, `reciprocal`, or `signbit`), only the default address mode is set:

**Both Wormhole and Blackhole:**
```
ADDR_MOD_7: { srca.incr = 0, srcb.incr = 0, dest.incr = 0 }
```

The dest increment of 0 means the SFPU does not auto-increment the DEST address between instructions. Instead, DEST address advancement is managed explicitly:
- **Within a face**: The SFPI `dst_reg++` at the end of each loop iteration advances the DEST read/write pointer by 1 sfpi row (= 2 physical DEST rows = 32 elements).
- **Between faces**: The params dispatch layer inserts `SETRWC` instructions (Wormhole) or calls `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` (Blackhole) to advance by 16 physical rows (1 face stride).

No additional address modes (ADDR_MOD_6, etc.) are configured for this operation.

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Determine compute kernel path, SFPU_OP_CHAIN_0 defines, and approximation mode for SINH
   **Key Findings**: SINH uses `eltwise_sfpu.cpp` compute kernel, expands to `sinh_tile_init(); sinh_tile({idst});`, macro guard `SFPU_OP_SINH_INCLUDE`, `get_op_approx_mode()` returns false (default case)

2. **File**: `tt_metal/hw/inc/api/compute/eltwise_unary/sinh.h`
   **Reason**: API header for sinh_tile and sinh_tile_init
   **Key Findings**: `sinh_tile(idst)` calls `llk_math_eltwise_unary_sfpu_sinh<APPROX>(idst)` inside MATH macro. `sinh_tile_init()` calls `llk_math_eltwise_unary_sfpu_sinh_init<APPROX>()`

3. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_sinh.h`
   **Reason**: LLK dispatch layer bridging API to ckernel
   **Key Findings**: Dispatch calls `_llk_math_eltwise_unary_sfpu_params_` with `calculate_sinh<APPROXIMATE, 8>` as the callable, VectorMode::RC, ITERATIONS=8

4. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_sinh.h`
   **Reason**: Core SFPU implementation containing `calculate_sinh` and `exp_21f`
   **Key Findings**: Implements sinh(x) = (exp(x)-exp(-x))/2 using exp_21f (Moroz 2^z algorithm) with Taylor fallback for |x|<0.5. Contains undefined reference to `_float_to_int32_positive_` (compilation blocker). Final BF16 rounding via `float_to_fp16b`.

5. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Parameters dispatch layer that loops over faces and manages DEST addressing
   **Key Findings**: VectorMode::RC loops 4 faces, calling sfpu_func once per face. Uses SETRWC for inter-face DEST advancement.

6. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: Address mode configuration and SFPU init
   **Key Findings**: `SfpuType::sinh` gets only ADDR_MOD_7 with all increments = 0 (no special cases). Init calls `_init_sfpu_config_reg()` and `reset_counters()`.

7. **File**: `runtime/sfpi/include/sfpi_lib.h`
   **Reason**: SFPI intrinsic-to-instruction mappings
   **Key Findings**: `addexp` -> `SFPDIVP2`, `exexp` -> `SFPEXEXP`, `exman9` -> `SFPEXMAN`, `setexp` -> `SFPSETEXP`, `setsgn` -> `SFPSETSGN`, `int32_to_float` -> `SFPCAST`, `float_to_fp16b` -> `SFPSTOCHRND`

8. **File**: `tt_metal/jit_build/genfiles.cpp`
   **Reason**: Verify how `math_approx_mode` becomes the `APPROX` constexpr
   **Key Findings**: Line 394 generates `constexpr bool APPROX = {math_approx_mode};` which is false for SINH

9. **File**: `.claude/references/sfpu-hardware-model.md`
   **Reason**: Authoritative SFPU hardware model reference for instruction semantics, addressing, and tile geometry
   **Key Findings**: Confirmed stride-2 addressing model, ITERATIONS=8 per face, 32 elements per sfpi row, instruction semantics for SFPMAD/SFPLOAD/SFPSTORE/etc.

10. **File**: `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h`
    **Reason**: Verify the include guard mechanism for SFPU_OP_SINH_INCLUDE
    **Key Findings**: `#if SFPU_OP_SINH_INCLUDE` guards `#include "api/compute/eltwise_unary/sinh.h"`
