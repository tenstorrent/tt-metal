## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### Unary Dispatch Summary
- **UnaryOpType**: `SINH`
- **Compute kernel**: `eltwise_sfpu.cpp`
- **SFPU_OP_CHAIN_0 expansion**: `sinh_tile(0)`

#### Approximation Mode Resolution
There are TWO independent controls for approximation. Both are reported below.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(UnaryOpType::SINH)` in `unary_op_utils.cpp` -- falls through to `default: return false` (no explicit case for SINH) |
| Template parameter (SFPU_OP_CHAIN) | none | `get_op_init_and_func_default()` returns `"sinh_tile_init();"` and `"sinh_tile({idst});"` -- no parameterized template argument, uses default |
| Effective SFPU path | `APPROXIMATION_MODE=false` throughout | `sinh_tile_init<APPROX>()` and `sinh_tile<APPROX>(idst)` where `APPROX` is the JIT-generated `constexpr bool APPROX = false`. The `exp_21f<APPROXIMATION_MODE>` helper receives `false`, but its code has no `if constexpr` branches that depend on `APPROXIMATION_MODE` -- the same code path executes regardless. |

### SFPU Abstraction Layers
| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/sinh.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_sinh.h` |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_sinh.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |

### Call Chain
1. **Compute kernel** invokes `sinh_tile(0)` via the `SFPU_OP_CHAIN_0` macro.
2. **API Header** (`sinh.h`): `sinh_tile(uint32_t idst)` calls `MATH((llk_math_eltwise_unary_sfpu_sinh<APPROX>(idst)))`.
3. **LLK Dispatch** (`llk_math_eltwise_unary_sfpu_sinh.h`): `llk_math_eltwise_unary_sfpu_sinh<APPROXIMATE, 8>(dst_index, VectorMode::RC)` calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::calculate_sinh<APPROXIMATE, 8>, dst_index, VectorMode::RC)`.
4. **Parameters Dispatch** (`llk_math_eltwise_unary_sfpu_params.h`): Sets up DEST addressing, stalls for SFPU, then loops 4 times (one per face in `VectorMode::RC`), calling `calculate_sinh<false, 8>()` each iteration, with `SETRWC` between faces.
5. **Core SFPU** (`ckernel_sfpu_sinh.h`): `calculate_sinh<false, 8>()` executes 8 iterations per face, processing 32 elements per iteration via SFPI abstractions (`dst_reg`, `vFloat`, `v_if`).

### Parameters Dispatch Summary
- **Vector mode**: `VectorMode::RC` (default) -- all 4 faces of the tile are processed (faces 0, 1, 2, 3).
- **Operation invocation**: The parameters dispatch loops `for (int face = 0; face < 4; face++)`, calling `calculate_sinh<false, 8>()` once per face. Each call processes 8 SFPI iterations (ITERATIONS=8), covering one full 16x16 face (8 iterations x 32 elements = 256 elements).
- **DEST address progression**: Standard DEST progression. On Wormhole, `set_addr_mod_base()` shifts to ADDR_MOD bank 4-7, then `ADDR_MOD_7` is used with `dest.incr = 0` (no auto-increment from the addr_mod -- the SFPI `dst_reg++` handles intra-face advancement). Between faces, `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` is called twice (advancing by 16 physical DEST rows = 1 face). On Blackhole, the same logic applies via `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` which calls `math::inc_dst_addr<8>()` twice.

### Annotated SFPU Kernel Source

The kernel uses SFPI abstractions (Style A). The Wormhole and Blackhole implementations are identical.

```cpp
// File: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_sinh.h

namespace ckernel::sfpu {

// Helper: compute 2^z using exp_21f algorithm (Moroz et al. 2022)
// Input z must be clamped to avoid overflow/underflow before calling.
// Returns 2^z as a vFloat.
template <bool APPROXIMATION_MODE> // APPROXIMATION_MODE=false
inline sfpi::vFloat exp_21f(sfpi::vFloat z) {
    // Scale by 2^23 to shift fractional bits into integer position
    z = sfpi::addexp(z, 23); // SFPDIVP2 with ADD mode: adds 23 to exponent field

    // Add IEEE 754 bias (0x3F800000 = 1.0f) and convert to int
    const sfpi::vFloat bias = sfpi::vFloat(0x3f800000); // SFPLOADI: load immediate 1.0f
    sfpi::vInt z_int = _float_to_int32_positive_(z + bias); // SFPMAD (z*1.0+bias) then float-to-int conversion

    // Decompose into exponent and mantissa parts
    sfpi::vInt exp_part = sfpi::exexp(sfpi::reinterpret<sfpi::vFloat>(z_int)); // SFPEXEXP: extract debiased exponent
    sfpi::vInt man_part = sfpi::exman9(sfpi::reinterpret<sfpi::vFloat>(z_int)); // SFPEXMAN: extract 9-bit mantissa

    // Polynomial refinement for 2^frac(z)
    sfpi::vFloat d1 = sfpi::vFloat(0.40196114e-7f); // SFPLOADI
    sfpi::vFloat d2 = sfpi::int32_to_float(sfpi::vInt(0xf94ee7) + man_part, 0); // SFPIADD + SFPCAST(INT32_TO_FP32)
    sfpi::vFloat d3 = sfpi::int32_to_float(sfpi::vInt(0x560e) + man_part, 0); // SFPIADD + SFPCAST(INT32_TO_FP32)

    d2 = d1 * d2; // SFPMAD (d1 * d2 + 0)
    sfpi::vInt frac_int = _float_to_int32_positive_(d2 * d3); // SFPMAD then float-to-int conversion

    // Reconstruct result = mantissa_frac * 2^exponent
    sfpi::vInt result_int =
        sfpi::reinterpret<sfpi::vInt>(sfpi::setexp(sfpi::reinterpret<sfpi::vFloat>(frac_int), 127U + exp_part));
        // SFPIADD (127 + exp_part) then SFPSETEXP: sets exponent field of frac_int

    return sfpi::reinterpret<sfpi::vFloat>(result_int);
}

// sinh(x) = (exp(x) - exp(-x)) / 2
//         = (2^(x * log2(e)) - 2^(-x * log2(e))) / 2
//
// Implementation notes, see the original file for more details
template <bool APPROXIMATION_MODE, int ITERATIONS = 8> // APPROXIMATION_MODE=false, ITERATIONS=8
inline void calculate_sinh() {
    constexpr float log2e = 1.4426950408889634f;
    const sfpi::vFloat v_log2e = log2e; // SFPLOADI
    const sfpi::vFloat v_half = 0.5f; // SFPLOADI
    const sfpi::vFloat v_low_threshold = -127.0f; // SFPLOADI
    const sfpi::vFloat v_sixth = 0.16666667f; // SFPLOADI

#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0]; // SFPLOAD: load from current DEST row

        // Compute z_pos = x * log2(e) for exp(x) = 2^z_pos
        sfpi::vFloat z_pos = x * v_log2e; // SFPMAD (x * log2e + 0)

        // Clamp to prevent underflow
        v_if(z_pos < v_low_threshold) { z_pos = v_low_threshold; } // SFPSETCC + conditional SFPMOV
        v_endif;

        sfpi::vFloat exp_pos = exp_21f<APPROXIMATION_MODE>(z_pos); // inline call to exp_21f

        // Compute z_neg = -x * log2(e) for exp(-x) = 2^z_neg
        sfpi::vFloat z_neg = -z_pos; // SFPMAD (z_pos * -1.0 + 0) or SFPSETSGN

        // Clamp to prevent underflow (z_neg could be very negative for large positive x)
        v_if(z_neg < v_low_threshold) { z_neg = v_low_threshold; } // SFPSETCC + conditional SFPMOV
        v_endif;

        sfpi::vFloat exp_neg = exp_21f<APPROXIMATION_MODE>(z_neg); // inline call to exp_21f

        // sinh(x) = (exp(x) - exp(-x)) / 2
        sfpi::vFloat y = (exp_pos - exp_neg) * v_half; // SFPMAD (exp_pos + (-exp_neg)) then SFPMAD (* 0.5)

        // For small |x|, override with Taylor: sinh(x) ~ x + x^3/6
        sfpi::vFloat abs_x = sfpi::setsgn(x, 0); // SFPSETSGN: clear sign bit to get |x|
        v_if(abs_x < v_half) { // SFPSETCC: set CC based on abs_x < 0.5
            sfpi::vFloat x_sq = x * x; // SFPMAD (x * x + 0)
            y = x + x_sq * x * v_sixth; // SFPMAD (x_sq * x + 0), SFPMAD (* v_sixth + x)
        }
        v_endif; // SFPENCC: restore CC

        // Convert to bfloat16 for deterministic rounding
        y = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(y, 0)); // SFPSTOCHRND: FP32 to FP16B with RNE

        sfpi::dst_reg[0] = y; // SFPSTORE: write back to current DEST row
        sfpi::dst_reg++; // advance SFPI address by 1 (= 2 physical DEST rows)
    }
}

template <bool APPROXIMATION_MODE>
inline void sinh_init() {
    // No programmable constants needed
}

} // namespace ckernel::sfpu
```

### SFPU Instructions Used

The following SFPU instructions are emitted by the SFPI compiler from the abstractions used in the kernel:

| Instruction | SFPI Abstraction | Description |
|-------------|-----------------|-------------|
| `SFPLOAD` | `dst_reg[0]` (read) | Load 32 elements from current DEST row pair into an LREG |
| `SFPSTORE` | `dst_reg[0] = y` (write) | Store 32 elements from LREG back to current DEST row pair |
| `SFPLOADI` | `vFloat(constant)` | Load an immediate floating-point constant into an LREG |
| `SFPMAD` | `a * b`, `a + b`, `a - b` | Fused multiply-add (a * b + c). Used for all float arithmetic: multiplication (c=0), addition (b=1.0), subtraction (via negation) |
| `SFPDIVP2` | `sfpi::addexp(z, 23)` | Divide/multiply by power of 2 by adding to the exponent field. Here adds 23 to scale for integer conversion |
| `SFPEXEXP` | `sfpi::exexp(v)` | Extract debiased exponent from IEEE 754 float as integer |
| `SFPEXMAN` | `sfpi::exman9(v)` | Extract 9-bit mantissa from IEEE 754 float, zero-padded to integer |
| `SFPSETEXP` | `sfpi::setexp(v, exp)` | Set the exponent field of a float to the given value |
| `SFPSETSGN` | `sfpi::setsgn(x, 0)` | Set or clear the sign bit of a float. Used here to compute `abs(x)` |
| `SFPCAST` | `sfpi::int32_to_float(v, 0)` | Convert int32 to FP32 (round nearest even mode) |
| `SFPIADD` | `vInt(constant) + man_part` | Integer addition of a constant and a vInt register |
| `SFPSTOCHRND` | `sfpi::float_to_fp16b(y, 0)` | Convert FP32 to BF16 (FP16B) with nearest-even rounding |
| `SFPSETCC` | `v_if(condition)` | Set per-lane condition codes based on comparison (e.g., less-than) |
| `SFPENCC` | `v_endif` | Restore/disable per-lane condition code predication |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **DEST rows** (via `dst_reg`) | Input tile data is read from and results written back to the current DEST row pair. Each iteration processes 2 physical rows (32 elements) via stride-2 addressing. |
| **LREGs (L0-L3)** | The SFPI compiler allocates LREGs for intermediate `vFloat`/`vInt` values. Key intermediates: `x` (input), `z_pos`/`z_neg` (scaled exponents), `exp_pos`/`exp_neg` (exponentials), `y` (result), and `exp_21f` internal variables (`z_int`, `exp_part`, `man_part`, `d1`, `d2`, `d3`, `frac_int`, `result_int`). The compiler manages register allocation and spills. |
| **Condition Code (CC)** | Per-lane CC bits are used by `v_if`/`v_endif` for three conditional branches: (1) clamp `z_pos` to -127, (2) clamp `z_neg` to -127, (3) Taylor override for small `|x|` < 0.5. |

### Address Mode Configuration

The address mode configuration is identical for Wormhole and Blackhole for `SfpuType::sinh`:

| Address Mode | Field | Value | Purpose |
|-------------|-------|-------|---------|
| `ADDR_MOD_7` | `srca.incr` | 0 | No source A auto-increment |
| `ADDR_MOD_7` | `srcb.incr` | 0 | No source B auto-increment |
| `ADDR_MOD_7` | `dest.incr` | 0 | No DEST auto-increment from addr_mod (SFPI `dst_reg++` handles intra-face progression) |

The init function `_llk_math_eltwise_unary_sfpu_init_<SfpuType::sinh>()` calls `eltwise_unary_sfpu_configure_addrmod<SfpuType::sinh>()` which only configures `ADDR_MOD_7` (the default for all SFPU unary ops). `SfpuType::sinh` does not match any of the special-case `if constexpr` branches (which are for `topk_local_sort`, `typecast`, `unary_max`, `unary_min`, `signbit`, etc.), so no `ADDR_MOD_6` is configured.

On Wormhole, `set_addr_mod_base()` is called before SFPU execution, which sets `ADDR_MOD_SET_Base = 1`, shifting the active addr_mod bank to mods 4-7. This means `ADDR_MOD_7` is the effective address mode for SFPU operations. On Blackhole, the same logical behavior applies through the `_llk_math_eltwise_unary_sfpu_start_()` function.

**Notable observation**: The helper function `_float_to_int32_positive_()` is called twice in `exp_21f()` but its definition is not present in the current codebase. It is expected to perform a float-to-int32 conversion for known-positive values (likely using SFPCAST or bit manipulation). This appears to be a missing definition that would need to be provided for compilation to succeed.

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Determine compute kernel path, SFPU_OP_CHAIN_0 defines, and approximation mode for SINH
   **Key Findings**: SINH uses `eltwise_sfpu.cpp`, expands to `sinh_tile_init()` + `sinh_tile({idst})`, `get_op_approx_mode()` returns false (default case), macro is `SFPU_OP_SINH_INCLUDE`

2. **File**: `tt_metal/hw/inc/api/compute/eltwise_unary/sinh.h`
   **Reason**: API header exposing `sinh_tile()` and `sinh_tile_init()` to compute kernels
   **Key Findings**: Calls `llk_math_eltwise_unary_sfpu_sinh<APPROX>(idst)` and `llk_math_eltwise_unary_sfpu_sinh_init<APPROX>()`

3. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_sinh.h`
   **Reason**: LLK dispatch layer bridging API to core SFPU implementation
   **Key Findings**: Calls `_llk_math_eltwise_unary_sfpu_params_` with `calculate_sinh<APPROXIMATE, 8>` as the callable, default `VectorMode::RC`

4. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_sinh.h`
   **Reason**: Core SFPU implementation containing the calculate_sinh and exp_21f functions
   **Key Findings**: Implements sinh(x) = (exp(x) - exp(-x)) / 2 via custom exp_21f (Moroz et al. 2022 power-of-2 algorithm). Uses Taylor approximation (x + x^3/6) for |x| < 0.5 to avoid catastrophic cancellation. Final result converted to bfloat16 via SFPSTOCHRND. `_float_to_int32_positive_` is called but undefined in the codebase.

5. **File**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_sinh.h`
   **Reason**: Verify Blackhole implementation matches Wormhole
   **Key Findings**: Identical to Wormhole implementation

6. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Understand the parameters dispatch (face loop, DEST addressing, vector mode handling)
   **Key Findings**: RC mode loops 4 faces, calls SFPU function per face, uses TTI_SETRWC to advance between faces

7. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: Understand address mode configuration and SFPU init
   **Key Findings**: `ADDR_MOD_7` configured with all increments = 0 for sinh (no special cases). `set_addr_mod_base()` shifts to upper bank (mods 4-7).

8. **File**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: Verify Blackhole address mode configuration matches Wormhole
   **Key Findings**: Same ADDR_MOD_7 configuration. Uses `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` instead of direct TTI_SETRWC for face advancement.

9. **File**: `tt_metal/jit_build/genfiles.cpp`
   **Reason**: Understand how `APPROX` constexpr is generated in JIT compilation
   **Key Findings**: `constexpr bool APPROX = {get_hlk_math_approx_mode()};` is emitted into `chlkc_descriptors.h`

10. **File**: `runtime/sfpi/include/sfpi_lib.h`
    **Reason**: Map SFPI abstractions to underlying SFPU builtins/instructions
    **Key Findings**: `addexp` -> `__builtin_rvtt_sfpdivp2`, `exexp` -> `__builtin_rvtt_sfpexexp`, `exman9` -> `__builtin_rvtt_sfpexman`, `setexp` -> `__builtin_rvtt_sfpsetexp_i`, `setsgn` -> `__builtin_rvtt_sfpsetsgn_i`, `int32_to_float` -> `__builtin_rvtt_sfpcast`, `float_to_fp16b` -> `__builtin_rvtt_sfpstochrnd_i`

11. **File**: `.claude/references/sfpu-hardware-model.md`
    **Reason**: Reference for SFPU tile/face geometry, stride-2 addressing model, and instruction semantics
    **Key Findings**: Used fixed hardware model values (32 sfpi iterations per tile, 8 per face, stride-2 addressing)
