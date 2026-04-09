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
| Template parameter (SFPU_OP_CHAIN) | none (uses `APPROX` constexpr) | `get_op_init_and_func_default()` returns `sinh_tile_init()` / `sinh_tile(0)` -- no explicit template parameter; the API header `sinh.h` uses `sinh_tile_init<APPROX>()` and `sinh_tile<APPROX>(idst)` where `APPROX` is a `constexpr bool` generated from `math_approx_mode` |
| Effective SFPU path | `APPROXIMATION_MODE=false` -- the `calculate_sinh` template parameter `APPROXIMATION_MODE` is `false`, but the kernel does not contain any `if constexpr (APPROXIMATION_MODE)` branches, so the same code path is taken regardless | `ckernel_sfpu_sinh.h` -- no conditional branches on `APPROXIMATION_MODE`; the parameter is forwarded to `exp_21f<APPROXIMATION_MODE>` which also does not branch on it |

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/sinh.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_sinh.h` (identical on Blackhole) |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_sinh.h` (identical on Blackhole) |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (Blackhole version at `tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu_params.h` uses `_llk_math_eltwise_unary_sfpu_start_` / `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_` / `_llk_math_eltwise_unary_sfpu_done_` helper functions instead of inline TTI instructions, but the logical behavior is identical) |

### Call Chain
1. The compute kernel's `SFPU_OP_CHAIN_0` macro expands to `sinh_tile(0)`, defined in the API header `sinh.h`.
2. `sinh_tile(uint32_t idst)` calls `llk_math_eltwise_unary_sfpu_sinh<APPROX>(idst)` (guarded by `MATH(...)` so it only runs on the math RISC-V core).
3. `llk_math_eltwise_unary_sfpu_sinh<APPROXIMATE, 8>(dst_index, VectorMode::RC)` in the LLK dispatch header calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::calculate_sinh<APPROXIMATE, 8>, dst_index, VectorMode::RC)`.
4. `_llk_math_eltwise_unary_sfpu_params_` in the parameters dispatch layer sets the DEST write address, stalls until SFPU is ready, then loops over 4 faces (for `VectorMode::RC`), calling `calculate_sinh<false, 8>()` once per face, with `SETRWC` advancing the DEST pointer between faces.
5. `calculate_sinh<false, 8>()` in `ckernel_sfpu_sinh.h` executes 8 iterations per face, reading from `dst_reg[0]`, computing sinh(x) via exp_21f-based approach with Taylor fallback for small |x|, and writing the result back to `dst_reg[0]` before advancing `dst_reg++`.

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` (all 4 faces of the tile are processed). This is the default mode passed from the LLK dispatch and processes all 1024 elements of the tile.
- **Operation invocation**: The dispatch layer loops `for (int face = 0; face < 4; face++)`, calling `calculate_sinh<false, 8>()` once per face. Each invocation processes 8 iterations (ITERATIONS=8), covering all 256 elements of one face.
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, `SETRWC` between faces). On Wormhole, the params dispatch uses `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` twice between faces (advancing by 16 sfpi rows = 1 face). On Blackhole, the equivalent `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` calls `math::inc_dst_addr<8>()` twice. The address mode `ADDR_MOD_7` is configured with `.dest = {.incr = 0}` (no auto-increment from addr_mod, since `dst_reg++` in the kernel code handles iteration-level advancement).

### Annotated SFPU Kernel Source

The kernel uses SFPI abstractions (`sfpi::vFloat`, `sfpi::vInt`, `sfpi::dst_reg`, `v_if`/`v_endif`), so Style A (inline-commented source) applies.

```cpp
// File: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_sinh.h

// Helper: compute 2^z using exp_21f algorithm (Moroz et al. 2022)
// Input z must be clamped to avoid overflow/underflow before calling.
// Returns 2^z as a vFloat.
template <bool APPROXIMATION_MODE> // APPROXIMATION_MODE=false (not branched on in this function)
inline sfpi::vFloat exp_21f(sfpi::vFloat z) {
    // Step 1: Scale by 2^23 to shift fractional bits into integer position
    z = sfpi::addexp(z, 23);  // SFPDIVP2: adds 23 to exponent of z, effectively z *= 2^23

    // Step 2: Add IEEE 754 bias (0x3F800000 = 1.0f) and convert to int
    const sfpi::vFloat bias = sfpi::vFloat(0x3f800000);
    sfpi::vInt z_int = _float_to_int32_positive_(z + bias);  // [UNVERIFIED] function not defined in codebase; intended as float-to-int conversion for positive values

    // Step 3: Decompose into exponent and mantissa parts
    sfpi::vInt exp_part = sfpi::exexp(sfpi::reinterpret<sfpi::vFloat>(z_int));  // SFPEXEXP: extract debiased exponent
    sfpi::vInt man_part = sfpi::exman9(sfpi::reinterpret<sfpi::vFloat>(z_int));  // SFPEXMAN: extract mantissa without hidden bit, 9-bit left padding

    // Step 4: Polynomial refinement for 2^frac(z)
    sfpi::vFloat d1 = sfpi::vFloat(0.40196114e-7f);
    sfpi::vFloat d2 = sfpi::int32_to_float(sfpi::vInt(0xf94ee7) + man_part, 0);  // SFPCAST: int32 -> fp32; SFPIADD for integer addition of constants
    sfpi::vFloat d3 = sfpi::int32_to_float(sfpi::vInt(0x560e) + man_part, 0);    // SFPCAST: int32 -> fp32; SFPIADD for integer addition

    d2 = d1 * d2;  // SFPMUL
    sfpi::vInt frac_int = _float_to_int32_positive_(d2 * d3);  // [UNVERIFIED] same undefined function

    // Step 5: Reconstruct result = mantissa_frac * 2^exponent
    sfpi::vInt result_int =
        sfpi::reinterpret<sfpi::vInt>(sfpi::setexp(sfpi::reinterpret<sfpi::vFloat>(frac_int), 127U + exp_part));  // SFPSETEXP: set exponent field to (127 + exp_part)

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
inline void calculate_sinh() {  // APPROXIMATION_MODE=false, ITERATIONS=8
    constexpr float log2e = 1.4426950408889634f;
    const sfpi::vFloat v_log2e = log2e;       // SFPLOADI: load immediate float constant
    const sfpi::vFloat v_half = 0.5f;          // SFPLOADI: load immediate float constant
    const sfpi::vFloat v_low_threshold = -127.0f;  // SFPLOADI: load immediate float constant
    const sfpi::vFloat v_sixth = 0.16666667f;  // SFPLOADI: load immediate float constant

#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];  // SFPLOAD: load 32 elements from current DEST row pair

        // Compute z_pos = x * log2(e) for exp(x) = 2^z_pos
        sfpi::vFloat z_pos = x * v_log2e;  // SFPMUL

        // Clamp to prevent underflow
        v_if(z_pos < v_low_threshold) { z_pos = v_low_threshold; }  // SFPMAD (subtract for comparison) + SFPSETCC + predicated SFPLOADI
        v_endif;

        sfpi::vFloat exp_pos = exp_21f<APPROXIMATION_MODE>(z_pos);  // inline expansion of exp_21f helper

        // Compute z_neg = -x * log2(e) for exp(-x) = 2^z_neg
        sfpi::vFloat z_neg = -z_pos;  // SFPSETSGN or SFPMAD (negate)

        // Clamp to prevent underflow (z_neg could be very negative for large positive x)
        v_if(z_neg < v_low_threshold) { z_neg = v_low_threshold; }  // same clamping as above
        v_endif;

        sfpi::vFloat exp_neg = exp_21f<APPROXIMATION_MODE>(z_neg);  // inline expansion of exp_21f helper

        // sinh(x) = (exp(x) - exp(-x)) / 2
        sfpi::vFloat y = (exp_pos - exp_neg) * v_half;  // SFPADD (subtract via negate+add) + SFPMUL

        // For small |x|, override with Taylor: sinh(x) ~ x + x^3/6
        sfpi::vFloat abs_x = sfpi::setsgn(x, 0);  // SFPSETSGN: clear sign bit to get |x|
        v_if(abs_x < v_half) {  // SFPMAD (subtract for comparison) + SFPSETCC
            sfpi::vFloat x_sq = x * x;         // SFPMUL
            y = x + x_sq * x * v_sixth;        // SFPMUL + SFPMUL + SFPADD
        }
        v_endif;

        // Convert to bfloat16 for deterministic rounding
        y = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(y, 0));  // SFPSTOCHRND: fp32 -> fp16b with stochastic rounding mode 0

        sfpi::dst_reg[0] = y;  // SFPSTORE: write 32 elements back to current DEST row pair
        sfpi::dst_reg++;       // advance DEST pointer by 1 sfpi row (= 2 physical DEST rows = 32 elements)
    }
}

template <bool APPROXIMATION_MODE>
inline void sinh_init() {
    // No programmable constants needed
}
```

### SFPU Instructions Used

| Instruction | SFPI Intrinsic / Abstraction | Description |
|-------------|------------------------------|-------------|
| `SFPLOAD` | `dst_reg[0]` (read) | Loads 32 elements (2 physical DEST rows) from the current DEST address into an LREG |
| `SFPSTORE` | `dst_reg[0] = y` (write) | Stores 32 elements from an LREG back to the current DEST address |
| `SFPLOADI` | `vFloat(constant)` | Loads an immediate floating-point constant into an LREG |
| `SFPMUL` | `vFloat * vFloat` / `flt_mul()` | Floating-point multiplication of two LREG vectors |
| `SFPADD` | `vFloat + vFloat` / `vFloat - vFloat` / `flt_add()` | Floating-point addition (subtraction is add of negated operand) |
| `SFPMAD` | Comparison in `v_if(a < b)` | Multiply-accumulate used for subtract-and-compare operations to set condition codes |
| `SFPDIVP2` | `sfpi::addexp(v, exp)` | Adds an 8-bit immediate value to the exponent field of a float (equivalent to multiplying by 2^exp) |
| `SFPEXEXP` | `sfpi::exexp(v)` | Extracts and debiases the 8-bit exponent from a float |
| `SFPEXMAN` | `sfpi::exman9(v)` | Extracts the mantissa without hidden bit, padded with 9 leading zeros |
| `SFPSETEXP` | `sfpi::setexp(v, exp)` | Replaces the exponent field of a float with the specified value |
| `SFPSETSGN` | `sfpi::setsgn(v, sgn)` | Replaces the sign bit of a float (used to compute absolute value and negate) |
| `SFPCAST` | `sfpi::int32_to_float(v, mode)` | Converts int32 to fp32 (with configurable rounding mode) |
| `SFPSTOCHRND` | `sfpi::float_to_fp16b(v, mode)` | Converts fp32 to fp16b (bfloat16) with stochastic rounding |
| `SFPIADD` | `vInt(literal) + man_part` | Integer addition of two vInt values (used for constant offsets in exp_21f polynomial) |
| `SFPSETCC` / `SFPENCC` | `v_if` / `v_endif` | Set/enable condition codes for predicated execution |

**Note**: `_float_to_int32_positive_` is used twice in the `exp_21f` helper but is **not defined anywhere** in the current codebase. This function would need to be implemented (likely mapping to `SFPCAST` with an appropriate mode for fp32-to-int32 conversion of positive values) for this kernel to compile.

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **DEST register** | Source and destination for tile data. Each iteration reads 32 elements (2 physical DEST rows) via `dst_reg[0]` and writes results back via `dst_reg[0] = y`. The DEST pointer advances by 1 sfpi row per iteration via `dst_reg++`. |
| **LREGs (L0-L3)** | The SFPU has 4 local registers (LREGs). The compiler allocates these for vFloat/vInt temporaries. This kernel is register-intensive: in `calculate_sinh`, the variables `x`, `z_pos`, `z_neg`, `exp_pos`, `exp_neg`, `y`, `abs_x`, `x_sq`, and the constants (`v_log2e`, `v_half`, `v_low_threshold`, `v_sixth`) compete for 4 LREGs. The `exp_21f` helper adds additional pressure with `z`, `bias`, `z_int`, `exp_part`, `man_part`, `d1`, `d2`, `d3`, `frac_int`, `result_int`. The compiler must spill some values to DEST or re-materialize constants. |
| **Condition Code (CC)** | Per-lane predication register used by `v_if`/`v_endif` for: (1) clamping `z_pos < v_low_threshold`, (2) clamping `z_neg < v_low_threshold`, and (3) the small-|x| Taylor approximation branch `abs_x < v_half`. Each `v_if` sets CC via comparison, and the guarded block only executes on lanes where the condition is true. |
| **Programmable constants** | None. `sinh_init()` is empty -- no `vConstFloatPrgm0`/`vConstFloatPrgm1` are configured. |

### Address Mode Configuration

The address mode is configured during initialization via `eltwise_unary_sfpu_configure_addrmod<SfpuType::sinh>()`, which is called from `_llk_math_eltwise_unary_sfpu_init_<SfpuType::sinh>()`.

Since `SfpuType::sinh` does not match any of the special-case `if constexpr` branches (which handle `topk_local_sort`, `typecast`, `unary_max/min`, `signbit`), only the default `ADDR_MOD_7` is configured:

**Wormhole B0 and Blackhole (identical):**

| ADDR_MOD | srca.incr | srcb.incr | dest.incr | Purpose |
|----------|-----------|-----------|-----------|---------|
| `ADDR_MOD_7` | 0 | 0 | 0 | Default SFPU addr mode -- no auto-increment. The kernel manages DEST advancement explicitly via `dst_reg++` in the inner loop (per-iteration) and `SETRWC`/`inc_dst_addr` between faces (per-face). |

No other `ADDR_MOD` slots are configured by this operation. The `ADDR_MOD_0` and `ADDR_MOD_2` slots are typically used by the A2D (unpack-to-DEST) path and are left untouched by the SFPU init.

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Determine compute kernel path, SFPU_OP_CHAIN_0 expansion, approximation mode, and macro defines for SINH
   **Key Findings**: `get_compute_kernel_path` returns `"eltwise_sfpu.cpp"` (default); `get_op_init_and_func_default` returns `sinh_tile_init()` / `sinh_tile(0)`; `get_op_approx_mode` returns `false` (default); `get_macro_definition` returns `"SFPU_OP_SINH_INCLUDE"`

2. **File**: `tt_metal/hw/inc/api/compute/eltwise_unary/sinh.h`
   **Reason**: API header that exposes `sinh_tile()` and `sinh_tile_init()` to compute kernels
   **Key Findings**: `sinh_tile(idst)` calls `llk_math_eltwise_unary_sfpu_sinh<APPROX>(idst)`; `sinh_tile_init()` calls `llk_math_eltwise_unary_sfpu_sinh_init<APPROX>()`

3. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_sinh.h`
   **Reason**: LLK dispatch layer bridging the API to the core SFPU implementation
   **Key Findings**: `llk_math_eltwise_unary_sfpu_sinh` dispatches to `_llk_math_eltwise_unary_sfpu_params_` with `calculate_sinh<APPROXIMATE, 8>` as the callable, `VectorMode::RC` as the default mode

4. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_sinh.h`
   **Reason**: Core SFPU kernel implementation containing `calculate_sinh` and `exp_21f` helper
   **Key Findings**: Implements sinh(x) = (exp(x) - exp(-x)) / 2 via the exp_21f algorithm (Moroz et al. 2022) for 2^z computation, with Taylor approximation (x + x^3/6) fallback for |x| < 0.5 to avoid catastrophic cancellation; `_float_to_int32_positive_` is called but not defined in the codebase

5. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Parameters dispatch layer that manages DEST addressing and per-face iteration
   **Key Findings**: For `VectorMode::RC`, loops over 4 faces calling the SFPU function, with `TTI_SETRWC` advancing DEST by 16 sfpi rows between faces

6. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: Contains `eltwise_unary_sfpu_configure_addrmod` and `_llk_math_eltwise_unary_sfpu_init_`
   **Key Findings**: For `SfpuType::sinh`, only `ADDR_MOD_7` is set with all increments = 0

7. **File**: `tt_metal/jit_build/genfiles.cpp`
   **Reason**: Determine how `APPROX` constexpr is generated from `math_approx_mode`
   **Key Findings**: `emit_math_scalar_descriptors` outputs `constexpr bool APPROX = {get_hlk_math_approx_mode()};` into `chlkc_descriptors.h`

8. **File**: `runtime/sfpi/include/sfpi_lib.h`
   **Reason**: Map SFPI intrinsics (`addexp`, `exexp`, `exman9`, `setexp`, `setsgn`, `int32_to_float`, `float_to_fp16b`) to their underlying `__builtin_rvtt_*` compiler builtins and SFPU instructions
   **Key Findings**: `addexp` -> `SFPDIVP2`, `exexp` -> `SFPEXEXP`, `exman9` -> `SFPEXMAN`, `setexp` -> `SFPSETEXP`, `setsgn` -> `SFPSETSGN`, `int32_to_float` -> `SFPCAST`, `float_to_fp16b` -> `SFPSTOCHRND`

9. **File**: `runtime/sfpi/include/sfpi.h`
   **Reason**: Map vFloat arithmetic operators to SFPU instructions
   **Key Findings**: `vFloat + vFloat` -> `__builtin_rvtt_sfpadd` (SFPADD), `vFloat * vFloat` -> `__builtin_rvtt_sfpmul` (SFPMUL), `vFloat - vFloat` -> negate + SFPADD

10. **File**: `.claude/references/sfpu-hardware-model.md`
    **Reason**: Authoritative reference for SFPU addressing model, tile/face geometry, and instruction semantics
    **Key Findings**: Confirmed stride-2 addressing (SFP_DESTREG_STRIDE=2), ITERATIONS=8 per face, 32 elements per iteration, FACE_SIZE=256
