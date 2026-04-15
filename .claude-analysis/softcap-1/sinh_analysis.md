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
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(SINH)` in `unary_op_utils.cpp` -- currently returns `false` for all ops (switch has only a `default: return false` case) |
| Template parameter (SFPU_OP_CHAIN) | none | `get_op_init_and_func_default()` -- non-parameterized: `sinh_tile_init()` / `sinh_tile(idst)` with default template args; the `APPROX` macro is resolved from `math_approx_mode` |
| Effective SFPU path | `APPROXIMATION_MODE=false` everywhere | `calculate_sinh<false, 8>()` and `exp_21f<false>()` are instantiated; however, neither function contains any `if constexpr (APPROXIMATION_MODE)` branches, so the `APPROXIMATION_MODE` template parameter is accepted but has no effect on execution |

### SFPU Abstraction Layers
| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/sinh.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_sinh.h` |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_sinh.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |

### Call Chain
1. **Compute kernel** (`eltwise_sfpu.cpp`): The `SFPU_OP_CHAIN_0` macro expands to `sinh_tile_init(); sinh_tile(0);`, calling the API header functions.
2. **API Header** (`sinh.h`): `sinh_tile(idst)` calls `llk_math_eltwise_unary_sfpu_sinh<APPROX>(idst)` on the MATH thread. `sinh_tile_init()` calls `llk_math_eltwise_unary_sfpu_sinh_init<APPROX>()`.
3. **LLK Dispatch** (`llk_math_eltwise_unary_sfpu_sinh.h`): The init function delegates to `llk_math_eltwise_unary_sfpu_init<SfpuType::sinh, APPROXIMATE>()` which configures address modes and SFPU config registers. The tile function delegates to `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::calculate_sinh<APPROXIMATE, 8>, dst_index, VectorMode::RC)`.
4. **Parameters Dispatch** (`llk_math_eltwise_unary_sfpu_params.h`): Sets up DEST write address, stalls for SFPU readiness, then calls `calculate_sinh<false, 8>()` once per face (4 faces for `VectorMode::RC`), advancing DEST address between faces via `SETRWC`/`inc_dst_addr`.
5. **Core SFPU** (`ckernel_sfpu_sinh.h`): `calculate_sinh<false, 8>()` processes 8 SFPU iterations (one face), computing `sinh(x) = (exp(x) - exp(-x)) / 2` using the `exp_21f` helper for exponentiation and a Taylor fallback for small inputs.

### Parameters Dispatch Summary
- **Vector mode**: `VectorMode::RC` (default) -- all 4 faces of the tile are processed.
- **Operation invocation**: `calculate_sinh<false, 8>()` is called 4 times (once per face) in a loop within `_llk_math_eltwise_unary_sfpu_params_`. Each invocation processes 8 SFPU iterations (ITERATIONS=8), covering one 16x16 face (256 elements).
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, `SETRWC` / `inc_dst_addr<8>` twice between faces). On Wormhole, `SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` is called twice between faces. On Blackhole, `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` calls `math::inc_dst_addr<8>()` twice. Both advance 16 physical DEST rows = 1 face stride. Address mode is `ADDR_MOD_7` with `{.srca.incr=0, .srcb.incr=0, .dest.incr=0}` on both architectures (no special address increment needed since `dst_reg++` in the SFPI code handles progression internally).

### Annotated SFPU Kernel Source

The kernel uses SFPI abstractions (`sfpi::vFloat`, `sfpi::vInt`, `sfpi::dst_reg`, `v_if`/`v_endif`), so Style A (inline-commented source) is used.

**Note on missing dependency**: The function `_float_to_int32_positive_()` is called twice in `exp_21f()` but is **not defined anywhere in the current codebase**. It is not found in any header file under `tt_metal/`, `runtime/sfpi/`, or `tt_metal/third_party/tt_llk/`. This is a missing dependency that would cause a compilation error. The intended semantics appear to be a float-to-int32 conversion for positive values, likely implemented via `SFPCAST` (sign-magnitude to 2's complement) or a bitwise reinterpretation.

```cpp
// File: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_sinh.h
// (Blackhole implementation is identical)

// Helper: compute 2^z using exp_21f algorithm (Moroz et al. 2022)
// Input z must be clamped to avoid overflow/underflow before calling.
// Returns 2^z as a vFloat.
template <bool APPROXIMATION_MODE> // APPROXIMATION_MODE=false (no branches depend on it)
inline sfpi::vFloat exp_21f(sfpi::vFloat z) {
    // Scale by 2^23 to shift fractional bits into integer position
    z = sfpi::addexp(z, 23); // SFPDIVP2 with ADD mode -- adds 23 to the exponent field

    // Add IEEE 754 bias (0x3F800000 = 1.0f) and convert to int
    const sfpi::vFloat bias = sfpi::vFloat(0x3f800000); // SFPLOADI to load 1.0f
    sfpi::vInt z_int = _float_to_int32_positive_(z + bias); // [UNVERIFIED] -- function not defined in codebase

    // Decompose into exponent and mantissa parts
    sfpi::vInt exp_part = sfpi::exexp(sfpi::reinterpret<sfpi::vFloat>(z_int)); // SFPEXEXP -- extract debiased exponent
    sfpi::vInt man_part = sfpi::exman9(sfpi::reinterpret<sfpi::vFloat>(z_int)); // SFPEXMAN with PAD9 mode -- extract 9-bit mantissa

    // Polynomial refinement for 2^frac(z)
    sfpi::vFloat d1 = sfpi::vFloat(0.40196114e-7f); // SFPLOADI
    sfpi::vFloat d2 = sfpi::int32_to_float(sfpi::vInt(0xf94ee7) + man_part, 0); // SFPIADD (int add) + SFPCAST (int32->fp32 RNE)
    sfpi::vFloat d3 = sfpi::int32_to_float(sfpi::vInt(0x560e) + man_part, 0);   // SFPIADD (int add) + SFPCAST (int32->fp32 RNE)

    d2 = d1 * d2; // SFPMAD (d1 * d2 + 0.0)
    sfpi::vInt frac_int = _float_to_int32_positive_(d2 * d3); // [UNVERIFIED] -- function not defined in codebase

    // Reconstruct result = mantissa_frac * 2^exponent
    sfpi::vInt result_int =
        sfpi::reinterpret<sfpi::vInt>(sfpi::setexp(sfpi::reinterpret<sfpi::vFloat>(frac_int), 127U + exp_part));
        // SFPIADD (127 + exp_part) + SFPSETEXP -- set exponent field of result

    return sfpi::reinterpret<sfpi::vFloat>(result_int);
}

// sinh(x) = (exp(x) - exp(-x)) / 2
// For small |x| (< 0.5), uses Taylor approximation sinh(x) ~ x + x^3/6
template <bool APPROXIMATION_MODE, int ITERATIONS = 8> // APPROXIMATION_MODE=false, ITERATIONS=8
inline void calculate_sinh() {
    constexpr float log2e = 1.4426950408889634f;
    const sfpi::vFloat v_log2e = log2e;         // SFPLOADI
    const sfpi::vFloat v_half = 0.5f;            // SFPLOADI
    const sfpi::vFloat v_low_threshold = -127.0f; // SFPLOADI -- prevents 2^z underflow
    const sfpi::vFloat v_sixth = 0.16666667f;    // SFPLOADI -- 1/6 for Taylor term

#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0]; // SFPLOAD -- load 32 elements from DEST

        // Compute z_pos = x * log2(e) for exp(x) = 2^z_pos
        sfpi::vFloat z_pos = x * v_log2e; // SFPMAD (x * log2e + 0.0)

        // Clamp to prevent underflow
        v_if(z_pos < v_low_threshold) { z_pos = v_low_threshold; } // CC: SFPSETCC + SFPENCC/SFPPUSHC/SFPPOPC sequence
        v_endif;

        sfpi::vFloat exp_pos = exp_21f<APPROXIMATION_MODE>(z_pos); // inline call to exp_21f

        // Compute z_neg = -x * log2(e) for exp(-x) = 2^z_neg
        sfpi::vFloat z_neg = -z_pos; // SFPMAD with sign inversion (InstrMod[0]=1)

        // Clamp to prevent underflow (z_neg could be very negative for large positive x)
        v_if(z_neg < v_low_threshold) { z_neg = v_low_threshold; } // CC: same pattern
        v_endif;

        sfpi::vFloat exp_neg = exp_21f<APPROXIMATION_MODE>(z_neg); // inline call to exp_21f

        // sinh(x) = (exp(x) - exp(-x)) / 2
        sfpi::vFloat y = (exp_pos - exp_neg) * v_half; // SFPMAD (exp_pos * 1.0 + (-exp_neg)) then SFPMAD (* 0.5)

        // For small |x|, override with Taylor: sinh(x) ~ x + x^3/6
        sfpi::vFloat abs_x = sfpi::setsgn(x, 0); // SFPSETSGN -- clear sign bit to get |x|
        v_if(abs_x < v_half) {                    // CC: compare |x| < 0.5
            sfpi::vFloat x_sq = x * x;            // SFPMAD (x * x + 0.0)
            y = x + x_sq * x * v_sixth;           // SFPMAD chain: x_sq*x, then *v_sixth, then +x
        }
        v_endif;

        // Convert to bfloat16 for deterministic rounding
        y = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(y, 0)); // SFP_STOCH_RND (FP32->FP16B, RNE mode)

        sfpi::dst_reg[0] = y; // SFPSTORE -- write 32 elements back to DEST
        sfpi::dst_reg++;      // advance DEST pointer by 1 sfpi row (2 physical rows, 32 elements)
    }
}
```

### SFPU Instructions Used

| Instruction | SFPI Abstraction | Usage in Kernel |
|-------------|-----------------|-----------------|
| `SFPLOAD` | `dst_reg[0]` (read) | Load 32 elements from current DEST position into an LREG at the start of each iteration |
| `SFPSTORE` | `dst_reg[0] = y` (write) | Store computed sinh result back to DEST at the end of each iteration |
| `SFPMAD` | `vFloat * vFloat`, `vFloat + vFloat`, `vFloat - vFloat`, `-vFloat` | Core arithmetic: all float multiply, add, subtract, and negate operations. Used extensively for `x * log2e`, `exp_pos - exp_neg`, `* v_half`, `x * x`, `x_sq * x * v_sixth`, `x + ...`, `d1 * d2`, `d2 * d3`, `z + bias` |
| `SFPLOADI` | `vFloat(constant)` | Load immediate constants: `log2e`, `0.5f`, `-127.0f`, `0.16666667f`, `0x3f800000`, `0.40196114e-7f` |
| `SFPDIVP2` | `sfpi::addexp(z, 23)` | Add 23 to the exponent field of `z`, effectively multiplying by 2^23 (used in exp_21f to shift fractional bits into integer position) |
| `SFPEXEXP` | `sfpi::exexp(v)` | Extract the debiased exponent field from a float (used in exp_21f to decompose the integer part) |
| `SFPEXMAN` | `sfpi::exman9(v)` | Extract the 9-bit mantissa with padding (used in exp_21f to get fractional part for polynomial refinement) |
| `SFPSETEXP` | `sfpi::setexp(v, exp)` | Set the exponent field of a float to a specified value (used in exp_21f to reconstruct `2^exp * mantissa`) |
| `SFPSETSGN` | `sfpi::setsgn(x, 0)` | Clear the sign bit to compute absolute value `|x|` (used before the Taylor branch threshold test) |
| `SFPCAST` | `sfpi::int32_to_float(vInt, 0)` | Convert INT32 to FP32 with Round-to-Nearest-Even (used twice in exp_21f for polynomial coefficient construction) |
| `SFP_STOCH_RND` | `sfpi::float_to_fp16b(y, 0)` | Convert FP32 to bfloat16 with Round-to-Nearest-Even rounding (final output conversion for deterministic rounding) |
| `SFPIADD` | `vInt + vInt` (integer) | Integer addition for constructing polynomial coefficients in exp_21f: `0xf94ee7 + man_part`, `0x560e + man_part`, `127 + exp_part` |
| `SFPSETCC` / `SFPENCC` / `SFPCOMPC` / `SFPPUSHC` / `SFPPOPC` | `v_if` / `v_endif` | Condition code manipulation for three conditional branches: (1) clamp z_pos to -127, (2) clamp z_neg to -127, (3) Taylor approximation for small |x| |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **DEST rows** | Source and destination for tile data. Each iteration reads 32 elements (2 physical rows) via `dst_reg[0]` and writes back the result. 8 iterations per face, 4 faces per tile. |
| **LREG0-LREG3** (LREGS1 bank) | General-purpose registers used for intermediate computations. The SFPI compiler allocates these for `x`, `z_pos`, `z_neg`, `exp_pos`, `exp_neg`, `y`, `abs_x`, `x_sq`, and all intermediate values in `exp_21f` (`z`, `bias`, `z_int`, `exp_part`, `man_part`, `d1`, `d2`, `d3`, `frac_int`, `result_int`). |
| **LREG4-LREG7** (LREGS2 bank) | Overflow registers used when >4 live values exist simultaneously. LREG7 may be used for indirect addressing by SFPMAD if the compiler emits indirect mode. |
| **CC stack** | Used by `v_if`/`v_endif` for the three conditional branches (underflow clamping for z_pos, underflow clamping for z_neg, Taylor fallback for small |x|). Each `v_if`/`v_endif` pair pushes/pops the CC stack. |

### Address Mode Configuration

The address mode is configured during `llk_math_eltwise_unary_sfpu_sinh_init<APPROX>()` which calls `eltwise_unary_sfpu_configure_addrmod<SfpuType::sinh>()`.

Since `SfpuType::sinh` does not match any of the special-cased types (`topk_local_sort`, `typecast`, `unary_max`, `unary_min`, etc.), only the default `ADDR_MOD_7` is configured:

| Hardware | Address Mode | Configuration |
|----------|-------------|---------------|
| **Wormhole B0** | `ADDR_MOD_7` | `{.srca.incr=0, .srcb.incr=0, .dest.incr=0}` |
| **Blackhole** | `ADDR_MOD_7` | `{.srca.incr=0, .srcb.incr=0, .dest.incr=0}` |

The `.dest.incr=0` means the hardware address mode does not auto-increment the DEST pointer between SFPU instructions. Instead, DEST progression is managed explicitly:
- **Within a face**: `dst_reg++` in the SFPI code advances the SFPU DEST pointer by 1 sfpi row (= 2 physical rows = 32 elements) per iteration.
- **Between faces**: The params dispatch layer uses `SETRWC` (Wormhole) or `inc_dst_addr<8>` twice (Blackhole) to advance by 16 physical DEST rows (1 face stride).

Both architectures use identical address mode configuration for this operation.

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Determine compute kernel path, SFPU_OP_CHAIN expansion, and approximation mode for SINH
   **Key Findings**: SINH uses `eltwise_sfpu.cpp` (default), expands to `sinh_tile_init(); sinh_tile({idst});`, `get_op_approx_mode` returns `false` (default case), uses `SFPU_OP_SINH_INCLUDE` macro guard

2. **File**: `tt_metal/hw/inc/api/compute/eltwise_unary/sinh.h`
   **Reason**: API header that exposes `sinh_tile()` and `sinh_tile_init()` to compute kernels
   **Key Findings**: `sinh_tile(idst)` calls `llk_math_eltwise_unary_sfpu_sinh<APPROX>(idst)`, `sinh_tile_init()` calls `llk_math_eltwise_unary_sfpu_sinh_init<APPROX>()`

3. **File**: `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_sinh.h`
   **Reason**: LLK dispatch layer bridging API to core SFPU implementation
   **Key Findings**: Init delegates to `llk_math_eltwise_unary_sfpu_init<SfpuType::sinh, APPROXIMATE>()`. Tile function delegates to `_llk_math_eltwise_unary_sfpu_params_` with `calculate_sinh<APPROXIMATE, 8>` as the callable. Both WH and BH files are identical.

4. **File**: `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_sinh.h`
   **Reason**: Core SFPU implementation containing `calculate_sinh()` and `exp_21f()` helper
   **Key Findings**: Uses SFPI abstractions (vFloat, dst_reg, v_if/v_endif). Computes `sinh(x) = (exp(x) - exp(-x)) / 2` via `exp_21f` (Moroz et al. 2022 2^z algorithm) with Taylor fallback (`x + x^3/6`) for `|x| < 0.5`. Both WH and BH implementations are identical. Critical issue: `_float_to_int32_positive_()` is called but not defined anywhere in the codebase.

5. **File**: `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Parameters dispatch layer that manages DEST addressing and per-face iteration
   **Key Findings**: For VectorMode::RC, calls sfpu_func 4 times (once per face) with SETRWC/inc_dst_addr between faces. WH and BH differ slightly in implementation details but achieve the same DEST progression.

6. **File**: `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: SFPU init and address mode configuration
   **Key Findings**: `ADDR_MOD_7` configured with `{.dest.incr=0}` for sinh. Both architectures use the same configuration since sinh is not special-cased.

7. **File**: `runtime/sfpi/include/sfpi_lib.h`
   **Reason**: SFPI library functions mapping to hardware instructions
   **Key Findings**: `addexp` maps to `SFPDIVP2` (ADD mode), `exexp` maps to `SFPEXEXP` (DEBIAS mode), `exman9` maps to `SFPEXMAN` (PAD9 mode), `setexp` maps to `SFPSETEXP`, `setsgn` maps to `SFPSETSGN`, `int32_to_float` maps to `SFPCAST`, `float_to_fp16b` maps to `SFP_STOCH_RND` (FP32_TO_FP16B mode)

8. **File**: `.claude/references/sfpu-hardware-model.md`
   **Reason**: Authoritative SFPU hardware model reference
   **Key Findings**: Tile geometry (32x32, 4 faces of 16x16), stride-2 addressing model (dst_tile_size_sfpi=32, 8 iterations per face), SFPMAD used for all float arithmetic, instruction semantics for SFPDIVP2/SFPEXEXP/SFPEXMAN/SFPSETEXP/SFPCAST/SFP_STOCH_RND
