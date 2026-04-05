## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### Unary Dispatch Summary
- **UnaryOpType**: `SELU`
- **Compute kernel**: `eltwise_sfpu.cpp` (default path; `get_compute_kernel_path()` has no explicit case for SELU)
- **SFPU_OP_CHAIN_0 expansion**: `selu_tile_init(); selu_tile(0);`

#### Approximation Mode Resolution
There are TWO independent controls for approximation. Both are reported below.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(SELU)` in `unary_op_utils.cpp` -- falls through to `default: return false` |
| Template parameter (SFPU_OP_CHAIN) | none (no parameterized variant) | `get_op_init_and_func_default()` returns `selu_tile_init()` and `selu_tile({idst})` with no template arguments; the API header `selu.h` uses the `APPROX` macro which resolves to the `math_approx_mode` compile define |
| Effective SFPU path | Non-approximate: `_calculate_exponential_piecewise_<false, false, false>` takes the `else` branch (line 388-397 of `ckernel_sfpu_exp.h`), calling `_sfpu_exp_()` + `_sfpu_reciprocal_<2>()` for negative inputs | The `if constexpr (APPROXIMATION_MODE)` branch at line 353 is NOT taken; the `else` branch at line 388 executes |

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/selu.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_selu.h` (identical on Blackhole) |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_selu.h` (identical on Blackhole) |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (Blackhole variant at `tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu_params.h`) |

### Call Chain

1. **Compute kernel** (`eltwise_sfpu.cpp`) invokes `SFPU_OP_CHAIN_0` which expands to `selu_tile(0)`.
2. **API header** (`selu.h`): `selu_tile(idst)` expands via the `SFPU_UNARY_NO_PARAM_KERNEL_FN` macro to `_llk_math_eltwise_unary_sfpu_params_<APPROX>(ckernel::sfpu::calculate_selu<APPROX>, idst, (int)VectorMode::RC)`.
3. **Parameters dispatch** (`llk_math_eltwise_unary_sfpu_params.h`): `_llk_math_eltwise_unary_sfpu_params_` sets the DEST write address, activates the address mod base, stalls until SFPU is available, then loops over 4 faces (RC mode), calling `calculate_selu<false>()` per face with `SETRWC` between faces.
4. **Core SFPU** (`ckernel_sfpu_selu.h`): `calculate_selu<false>()` iterates 8 times per face (ITERATIONS=8), loading each DEST row pair, conditionally computing `alpha * (exp(x) - 1)` for negative lanes via `_calculate_exponential_piecewise_`, then unconditionally multiplying all lanes by `scale`.

For initialization: `selu_tile_init()` expands via `SFPU_INIT_KERNEL_CALL` to `llk_math_eltwise_unary_sfpu_init<SfpuType::selu, APPROX>(ckernel::sfpu::selu_init<APPROX>)`. This calls `_llk_math_eltwise_unary_sfpu_init_<SfpuType::selu>()` (configures ADDR_MOD_7, resets counters), then `selu_init<false>()` which calls `_init_exponential_<false, false, 0x3F800000>()`. Since both `APPROXIMATION_MODE` and `FAST_APPROX` are false, this takes the final `else` branch, calling `_init_sfpu_reciprocal_<false>()` to load the polynomial coefficients for the Newton-Raphson reciprocal into programmable constant registers.

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` -- all 4 faces of the tile are processed (full 32x32 tile).
- **Operation invocation**: The params dispatch loops `for (int face = 0; face < 4; face++)`, calling `calculate_selu<false>()` once per face. Each invocation runs ITERATIONS=8 inner loop iterations, processing 8 sfpi rows (= 256 elements = 1 face).
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, `SETRWC` between faces). On Wormhole, `set_addr_mod_base()` activates ADDR_MODs 4..7, so ADDR_MOD_7 (dest.incr=0) is used; the SFPU manages row advancement via `dst_reg++` in the SFPI abstraction. On Blackhole, `_llk_math_eltwise_unary_sfpu_start_` does not call `set_addr_mod_base()` but ADDR_MOD_7 is still configured identically with dest.incr=0.

### Annotated SFPU Kernel Source

The kernel uses SFPI abstractions (`sfpi::vFloat`, `sfpi::dst_reg`, `v_if`/`v_endif`, etc.) -- Style A applies.

```cpp
// File: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_selu.h

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_selu() { // APPROXIMATION_MODE=false, ITERATIONS=8
    // Implementation notes, see the original file for more details

    constexpr bool SCALE_EN = false;
    constexpr bool SKIP_POSITIVE_CHECK = false;
    constexpr std::uint16_t exp_base_scale_factor = p_sfpu::kCONST_1_FP16B; // 0x3F80 = 1.0 in BF16

    // alpha = 1.6732632... in FP32: 0x3FD63840
    sfpi::vFloat v_alpha = Converter::as_float(0x3FD63840); // SFPLOADI x2 to load 32-bit immediate into LREG
    // scale = 1.0507009... in FP32: 0x3F868640
    sfpi::vFloat v_scale = Converter::as_float(0x3F868640); // SFPLOADI x2 to load 32-bit immediate into LREG

#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0]; // SFPLOAD from current DEST row pair (32 elements)

        v_if(v < 0.0f) { // SFPSETCC with LT0 mode; enables CC for negative lanes only
            // Negative branch: compute alpha * (exp(x) - 1)
            sfpi::vFloat v_exp = _calculate_exponential_piecewise_<APPROXIMATION_MODE, SCALE_EN, SKIP_POSITIVE_CHECK>(
                v, exp_base_scale_factor);
            // With APPROXIMATION_MODE=false: calls _sfpu_exp_(setsgn(v, 0)) then _sfpu_reciprocal_<2> for v<0
            v = v_alpha * (v_exp - 1.0f); // SFPMAD: (v_exp * 1.0 + (-1.0)), then SFPMAD: v_alpha * result + 0.0
        }
        v_endif; // SFPENCC to restore all lanes

        // Unconditionally multiply all lanes by scale:
        //   positive lanes: scale * x (original value)
        //   negative lanes: scale * alpha * (exp(x) - 1)
        v = v_scale * v; // SFPMAD: v_scale * v + 0.0

        sfpi::dst_reg[0] = v; // SFPSTORE to current DEST row pair
        sfpi::dst_reg++;      // advance DEST pointer by 1 sfpi row (= 2 physical rows = 32 elements)
    }
}

template <bool APPROXIMATION_MODE>
inline void selu_init() { // APPROXIMATION_MODE=false
    const std::uint32_t EXP_BASE_SCALE_FACTOR = 0x3F800000; // 1.0f in FP32
    const bool FAST_APPROX = false;
    _init_exponential_<APPROXIMATION_MODE, FAST_APPROX, EXP_BASE_SCALE_FACTOR>();
    // With APPROX=false, FAST_APPROX=false: takes the final else branch,
    // calling _init_sfpu_reciprocal_<false>() which loads:
    //   vConstFloatPrgm0 = 0.3232325... (k0 for reciprocal polynomial)
    //   vConstFloatPrgm1 = 1.4545459... (k1 for reciprocal polynomial)
    //   vConstFloatPrgm2 = 2.1212124... (k2 for reciprocal polynomial)
}

}  // namespace sfpu
}  // namespace ckernel
```

**Helper: `_calculate_exponential_piecewise_` (non-approximate path)**

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_exp.h

template <bool APPROXIMATION_MODE, bool SCALE_EN, bool SKIP_POSITIVE_CHECK>
inline sfpi::vFloat _calculate_exponential_piecewise_(sfpi::vFloat in, const std::uint16_t exp_base_scale_factor)
{ // APPROXIMATION_MODE=false, SCALE_EN=false, SKIP_POSITIVE_CHECK=false
    sfpi::vFloat result = 0.0f;
    // SCALE_EN=false: no input scaling
    // APPROXIMATION_MODE=false: takes the else branch (line 388)

    // Non-approximate path: compute exp(|x|) via Horner series, then reciprocal for negative inputs
    result = _sfpu_exp_(sfpi::setsgn(in, 0)); // SFPSETSGN to force positive, then Horner exp

    v_if (in < 0) { // CC guard: only negative lanes
        result = _sfpu_reciprocal_<2>(result); // Newton-Raphson reciprocal (2 iterations) for exp(-|x|)
    }
    v_endif;

    return result;
}
```

**Helper: `_sfpu_exp_` (Horner series exponential)**

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_exp.h

sfpi_inline sfpi::vFloat _sfpu_exp_(sfpi::vFloat val)
{
    // If exponent is > -1 extract it and replace with -1
    sfpi::vInt exp = exexp(val);        // SFPEXEXP: extract biased exponent
    v_if (exp >= 0) {                   // CC guard for large magnitude inputs
        val = setexp(val, 126);         // SFPSETEXP: clamp exponent to -1 (bias 127 - 1 = 126)
    }
    v_endif;

    // Run series in Horner form: exp(x) ~ ((x * 0.8373) + 0.8633) * x + 1.0
    sfpi::vFloat tmp = val * sfpi::vConst0p8373 + sfpi::s2vFloat16b(0.863281); // SFPMAD
    val              = val * tmp + sfpi::vConst1;                               // SFPMAD

    v_if (exp >= 0) {                   // CC guard: repeated squaring for large exponents
        val = val * val;                // SFPMAD: first squaring
        for (int s_iter = 0; s_iter < 7; s_iter++) {
            exp = exp - 1;             // SFPIADD: integer decrement
            v_and(exp >= 0);           // narrow predication: CC AND (exp >= 0)
            val = val * val;           // SFPMAD: conditional squaring
        }
    }
    v_endif;

    return val;
}
```

**Helper: `_sfpu_reciprocal_<2>` (Newton-Raphson reciprocal, 2 iterations)**

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_recip.h

template <int max_iter = 2>
sfpi_inline sfpi::vFloat _sfpu_reciprocal_(const sfpi::vFloat in)
{ // max_iter=2 (FP32 precision)
    // Scale input to [-2, -1) range via mantissa injection from -1.0
    sfpi::vFloat negative_x = sfpi::setman(sfpi::vConstNeg1, sfpi::reinterpret<sfpi::vInt>(in)); // SFPSETMAN

    // Quadratic initial estimate: y = k2 - k1*x + k0*x^2
    // Uses programmable constants loaded by _init_sfpu_reciprocal_:
    //   vConstFloatPrgm0 = 0.3232325 (k0)
    //   vConstFloatPrgm1 = 1.4545459 (k1)
    //   vConstFloatPrgm2 = 2.1212124 (k2)
    sfpi::vFloat y = sfpi::vConstFloatPrgm1 + sfpi::vConstFloatPrgm0 * negative_x; // SFPMAD

    // Compute scale factor: ~in (bitwise NOT) gives 255-exponent
    sfpi::vInt scale_bits = ~sfpi::reinterpret<sfpi::vUInt>(in); // SFPNOT

    // Continue quadratic estimate
    y = sfpi::vConstFloatPrgm2 + y * negative_x; // SFPMAD

    // Clear mantissa of scale factor
    sfpi::vFloat scale = sfpi::setman(sfpi::reinterpret<sfpi::vFloat>(scale_bits), 0); // SFPSETMAN

    // First Newton-Raphson iteration: t = 1.0 - x*y
    sfpi::vFloat t = sfpi::vConst1 + negative_x * y; // SFPMAD

    // Adjust scale: scale = scale * 0.5 (handles edge cases for 0 and inf)
    scale *= 0.5f; // SFPMAD (or SFPMUL)

    // y = y + y*t (complete first NR iteration)
    y = y + y * t; // SFPMAD

    // max_iter=2: second Newton-Raphson iteration
    t = sfpi::vConst1 + negative_x * y; // SFPMAD
    y = y + y * t;                       // SFPMAD

    // Apply scaling and restore sign
    y = y * scale;             // SFPMAD
    y = sfpi::setsgn(y, in);  // SFPSETSGN: restore original sign

    return y;
}
```

### SFPU Instructions Used

The following SFPU instructions are emitted by the SFPI compiler from the C++ abstractions used in `calculate_selu` and its helper functions:

| Instruction | Description | Used By |
|-------------|-------------|---------|
| `SFPLOAD` | Load 32 elements from current DEST row pair into LREG | `dst_reg[0]` read at start of each iteration |
| `SFPSTORE` | Store 32 elements from LREG back to current DEST row pair | `dst_reg[0] = v` write at end of each iteration |
| `SFPLOADI` | Load 16-bit immediate into LREG (used in pairs for 32-bit constants) | Loading `v_alpha` (0x3FD63840), `v_scale` (0x3F868640), and intermediate constants |
| `SFPMAD` | Fused multiply-add (VD = VA * VB + VC) | All floating-point arithmetic: Horner polynomial, Newton-Raphson, `v_alpha * (v_exp - 1.0f)`, `v_scale * v` |
| `SFPSETCC` | Set condition code based on register comparison | `v_if(v < 0.0f)` -- sets CC.Res = (LREG < 0) for sign-bit test |
| `SFPENCC` | Enable/disable condition code masking | `v_endif` -- restores all lanes to active |
| `SFPCOMPC` | Complement CC.Res for else-branch handling | Used internally by `v_if`/`v_endif` CC management |
| `SFPPUSHC` | Push CC state onto stack for nested conditionals | Nested `v_if` in `_sfpu_exp_` and `_calculate_exponential_piecewise_` |
| `SFPPOPC` | Pop CC state from stack | Restoring CC after nested conditional blocks |
| `SFPEXEXP` | Extract exponent field from FP32 value | `exexp(val)` in `_sfpu_exp_` to check if exponent > -1 |
| `SFPSETEXP` | Set exponent field of FP32 value | `setexp(val, 126)` in `_sfpu_exp_` to clamp exponent |
| `SFPSETSGN` | Set sign bit of FP32 value | `setsgn(in, 0)` to force positive for exp; `setsgn(y, in)` to restore sign in reciprocal |
| `SFPSETMAN` | Set mantissa field of FP32 value | `setman(vConstNeg1, ...)` in `_sfpu_reciprocal_` for range normalization; `setman(..., 0)` to clear mantissa of scale |
| `SFPNOT` | Bitwise NOT on register | `~reinterpret<vUInt>(in)` in `_sfpu_reciprocal_` to compute 255-exponent for scale factor |
| `SFPIADD` | Integer add/subtract (sets CC.Res) | `exp = exp - 1` in `_sfpu_exp_` repeated squaring loop |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **LREG0-3** | General-purpose working registers used by SFPI compiler for intermediate values (`v`, `v_exp`, `v_alpha`, `v_scale`, `negative_x`, `y`, `t`, `scale`, `exp`, `tmp`) |
| **Programmable Const 0** (`vConstFloatPrgm0`) | Loaded by `_init_sfpu_reciprocal_` with 0.3232325 (k0 coefficient for reciprocal polynomial initial estimate) |
| **Programmable Const 1** (`vConstFloatPrgm1`) | Loaded by `_init_sfpu_reciprocal_` with 1.4545459 (k1 coefficient for reciprocal polynomial) |
| **Programmable Const 2** (`vConstFloatPrgm2`) | Loaded by `_init_sfpu_reciprocal_` with 2.1212124 (k2 coefficient for reciprocal polynomial) |
| **Fixed Const 2** (`vConst1`) | Hardware constant 1.0f -- used in Horner series final term and Newton-Raphson `t = 1.0 + negative_x * y` |
| **Fixed Const 0** (`vConst0p8373`) | Hardware constant 0.8373 -- used as coefficient in Horner series for `_sfpu_exp_` |
| **Fixed Const 2** (`vConstNeg1`) | Hardware constant -1.0f -- used in `_sfpu_reciprocal_` via `setman(vConstNeg1, ...)` to normalize input range |
| **DEST register** | Source and destination for tile data; accessed via `dst_reg[0]` (SFPLOAD/SFPSTORE), advanced by `dst_reg++` each iteration |

### Address Mode Configuration

The address mode is configured identically on both Wormhole and Blackhole:

**ADDR_MOD_7** (configured by `eltwise_unary_sfpu_configure_addrmod<SfpuType::selu>()`):
- `srca.incr = 0`
- `srcb.incr = 0`
- `dest.incr = 0`

Since `SfpuType::selu` does not match any special-cased `if constexpr` checks (only `topk_local_sort`, `typecast`, and min/max ops get custom ADDR_MOD_6 configuration), only ADDR_MOD_7 is set with zero increments.

**Wormhole-specific**: `set_addr_mod_base()` is called in `_llk_math_eltwise_unary_sfpu_params_`, setting the address mod base register to 1. This means physical ADDR_MODs 4-7 are used at runtime (ADDR_MOD_7 maps to physical slot 7, which is always in the 4-7 range regardless of the base bit).

**Blackhole-specific**: `_llk_math_eltwise_unary_sfpu_start_` does NOT call `set_addr_mod_base()`. ADDR_MOD_7 is configured the same way (all increments = 0).

DEST row advancement within each face is handled entirely by `dst_reg++` in the SFPI abstraction (which emits the appropriate DEST pointer increment), not by the address mode auto-increment. Between faces, `SETRWC` with increment 8 is used twice (= 16 physical rows = 1 face stride).

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Determine compute kernel path, SFPU_OP_CHAIN_0 defines, approximation mode, and macro definitions for SELU
   **Key Findings**: SELU uses `eltwise_sfpu.cpp` (default), `selu_tile_init()`/`selu_tile(idst)` with no parameters, approx mode = false (default case), macro `SFPU_OP_SELU_INCLUDE`

2. **File**: `tt_metal/hw/inc/api/compute/eltwise_unary/selu.h`
   **Reason**: API header defining `selu_tile()` and `selu_tile_init()` entry points
   **Key Findings**: Uses `SFPU_UNARY_NO_PARAM_KERNEL_FN(calculate_selu, RC, APPROX, idst)` for tile function and `SFPU_INIT_KERNEL_CALL(selu, ckernel::sfpu::selu_init, APPROX)` for init

3. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_selu.h`
   **Reason**: LLK dispatch layer bridging API to ckernel SFPU implementation
   **Key Findings**: Calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::calculate_selu<APPROXIMATE>, dst_index, vector_mode)` with VectorMode::RC default

4. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_selu.h`
   **Reason**: Core SFPU kernel implementation for SELU
   **Key Findings**: WH and BH implementations are identical. Uses SFPI abstractions. Computes SELU via: (1) load from DEST, (2) for negative lanes: exp(x) via `_calculate_exponential_piecewise_` then `alpha * (exp(x) - 1)`, (3) unconditional multiply by scale. Fixed constants alpha=1.6732632 (0x3FD63840) and scale=1.0507009 (0x3F868640).

5. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_exp.h`
   **Reason**: Contains `_calculate_exponential_piecewise_`, `_sfpu_exp_`, `_init_exponential_` used by SELU
   **Key Findings**: Non-approximate path computes exp(|x|) via Horner series with repeated squaring, then reciprocal for negative inputs. Init function (FAST_APPROX=false, APPROX=false) calls `_init_sfpu_reciprocal_<false>()`.

6. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_recip.h`
   **Reason**: Contains `_sfpu_reciprocal_<2>` and `_init_sfpu_reciprocal_` used in the non-approximate exponential path
   **Key Findings**: Newton-Raphson reciprocal with 2 iterations for FP32 precision. Quadratic initial estimate using Sollya-optimized polynomial. Init loads k0=0.3232325, k1=1.4545459, k2=2.1212124 into programmable constants.

7. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Parameters dispatch layer controlling per-face iteration and DEST address progression
   **Key Findings**: VectorMode::RC loops 4 faces, calling the SFPU function once per face. Uses SETRWC(CR_D, 8) x2 between faces. Stalls SFPU before starting, waits for SFPU completion at end.

8. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: ADDR_MOD configuration and SFPU init infrastructure
   **Key Findings**: ADDR_MOD_7 configured with all increments = 0 for selu. `set_addr_mod_base()` sets base to 1 (ADDR_MODs 4-7 active) on Wormhole.

9. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h`
   **Reason**: Macro definitions for `SFPU_UNARY_NO_PARAM_KERNEL_FN` and `SFPU_INIT_KERNEL_CALL`
   **Key Findings**: `SFPU_UNARY_NO_PARAM_KERNEL_FN(FN, MODE, APPROXIMATE, DST_IDX)` expands to `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::FN<APPROXIMATE>, DST_IDX, (int)VectorMode::MODE)`.

10. **File**: `.claude/references/sfpu-hardware-model.md`
    **Reason**: Authoritative reference for SFPU hardware architecture, register layout, instruction semantics, and addressing model
    **Key Findings**: Used for tile/face geometry (32x32 tile, 4 faces of 16x16), SFPU stride-2 addressing model, LREG organization, instruction latencies, and CC mechanism documentation.
