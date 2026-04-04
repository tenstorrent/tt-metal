## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### Unary Dispatch Summary
- **UnaryOpType**: `SELU`
- **Compute kernel**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`
- **SFPU_OP_CHAIN_0 expansion**: `selu_tile(0)`

#### Approximation Mode Resolution
There are TWO independent controls for approximation. You MUST report both.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(SELU)` in `unary_op_utils.cpp` -- currently returns `false` for all ops (switch has only a `default: return false` case) |
| Template parameter (SFPU_OP_CHAIN) | none | `get_op_init_and_func_default()` -- non-parameterized case: `selu_tile_init()` / `selu_tile(idst)` with default template args. The `APPROX` macro in the compute kernel resolves to `false`. |
| Effective SFPU path | Non-approximate exponential: `_sfpu_exp_()` + `_sfpu_reciprocal_<2>()` for negative inputs | The `if constexpr (APPROXIMATION_MODE)` branch in `_calculate_exponential_piecewise_` is NOT taken; the `else` branch at line 388 of `ckernel_sfpu_exp.h` is executed. |

### SFPU Abstraction Layers
List the file path for each abstraction layer.

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/selu.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_selu.h` |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_selu.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |

### Call Chain
1. **Compute kernel** calls `selu_tile(idst)` (defined in `selu.h` API header).
2. **API Header** expands `SFPU_UNARY_NO_PARAM_KERNEL_FN(calculate_selu, RC, APPROX, idst)` which calls `_llk_math_eltwise_unary_sfpu_params_<APPROX>(ckernel::sfpu::calculate_selu<APPROX>, idst, (int)VectorMode::RC)`.
3. **Parameters dispatch** (`llk_math_eltwise_unary_sfpu_params.h`) sets up the DEST write address, stalls until SFPU is ready, then loops over 4 faces (VectorMode::RC), calling `calculate_selu<false>()` once per face and advancing the DEST face address between calls via `SETRWC`.
4. **Core SFPU** (`ckernel_sfpu_selu.h`) implements the SELU formula: for each of 8 iterations per face, it loads from `dst_reg[0]`, conditionally computes `alpha * (exp(x) - 1)` for negative lanes via `_calculate_exponential_piecewise_`, then unconditionally multiplies by `scale` and stores back.
5. **Exponential sub-call**: `_calculate_exponential_piecewise_<false, false, false>()` (in `ckernel_sfpu_exp.h`) calls `_sfpu_exp_(setsgn(in, 0))` to compute `exp(|x|)`, then for lanes where `in < 0`, calls `_sfpu_reciprocal_<2>()` to get `exp(x) = 1/exp(|x|)`.

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` -- all 4 faces of the tile are processed (face 0 through face 3), covering all 1024 elements.
- **Operation invocation**: The params dispatch loops `for (int face = 0; face < 4; face++)`, calling `calculate_selu<false>()` once per face. Each call processes 8 SFPU iterations (ITERATIONS=8), covering 8 sfpi rows x 32 elements = 256 elements per face.
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, `SETRWC` between faces). The ADDR_MOD_7 is configured with all-zero increments on both Wormhole and Blackhole. On Wormhole, `set_addr_mod_base()` sets the base to 1 (selecting ADDR_MOD 4..7); on Blackhole, no addr_mod base remapping is performed, but the same ADDR_MOD_7 with zero increments is used. The actual DEST address advancement is done by `dst_reg++` within the kernel (per iteration) and `SETRWC(CR_D, 8)` x2 between faces (advancing 16 physical rows = 1 face).

### Annotated SFPU Kernel Source

The kernel uses SFPI abstractions (`vFloat`, `dst_reg`, `v_if`/`v_endif`), so Style A (inline-commented source) is used.

```cpp
// File: tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_selu.h
// (WH and BH implementations are identical)

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_selu() { // APPROXIMATION_MODE=false, ITERATIONS=8
    // SELU(x) = scale * (max(0,x) + min(0, alpha*(exp(x)-1)))
    // Implementation notes, see the original file for more details

    constexpr bool SCALE_EN = false;
    constexpr bool SKIP_POSITIVE_CHECK = false;
    constexpr std::uint16_t exp_base_scale_factor = p_sfpu::kCONST_1_FP16B; // 0x3F80 = 1.0 in BF16

    // alpha = 1.6732632... in FP32: 0x3FD63840
    sfpi::vFloat v_alpha = Converter::as_float(0x3FD63840); // SFPLOADI x2 to load 32-bit immediate
    // scale = 1.0507009... in FP32: 0x3F868640
    sfpi::vFloat v_scale = Converter::as_float(0x3F868640); // SFPLOADI x2 to load 32-bit immediate

#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {         // 8 iterations per face
        sfpi::vFloat v = sfpi::dst_reg[0];          // SFPLOAD from DEST

        v_if(v < 0.0f) {                           // SFPSETCC(LT0) + CC push -- enables only negative lanes
            // Negative branch: compute alpha * (exp(x) - 1)
            sfpi::vFloat v_exp = _calculate_exponential_piecewise_<APPROXIMATION_MODE, SCALE_EN, SKIP_POSITIVE_CHECK>(
                v, exp_base_scale_factor);          // See sub-function analysis below
            v = v_alpha * (v_exp - 1.0f);           // SFPMAD: v_exp - 1.0 = v_exp * 1.0 + (-1.0); then SFPMAD: v_alpha * result + 0.0
        }
        v_endif;                                    // CC pop -- restores all-lanes-active state

        // Unconditionally multiply all lanes by scale:
        //   positive: scale * x
        //   negative: scale * alpha * (exp(x) - 1)
        v = v_scale * v;                            // SFPMAD: v_scale * v + 0.0

        sfpi::dst_reg[0] = v;                       // SFPSTORE back to DEST
        sfpi::dst_reg++;                            // advance 1 sfpi row = 2 physical DEST rows = 32 elements
    }
}

template <bool APPROXIMATION_MODE>
inline void selu_init() { // APPROXIMATION_MODE=false
    const std::uint32_t EXP_BASE_SCALE_FACTOR = 0x3F800000; // 1.0f in FP32
    const bool FAST_APPROX = false;
    _init_exponential_<APPROXIMATION_MODE, FAST_APPROX, EXP_BASE_SCALE_FACTOR>();
    // With APPROXIMATION_MODE=false and FAST_APPROX=false, this enters the else branch
    // which calls _init_sfpu_reciprocal_<false>(), setting:
    //   vConstFloatPrgm0 = 0.3232325... (k0 for reciprocal polynomial)
    //   vConstFloatPrgm1 = 1.4545459... (k1 for reciprocal polynomial)
    //   vConstFloatPrgm2 = 2.1212124... (k2 for reciprocal polynomial)
}
```

#### Sub-function: `_calculate_exponential_piecewise_` (non-approximate path)

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/common/inc/sfpu/ckernel_sfpu_exp.h
// Only the else branch (APPROXIMATION_MODE=false) is shown, as that is what SELU uses.

template <bool APPROXIMATION_MODE, bool SCALE_EN, bool SKIP_POSITIVE_CHECK>
inline sfpi::vFloat _calculate_exponential_piecewise_(sfpi::vFloat in, const std::uint16_t exp_base_scale_factor)
{ // APPROXIMATION_MODE=false, SCALE_EN=false, SKIP_POSITIVE_CHECK=false
    sfpi::vFloat result = 0.0f;
    // SCALE_EN=false: skip scaling
    // APPROXIMATION_MODE=false: take the else branch

    result = _sfpu_exp_(sfpi::setsgn(in, 0));   // Compute exp(|x|) via Horner polynomial + repeated squaring

    v_if (in < 0) {                              // SFPSETCC(LT0) -- for lanes where original input was negative
        result = _sfpu_reciprocal_<2>(result);   // exp(x) = 1/exp(|x|) via Newton-Raphson (2 iterations)
    }
    v_endif;

    return result;
}
```

#### Sub-function: `_sfpu_exp_` (Horner polynomial + repeated squaring)

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/common/inc/sfpu/ckernel_sfpu_exp.h

sfpi_inline sfpi::vFloat _sfpu_exp_(sfpi::vFloat val)
{
    // If exponent >= 0, extract it and replace with -1 (exponent bias 126)
    sfpi::vInt exp = exexp(val);                     // SFPEXEXP: extract biased exponent
    v_if (exp >= 0) {                                // SFPSETCC(GTE0) on integer exponent
        val = setexp(val, 126);                      // SFPSETEXP: force exponent to -1 (bias 126), normalizing val to [-1, -0.5) or [0.5, 1)
    }
    v_endif;

    // Run series in Horner form: exp(x) approx = ((x * 0.8373) + 0.863281) * x + 1.0
    sfpi::vFloat tmp = val * sfpi::vConst0p8373 + sfpi::s2vFloat16b(0.863281);  // SFPMAD: val * 0.8373 + 0.863281
    val              = val * tmp + sfpi::vConst1;                                 // SFPMAD: val * tmp + 1.0

    v_if (exp >= 0) {                                // SFPSETCC(GTE0) on integer exponent again
        val = val * val;                             // SFPMAD: val * val + 0.0 (first squaring)
        for (int s_iter = 0; s_iter < 7; s_iter++) {
            exp = exp - 1;                           // SFPIADD: integer decrement
            // Narrow predication on each loop -- only lanes with exp still >= 0 continue squaring
            v_and(exp >= 0);                         // SFPPUSHC + SFPSETCC(GTE0) + SFPPOPC(AND) -- narrows active lanes
            val = val * val;                         // SFPMAD: repeated squaring (CC-guarded)
        }
    }
    v_endif;

    return val;
}
```

#### Sub-function: `_sfpu_reciprocal_<2>` (Newton-Raphson reciprocal, 2 iterations)

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/common/inc/sfpu/ckernel_sfpu_recip.h

template <int max_iter = 2>
sfpi_inline sfpi::vFloat _sfpu_reciprocal_(const sfpi::vFloat in) // max_iter=2
{
    // Scale input to [-2, -1) by combining sign/exponent of -1.0 with mantissa of input
    sfpi::vFloat negative_x = sfpi::setman(sfpi::vConstNeg1, sfpi::reinterpret<sfpi::vInt>(in)); // SFPSETMAN

    // Quadratic initial estimate: y = k2 + k0 * negative_x (first part)
    // vConstFloatPrgm0 = k0 = 0.3232325, vConstFloatPrgm1 = k1 = 1.4545459
    sfpi::vFloat y = sfpi::vConstFloatPrgm1 + sfpi::vConstFloatPrgm0 * negative_x; // SFPMAD: k0 * neg_x + k1

    // Compute scale factor: scale.Exp = ~in.Exp (via SFPNOT)
    sfpi::vInt scale_bits = ~sfpi::reinterpret<sfpi::vUInt>(in); // SFPNOT: bitwise complement

    // Continue quadratic estimate: y = k2 + y * negative_x
    // vConstFloatPrgm2 = k2 = 2.1212124
    y = sfpi::vConstFloatPrgm2 + y * negative_x; // SFPMAD: y * neg_x + k2

    // Clear mantissa of scale factor
    sfpi::vFloat scale = sfpi::setman(sfpi::reinterpret<sfpi::vFloat>(scale_bits), 0); // SFPSETMAN

    // First Newton-Raphson iteration: t = 1.0 + negative_x * y (= 1.0 - x*y)
    sfpi::vFloat t = sfpi::vConst1 + negative_x * y; // SFPMAD: neg_x * y + 1.0

    // Adjust scale: scale *= 0.5 (decrements exponent by 1, giving 254-in.Exp)
    scale *= 0.5f; // SFPMAD: scale * 0.5 + 0.0

    // Continue NR: y = y + y * t
    y = y + y * t; // SFPMAD: y * t + y

    // max_iter=2: second Newton-Raphson iteration
    t = sfpi::vConst1 + negative_x * y; // SFPMAD: neg_x * y + 1.0
    y = y + y * t;                       // SFPMAD: y * t + y

    // Apply scaling factor and restore sign
    y = y * scale;          // SFPMAD: y * scale + 0.0
    y = sfpi::setsgn(y, in); // SFPSETSGN: copy sign from original input

    return y;
}
```

### SFPU Instructions Used

| Instruction | Context | Description |
|-------------|---------|-------------|
| `SFPLOAD` | `dst_reg[0]` read | Loads 32 elements (2 physical DEST rows) into an LREG for SFPU processing |
| `SFPSTORE` | `dst_reg[0] = v` write | Stores an LREG value back to 32 elements in DEST |
| `SFPLOADI` | `Converter::as_float(...)`, constant loading | Loads a 16-bit immediate into an LREG (two calls needed for a 32-bit FP constant) |
| `SFPMAD` | All float arithmetic (`*`, `+`, `-`) | Fused multiply-add: `VD = VA * VB + VC`. Used for Horner polynomial, Newton-Raphson, and all float addition/multiplication in the kernel |
| `SFPSETCC` | `v_if(v < 0.0f)`, `v_if(exp >= 0)` | Sets per-lane CC.Res based on comparison (LT0 for negative check, GTE0 for exponent check) |
| `SFPENCC` | `v_if` / `v_endif` preamble/epilogue | Enables/disables condition code masking for predicated execution |
| `SFPPUSHC` | `v_if` nesting, `v_and` | Pushes current CC state onto the CC stack for nested conditional blocks |
| `SFPPOPC` | `v_endif`, `v_and` | Pops CC state from the CC stack, restoring prior conditional context |
| `SFPCOMPC` | `v_else` (if used internally) | Complements CC.Res for else-branches (used internally by `v_elseif` in `_calculate_exponential_piecewise_`) |
| `SFPEXEXP` | `exexp(val)` in `_sfpu_exp_` | Extracts the biased exponent field from a floating-point value |
| `SFPSETEXP` | `setexp(val, 126)` in `_sfpu_exp_` | Sets the exponent field of a floating-point value (used to normalize input to [-1, 1) range) |
| `SFPSETMAN` | `setman(...)` in `_sfpu_reciprocal_` | Sets the mantissa field of a float (used to scale input and clear scale mantissa) |
| `SFPNOT` | `~reinterpret<vUInt>(in)` in `_sfpu_reciprocal_` | Bitwise NOT of a register (used to compute `255 - exponent` for scale factor) |
| `SFPSETSGN` | `setsgn(y, in)` in `_sfpu_reciprocal_` | Sets the sign bit of the result to match the input sign |
| `SFPIADD` | `exp = exp - 1` in `_sfpu_exp_` | Integer subtraction (2's complement). Used to decrement the exponent counter in repeated squaring loop |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **LREG0-3** | General-purpose working registers. Used by SFPI compiler for `vFloat`/`vInt` temporaries: `v`, `v_exp`, `v_alpha`, `v_scale`, `result`, `y`, `t`, `negative_x`, `scale`, `exp`, `tmp`, etc. The SFPI compiler manages LREG allocation automatically. |
| **Programmable Constant 0** (`vConstFloatPrgm0`) | Set to `0.3232325...` by `_init_sfpu_reciprocal_<false>()` -- this is `k0` in the quadratic reciprocal initial estimate |
| **Programmable Constant 1** (`vConstFloatPrgm1`) | Set to `1.4545459...` by `_init_sfpu_reciprocal_<false>()` -- this is `k1` in the quadratic reciprocal initial estimate |
| **Programmable Constant 2** (`vConstFloatPrgm2`) | Set to `2.1212124...` by `_init_sfpu_reciprocal_<false>()` -- this is `k2` in the quadratic reciprocal initial estimate |
| **Fixed Constant `vConst1`** | Value `1.0f` -- used in Horner polynomial (`val * tmp + 1.0`) and Newton-Raphson (`1.0 + negative_x * y`) |
| **Fixed Constant `vConstNeg1`** | Value `-1.0f` -- used in `_sfpu_reciprocal_` to combine sign/exponent of -1.0 with input mantissa |
| **Fixed Constant `vConst0p8373`** | Value `0.8373f` -- coefficient in the Horner series for `_sfpu_exp_` |
| **DEST register** | Source and destination for tile data. Each iteration processes 32 elements (2 physical rows x 16 columns). SFPLOAD reads from DEST into LREGs; SFPSTORE writes results back. |
| **CC stack** | Used for nested `v_if`/`v_endif` blocks. The SELU kernel has up to 3 levels of nesting: outer `v_if(v < 0)` in `calculate_selu`, inner `v_if(in < 0)` in `_calculate_exponential_piecewise_`, and nested `v_if(exp >= 0)` with `v_and` narrowing in `_sfpu_exp_`. |

### Address Mode Configuration

The ADDR_MOD configuration for SELU is the standard unary SFPU configuration, identical across Wormhole and Blackhole:

| Property | Value | Notes |
|----------|-------|-------|
| **ADDR_MOD_7** | `srca.incr=0, srcb.incr=0, dest.incr=0` | Configured by `eltwise_unary_sfpu_configure_addrmod<SfpuType::selu>()` in `llk_math_eltwise_unary_sfpu.h`. All increments are zero because DEST address advancement is handled explicitly by `dst_reg++` (per SFPU iteration) and `SETRWC` (per face), not by auto-increment. |
| **WH addr_mod base** | Base = 1 (active ADDR_MOD range 4..7) | Set by `math::set_addr_mod_base()` in the WH params dispatch. ADDR_MOD_7 is in the active range. |
| **BH addr_mod base** | No base remapping | Blackhole does not use `set_addr_mod_base()`; ADDR_MOD_7 is accessed directly. |

No additional ADDR_MOD slots (e.g., ADDR_MOD_6) are configured for this operation because SELU is not a topk, typecast, or min/max operation.

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Determine compute kernel path, SFPU_OP_CHAIN_0 expansion, and approximation mode for SELU
   **Key Findings**: SELU uses `eltwise_sfpu.cpp`, expands to `selu_tile_init()` / `selu_tile(idst)`, `math_approx_mode=false` (default case), include guard `SFPU_OP_SELU_INCLUDE`

2. **File**: `tt_metal/hw/inc/api/compute/eltwise_unary/selu.h`
   **Reason**: API header defining `selu_tile()` and `selu_tile_init()` entry points
   **Key Findings**: Uses `SFPU_UNARY_NO_PARAM_KERNEL_FN(calculate_selu, RC, APPROX, idst)` and `SFPU_INIT_KERNEL_CALL(selu, ckernel::sfpu::selu_init, APPROX)`

3. **File**: `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_selu.h`
   **Reason**: LLK dispatch layer bridging API to ckernel
   **Key Findings**: WH and BH versions are identical. Calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(calculate_selu<APPROXIMATE>, dst_index, vector_mode)`

4. **File**: `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_selu.h`
   **Reason**: Core SFPU kernel implementation
   **Key Findings**: WH and BH versions are identical. Implements SELU formula using SFPI abstractions. For negative lanes: calls `_calculate_exponential_piecewise_` then computes `alpha * (exp(x) - 1)`. Unconditionally multiplies by `scale`. Constants: `alpha=0x3FD63840` (1.6732632...), `scale=0x3F868640` (1.0507009...)

5. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_exp.h`
   **Reason**: Shared exponential implementation called by SELU
   **Key Findings**: `_calculate_exponential_piecewise_<false,false,false>` takes the non-approximate path: calls `_sfpu_exp_()` (Horner polynomial + repeated squaring on |x|), then `_sfpu_reciprocal_<2>()` for negative inputs. `_init_exponential_<false,false,0x3F800000>()` initializes reciprocal constants only.

6. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_recip.h`
   **Reason**: Reciprocal implementation called from the non-approximate exp path
   **Key Findings**: `_sfpu_reciprocal_<2>()` uses quadratic initial estimate with programmable constants (k0, k1, k2 set by `_init_sfpu_reciprocal_`) followed by 2 Newton-Raphson iterations. Achieves float32-precision reciprocal.

7. **File**: `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Parameters dispatch layer that manages face iteration and DEST addressing
   **Key Findings**: VectorMode::RC loops over 4 faces. WH version calls `set_addr_mod_base()` inline; BH version delegates to `_llk_math_eltwise_unary_sfpu_start_` which does not set addr_mod base.

8. **File**: `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: ADDR_MOD configuration and SFPU init
   **Key Findings**: ADDR_MOD_7 configured with all-zero increments for SELU (no special-case ADDR_MOD_6 configuration needed)

9. **File**: `.claude/references/sfpu-hardware-model.md`
   **Reason**: SFPU hardware model reference for tile geometry, DEST layout, instruction semantics
   **Key Findings**: Standard tile/face geometry (32x32 tile, 4 faces of 16x16, 8 iterations per face), stride-2 addressing model, SFPMAD used for all float arithmetic

10. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h`
    **Reason**: Macro definitions for SFPU dispatch patterns
    **Key Findings**: `SFPU_UNARY_NO_PARAM_KERNEL_FN` expands to `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::FN<APPROXIMATE>, DST_IDX, (int)VectorMode::MODE)`. `SFPU_INIT_KERNEL_CALL` calls `llk_math_eltwise_unary_sfpu_init` then the init callback.
