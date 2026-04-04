## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### Unary Dispatch Summary
- **UnaryOpType**: `SELU`
- **Compute kernel**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`
- **SFPU_OP_CHAIN_0 expansion**: `selu_tile_init(); selu_tile(0);`

#### Approximation Mode Resolution
There are TWO independent controls for approximation. Both are reported below.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(SELU)` in `unary_op_utils.cpp` -- currently returns `false` for all ops (switch has only a `default: return false` case) |
| Template parameter (SFPU_OP_CHAIN) | none | `get_op_init_and_func_default()` -- non-parameterized case: `selu_tile_init()` / `selu_tile(idst)` with default template args; `APPROX` is a `constexpr bool` generated from `math_approx_mode` |
| Effective SFPU path | `APPROXIMATION_MODE=false`: `_calculate_exponential_piecewise_` takes the `else` branch calling `_sfpu_exp_(setsgn(in, 0))` then `_sfpu_reciprocal_<2>(result)` for negative inputs | `ckernel_sfpu_exp.h` lines 388-397: `else { result = _sfpu_exp_(setsgn(in, 0)); v_if (in < 0) { result = _sfpu_reciprocal_<2>(result); } v_endif; }` |

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/selu.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_selu.h` |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_selu.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |

### Call Chain

1. **Compute kernel** (`eltwise_sfpu.cpp`): The `SFPU_OP_CHAIN_0` macro expands to `selu_tile_init(); selu_tile(0);`. The init call runs once; the tile call runs per tile.
2. **API Header** (`selu.h`): `selu_tile(idst)` expands via the `SFPU_UNARY_NO_PARAM_KERNEL_FN` macro to `_llk_math_eltwise_unary_sfpu_params_<APPROX>(ckernel::sfpu::calculate_selu<APPROX>, idst, (int)VectorMode::RC)`. Similarly `selu_tile_init()` expands via `SFPU_INIT_KERNEL_CALL` to `llk_math_eltwise_unary_sfpu_init<SfpuType::selu, APPROX>(ckernel::sfpu::selu_init<APPROX>)`.
3. **LLK Dispatch** (`llk_math_eltwise_unary_sfpu_selu.h`): `llk_math_eltwise_unary_sfpu_selu<APPROXIMATE>(dst_index, VectorMode::RC)` calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::calculate_selu<APPROXIMATE>, dst_index, vector_mode)`. The init function calls `llk_math_eltwise_unary_sfpu_init<SfpuType::selu, APPROXIMATE>(ckernel::sfpu::selu_init<APPROXIMATE>)`.
4. **Parameters Dispatch** (`llk_math_eltwise_unary_sfpu_params.h`): For `VectorMode::RC`, iterates over 4 faces, calling `calculate_selu<APPROXIMATE>()` per face, with `SETRWC` between faces.
5. **Core SFPU** (`ckernel_sfpu_selu.h`): `calculate_selu<false>()` runs 8 iterations per face. For each 32-element chunk: loads from DEST, conditionally computes `alpha * (exp(x) - 1)` for negative lanes, then unconditionally multiplies by `scale`.

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` -- all 4 faces of the tile are processed (full 32x32 tile).
- **Operation invocation**: The params dispatch calls `calculate_selu<APPROXIMATE>()` once per face (4 times total). Each invocation processes 8 sfpi iterations (ITERATIONS=8 default), covering one full 16x16 face.
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, `SETRWC` between faces). The address mode is `ADDR_MOD_7` on both Wormhole and Blackhole, configured with all-zero increments (`srca.incr=0, srcb.incr=0, dest.incr=0`). SFPU addressing uses `dst_reg++` within the kernel loop for intra-face advancement and `TTI_SETRWC` with stride 8 (x2) between faces.

### Annotated SFPU Kernel Source

The core SFPU implementation uses SFPI abstractions (`vFloat`, `dst_reg`, `v_if`/`v_endif`, `Converter`). The Wormhole B0 and Blackhole implementations are identical.

```cpp
// File: tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_selu.h

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_selu() { // APPROXIMATION_MODE=false, ITERATIONS=8
    // SELU(x) = scale * (max(0,x) + min(0, alpha*(exp(x)-1)))
    // Implementation notes, see the original file for more details

    constexpr bool SCALE_EN = false;               // No pre-scaling of exp input
    constexpr bool SKIP_POSITIVE_CHECK = false;     // Check for overflow in exp
    constexpr std::uint16_t exp_base_scale_factor = p_sfpu::kCONST_1_FP16B; // 0x3F80 = 1.0 in BF16

    // alpha = 1.6732632... in FP32: 0x3FD63840
    sfpi::vFloat v_alpha = Converter::as_float(0x3FD63840); // SFPLOADI x2 (lo16 + hi16)
    // scale = 1.0507009... in FP32: 0x3F868640
    sfpi::vFloat v_scale = Converter::as_float(0x3F868640); // SFPLOADI x2 (lo16 + hi16)

#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {          // 8 iterations per face
        sfpi::vFloat v = sfpi::dst_reg[0];          // SFPLOAD from DEST

        v_if(v < 0.0f) {                           // SFPSETCC (LT0) + CC enable
            // Negative branch: compute alpha * (exp(x) - 1)
            sfpi::vFloat v_exp = _calculate_exponential_piecewise_<APPROXIMATION_MODE, SCALE_EN, SKIP_POSITIVE_CHECK>(
                v, exp_base_scale_factor);          // exp(x) via _sfpu_exp_ + _sfpu_reciprocal_<2>
            v = v_alpha * (v_exp - 1.0f);           // SFPMAD (sub 1.0), SFPMAD (mul alpha)
        }
        v_endif;                                    // CC restore

        // Unconditionally multiply all lanes by scale:
        //   positive: scale * x
        //   negative: scale * alpha * (exp(x) - 1)
        v = v_scale * v;                            // SFPMAD (mul scale)

        sfpi::dst_reg[0] = v;                       // SFPSTORE to DEST
        sfpi::dst_reg++;                            // Advance 1 sfpi row = 2 physical DEST rows
    }
}

template <bool APPROXIMATION_MODE>
inline void selu_init() { // APPROXIMATION_MODE=false
    const std::uint32_t EXP_BASE_SCALE_FACTOR = 0x3F800000; // 1.0f in FP32
    const bool FAST_APPROX = false;
    _init_exponential_<APPROXIMATION_MODE, FAST_APPROX, EXP_BASE_SCALE_FACTOR>();
    // With APPROXIMATION_MODE=false, FAST_APPROX=false: takes the else branch,
    // calls _init_sfpu_reciprocal_<false>() which sets:
    //   vConstFloatPrgm0 = 0.3232325... (k0 for recip polynomial)
    //   vConstFloatPrgm1 = 1.4545459... (k1 for recip polynomial)
    //   vConstFloatPrgm2 = 2.1212124... (k2 for recip polynomial)
}
```

The sub-functions called by `calculate_selu` are from shared SFPU libraries:

**`_calculate_exponential_piecewise_` (APPROXIMATION_MODE=false path)**:
```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_exp.h

template <bool APPROXIMATION_MODE, bool SCALE_EN, bool SKIP_POSITIVE_CHECK>
inline sfpi::vFloat _calculate_exponential_piecewise_(sfpi::vFloat in, const std::uint16_t exp_base_scale_factor)
{ // APPROXIMATION_MODE=false, SCALE_EN=false, SKIP_POSITIVE_CHECK=false
    sfpi::vFloat result = 0.0f;
    // SCALE_EN=false: skip scaling
    // APPROXIMATION_MODE=false: take the else branch
    // else branch:
    result = _sfpu_exp_(sfpi::setsgn(in, 0));       // exp(|x|) via Horner series + repeated squaring

    v_if (in < 0)                                    // For negative inputs...
    {
        result = _sfpu_reciprocal_<2>(result);       // ...compute 1/exp(|x|) = exp(x)
    }
    v_endif;

    return result;
}
```

**`_sfpu_exp_` (Horner series + repeated squaring)**:
```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_exp.h

sfpi_inline sfpi::vFloat _sfpu_exp_(sfpi::vFloat val)
{
    // If exponent is > -1 extract it and replace with -1
    sfpi::vInt exp = exexp(val);                     // SFPEXEXP: extract exponent
    v_if (exp >= 0)
    {
        val = setexp(val, 126);                      // SFPSETEXP: force exponent to -1 (bias 127 - 1 = 126)
    }
    v_endif;

    // Run series in Horner form: val = val * (val * 0.8373 + 0.8633) + 1.0
    sfpi::vFloat tmp = val * sfpi::vConst0p8373 + sfpi::s2vFloat16b(0.863281); // SFPMAD
    val              = val * tmp + sfpi::vConst1;                                // SFPMAD

    v_if (exp >= 0)
    {
        val = val * val;                             // SFPMAD: square once unconditionally
        for (int s_iter = 0; s_iter < 7; s_iter++)
        {
            exp = exp - 1;                           // SFPIADD: integer decrement
            v_and(exp >= 0);                         // Narrow predication
            val = val * val;                         // SFPMAD: repeated squaring
        }
    }
    v_endif;

    return val;
}
```

**`_sfpu_reciprocal_<2>` (Newton-Raphson with 2 iterations)**:
```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_recip.h

template <int max_iter = 2>
sfpi_inline sfpi::vFloat _sfpu_reciprocal_(const sfpi::vFloat in)
{ // max_iter=2 (float32 precision)
    // Scale input to [1,2) and negate: negative_x = -setman(vConstNeg1, man(in))
    sfpi::vFloat negative_x = sfpi::setman(sfpi::vConstNeg1, sfpi::reinterpret<sfpi::vInt>(in)); // SFPSETMAN

    // Quadratic initial estimate: y = k2 - k1*x + k0*x^2
    sfpi::vFloat y = sfpi::vConstFloatPrgm1 + sfpi::vConstFloatPrgm0 * negative_x; // SFPMAD

    // Scale factor: ~in gives 255-in.Exp, then clear mantissa
    sfpi::vInt scale_bits = ~sfpi::reinterpret<sfpi::vUInt>(in);                    // SFPNOT

    // Continue quadratic estimate
    y = sfpi::vConstFloatPrgm2 + y * negative_x;                                    // SFPMAD

    // Set mantissa to zero for scale factor
    sfpi::vFloat scale = sfpi::setman(sfpi::reinterpret<sfpi::vFloat>(scale_bits), 0); // SFPSETMAN

    // Newton-Raphson iteration 1: t = 1.0 + (-x) * y
    sfpi::vFloat t = sfpi::vConst1 + negative_x * y;                                // SFPMAD

    // Scale adjustment: scale *= 0.5
    scale *= 0.5f;                                                                    // SFPMAD

    // y = y + y * t
    y = y + y * t;                                                                    // SFPMAD

    // Newton-Raphson iteration 2 (max_iter > 1): t = 1.0 + (-x) * y; y = y + y * t
    t = sfpi::vConst1 + negative_x * y;                                              // SFPMAD
    y = y + y * t;                                                                    // SFPMAD

    // Apply scaling and restore sign
    y = y * scale;                                                                    // SFPMAD
    y = sfpi::setsgn(y, in);                                                          // SFPSETSGN

    return y;
}
```

### SFPU Instructions Used

| Instruction | Context | Description |
|-------------|---------|-------------|
| `SFPLOAD` | `dst_reg[0]` read | Load 32 elements from DEST into LREG for processing |
| `SFPSTORE` | `dst_reg[0] = v` write | Store 32 elements from LREG back to DEST |
| `SFPLOADI` | `Converter::as_float()`, constant loading | Load 16-bit immediate into LREG (used for alpha, scale, and polynomial coefficients) |
| `SFPMAD` | `v * v_scale`, `val * tmp + vConst1`, etc. | Fused multiply-add; used for all float arithmetic (adds emitted as `a * 1.0 + b`, multiplies as `a * b + 0.0`) |
| `SFPSETCC` | `v_if(v < 0.0f)`, `v_if(exp >= 0)` | Set per-lane condition code based on comparison (LT0, GTE0) |
| `SFPENCC` | `v_if` / `v_endif` | Enable/disable per-lane predicated execution |
| `SFPPUSHC` | `v_if` nesting | Push condition code state onto CC stack for nested conditionals |
| `SFPPOPC` | `v_endif` | Pop condition code state from CC stack |
| `SFPCOMPC` | implicit in `v_else` / `v_elseif` within exp | Complement CC for else-branch logic |
| `SFPEXEXP` | `exexp(val)` in `_sfpu_exp_` | Extract exponent field from float value |
| `SFPSETEXP` | `setexp(val, 126)` in `_sfpu_exp_` | Set exponent field of float value |
| `SFPSETMAN` | `setman()` in `_sfpu_reciprocal_` | Set mantissa field of float value (used to scale input to [1,2) and clear mantissa for scale factor) |
| `SFPSETSGN` | `setsgn(in, 0)` in piecewise, `setsgn(y, in)` in reciprocal | Set sign bit of float value |
| `SFPNOT` | `~reinterpret<vUInt>(in)` in `_sfpu_reciprocal_` | Bitwise NOT to compute `255 - exponent` for reciprocal scale factor |
| `SFPIADD` | `exp = exp - 1` in `_sfpu_exp_` | Integer subtract used for loop counter in repeated squaring |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **LREG0-3** | General-purpose working registers. Used for intermediate values: input value `v`, exponential result `v_exp`, reciprocal intermediates (`negative_x`, `y`, `t`, `scale`), and arithmetic temporaries. The SFPI compiler allocates these dynamically. |
| **LREG4-7** | Additional general-purpose registers available for spills and complex sub-expressions (e.g., the reciprocal function uses many simultaneous live values). LREG7 may be used for SFPMAD indirect addressing if the compiler chooses. |
| **DEST** | Source and destination for tile data. Each sfpi iteration accesses 2 physical DEST rows (32 elements) via `dst_reg[0]`. Read at loop start (`SFPLOAD`), written at loop end (`SFPSTORE`). |
| **vConstFloatPrgm0** | Programmable constant register 0. Set by `_init_sfpu_reciprocal_<false>()` to `0.3232325...` (k0 coefficient for reciprocal quadratic estimate). |
| **vConstFloatPrgm1** | Programmable constant register 1. Set to `1.4545459...` (k1 coefficient for reciprocal quadratic estimate). |
| **vConstFloatPrgm2** | Programmable constant register 2. Set to `2.1212124...` (k2 coefficient for reciprocal quadratic estimate). |
| **vConst0p8373** | Fixed constant register. Value `0.8373` -- used by `_sfpu_exp_` in the Horner series first coefficient. |
| **vConst1** | Fixed constant register. Value `1.0` -- used as addend in `_sfpu_exp_` and Newton-Raphson iterations. |
| **vConstNeg1** | Fixed constant register. Value `-1.0` -- used by `_sfpu_reciprocal_` to construct negative scaled input via `setman`. |
| **CC stack** | Per-lane condition code stack (8-entry). Used for nested `v_if`/`v_endif` blocks: outer `v_if(v < 0)` in `calculate_selu`, inner `v_if(in < 0)` in `_calculate_exponential_piecewise_`, and `v_if(exp >= 0)` / `v_and(exp >= 0)` in `_sfpu_exp_`. |

### Address Mode Configuration

The address mode for SELU is configured in `eltwise_unary_sfpu_configure_addrmod<SfpuType::selu>()`. Since `SfpuType::selu` does not match any special case (`topk_local_sort`, `typecast`, `reciprocal`, min/max variants), only the default `ADDR_MOD_7` is configured.

**Wormhole B0 and Blackhole (identical)**:

| ADDR_MOD | srca.incr | srcb.incr | dest.incr | Purpose |
|----------|-----------|-----------|-----------|---------|
| `ADDR_MOD_7` | 0 | 0 | 0 | Default SFPU address mode; no auto-increment. DEST advancement is handled explicitly by `dst_reg++` within the SFPI kernel loop and `TTI_SETRWC` between faces in the params dispatch. |

The address mode is set once during `_llk_math_eltwise_unary_sfpu_init_<SfpuType::selu>()` via `eltwise_unary_sfpu_configure_addrmod<SfpuType::selu>()`, and activated during tile processing by `math::set_addr_mod_base()` in the params dispatch layer.

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Determine compute kernel path, SFPU_OP_CHAIN_0 expansion, and approximation mode for SELU
   **Key Findings**: SELU uses `eltwise_sfpu.cpp`, expands to `selu_tile_init(); selu_tile(idst);`, include guard `SFPU_OP_SELU_INCLUDE`, approx mode `false` (default case)

2. **File**: `tt_metal/hw/inc/api/compute/eltwise_unary/selu.h`
   **Reason**: API header defining `selu_tile()` and `selu_tile_init()`
   **Key Findings**: Uses `SFPU_UNARY_NO_PARAM_KERNEL_FN(calculate_selu, RC, APPROX, idst)` and `SFPU_INIT_KERNEL_CALL(selu, ckernel::sfpu::selu_init, APPROX)`

3. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_selu.h`
   **Reason**: LLK dispatch layer bridging API to ckernel
   **Key Findings**: Calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::calculate_selu<APPROXIMATE>, ...)` for tile processing and `llk_math_eltwise_unary_sfpu_init<SfpuType::selu, APPROXIMATE>(ckernel::sfpu::selu_init<APPROXIMATE>)` for init

4. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_selu.h`
   **Reason**: Core SFPU implementation of SELU
   **Key Findings**: SFPI-based kernel. Loads from DEST, conditionally computes `alpha * (exp(x) - 1)` for negative lanes via `_calculate_exponential_piecewise_`, unconditionally multiplies by scale. WH and BH implementations identical.

5. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_exp.h`
   **Reason**: Shared exponential implementation used by SELU
   **Key Findings**: `_calculate_exponential_piecewise_<false, false, false>()` takes the else branch: computes `exp(|x|)` via `_sfpu_exp_` (Horner + repeated squaring), then `1/exp(|x|)` via `_sfpu_reciprocal_<2>` for negative inputs. `_init_exponential_<false, false, 0x3F800000>()` takes the else branch calling `_init_sfpu_reciprocal_<false>()`

6. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_recip.h`
   **Reason**: Shared reciprocal implementation used by the non-approximate exponential path
   **Key Findings**: `_sfpu_reciprocal_<2>()` uses quadratic initial estimate + 2 Newton-Raphson iterations for float32 precision. Init sets programmable constants vConstFloatPrgm0/1/2 to polynomial coefficients.

7. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Parameters dispatch layer that drives per-face SFPU invocation
   **Key Findings**: For VectorMode::RC, loops 4 faces calling the SFPU function once per face with TTI_SETRWC between faces

8. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: Base LLK dispatch with ADDR_MOD configuration and init
   **Key Findings**: ADDR_MOD_7 configured with all-zero increments for SfpuType::selu (no special case match)

9. **File**: `tt_metal/jit_build/genfiles.cpp` (line 394)
   **Reason**: Confirm how `APPROX` compile-time constant is generated
   **Key Findings**: `constexpr bool APPROX = {math_approx_mode};` generated from `ComputeConfig.math_approx_mode`
