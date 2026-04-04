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
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(SELU)` in `unary_op_utils.cpp` -- falls through to `default: return false` |
| Template parameter (SFPU_OP_CHAIN) | `false` (via `APPROX` constexpr) | `get_op_init_and_func_default()` returns `selu_tile_init()` / `selu_tile(idst)` -- non-parameterized, uses `APPROX` which is generated as `constexpr bool APPROX = false` from `math_approx_mode` |
| Effective SFPU path | Non-approximate: `_calculate_exponential_piecewise_` takes the `else` branch (line 388-397 of `ckernel_sfpu_exp.h`), which uses `_sfpu_exp_` (Horner polynomial) + `_sfpu_reciprocal_<2>` (2-iteration Newton-Raphson) for negative inputs | `if constexpr (APPROXIMATION_MODE)` is `false`, so the accurate exp path is used |

### SFPU Abstraction Layers
| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/selu.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_selu.h` |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_selu.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |

### Call Chain
1. The compute kernel (`eltwise_sfpu.cpp`) invokes the `SFPU_OP_CHAIN_0` macro, which first calls `selu_tile_init()` during initialization and then `selu_tile(0)` per tile.
2. `selu_tile(idst)` (in `selu.h`) expands via `SFPU_UNARY_NO_PARAM_KERNEL_FN(calculate_selu, RC, APPROX, idst)` to `_llk_math_eltwise_unary_sfpu_params_<false>(ckernel::sfpu::calculate_selu<false>, idst, (int)VectorMode::RC)`.
3. `_llk_math_eltwise_unary_sfpu_params_` (in `llk_math_eltwise_unary_sfpu_params.h`) sets up DEST addressing, stalls for SFPU readiness, then invokes `calculate_selu<false>()` once per face (4 faces for RC mode), advancing DEST address between faces via `TTI_SETRWC` (WH) or `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_` (BH).
4. `selu_tile_init()` (in `selu.h`) expands via `SFPU_INIT_KERNEL_CALL(selu, ckernel::sfpu::selu_init, APPROX)` to `llk_math_eltwise_unary_sfpu_init<SfpuType::selu, false>(ckernel::sfpu::selu_init<false>)`.
5. `llk_math_eltwise_unary_sfpu_init` (in `llk_math_eltwise_unary_sfpu_init.h`) first calls `_llk_math_eltwise_unary_sfpu_init_<SfpuType::selu>()` to configure addr_mod and SFPU config, then calls the callback `selu_init<false>()`.
6. `selu_init<false>()` (in `ckernel_sfpu_selu.h`) calls `_init_exponential_<false, false, 0x3F800000>()`, which takes the final `else` branch and calls `_init_sfpu_reciprocal_<false>()` to program the reciprocal polynomial coefficients into programmable constant registers.

### Parameters Dispatch Summary
- **Vector mode**: `VectorMode::RC` -- all 4 faces of the tile are processed (the standard mode for element-wise unary operations).
- **Operation invocation**: The params dispatch calls `calculate_selu<false>()` once per face in a loop of 4 iterations. Each call to `calculate_selu` internally iterates 8 times (ITERATIONS=8), processing one full face (8 iterations x 32 elements = 256 elements).
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, `TTI_SETRWC` between faces on WH / `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_` on BH). ADDR_MOD_7 is configured with all increments = 0 (srca=0, srcb=0, dest=0) since SFPI abstractions manage addressing through `dst_reg++`.

### Annotated SFPU Kernel Source
The kernel uses SFPI abstractions (`sfpi::vFloat`, `sfpi::dst_reg`, `v_if`/`v_endif`). Style A is used.

```cpp
// File: tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_selu.h

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_selu() { // APPROXIMATION_MODE=false, ITERATIONS=8
    // Implementation notes, see the original file for more details

    constexpr bool SCALE_EN = false;
    constexpr bool SKIP_POSITIVE_CHECK = false;
    constexpr std::uint16_t exp_base_scale_factor = p_sfpu::kCONST_1_FP16B; // 0x3F80 = 1.0 in BF16

    // alpha = 1.6732632... in FP32: 0x3FD63840
    sfpi::vFloat v_alpha = Converter::as_float(0x3FD63840); // SFPLOADI x2 -> SFPMAD to load into vFloat
    // scale = 1.0507009... in FP32: 0x3F868640
    sfpi::vFloat v_scale = Converter::as_float(0x3F868640); // SFPLOADI x2 -> SFPMAD to load into vFloat

#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) { // 8 iterations per face
        sfpi::vFloat v = sfpi::dst_reg[0]; // SFPLOAD from current DEST row pair

        v_if(v < 0.0f) { // SFPSETCC(LT0) + SFPENCC: enables CC, masks lanes where v >= 0
            // Negative branch: compute alpha * (exp(x) - 1)
            sfpi::vFloat v_exp = _calculate_exponential_piecewise_<APPROXIMATION_MODE, SCALE_EN, SKIP_POSITIVE_CHECK>(
                v, exp_base_scale_factor); // Full accuracy exp: _sfpu_exp_(|x|), then reciprocal for negative x
            v = v_alpha * (v_exp - 1.0f); // SFPMAD(v_exp, 1.0, -1.0) then SFPMAD(v_alpha, result, 0.0)
        }
        v_endif; // SFPENCC: restores all-lanes-active state

        // Unconditionally multiply all lanes by scale:
        //   positive: scale * x
        //   negative: scale * alpha * (exp(x) - 1)
        v = v_scale * v; // SFPMAD(v_scale, v, 0.0)

        sfpi::dst_reg[0] = v; // SFPSTORE to current DEST row pair
        sfpi::dst_reg++;      // advance 1 sfpi row = 2 physical DEST rows
    }
}

template <bool APPROXIMATION_MODE>
inline void selu_init() { // APPROXIMATION_MODE=false
    const std::uint32_t EXP_BASE_SCALE_FACTOR = 0x3F800000; // 1.0f in FP32
    const bool FAST_APPROX = false;
    _init_exponential_<APPROXIMATION_MODE, FAST_APPROX, EXP_BASE_SCALE_FACTOR>();
    // With APPROXIMATION_MODE=false and FAST_APPROX=false, this hits the final else branch
    // which calls _init_sfpu_reciprocal_<false>(), programming:
    //   vConstFloatPrgm0 = 0.3232325... (k0 for reciprocal polynomial)
    //   vConstFloatPrgm1 = 1.4545459... (k1 for reciprocal polynomial)
    //   vConstFloatPrgm2 = 2.1212124... (k2 for reciprocal polynomial)
}
```

The `_calculate_exponential_piecewise_` function called by SELU (with `APPROXIMATION_MODE=false`, `SCALE_EN=false`, `SKIP_POSITIVE_CHECK=false`):

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_exp.h

template <bool APPROXIMATION_MODE, bool SCALE_EN, bool SKIP_POSITIVE_CHECK>
inline sfpi::vFloat _calculate_exponential_piecewise_(sfpi::vFloat in, const std::uint16_t exp_base_scale_factor)
{ // APPROXIMATION_MODE=false, SCALE_EN=false, SKIP_POSITIVE_CHECK=false
    sfpi::vFloat result = 0.0f; // SFPLOADI: initialize result to 0

    // SCALE_EN=false: no scaling applied (skip)
    // APPROXIMATION_MODE=false: take the else branch (accurate path)

    result = _sfpu_exp_(sfpi::setsgn(in, 0)); // Compute exp(|x|) via Horner polynomial

    v_if (in < 0) // SFPSETCC(LT0): mask lanes where in >= 0
    {
        result = _sfpu_reciprocal_<2>(result); // For negative inputs: exp(x) = 1/exp(|x|)
    }
    v_endif; // SFPENCC: restore all-lanes-active

    return result;
}
```

The `_sfpu_exp_` function (Horner polynomial for exp on positive inputs):

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_exp.h

sfpi_inline sfpi::vFloat _sfpu_exp_(sfpi::vFloat val)
{
    // If exponent is > -1 extract it and replace with -1
    sfpi::vInt exp = exexp(val);   // SFPEXEXP: extract exponent field
    v_if (exp >= 0)                // SFPSETCC(GTE0)
    {
        val = setexp(val, 126);    // SFPSETEXP: clamp exponent to -1 (bias 126), range-reduce to [-1, 0)
    }
    v_endif;

    // Run series in Horner form: approximates 2^val for val in [-1, 0)
    sfpi::vFloat tmp = val * sfpi::vConst0p8373 + sfpi::s2vFloat16b(0.863281); // SFPMAD
    val              = val * tmp + sfpi::vConst1;                                // SFPMAD

    v_if (exp >= 0) // SFPSETCC(GTE0): only square for large exponents
    {
        val = val * val; // SFPMAD: square once unconditionally
        for (int s_iter = 0; s_iter < 7; s_iter++)
        {
            exp = exp - 1;       // SFPIADD: integer decrement
            v_and(exp >= 0);     // SFPPUSHC + narrow predication
            val = val * val;     // SFPMAD: repeated squaring (CC-guarded)
        }
    }
    v_endif;

    return val;
}
```

The `_sfpu_reciprocal_<2>` function (Newton-Raphson reciprocal with 2 iterations):

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_recip.h

template <int max_iter = 2>
sfpi_inline sfpi::vFloat _sfpu_reciprocal_(const sfpi::vFloat in) // max_iter=2
{
    // Implementation notes, see the original file for more details
    sfpi::vFloat negative_x = sfpi::setman(sfpi::vConstNeg1, sfpi::reinterpret<sfpi::vInt>(in)); // SFPSETMAN: scale to [-2,-1)

    // Quadratic initial estimate: y = k2 - k1*x + k0*x^2
    sfpi::vFloat y = sfpi::vConstFloatPrgm1 + sfpi::vConstFloatPrgm0 * negative_x; // SFPMAD

    // Scale factor: scale.Exp = ~in.Exp (efficient via SFPNOT)
    sfpi::vInt scale_bits = ~sfpi::reinterpret<sfpi::vUInt>(in); // SFPNOT

    // Continue quadratic estimate
    y = sfpi::vConstFloatPrgm2 + y * negative_x; // SFPMAD

    // Scale factor: clear mantissa
    sfpi::vFloat scale = sfpi::setman(sfpi::reinterpret<sfpi::vFloat>(scale_bits), 0); // SFPSETMAN

    // First Newton-Raphson iteration: t = 1.0 - x*y
    sfpi::vFloat t = sfpi::vConst1 + negative_x * y; // SFPMAD

    // Scale adjustment: scale = scale * 0.5
    scale *= 0.5f; // SFPMAD

    // Continue NR: y = y + y*t
    y = y + y * t; // SFPMAD

    // Second Newton-Raphson iteration (max_iter=2)
    t = sfpi::vConst1 + negative_x * y; // SFPMAD
    y = y + y * t;                       // SFPMAD

    // Apply scaling and restore sign
    y = y * scale;          // SFPMAD
    y = sfpi::setsgn(y, in); // SFPSETSGN: copy sign from original input

    return y;
}
```

### SFPU Instructions Used

| Instruction | Description | Where Used |
|-------------|-------------|------------|
| `SFPLOAD` | Load 32 elements from DEST row pair into LREG | `dst_reg[0]` reads in `calculate_selu`, `_sfpu_exp_`, `_sfpu_reciprocal_` |
| `SFPSTORE` | Store LREG contents back to DEST row pair | `dst_reg[0] = v` writes in `calculate_selu` |
| `SFPLOADI` | Load 16-bit immediate into LREG (used for constants) | Loading alpha (0x3FD63840), scale (0x3F868640), and intermediate constants |
| `SFPMAD` | Fused multiply-add (a * b + c) | All floating-point arithmetic: multiplication by alpha/scale, Horner polynomial steps in `_sfpu_exp_`, Newton-Raphson iterations in `_sfpu_reciprocal_`, subtraction (exp-1.0) |
| `SFPSETCC` | Set CC.Res based on comparison | `v_if(v < 0.0f)` (LT0 mode), `v_if(exp >= 0)` (GTE0 mode) in `_sfpu_exp_` |
| `SFPENCC` | Enable/disable condition code masking | `v_if` (enable CC) and `v_endif` (disable CC, restore all-lanes-active) |
| `SFPCOMPC` | Complement CC.Res for else-branch | Implicit in `v_if`/`v_endif` SFPI abstraction |
| `SFPPUSHC` | Push CC state onto stack for nested conditionals | Nested `v_if` blocks: outer (v < 0) containing inner `_calculate_exponential_piecewise_` which has its own `v_if(in < 0)`, and `v_and` in `_sfpu_exp_` |
| `SFPPOPC` | Pop CC state from stack | Restoring CC after nested conditional blocks |
| `SFPEXEXP` | Extract exponent field from float | `exexp(val)` in `_sfpu_exp_` to extract exponent for range reduction |
| `SFPSETEXP` | Set exponent field of float | `setexp(val, 126)` in `_sfpu_exp_` to clamp input to [-1, 0) range |
| `SFPSETMAN` | Set mantissa field of float | `setman()` in `_sfpu_reciprocal_` for input scaling and scale factor construction |
| `SFPSETSGN` | Set sign bit of float | `setsgn(in, 0)` in `_calculate_exponential_piecewise_` (force positive), `setsgn(y, in)` in `_sfpu_reciprocal_` (restore sign) |
| `SFPNOT` | Bitwise NOT | `~reinterpret<vUInt>(in)` in `_sfpu_reciprocal_` to compute scale exponent = 255 - in.Exp |
| `SFPIADD` | Integer add/subtract | `exp = exp - 1` in `_sfpu_exp_` repeated squaring loop |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **LREG0-LREG3** | General-purpose working registers for intermediate values. Used by SFPI abstraction to hold `v`, `v_exp`, `v_alpha`, `v_scale`, `result`, `y`, `t`, `scale`, `negative_x`, etc. The compiler maps SFPI `vFloat`/`vInt` variables to these registers. |
| **DEST rows** | Input/output: `dst_reg[0]` loads from and stores to the current DEST row pair. Each iteration processes 32 elements (2 physical rows x 16 elements/row). |
| **vConstFloatPrgm0** | Programmable constant register 0: loaded with `0.3232325...` (k0 coefficient for reciprocal initial estimate polynomial) by `_init_sfpu_reciprocal_`. |
| **vConstFloatPrgm1** | Programmable constant register 1: loaded with `1.4545459...` (k1 coefficient for reciprocal initial estimate polynomial) by `_init_sfpu_reciprocal_`. |
| **vConstFloatPrgm2** | Programmable constant register 2: loaded with `2.1212124...` (k2 coefficient for reciprocal initial estimate polynomial) by `_init_sfpu_reciprocal_`. |
| **vConst0p8373** | Fixed constant register: `0.8373` (coefficient in Horner polynomial for exp approximation). |
| **vConst1** | Fixed constant register: `1.0` (used in Newton-Raphson: `1.0 + negative_x * y`, and in Horner: `val * tmp + 1.0`). |
| **vConstNeg1** | Fixed constant register: `-1.0` (used in `_sfpu_reciprocal_` to construct `negative_x` via `setman`). |

### Address Mode Configuration

**Wormhole B0:**
- `ADDR_MOD_7` is configured with: `srca.incr = 0`, `srcb.incr = 0`, `dest.incr = 0`
- This is the standard unary SFPU address mode. It does not auto-increment DEST addressing -- all address progression is handled by the SFPI abstraction (`dst_reg++`) and the `TTI_SETRWC` instructions between faces in the params dispatch.
- The SFPU kernel itself uses `dst_reg++` which increments by 1 sfpi row (= 2 physical DEST rows) per iteration, covering 32 elements each time.
- Between faces, the params dispatch issues two `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` instructions, each advancing DEST by 8 physical rows, totaling 16 physical rows = 1 face stride.

**Blackhole:**
- Same `ADDR_MOD_7` configuration: `srca.incr = 0`, `srcb.incr = 0`, `dest.incr = 0`
- Between faces, the params dispatch calls `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` which executes two `math::inc_dst_addr<8>()` calls (equivalent to the WH `TTI_SETRWC` approach).

**Note:** For `SfpuType::selu`, no special ADDR_MOD_6 configuration is applied (that is only for `topk_local_sort`, `typecast`, and min/max operations).

## Local Knowledge Sources
### Local References
1. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Determine compute kernel path, approximation mode, and SFPU_OP_CHAIN_0 expansion for SELU
   **Key Findings**: SELU uses `eltwise_sfpu.cpp` compute kernel, `math_approx_mode = false` (default case), macro definition is `SFPU_OP_SELU_INCLUDE`, init/func pair is `selu_tile_init()` / `selu_tile(idst)`

2. **File**: `tt_metal/hw/inc/api/compute/eltwise_unary/selu.h`
   **Reason**: API header that defines the tile-level functions `selu_tile()` and `selu_tile_init()`
   **Key Findings**: Uses `SFPU_UNARY_NO_PARAM_KERNEL_FN` and `SFPU_INIT_KERNEL_CALL` macros with `calculate_selu` and `selu_init` as the SFPU function and init callback

3. **File**: `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_selu.h`
   **Reason**: LLK dispatch layer that bridges API to core SFPU implementation
   **Key Findings**: Identical on both architectures; calls `_llk_math_eltwise_unary_sfpu_params_` with `calculate_selu<APPROXIMATE>` and `VectorMode::RC`

4. **File**: `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_selu.h`
   **Reason**: Core SFPU implementation containing `calculate_selu` and `selu_init`
   **Key Findings**: SFPI-based kernel identical on WH and BH. Uses `v_if(v < 0)` for negative branch, calls `_calculate_exponential_piecewise_` for exp computation, multiplies by alpha constant (0x3FD63840) for negative branch and scale constant (0x3F868640) for all lanes

5. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_exp.h`
   **Reason**: Contains `_calculate_exponential_piecewise_`, `_sfpu_exp_`, and `_init_exponential_` used by SELU
   **Key Findings**: With APPROXIMATION_MODE=false, uses accurate path: `_sfpu_exp_` (Horner polynomial with repeated squaring for range reduction) + `_sfpu_reciprocal_<2>` for negative inputs. Init function programs reciprocal polynomial constants.

6. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_recip.h`
   **Reason**: Contains `_sfpu_reciprocal_<2>` and `_init_sfpu_reciprocal_` used for computing exp(x) for negative x
   **Key Findings**: Newton-Raphson reciprocal with quadratic initial estimate and 2 iterations. Programs k0, k1, k2 polynomial coefficients into vConstFloatPrgm0/1/2.

7. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_converter.h`
   **Reason**: Contains the `Converter::as_float` utility used to reinterpret uint32_t as float for loading SELU constants
   **Key Findings**: Simple union-based type punning from uint32_t to float

8. **File**: `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Parameters dispatch function that manages per-face iteration and DEST addressing
   **Key Findings**: For VectorMode::RC, loops 4 faces calling the SFPU function once per face. WH uses `TTI_SETRWC` for face advancement; BH uses `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()`

9. **File**: `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: Contains addr_mod configuration and SFPU init/done functions
   **Key Findings**: ADDR_MOD_7 configured with all increments = 0 for standard SFPU operations. `SfpuType::selu` does not trigger any special ADDR_MOD_6 configuration.

10. **File**: `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h`
    **Reason**: Contains macro definitions used by the API header (SFPU_UNARY_NO_PARAM_KERNEL_FN, SFPU_INIT_KERNEL_CALL)
    **Key Findings**: `SFPU_UNARY_NO_PARAM_KERNEL_FN(FN, MODE, APPROXIMATE, DST_IDX)` expands to `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::FN<APPROXIMATE>, DST_IDX, (int)VectorMode::MODE)`

11. **File**: `.claude/references/sfpu-hardware-model.md`
    **Reason**: Authoritative reference for SFPU hardware model, instruction semantics, and register layout
    **Key Findings**: Confirmed stride-2 addressing, per-face iteration count (8), tile geometry (32x32), SFPMAD semantics for float add/multiply, SFPIADD for integer-only operations

12. **File**: `.claude/references/sfpu-dest-addressing-explained.md`
    **Reason**: Detailed explanation of stride-2 DEST addressing mechanism
    **Key Findings**: Confirmed `dst_tile_size_sfpi = 32`, each `dst_reg++` advances 2 physical rows (32 elements), 8 iterations per face

13. **File**: `tt_metal/jit_build/genfiles.cpp`
    **Reason**: Understand how `APPROX` constexpr is generated
    **Key Findings**: `APPROX` is emitted as `constexpr bool APPROX = {math_approx_mode};` in the generated `chlkc_descriptors.h` file during JIT compilation
