# Softcap Implementation Notes

## Operation Definition
`softcap(x, cap) = cap * tanh(x / cap)` where `cap` is a positive float parameter (default 50.0).

## Implementation Strategy

### SFPU Kernel Algorithm
The kernel computes `tanh(u)` where `u = x / cap` using two regimes:

1. **Small |u| (< 1.0)**: Degree-7 Taylor series
   - `tanh(u) = u * (1 + u² * (-1/3 + u² * (2/15 + u² * (-17/315))))`
   - Evaluated in Horner form for numerical stability
   - Provides <1 ULP accuracy in bfloat16 for |u| < 0.85, and ~2 ULP at |u| = 1.0

2. **Moderate/large |u| (≥ 1.0)**: Exponential series
   - Let `e = exp(-2|u|)`, computed via the Moroz et al. 2022 `exp_21f` algorithm (2^z)
   - `tanh(|u|) = 1 - 2e + 2e² - 2e³` (geometric series expansion truncated at degree 3)
   - For |u| ≥ 1.0, `e ≤ 0.135`, giving truncation error < 2*e⁴ ≈ 0.0007 (< 0.2 ULP)
   - For |u| ≥ 4.0, `e < 1e-5`, and tanh naturally rounds to 1.0 in bfloat16

The exp-based formula is computed for ALL SIMD lanes (SFPU processes in lockstep), then the Taylor series overrides small-|u| lanes via `v_if` conditional predication.

### Parameter Passing
The `cap` parameter flows through the standard parameterized unary path:
- Host: `get_op_init_and_func_parameterized()` embeds the float as a `std::bit_cast<uint32_t>` literal in the kernel define string
- Device: The SFPU kernel decodes the uint32_t back to float via union reinterpretation
- `inv_cap = 1.0f / cap` is computed once per SFPU function call (once per face, 4 times per tile)

### exp_21f Helper
The `softcap_exp_21f_` helper is a local copy of the Moroz et al. 2022 algorithm from `ckernel_sfpu_sinh.h`. It computes `2^z` using IEEE 754 decomposition and a degree-2 polynomial refinement. This is copied locally to avoid cross-include dependencies, following the established codebase pattern.

## Reference Operations Used
- **sinh** (most useful): Provided the `exp_21f` helper algorithm, the dual-regime (exp + Taylor override) pattern, and the `v_if` conditional override pattern for small-argument special casing.
- **atanh**: Provided the standard abstraction layer pattern (ckernel → LLK → API → split-include) and the `SfpuType` enum registration pattern. Also demonstrated programmable constant usage (though softcap doesn't need them).
- **swish**: Provided the SFPI piecewise computation pattern with `v_if`/`v_endif` for segment selection, and the `abs`/comparison workflow.
- **hardshrink/tanhshrink**: Provided context on parameterized operations and custom compute kernel patterns (not directly used since softcap uses the standard `eltwise_sfpu.cpp` path).

## Deviations from Standard Patterns
- **Parameterized SFPU op via `eltwise_sfpu.cpp`**: Most parameterized operations (hardshrink, tanhshrink) use custom compute kernels. Softcap uses the standard `eltwise_sfpu.cpp` with the parameter embedded in the `SFPU_OP_CHAIN_0` macro expansion, which is simpler and follows the same path as non-parameterized ops.
- **`#pragma GCC unroll 0`**: Used instead of `#pragma GCC unroll 8` to reduce register pressure, since the kernel has high register usage (exp_21f uses ~10 intermediates). This follows the sinh kernel's pattern.
- **No programmable constants**: All coefficients are local `constexpr` or computed from the runtime parameter. The `softcap_init()` function is empty.

## Known Limitations
- **fp32 precision**: Like all SFPU operations in this codebase, the fp32 path computes at approximately bfloat16 precision due to the polynomial approximations in `exp_21f`. The fp32 result is bfloat16-quality stored in fp32 format.
- **Taylor-exp transition**: At |u| = 1.0, the Taylor degree-7 series has ~2 ULP error in bfloat16. The exp-based formula has ~0.2 ULP at |u| = 1.0. The transition is handled by `v_if` predication with the Taylor override taking priority for |u| < 1.0. There is no smoothing at the boundary, but both approximations agree to within ~2 ULP at the transition point.
- **Very large |u|**: For |u| > ~88 (where `exp(-2|u|)` underflows), the `exp_21f` result is clamped to `2^(-127)` ≈ 5.9e-39, and tanh correctly evaluates to 1.0.

### New Files
tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_softcap.h
tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_softcap.h
tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_softcap.h
tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_softcap.h
tt_metal/hw/inc/api/compute/eltwise_unary/softcap.h

### Modified Files
tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h
tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h
tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h
ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp
ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp
ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp
ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp
ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp
ttnn/ttnn/operations/unary.py
ttnn/ttnn/experimental_loader/golden_functions.py
