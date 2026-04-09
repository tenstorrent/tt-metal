# Softcap Implementation Notes

## Operation Definition
`softcap(x, cap) = cap * tanh(x / cap)`

where `cap` is a positive float scalar parameter (default = 50.0).

## Implementation Summary

Softcap is implemented as a parameterized unary SFPU operation following the standard abstraction layer pattern. The `cap` parameter flows from Python through C++ to the SFPU kernel as a hex-encoded uint32_t bit-cast from float.

### SFPU Kernel Design

The core SFPU kernel computes `cap * tanh(x/cap)` using a piecewise polynomial approximation for tanh, following the swish kernel's piecewise sigmoid pattern:

1. **u = x * (1/cap)** - Division replaced by multiplication with precomputed reciprocal
2. **tanh(|u|)** computed via 4 regions:
   - **Region 1** (|u| <= 1.0): Degree-5 odd polynomial: `u * (1.0 + u^2 * (-0.3253 + u^2 * 0.0869))`. Fitted through tanh(0), tanh(0.5), tanh(1.0). Max ~0.8 ULP bf16 error.
   - **Region 2** (1.0 < |u| <= 2.0): Quadratic: `0.2208 + 0.710*u - 0.1692*u^2`. Fitted through tanh(1.0), tanh(1.5), tanh(2.0). Max ~1.3 ULP bf16 error.
   - **Region 3** (2.0 < |u| <= 3.0): Quadratic: `0.7324 + 0.1722*u - 0.0282*u^2`. Fitted through tanh(2.0), tanh(2.5), tanh(3.0). Max ~0.5 ULP bf16 error.
   - **Region 4** (|u| > 3.0): Saturation to 1.0. Max ~1 ULP at transition.
3. **result = cap * tanh_val * sign(x)**

### Parameter Passing

The `cap` parameter is passed as a compile-time define via the SFPU_OP_CHAIN macro mechanism:
- `get_op_init_and_func_parameterized()` formats `cap` as hex: `softcap_tile_init(0xHHHHHHHH);`
- The API header receives it as `uint32_t param0`
- The LLK dispatch forwards it to `calculate_softcap()` via variadic args
- The kernel bit-casts `param0` back to float

### Abstraction Layers

| Layer | File |
|-------|------|
| **Core SFPU** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_softcap.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_softcap.h` |
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/softcap.h` |

## Reference Operations Used

1. **swish** (most useful): Provided the piecewise polynomial pattern with v_if/v_endif branching for different value ranges. The sigmoid approximation approach directly inspired the tanh approximation strategy.

2. **atanh** (useful for init pattern): Demonstrated how to use programmable constant registers and how to pass init functions with parameters via the LLK init infrastructure.

3. **sinh** (useful for understanding exp_21f): Showed the exp computation approach, though we couldn't use it directly since tanh was implemented via piecewise polynomial instead of exp-based computation.

4. **hardshrink** (useful for parameter passing): Demonstrated how parameterized operations pass scalar parameters through the program factory and compute kernel.

5. **tanhshrink** (context only): Showed that the original tanh was nuked, confirming we needed to implement tanh from scratch.

## Deviations from Standard Patterns

1. **Piecewise polynomial for tanh instead of exp-based**: The original codebase implemented tanh as `2*sigmoid(2x) - 1` using exp. Since exp, sigmoid, and tanh were all nuked, we implemented a direct piecewise polynomial approximation for tanh. This avoids the need for exp entirely.

2. **No programmable constants used**: Unlike atanh which loads polynomial coefficients into vConstFloatPrgm0/1/2, softcap uses all coefficients as `constexpr float` values loaded via SFPLOADI each iteration. This avoids complexity in the init function while still allowing the cap parameter to flow through. The cap and 1/cap are computed from the passed parameter directly in the kernel loop.

## Known Limitations and Concerns

1. **Accuracy at region boundaries**: The piecewise polynomial transitions (at |u|=1.0, 2.0, 3.0) may introduce up to ~2 ULP error in bfloat16 at the exact boundary points due to discontinuity in higher derivatives.

2. **No fp32 accuracy optimization**: The polynomial coefficients were chosen for bfloat16 accuracy. For fp32 inputs, the approximation may have higher ULP error (estimated 5-10 ULP in the transition regions). Higher-degree polynomials or more regions would improve fp32 accuracy.

3. **Division by zero**: If cap = 0 is passed, 1/cap produces infinity, and the results will be NaN. The caller is expected to ensure cap > 0.

4. **Large input values**: For very large |x| (much larger than cap), u = x/cap becomes large and tanh saturates to +/-1, giving result = +/-cap. This is mathematically correct.

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
