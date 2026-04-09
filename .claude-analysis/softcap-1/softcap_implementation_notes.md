# Softcap Implementation Notes

## Operation Definition
softcap(x, cap) = cap * tanh(x / cap)

Where cap is a positive float scalar (default 50.0).

## Algorithm

Two-regime tanh approximation for computing tanh(u) where u = x/cap:

### Regime 1: Small |u| (|u| < 1.0) - Degree-9 Taylor Polynomial
```
tanh(u) = u * (1 + u^2*(-1/3 + u^2*(2/15 + u^2*(-17/315 + u^2*(62/2835)))))
```
Evaluated in Horner form on u^2. Converges within the Taylor series radius of convergence (pi/2 ~ 1.57). At the boundary |u|=1.0, the error is ~0.8% relative, which is within ~2 ULP for bfloat16.

### Regime 2: Large |u| (|u| >= 1.0) - Exp-based Formula
Uses the identity tanh(|u|) = (1-f)/(1+f) where f = exp(-2|u|).

The reciprocal 1/(1+f) is approximated via 5-term geometric series:
```
tanh(|u|) = 1 - 2f + 2f^2 - 2f^3 + 2f^4
```
Evaluated in Horner form: `1 + 2*f*(-1 + f*(1 + f*(-1 + f)))`

exp(-2|u|) is computed via the Moroz et al. 2022 `exp_21f` algorithm (2^z computation). This is a self-contained helper copied from the sinh kernel.

### Parameter Passing
- `cap` is passed as a compile-time constant embedded in the init/func strings
- `softcap_init(cap)` stores `1/cap` in `vConstFloatPrgm0` and `cap` in `vConstFloatPrgm1`
- The compute loop reads these programmable constant registers

## Reference Operations Used
1. **sinh** (most useful): Provided the `exp_21f` helper function and the two-regime pattern (Taylor for small inputs, exp-based for large inputs). The overall kernel structure closely follows sinh.
2. **swish**: Provided patterns for `v_if`/`v_endif` conditional execution and the piecewise SFPU computation approach.
3. **atanh**: Provided patterns for programmable constant register usage in init functions.
4. **hardshrink**: Provided patterns for how parameterized ops pass scalar values through the dispatch chain.

## Deviations from Standard Patterns
1. **Parameterized init via compile-time embedding**: The cap value is formatted into the init string (e.g., `softcap_tile_init(50.0f)`) rather than passed as a runtime arg. This is simpler than the runtime arg approach used by hardshrink (which uses a custom compute kernel), and is enabled by the standard `eltwise_sfpu.cpp` dispatch path.
2. **Self-contained exp_21f helper**: Copied from sinh rather than imported, to avoid cross-kernel header dependencies. Named `exp_21f_softcap` to avoid ODR conflicts.

## Known Limitations
- The Taylor polynomial at the boundary (|u|=1.0) has ~0.8% relative error, translating to ~2 ULP in bfloat16. For fp32, the relative error is small but ULP count is higher due to fp32's finer granularity.
- The geometric series approximation of 1/(1+f) is accurate to O(f^5). At |u|=1.0 (f~0.135), the error from this term is ~5e-5, negligible.
- For very large |x/cap| (> ~64), the exp computation clamps to -127 in log2 space, producing tanh(u) ≈ 1.0 exactly.
- `#pragma GCC unroll 0` is used (no unrolling) to manage instruction cache pressure, following the sinh pattern.

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
