# rpow Implementation Notes

## Overview
- **Operation**: rpow (reverse power)
- **Math definition**: base^x where base is a float parameter and x is each tensor element
- **Parameter**: base (float) - the scalar base raised to the power of each element

## Algorithm
The implementation computes `base^x = 2^(x * log2(base))` using the exp_21f algorithm from Moroz et al. 2022.

Key optimization: since `base` is a constant scalar, `log2(base)` is precomputed once before the SFPU vector loop. Only the `2^(x * log2_base)` computation runs in the vector loop.

### Algorithm Steps
1. Precompute `log2(|base|)` using IEEE 754 decomposition and polynomial approximation
2. For each element x: compute `z = x * log2(base)`
3. Clamp z to [-127, ...] to prevent overflow
4. Compute `2^z` using the exp_21f algorithm
5. Handle special cases (base=0, negative base, non-integer exponents)

## Reference Operations Used
- **power** (most useful): Provided the core _sfpu_unary_power_21f_ algorithm with log2 polynomial coefficients and exp_21f implementation
- **hardtanh**: Showed the parameterized operation pattern (is_parametrized_type, get_op_init_and_func_parameterized)
- **cbrt**: Showed the complete 12-layer file creation pattern

## Deviations from Standard Patterns
- No `_init_exponential_` call needed since we don't use the standard exp path
- rpow_init() is a no-op (no programmable constants needed - log2(base) is computed from the parameter at runtime)

## Known Limitations
- For negative base values, only integer exponents produce real results (non-integer exponents return NaN)
- Large exponents may overflow or underflow (clamped to [-127, ...] threshold)
- The bfloat16 rounding step may reduce precision for some edge cases

### New Files
- tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_rpow.h
- tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_rpow.h
- tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_rpow.h
- tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_rpow.h
- tt_metal/hw/inc/api/compute/eltwise_unary/rpow.h

### Modified Files
- tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_math_unary_sfpu_api.h
- tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_math_unary_sfpu_api.h
- tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h
- tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h
- tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h
- ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp
- ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp
- ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp
- ttnn/ttnn/operations/unary.py
