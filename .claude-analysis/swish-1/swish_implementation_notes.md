# Implementation Notes: swish

## Math Definition
swish(x) = x * sigmoid(x) = x / (1 + exp(-x))

## Implementation Approach
Compute sigmoid(x) = 1 / (1 + exp(-x)) using:
1. `_sfpu_exp_21f_bf16_(-x)` for exp(-x) computation
2. Add 1.0 to get denominator: `1 + exp(-x)`
3. `_sfpu_reciprocal_<2>(denom)` for 1/denominator
4. Multiply by x to get final result

## Reference Operations Used
- **cosh**: Pattern for using `_sfpu_exp_21f_bf16_` directly
- **softsign**: Pattern for using `_sfpu_reciprocal_<2>` for division
- **selu**: Pattern for exp init and init function with `_init_exponential_`
- **hardswish**: Structural template (x * f(x) pattern)

## New Files
- tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_swish.h
- tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_swish.h
- tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_swish.h
- tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_swish.h
- tt_metal/hw/inc/api/compute/eltwise_unary/swish.h

## Modified Files
- tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h
- tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h
- tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h
- ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp
- ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp
- ttnn/cpp/ttnn/operations/eltwise/unary_ng/common/unary_ng_op_utils.cpp
- ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp
- ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp
- ttnn/ttnn/operations/unary.py

## Key Design Decisions
1. Used `_sfpu_exp_21f_bf16_` instead of `_calculate_exponential_piecewise_` for better accuracy in bf16 mode
2. Used `_sfpu_reciprocal_<2>` (2 Newton-Raphson iterations) for good precision on the division
3. Used `#pragma GCC unroll 0` (no unrolling) since exp computation is complex
4. Both exp and reciprocal init functions called in `swish_init()`
5. Removed the old `swish` inline alias for `silu` and replaced with a first-class SFPU operation

## Known Limitations
- For very large negative x values, exp(-x) may overflow, but `_sfpu_exp_21f_bf16_` handles clamping internally
- For very large positive x values, exp(-x) approaches 0, sigmoid approaches 1, so swish approaches x (correct behavior)
