# Implementation Notes: sinh

## Math Definition
sinh(x) = (exp(x) - exp(-x)) / 2 (hyperbolic sine)

## Implementation Strategy
The SFPU kernel computes sinh(x) by:
1. Computing exp(x) using `_sfpu_exp_21f_bf16_` polynomial approximation
2. Computing exp(-x) using the same function on negated input
3. Subtracting: exp(x) - exp(-x)
4. Halving: multiply by 0.5f

This preserves the odd symmetry sinh(-x) = -sinh(x) naturally through the subtraction.
The init function uses `_init_exponential_` with the same parameters as cosh.

## Reference Operations Used
- **cosh**: Primary reference -- identical layer structure, same exp-based approach, just + vs - in the formula
- **selu**: Secondary reference -- shows exp usage patterns and SFPU conditional branching

## New Files
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_sinh.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_sinh.h`
- `tt_metal/hw/inc/api/compute/eltwise_unary/sinh.h`

## Modified Files
- `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h` -- added SFPU_OP_SINH_INCLUDE guard
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp` -- registered SINH in get_macro_definition, get_op_init_and_func_default, and string_to_unary_with_param
- `ttnn/cpp/ttnn/operations/eltwise/unary_ng/common/unary_ng_op_utils.cpp` -- registered SINH in get_macro_definition and get_op_init_and_func
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp` -- added Python nanobind binding
- `ttnn/ttnn/operations/unary.py` -- added golden function using torch.sinh

## Deviations from Standard Patterns
None -- follows the exact same patterns as cosh across all layers.

## Known Limitations
- Supported range is approximately [-9, 9] for bfloat16 due to exp overflow limits (same as cosh)
- The `_sfpu_exp_21f_bf16_` function clamps exp results for inputs outside approximately [-88.5, 88.5]
