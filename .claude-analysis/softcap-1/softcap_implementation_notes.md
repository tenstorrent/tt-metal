# softcap Implementation Notes

## Overview
softcap(x, cap) = cap * tanh(x / cap) where cap is a positive float scalar (default 50.0).

## Implementation Approach
The implementation follows the standard SFPU kernel pattern used by swish, sinh, and other custom operations. The cap parameter is passed as a bfloat16-encoded uint32 through the parameterized dispatch chain.

The SFPU kernel uses three regions for computing tanh:
1. **Saturation** (|y| >= 9): tanh(y) = sign(y)
2. **Exp-based** (0.1 <= |y| < 9): tanh(y) = 1 - 2/(exp(2|y|) + 1) using a Cody-Waite range-reduced exp with degree-7 Horner polynomial
3. **Taylor** (|y| < 0.1): tanh(y) = y - y^3/3 + 2y^5/15 - 17y^7/315

Division by cap is implemented as multiplication by 1/cap where the reciprocal uses Newton-Raphson iteration (3 iterations for ~24-bit precision).

## Reference Operations Used
- **swish**: Template for custom SFPU kernel structure, SFPU_OP_CHAIN dispatch pattern
- **sinh**: exp_21f helper function (adapted to Cody-Waite exp for higher accuracy)
- **hardtanh**: Parameterized UnaryOpType pattern (is_parametrized_type, get_op_init_and_func_parameterized)
- **atanh**: Polynomial coefficient initialization patterns
- **tanhshrink**: tanh_tile usage pattern at compute kernel level

## Key Design Decisions
1. Used Cody-Waite range reduction with degree-7 polynomial for exp instead of the Moroz exp_21f - provides higher accuracy needed for ULP <= 2
2. Used Newton-Raphson reciprocal instead of hardware division - more predictable precision
3. Used bfloat16 parameter encoding (s2vFloat16b) matching the existing hardtanh pattern
4. Three-region piecewise tanh avoids catastrophic cancellation near zero

### New Files
tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_softcap.h
tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_softcap.h
tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_softcap.h
tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_softcap.h
tt_metal/hw/inc/api/compute/eltwise_unary/softcap.h

### Modified Files
ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp
ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp
ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp
ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp
ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp
ttnn/ttnn/operations/unary.py
tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h
tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h
tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h

## Known Limitations
- The Newton-Raphson reciprocal has ~24-bit precision which may introduce slight error for extreme cap values
- The Taylor series region boundary at |y| = 0.1 is conservative; could potentially be extended
