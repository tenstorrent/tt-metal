# softcap Implementation Notes

## Overview
softcap(x, cap) = cap * tanh(x / cap) where cap is a positive float scalar (default 50.0).

## Implementation Approach
The implementation uses a three-region piecewise tanh approximation:
1. **Saturation** (|y| >= 9): tanh(y) = sign(y)
2. **Exp-based** (0.6 <= |y| < 9): tanh(y) = 1 - 2/(exp(2|y|) + 1)
   - Cody-Waite range-reduced exp with degree-7 Horner polynomial
   - Newton-Raphson reciprocal (3 iterations)
3. **Polynomial** (|y| < 0.6): tanh(y) = y * P(y^2) with Sollya minimax degree-4 polynomial

Two parameters are passed from host: cap and 1/cap (pre-computed as full FP32 bit patterns), loaded via `sfpi::reinterpret<sfpi::vFloat>(sfpi::vInt(param))`.

## Test Results
- **BF16**: All 3 cap values PASS (ULP <= 2)
- **FP32**: All 3 cap values FAIL (ULP ~6107) -- SFPU hardware computes in bfloat16 intermediate precision, limiting FP32 accuracy

## Reference Operations Used
- **swish**: Template for custom SFPU kernel structure and SFPU_OP_CHAIN dispatch
- **sinh**: exp_21f helper pattern (adapted to Cody-Waite for higher accuracy)
- **hardtanh**: Parameterized UnaryOpType pattern
- **atanh**: Polynomial coefficient initialization, exponent manipulation
- **tanhshrink**: tanh_tile usage patterns

## Key Design Decisions
1. Cody-Waite range reduction for exp (higher accuracy than Moroz exp_21f)
2. Full FP32 parameter encoding (not bfloat16) to preserve precision for host-side constants
3. Sollya minimax polynomial for small |y| avoids cancellation in 2*sigmoid(2y)-1
4. Specialized compute kernel (softcap_sfpu.cpp) to avoid missing include issues in eltwise_sfpu.cpp
5. SfpuType enum extended for both Wormhole and Blackhole with required tt_llk values

### New Files
tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_softcap.h
tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_softcap.h
tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_softcap.h
tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_softcap.h
tt_metal/hw/inc/api/compute/eltwise_unary/softcap.h
ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/softcap_sfpu.cpp

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
- FP32 precision is limited by SFPU bfloat16 intermediate arithmetic (~10 mantissa bits)
- Achieving FP32 ULP <= 2 requires 23+ mantissa bits, fundamentally incompatible with SFPU vFloat precision
- Newton-Raphson reciprocal gives ~24 bits but each subsequent multiply truncates to bfloat16
