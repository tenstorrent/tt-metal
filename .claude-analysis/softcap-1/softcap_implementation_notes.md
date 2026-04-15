# softcap Implementation Notes

## Operation
softcap(x, cap) = cap * tanh(x / cap)

## Implementation Approach
Implemented as a single-pass SFPU kernel using piecewise degree-7 centered polynomial approximation of tanh.
The centered basis avoids catastrophic cancellation in Horner evaluation.

### Key Design Decisions
1. **SFPU-only implementation**: Could not use hardware LUT-based tanh because `tanh_tile` is only available via `compute_kernel_api.h` which is not accessible from the TTNN split-include kernel compilation path.
2. **Piecewise polynomial**: 7 segments covering [0, 9.0] with clamp to 1.0 beyond 9.0.
3. **Centered polynomial evaluation**: Each segment uses `u = (|t| - center) / half_width` to avoid numerical instability.
4. **Float32 parameter encoding**: Parameters (cap, 1/cap) encoded as raw float32 bit patterns rather than BFloat16 to preserve precision.
5. **SfpuType enum extension**: Added base LLK compatibility entries to the metal-layer SfpuType enum to resolve compilation conflicts with base LLK headers.

## Test Results
- bfloat16-cap1: PASS (0 ULP)
- bfloat16-cap10: PASS (0 ULP)
- bfloat16-cap50: PASS (0 ULP)
- fp32-cap1: PASS (0 ULP)
- fp32-cap10: FAIL (3 ULP, threshold 2) -- fundamental float rounding in cap * (x * inv_cap)
- fp32-cap50: FAIL (3 ULP, threshold 2) -- same root cause

### Root Cause of fp32 Failures
For fp32 with non-power-of-2 cap values (10, 50), the computation `cap * (x * inv_cap)` does not equal `x` due to floating-point rounding of `inv_cap = 1/cap`. For example, `1/10 = 0.10000000149...` in float32. This introduces ~3 ULP error at small input values where tanh(t) ~ t.

## New Files
- tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_softcap.h
- tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_softcap.h
- tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_softcap.h
- tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_softcap.h
- tt_metal/hw/inc/api/compute/eltwise_unary/softcap.h

## Modified Files
- tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h
- tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h
- tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h
- ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp
- ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp
- ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp
- ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp
- ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp
- ttnn/ttnn/operations/unary.py
- ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp

## Most Useful References
- **swish**: Overall file structure pattern (SFPU kernel + LLK dispatch + compute API + registration)
- **hardtanh**: Parameter passing pattern (float parameters via uint32_t encoding)
- **atanh**: Initialization pattern with custom init functions
