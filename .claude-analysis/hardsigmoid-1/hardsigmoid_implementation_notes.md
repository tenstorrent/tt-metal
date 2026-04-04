# Implementation Notes: hardsigmoid

## Math Definition
hardsigmoid(x) = max(0, min(1, x/6 + 0.5))

## Implementation Approach
Piecewise linear function computed as:
1. result = x * (1/6) + 0.5
2. Clamp result to [0, 1] using v_if conditionals

Uses SFPI vector instructions (v_if/v_endif) for clean conditional clamping.

## Reference Operations Used
- **hardtanh**: Clamping pattern (SFPSWAP-based min/max)
- **relu**: Conditional assignment with v_if
- **heaviside**: Three-region piecewise function with v_if/v_elseif/v_else
- **silu**: Full LLK stack wiring pattern

## Key Design Decisions
- Used SFPI conditionals (v_if/v_endif) rather than TTI_SFPSWAP for clamping. The piecewise linear compute + clamp approach is simpler and avoids needing LREG parameter loading.
- No custom init function needed (no LUTs or special configuration).
- No runtime parameters (alpha=1/6 and beta=0.5 are compile-time constants).
- Used `#pragma GCC unroll 8` for full unrolling (matches silu, sigmoid patterns).

## Files

### New Files
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_hardsigmoid.h`
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_hardsigmoid.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_hardsigmoid.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_hardsigmoid.h`
- `tt_metal/hw/inc/api/compute/eltwise_unary/hardsigmoid.h`
- `tt_metal/hw/inc/api/compute/eltwise_unary/activations.h`

### Modified Files
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h`
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_math_unary_sfpu_api.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_math_unary_sfpu_api.h`
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp`
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp`
- `ttnn/ttnn/operations/unary.py`

## Known Limitations
- Only supports BFLOAT16 and FLOAT32 data types
- No approximation mode variant (the operation is already a simple linear function with clamps)
- bfloat16 precision: the 1/6 constant will be truncated in bfloat16 representation

## Test Results
(To be filled after Phase 4)
