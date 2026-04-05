# SFPU Analysis: selu

## Overview
SELU(x) = scale * (max(0,x) + min(0, alpha*(exp(x)-1)))

## SFPU Kernel
- **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_selu.h`
- **Function**: `calculate_selu<APPROXIMATION_MODE, ITERATIONS>()`
- **Init**: `selu_init()` calls `_init_exponential_<APPROXIMATION_MODE, FAST_APPROX, EXP_BASE_SCALE_FACTOR>()`

### Key Pattern
Uses `_calculate_exponential_piecewise_` for the negative branch with conditional `v_if(v < 0.0f)` / `v_endif`.
Shows how to use SFPI conditional execution and FP32 constant encoding via `Converter::as_float()`.

## Compute API
- **File**: `tt_metal/hw/inc/api/compute/eltwise_unary/selu.h`
- Standard `selu_tile_init()` and `selu_tile(idst)` pattern

## Relevance to sinh
Shows exp-based computation with arithmetic. Different init approach using `_calculate_exponential_piecewise_` vs `_sfpu_exp_21f_bf16_`.
