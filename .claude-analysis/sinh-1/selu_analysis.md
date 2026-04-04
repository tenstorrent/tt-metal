# SFPU Operation Analysis: selu

## Overview
- **Operation**: selu (Scaled Exponential Linear Unit)
- **Math**: selu(x) = scale * (max(0,x) + min(0, alpha*(exp(x)-1)))
- **Parameters**: None (scale and alpha are fixed constants)

## SFPU Kernel
- Uses `_calculate_exponential_piecewise_` for the exp computation
- Init: `_init_exponential_<APPROXIMATION_MODE, FAST_APPROX, EXP_BASE_SCALE_FACTOR>()`
- Uses `#pragma GCC unroll 0` (no unrolling due to complexity)
- Uses conditional `v_if(v < 0.0f)` for negative branch

## Compute API
- Uses same macro pattern: `SFPU_INIT_KERNEL_CALL` and `SFPU_THREE_PARAM_KERNEL_FP32_FIRST`

## Key Takeaway
- Shows how to use exp init functions
- Different exp helper function than cosh (`_calculate_exponential_piecewise_` vs `_sfpu_exp_21f_bf16_`)
