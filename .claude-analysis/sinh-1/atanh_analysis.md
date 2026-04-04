# SFPU Operation Analysis: atanh

## Overview
- **Operation**: atanh (inverse hyperbolic tangent)
- **Math**: atanh(x) = 0.5 * ln((1+x)/(1-x))
- **Parameters**: None

## SFPU Kernel
- Uses log and reciprocal helpers
- Has boundary condition handling (NaN for |x| > 1, inf for |x| = 1)
- Init: `_init_sfpu_reciprocal_<APPROXIMATION_MODE>()`

## Registration Stack
Full modern split-include pattern:
1. SFPU kernel: `ckernel_sfpu_atanh.h`
2. Compute API: `atanh.h` with `SFPU_INIT_KERNEL_CALL` / `SFPU_THREE_PARAM_KERNEL_FP32_FIRST`
3. Split includes: `SFPU_OP_ATANH_INCLUDE`
4. unary_op_utils.cpp: registered in all three functions
5. unary_ng_op_utils.cpp: registered in both functions
6. Golden function in unary.py

## Key Takeaway
- Shows the full registration pattern for a hyperbolic function
- Uses the same macro dispatch as cosh
