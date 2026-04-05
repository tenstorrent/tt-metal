# Reference Analysis: cosh

## Math Definition
cosh(x) = (exp(x) + exp(-x)) / 2

## SFPU Kernel Pattern
- File: `ckernel_sfpu_cosh.h`
- Includes: `ckernel.h`, `sfpu/ckernel_sfpu_exp.h`, `sfpi.h`
- Template: `<bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>`
- Function: `calculate_cosh()`
- Uses `#pragma GCC unroll 8`
- Uses `_sfpu_exp_21f_bf16_<is_fp32_dest_acc_en>(v)` directly for exp computation
- Has `cosh_init()` that calls `_init_exponential_<APPROXIMATION_MODE, false, p_sfpu::kCONST_1_FP16B>()`

## Key Insight for Swish
- Shows direct use of `_sfpu_exp_21f_bf16_` for exp computation
- Simple pattern: read from dst_reg, compute, write back
- `#pragma GCC unroll 8` for performance
