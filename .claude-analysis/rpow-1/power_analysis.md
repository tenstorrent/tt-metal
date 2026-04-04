# Reference Analysis: power (POWER)

## Overview
The `power` operation computes `x^exponent` where x is each tensor element and exponent is a scalar parameter. This is the inverse of rpow (`base^x`).

## SFPU Kernel Pattern
Located in the build directory at `ckernel_sfpu_unary_power.h`.

### Algorithm
Uses `_sfpu_unary_power_21f_` based on Moroz et al. 2022 ("Simple Multiple Precision Algorithms for Exponential Functions"):
1. Compute log2(base) using polynomial approximation
2. Compute 2^(pow * log2(base)) using exp_21f algorithm

### Key Implementation Details
- Takes `uint32_t exponent` parameter (IEEE 754 float bits)
- Uses `Converter::as_float(exponent)` to decode
- Requires programmable constants: vConstFloatPrgm0 = 1/ln(2), vConstFloatPrgm1 = -127.0, vConstFloatPrgm2 = NaN
- Init function `sfpu_unary_pow_init()` sets these constants
- Handles negative base, zero base, and overflow cases
- Uses `sfpi::float_to_fp16b()` for bfloat16 rounding

### Parameter Passing Chain
1. `unary_op_utils.cpp`: `"power_tile_init();"`, `"power_tile({idst}, {param0_hex}u);"`
2. `compute_kernel_api.h`: `power_tile(uint32_t idst, uint32_t param0)` -> `llk_math_eltwise_unary_sfpu_power<APPROX, DST_ACCUM_MODE>(idst, param0)`
3. LLK: `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(calculate_unary_power<APPROXIMATE, 8>, dst_index, vector_mode, exponent)`
4. Kernel: `calculate_unary_power(const uint32_t exponent)` -> `_sfpu_unary_power_<ITERATIONS>(exponent)`

## Relevance to rpow
For rpow (base^x), the algorithm is identical but with swapped operands:
- power: base = dst_reg[0] (tensor element), pow = parameter
- rpow: base = parameter (scalar), pow = dst_reg[0] (tensor element)

The `_sfpu_unary_power_21f_` function takes (base, pow) as vFloat args, so for rpow we pass (scalar_base, x) instead of (x, scalar_exponent).
