# Execution Log: ttnn-unary-sfpu-operation-analyzer

## Operation: atanh
## Date: 2026-04-09
## Status: SUCCESS

### Summary
Analyzed the SFPU kernel implementation for the `atanh` unary operation. The kernel uses SFPI abstractions (Style A) with IEEE 754 decomposition for natural logarithm computation via cubic minimax polynomial approximation. The implementation computes `atanh(x) = 0.5 * (ln(1+x) - ln(1-x))` using `exexp`, `setexp`, `int32_to_float` intrinsics and Horner-form polynomial evaluation.

### Key Findings
- **Compute kernel**: `eltwise_sfpu.cpp` (standard unary)
- **SFPU chain**: `atanh_tile_init(); atanh_tile(0);`
- **Approximation mode**: `APPROX = false` (default); kernel does not branch on APPROXIMATION_MODE
- **Kernel style**: A_sfpi (pure SFPI abstractions, no raw TTI instructions)
- **WH/BH identical**: Both architectures use the same implementation
- **ADDR_MOD**: ADDR_MOD_7 (all zero increments) for both WH and BH
- **Key instructions**: SFPLOAD, SFPSTORE, SFPMAD (~12/iter), SFPEXEXP (2/iter), SFPSETEXP (2/iter), SFPCAST (2/iter), SFPLOADI (3, init only)

### Files Produced
- `.claude-analysis/softcap-1/atanh_analysis.md`

### Verification
- All function names verified via grep (calculate_atanh, atanh_init, llk_math_eltwise_unary_sfpu_atanh, llk_math_eltwise_unary_sfpu_atanh_init)
- All SFPI intrinsics verified present in ckernel_sfpu_atanh.h (exexp, setexp, int32_to_float, vConst1, vConstFloatPrgm0/1/2, dst_reg)
- All file paths verified to exist
