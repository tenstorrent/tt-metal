# Execution Log: ttnn-unary-sfpu-operation-analyzer

## Operation: atanh
## Date: 2026-04-15

### Summary
Analyzed the SFPU kernel implementation for the `atanh` unary operation. The kernel computes `atanh(x) = 0.5 * (ln(1+x) - ln(1-x))` using IEEE 754 float decomposition for the natural logarithm, with a cubic minimax polynomial approximation for `ln(mantissa)` on [1, 2).

### Key Findings
- **Compute kernel**: `eltwise_sfpu.cpp` (standard unary dispatch)
- **SFPU_OP_CHAIN_0**: `atanh_tile_init(); atanh_tile(0);`
- **Approximation mode**: `false` (unused -- kernel has no `if constexpr` on APPROXIMATION_MODE)
- **Core SFPU function**: `calculate_atanh<false, 8>()` in `ckernel_sfpu_atanh.h`
- **Kernel style**: SFPI abstractions (Style A)
- **WH/BH parity**: Identical implementations on both platforms
- **SFPU instructions**: SFPLOAD, SFPSTORE, SFPMAD (dominant -- Horner chain for two ln() evaluations), SFPEXEXP (2x per iteration), SFPSETEXP (2x), SFPCAST (2x), SFPCONFIG (3x in init)
- **Constant registers**: Three programmable constants for cubic polynomial coefficients (c0, c1, c2), plus fixed constant vConst1 for value 1.0

### Files Read
1. `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
2. `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h`
3. `tt_metal/hw/inc/api/compute/eltwise_unary/atanh.h`
4. `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_atanh.h`
5. `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_atanh.h`
6. `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_atanh.h`
7. `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_atanh.h`
8. `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
9. `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
10. `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`
11. `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu.h`
12. `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_init.h`
13. `runtime/sfpi/include/sfpi_lib.h`
14. `runtime/sfpi/include/sfpi.h`
15. `.claude/references/sfpu-hardware-model.md`
16. `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`
17. `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp`

### Verification Results
- `calculate_atanh`: VERIFIED in both WH and BH ckernels
- `atanh_init`: VERIFIED in both WH and BH ckernels
- All file paths in abstraction layers table: VERIFIED (all exist)
- SFPU instructions: VERIFIED via SFPI-to-instruction mappings in sfpi_lib.h (no raw TTI instructions in this kernel; all instructions inferred from SFPI abstraction layer)

### Output
- Analysis file: `.claude-analysis/softcap-1/atanh_analysis.md`
