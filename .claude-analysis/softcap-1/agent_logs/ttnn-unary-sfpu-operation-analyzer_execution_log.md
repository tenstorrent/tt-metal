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

---

## Operation: swish
## Date: 2026-04-09
## Status: SUCCESS

### Summary
Analyzed the SFPU kernel implementation for the `swish` unary operation. The kernel uses SFPI abstractions (Style A) with a custom piecewise sigmoid approximation: degree-3 polynomial for |x| <= 2.5, linear interpolation for 2.5 < |x| <= 5.0, and saturation to 1.0 for |x| > 5.0. Final result: swish(x) = x * sigmoid(x). Three sequential `v_if` blocks handle piecewise selection.

### Key Findings
- **Compute kernel**: `eltwise_sfpu.cpp` (standard unary)
- **SFPU chain**: `swish_tile_init(); swish_tile(0);`
- **Approximation mode**: `APPROX = false` (default); kernel accepts APPROXIMATION_MODE template parameter but never branches on it
- **Kernel style**: A_sfpi (pure SFPI abstractions, no raw TTI instructions)
- **WH/BH identical**: Both architectures use byte-identical implementations
- **ADDR_MOD**: ADDR_MOD_7 (all zero increments) for both WH and BH
- **Key instructions**: SFPLOAD, SFPSTORE, SFPABS, SFPMAD (polynomial eval + linear + arithmetic), SFPLOADI (float constants), SFPPUSHC/SFPPOPC (v_if/v_endif), SFPSETCC (comparisons), SFPENCC
- **No SFPNONLINEAR**: Does not use hardware-accelerated exp/sigmoid; uses custom polynomial approximation instead
- **CC stack depth**: Maximum 1 (three sequential non-nested v_if blocks)

### Files Produced
- `.claude-analysis/softcap-1/swish_analysis.md`

### Verification
- Function name `calculate_swish` verified via grep in `tt_metal/hw/ckernels/` (found in WH and BH)
- All file paths verified to exist (API header, LLK dispatch, core SFPU, params dispatch)
- `sfpi::abs` verified to compile to `__builtin_rvtt_sfpabs` (SFPABS instruction) via `runtime/sfpi/include/sfpi_lib.h`
- `vConst1` verified as constant register index 10 (Fixed Const 2 = 1.0) via `runtime/sfpi/include/sfpi.h`

---

## Operation: tanhshrink
## Date: 2026-04-09
## Status: SUCCESS

### Summary
Analyzed the SFPU kernel implementation for the `tanhshrink` unary operation. This operation uses a dedicated compute kernel (`tanhshrink_kernel.cpp`) that computes `x - tanh(x)` in two phases: (1) SFPU tanh via `tanh_tile(0)`, then (2) FPU subtraction via `binary_dest_reuse_tiles<ELWSUB, DEST_TO_SRCB>`. The core SFPU tanh implementation (`ckernel_sfpu_tanh.h`) was deleted in Phase 1 of the deep nuke, along with all its dependencies (exp, sigmoid).

### Key Findings
- **Compute kernel**: `tanhshrink_kernel.cpp` (dedicated, NOT `eltwise_sfpu.cpp`)
- **Two-phase operation**: SFPU tanh (in-place on DEST) + FPU subtract (x - tanh(x))
- **NUKED**: `ckernel_sfpu_tanh.h`, LLK dispatch `llk_math_eltwise_unary_sfpu_tanh`, exp/sigmoid dependencies all deleted
- **Surviving**: Compute kernel, API declarations, FPU binary_dest_reuse infrastructure
- **Approximation mode**: Both controls `false` (default)
- **SFPNONLINEAR tanh**: Hardware-accelerated (instr_mod=5) on Quasar ONLY, not WH/BH
- **FPU subtraction**: Uses `MathFidelity::LoFi` for the ELWSUB step

### Files Produced
- `.claude-analysis/softcap-1/tanhshrink_analysis.md`

### Verification
- Compute kernel path verified to exist
- API header tanh_tile declarations verified in `compute_kernel_api.h`
- LLK dispatch `llk_math_eltwise_unary_sfpu_tanh` confirmed NUKED (grep returns 0 results in `tt_metal/hw/ckernels/`)
- `_calculate_tanh_` confirmed NUKED (grep returns 0 results in `tt_metal/hw/ckernels/`)
- `SFPNONLINEAR` confirmed Quasar-only (only in `tt_llk_quasar/common/inc/ckernel_ops.h`)
