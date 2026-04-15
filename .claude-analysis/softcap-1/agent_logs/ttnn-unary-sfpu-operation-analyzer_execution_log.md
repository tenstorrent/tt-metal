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

---

## Operation: softshrink
## Date: 2026-04-15

### Summary
Analyzed the softshrink unary operation and discovered that its **entire SFPU kernel stack has been deleted** from the codebase during the Phase 1 deep nuke (commit `efdc0ad853`). Only residual host-side infrastructure remains (enum value, registration macro, parametrized type flag). The analysis documents the deletion comprehensively and provides hypothetical re-implementation guidance.

### Key Findings
- **SFPU kernel**: DELETED -- all implementation files removed (compute API, ckernel, LLK for WH and BH)
- **Dispatch path**: BROKEN -- `get_op_init_and_func_parameterized()` has no case for SOFTSHRINK (throws TT_THROW)
- **Operation family**: Part of `SFPU_OP_ACTIVATIONS_INCLUDE` (along with SOFTSIGN, HARDSIGMOID, CELU)
- **Classification**: Piecewise Linear activation function
- **Formula**: x - lambda if x > lambda, x + lambda if x < -lambda, 0 otherwise
- **Parameter**: lambda (float), default 0.5
- **Approximation mode**: `false` from get_op_approx_mode() (default case)
- **Residual infrastructure**: UnaryOpType::SOFTSHRINK enum, is_parametrized_type() returns true, REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER macro, Python nanobind binding

### Files Read
1. `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
2. `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp`
3. `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp`
4. `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp`
5. `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`
6. `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h`
7. `tt_metal/hw/inc/api/compute/eltwise_unary/activations.h`
8. `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_swish.h` (reference)
9. `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_swish.h` (reference)
10. `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
11. `DEEP_NUKE_MANIFEST.md`
12. `nuke_op_comparison.md`
13. `docs/sfpu_operations/key_notes/softshrink_key_notes.md`
14. `.claude/references/sfpu-hardware-model.md`
15. `.claude/references/logging/sfpu-operation-analyzer.md`

### Verification Results
- SOFTSHRINK enum: VERIFIED exists in `unary_op_types.hpp:113`
- is_parametrized_type: VERIFIED returns true in `unary_op_utils.hpp:47`
- Registration macro: VERIFIED in `unary.hpp:165`
- No SFPU kernel files found: VERIFIED via grep across entire `tt_metal/hw/ckernels/` and `tt_metal/third_party/tt_llk/`
- Deletion confirmed: VERIFIED via `DEEP_NUKE_MANIFEST.md` Phase 1 table

### Output
- Analysis file: `.claude-analysis/softcap-1/softshrink_analysis.md`

---

## Operation: swish
## Date: 2026-04-15

### Summary
Analyzed the SFPU kernel implementation for the `swish` unary operation. The kernel computes `swish(x) = x * sigmoid(x)` using a piecewise approximation of sigmoid based on the absolute value of x: a degree-3 polynomial for |x| <= 2.5, a linear segment for 2.5 < |x| <= 5.0, and saturation to 1.0 for |x| > 5.0. Sign correction is applied for negative inputs.

### Key Findings
- **Compute kernel**: `eltwise_sfpu.cpp` (standard unary dispatch)
- **SFPU_OP_CHAIN_0**: `swish_tile_init(); swish_tile(0);`
- **Approximation mode**: `false` (unused -- kernel has no `if constexpr` on APPROXIMATION_MODE)
- **Core SFPU function**: `calculate_swish<false, 8>()` in `ckernel_sfpu_swish.h`
- **Kernel style**: SFPI abstractions (Style A)
- **WH/BH parity**: Identical implementations on both platforms (byte-identical files)
- **SFPU instructions**: SFPLOAD, SFPSTORE, SFPABS, SFPMUL, SFPADD, SFPMAD (Horner polynomial chain), SFPLOADI (float constants), SFPSETCC, SFPENCC, SFPPUSHC, SFPPOPC (CC management for 4 v_if blocks)
- **Address mode**: ADDR_MOD_7 only (all-zero increments), same on WH and BH
- **Mathematical approach**: 3-segment piecewise sigmoid approximation; max ULP error ~4 for bfloat16

### Files Read
1. `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
2. `tt_metal/hw/inc/api/compute/eltwise_unary/swish.h`
3. `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_swish.h`
4. `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_swish.h`
5. `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_swish.h`
6. `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_swish.h`
7. `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
8. `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
9. `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`
10. `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu.h`
11. `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_init.h`
12. `runtime/sfpi/include/sfpi.h`
13. `runtime/sfpi/include/sfpi_lib.h`
14. `runtime/sfpi/include/sfpi_constants.h`
15. `.claude/references/sfpu-hardware-model.md`
16. `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`
17. `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp`
18. `tt_metal/jit_build/genfiles.cpp`
19. `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h`
20. `tt_metal/hw/inc/api/compute/common_globals.h`

### Verification Results
- `calculate_swish`: VERIFIED in both WH and BH ckernels
- `llk_math_eltwise_unary_sfpu_swish`: VERIFIED in both WH and BH
- `llk_math_eltwise_unary_sfpu_swish_init`: VERIFIED in both WH and BH
- All file paths in abstraction layers table: VERIFIED (all exist)
- SFPU instructions: VERIFIED via SFPI-to-instruction mappings in sfpi.h and sfpi_lib.h (no raw TTI instructions in kernel; all instructions inferred from SFPI abstraction layer)
- `SfpuType::swish`: VERIFIED in `llk_sfpu_types.h` line 10

### Output
- Analysis file: `.claude-analysis/softcap-1/swish_analysis.md`
