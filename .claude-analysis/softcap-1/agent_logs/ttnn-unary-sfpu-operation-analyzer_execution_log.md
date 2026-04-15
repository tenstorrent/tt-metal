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

---

## Operation: hardtanh
## Date: 2026-04-15

### Summary
Analyzed the SFPU kernel implementation for the `hardtanh` unary operation. The operation has a core SFPU kernel (`_calculate_hardtanh_`) in the tt_llk layer but its dispatch chain through the Metal compute API is incomplete -- no API header, no LLK dispatch, no SfpuType in production Metal. The kernel implements piecewise clamping using 3 additions and 2 conditional zeroing blocks with SFPI abstractions.

### Key Findings
- **Compute kernel**: `eltwise_sfpu.cpp` (default, via `get_compute_kernel_path()`)
- **SFPU_OP_CHAIN_0**: NOT WIRED -- `get_op_init_and_func_parameterized()` has no HARDTANH case, would throw at runtime
- **Approximation mode**: `false` from `get_op_approx_mode()` (default case); kernel does not use APPROXIMATION_MODE in any `if constexpr` branch
- **Core SFPU function**: `_calculate_hardtanh_<false, 8>()` in `ckernel_sfpu_hardtanh.h`
- **Kernel style**: SFPI abstractions (Style A)
- **WH/BH parity**: Identical implementations on both platforms
- **Dispatch chain status**: INCOMPLETE -- UnaryOpType::HARDTANH enum exists, is_parametrized_type returns true, but no API header, no LLK dispatch, no SfpuType::hardtanh in production Metal
- **SFPU instructions**: SFPLOADI (parameter loading and conditional zeros), SFPLOAD, SFPSTORE, SFPMAD (3 additions via FMA), SFPSETCC (LT0 and GTE0), SFPENCC, SFPPUSHC, SFPPOPC
- **Algorithm concern**: Parameter comments in the kernel suggest all three params are negated, which produces incorrect results upon mathematical trace-through. The kernel likely requires param2 = max_val (positive) for correctness.
- **Address mode**: Would use ADDR_MOD_7 (dest.incr=0, standard for generic SFPU ops)

### Files Read
1. `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
2. `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp`
3. `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp`
4. `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp`
5. `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`
6. `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp`
7. `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_hardtanh.h`
8. `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_hardtanh.h`
9. `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_clamp.h` (reference)
10. `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
11. `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
12. `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`
13. `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu.h`
14. `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h`
15. `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h`
16. `tt_metal/third_party/tt_llk/tests/helpers/include/llk_sfpu_types.h`
17. `tt_metal/hw/inc/api/compute/eltwise_unary/eltwise_unary.h`
18. `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h`
19. `tt_metal/hw/inc/api/compute/eltwise_unary/activations.h`
20. `tt_metal/hw/inc/api/compute/eltwise_unary/swish.h` (reference)
21. `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_swish.h` (reference)
22. `runtime/sfpi/include/sfpi_fp16.h`
23. `.claude/references/sfpu-hardware-model.md`
24. `.claude/references/logging/sfpu-operation-analyzer.md`

### Verification Results
- `_calculate_hardtanh_`: VERIFIED in both WH and BH tt_llk trees
- All file paths in abstraction layers table: VERIFIED (existing files exist, missing files confirmed absent)
- `SfpuType::hardtanh`: VERIFIED absent from production Metal `llk_sfpu_types.h` (both WH and BH)
- `SfpuType::hardtanh`: VERIFIED present in test helpers `llk_sfpu_types.h`
- SFPU instructions: VERIFIED via SFPI-to-instruction mappings (no raw TTI instructions in kernel)
- No `hardtanh_tile` function found in production codebase: VERIFIED via grep

### Output
- Analysis file: `.claude-analysis/softcap-1/hardtanh_analysis.md`

---

## Operation: tanhshrink
## Date: 2026-04-15

### Summary
Analyzed the SFPU kernel implementation for the `tanhshrink` unary operation. Tanhshrink has a unique architecture: it uses dedicated compute kernels (two variants) rather than the standard `SFPU_OP_CHAIN_0` dispatch. Both variants compute `x - tanh(x)` by combining a tanh SFPU step with a subtraction step. However, the tanh component (`tanh_tile()`) calls `llk_math_eltwise_unary_sfpu_tanh()` which has NO implementation in the codebase. The analysis focuses on the fully-defined SFPU binary subtraction path (`_calculate_sfpu_binary_` with `BinaryOp::SUB`).

### Key Findings
- **Compute kernel**: Dedicated kernel (NOT `eltwise_sfpu.cpp`), two variants:
  - `tanhshrink_kernel.cpp` -- hybrid FPU+SFPU: tanh via SFPU, then FPU binary_dest_reuse subtraction
  - `tanhshrink_sfpu_kernel.cpp` -- pure SFPU: tanh via SFPU, then SFPU sub_binary_tile subtraction
- **SFPU_OP_CHAIN_0**: NOT USED -- dedicated kernel with direct API calls
- **Dispatch path**: BROKEN -- `get_op_init_and_func_default()` has no TANHSHRINK case (would throw)
- **Tanh component**: BROKEN -- `llk_math_eltwise_unary_sfpu_tanh` is referenced but never defined; no `ckernel_sfpu_tanh.h` exists; `SfpuType::tanh` not in Metal enum
- **Subtraction component (SFPU)**: FULLY DEFINED -- `_calculate_sfpu_binary_<APPROX, BinaryOp::SUB, 8>` in `ckernel_sfpu_binary.h`
- **Subtraction component (FPU)**: FULLY DEFINED -- `binary_dest_reuse_tiles<ELWSUB, DEST_TO_SRCB>` uses FPU math unit
- **Kernel style**: SFPI abstractions (Style A) for binary subtraction
- **WH/BH parity**: Identical `ckernel_sfpu_binary.h` on both platforms
- **SFPU instructions** (binary subtraction only): SFPLOAD (2 reads), SFPMAD (subtraction as a*1.0+(-b)), SFPSTORE (1 write)
- **Address mode**: ADDR_MOD_7 with all-zero increments (binary SFPU uses SfpuType::unused)
- **Approximation mode**: `false` from `get_op_approx_mode()`; `tanh_tile()` uses default `fast_and_approx=false`

### Files Read
1. `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
2. `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp`
3. `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp`
4. `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp`
5. `ttnn/cpp/ttnn/operations/eltwise/unary/unary.cpp`
6. `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp`
7. `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_device_operation.cpp`
8. `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/tanhshrink_kernel.cpp`
9. `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/tanhshrink_sfpu_kernel.cpp`
10. `tt_metal/hw/inc/api/compute/compute_kernel_api.h`
11. `tt_metal/hw/inc/api/compute/eltwise_binary_sfpu.h`
12. `tt_metal/hw/inc/api/compute/eltwise_binary.h`
13. `tt_metal/hw/inc/api/compute/eltwise_unary/eltwise_unary.h`
14. `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_binop.h`
15. `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_init.h`
16. `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h`
17. `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_math_unary_sfpu_api.h`
18. `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_binary_sfpu_params.h`
19. `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_binary_sfpu.h`
20. `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`
21. `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
22. `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_binary.h`
23. `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_binary.h`
24. `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/ckernel_sfpu.h`
25. `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/ckernel_defs.h`
26. `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_activations.h`
27. `tt_metal/third_party/tt_llk/tests/helpers/include/llk_sfpu_types.h`
28. `.claude/references/sfpu-hardware-model.md`
29. `.claude/references/logging/sfpu-operation-analyzer.md`

### Verification Results
- `_calculate_sfpu_binary_`: VERIFIED in `ckernel_sfpu_binary.h` (both WH and BH)
- `_sfpu_binary_init_`: VERIFIED in `ckernel_sfpu_binary.h` (both WH and BH)
- `sub_binary_tile`: VERIFIED in `eltwise_binary_sfpu.h`
- `sub_binary_tile_init`: VERIFIED in `eltwise_binary_sfpu.h`
- `tanh_tile`: VERIFIED as DECLARED in `compute_kernel_api.h` but calls UNDEFINED function
- `llk_math_eltwise_unary_sfpu_tanh`: VERIFIED as UNDEFINED -- no implementation anywhere in codebase
- No `ckernel_sfpu_tanh.h`: VERIFIED absent via find across all tt_metal directories
- `SfpuType::tanh`: VERIFIED present in test helpers `llk_sfpu_types.h`, ABSENT from production Metal `llk_sfpu_types.h`
- All file paths in abstraction layers table: VERIFIED
- SFPU instructions: VERIFIED via SFPI abstraction mappings (dst_reg reads -> SFPLOAD, vFloat arithmetic -> SFPMAD, dst_reg writes -> SFPSTORE)

### Output
- Analysis file: `.claude-analysis/softcap-1/tanhshrink_analysis.md`
