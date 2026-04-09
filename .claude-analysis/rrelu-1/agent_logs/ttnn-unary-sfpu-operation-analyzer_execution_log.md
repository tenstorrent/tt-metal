# Execution Log: ttnn-unary-sfpu-operation-analyzer

## Session Info (prelu_sfpu - prior run)
- **Operation**: prelu_sfpu
- **Agent**: ttnn-unary-sfpu-operation-analyzer
- **Date**: 2026-04-09
- **Status**: SUCCESS

## Summary (prelu_sfpu)
Analyzed the SFPU kernel implementation for the `PRELU_SFPU` unary operation. The operation was deep-nuked from this repository, so the analysis was reconstructed from pre-nuke documentation, surviving reference implementations (threshold, hardtanh, clamp), and the shared dispatch infrastructure.

## Key Findings (prelu_sfpu)
- **Formula**: `max(0, x) + weight * min(0, x)` = `x if x >= 0, weight * x if x < 0`
- **Macro group**: `SFPU_OP_PRELU_INCLUDE` (standalone)
- **API signature**: `prelu_tile(uint32_t idst, uint32_t param0)` + `prelu_tile_init()`
- **Kernel style**: SFPI-based (Style A) using `v_if`/`v_endif` for sign-conditional branching
- **Parameter**: Single `uint32_t param0` (bit-cast weight float)
- **Approximation mode**: `false` (no approximation-dependent branches in the algorithm)
- **SFPU instructions**: SFPLOAD, SFPSTORE, SFPMAD (multiply), SFPSETCC/SFPENCC/SFPPUSHC/SFPPOPC (CC management)
- **Address mode**: ADDR_MOD_7 with dest.incr=0 (standard for most unary SFPU ops)

## Deep-Nuke Impact (prelu_sfpu)
The following files were confirmed removed:
- `unary_op_types.hpp`: PRELU_SFPU enum entry removed
- `unary_op_utils.cpp`: get_op_init_and_func / get_macro_definition cases removed
- `prelu.h`: API header removed from `tt_metal/hw/inc/api/compute/eltwise_unary/`
- `ckernel_sfpu_prelu.h`: Core SFPU kernel removed from both WH and BH tt_llk directories
- `ckernel_sfpu_relu.h`: Emptied (just `#pragma once`)
- Per-operation LLK dispatch files: Removed from both tt_llk and hw/ckernels

## Reconstruction Confidence: HIGH
The reconstruction is based on well-documented formula, confirmed API signature from Doxygen docs, and structurally identical surviving operations.

## Output (prelu_sfpu)
- **Analysis file**: `.claude-analysis/rrelu-1/prelu_sfpu_analysis.md`

---

## Session Info (threshold)
- **Operation**: threshold
- **Agent**: ttnn-unary-sfpu-operation-analyzer
- **Date**: 2026-04-09
- **Status**: SUCCESS

## Summary (threshold)
Analyzed the SFPU kernel implementation for the `THRESHOLD` unary operation. While the API header and LLK dispatch files have been nuked, the core SFPU kernel (`ckernel_sfpu_threshold.h`) survives intact in both Wormhole B0 and Blackhole tt_llk directories. The analysis covers the complete SFPU implementation.

## Key Findings (threshold)
- **Formula**: `if (in <= threshold) { out = value } else { out = in }`
- **API signature**: `threshold_tile(uint32_t idst, uint32_t param0, uint32_t param1)` + `threshold_tile_init()`
- **Kernel style**: SFPI-based (Style A) using `v_if`/`v_endif` for LTE conditional
- **Parameters**: Two `uint32_t` params (threshold value and replacement value, both bitcast from float)
- **Approximation mode**: `false` (APPROXIMATION_MODE template param is unused in the function body)
- **SFPU instructions**: SFPLOAD, SFPSTORE, SFPMAD, SFPSETCC, SFPENCC, SFPPUSHC, SFPPOPC, SFPLOADI
- **Address mode**: ADDR_MOD_7 with dest.incr=0 (same on WH and BH)
- **WH/BH parity**: Core kernel is byte-for-byte identical on both architectures

## Execution Timeline (threshold)
1. Read reference files (sfpu-hardware-model.md, diagram-templates.md, logging docs)
2. Traced dispatch in unary_op_utils.cpp -- THRESHOLD in enum but dispatch cases nuked
3. Found core SFPU kernel in tt_llk (WH + BH), confirmed identical
4. Read LLK dispatch infrastructure (params, init, addr_mod)
5. Read sfpi.h for v_if/v_endif, vFloat comparison instruction mapping
6. Verified all identifiers and file paths via grep
7. Wrote threshold_analysis.md

## Output (threshold)
- **Analysis file**: `.claude-analysis/rrelu-1/threshold_analysis.md`

---

## Session Info (leaky_relu)
- **Operation**: leaky_relu
- **Agent**: ttnn-unary-sfpu-operation-analyzer
- **Date**: 2026-04-09
- **Status**: SUCCESS

## Summary (leaky_relu)
Analyzed the SFPU kernel implementation for the `LEAKY_RELU` unary operation. Like prelu_sfpu, the operation was deep-nuked from this branch (Phase 2 nuke), so the analysis was reconstructed from the nuke manifest, surviving structurally-identical kernels (threshold, sign), and the shared LLK dispatch infrastructure.

## Key Findings (leaky_relu)
- **Formula**: `max(0, x) + negative_slope * min(0, x)` = `x if x >= 0, slope * x if x < 0`
- **Macro group**: `SFPU_OP_RELU_FAMILY_INCLUDE` or `SFPU_OP_COMPUTE_KERNEL_API_INCLUDE` (original uncertain)
- **API signature**: `leaky_relu_tile(uint32_t idst, uint32_t param0)` + `leaky_relu_tile_init()`
- **Kernel style**: SFPI-based (Style A) using `v_if`/`v_endif` for sign-conditional branching
- **Parameter**: Single `uint32_t param0` (bit-cast negative_slope float, default=0.01)
- **Approximation mode**: `false` (no approximation-dependent branches)
- **Core function**: `_calculate_lrelu_<APPROXIMATION_MODE, ITERATIONS>` in `ckernel_sfpu_relu.h` (DELETED)
- **SFPU instructions**: SFPLOAD, SFPSTORE, SFPMAD (multiply), SFPSETCC/SFPENCC (CC management), SFPLOADI (constants)
- **Address mode**: ADDR_MOD_7 with dest.incr=0 (standard for most unary SFPU ops)

## Deep-Nuke Impact (leaky_relu)
The following files were confirmed removed:
- `unary_op_types.hpp`: LEAKY_RELU enum entry removed (Phase 2)
- `unary_op_utils.cpp`: get_op_init_and_func_parameterized case removed
- `ckernel_sfpu_relu.h` (WH+BH): gutted to `#pragma once` only
- `ckernel_sfpu_lrelu.h` (Quasar): deleted entirely
- LLK dispatch files: removed from both tt_llk and hw/ckernels
- Compute API header: removed from activations.h include chain

## Reconstruction Confidence: HIGH
Leaky ReLU is a trivially simple operation (conditional multiply) with well-documented formula. The reconstruction is based on the exact same SFPI patterns demonstrated by surviving threshold and sign kernels. The nuke manifest confirms the function name (`_calculate_lrelu_`) and file location.

## Verification Results (leaky_relu)
- `_calculate_lrelu_`: NOT FOUND (confirmed nuked)
- `_calculate_threshold_`: FOUND in wh+bh (pattern reference)
- `_calculate_sign_`: FOUND in wh+bh (pattern reference)
- `Converter::as_float`: FOUND in wh+bh
- `_llk_math_eltwise_unary_sfpu_params_`: FOUND in wh+bh+quasar
- `_llk_math_eltwise_unary_sfpu_init_`: FOUND in wh+bh+quasar
- `eltwise_unary_sfpu_configure_addrmod`: FOUND in wh+bh
- `SFPU_UNARY_ONE_PARAM_KERNEL_FN`: FOUND in macros.h

## Output (leaky_relu)
- **Analysis file**: `.claude-analysis/rrelu-1/leaky_relu_analysis.md`

---

## Session Info (dropout)
- **Operation**: dropout
- **Agent**: ttnn-unary-sfpu-operation-analyzer
- **Date**: 2026-04-09
- **Status**: SUCCESS

## Summary (dropout)
Analyzed the SFPU kernel implementation for the `DROPOUT` operation. This is a non-standard operation -- it uses its own experimental program factory (`DropoutProgramFactory`) rather than the standard `UnaryProgramFactory`. The API header and LLK dispatch files have been nuked, but the core SFPU kernel (`ckernel_sfpu_dropout.h`) survives intact in both Wormhole B0 and Blackhole tt_llk directories.

## Key Findings (dropout)
- **Formula**: For each element: `output = (rand > probability) ? input * scale : 0.0` where scale = `1 / (1 - prob)`
- **API signature**: `dropout_tile(uint32_t idst, uint32_t probability, uint32_t scale_factor)` + `dropout_kernel_init(uint32_t seed)`
- **Kernel style**: Raw TTI instructions (Style B-adjacent, but simple enough CC logic for Style A annotation)
- **Parameters**: Three params -- probability (uint32 = INT_MAX * prob), scale (bitcast float), seed (uint32 runtime arg)
- **Approximation mode**: `false` (hardcoded in program factory; APPROXIMATION_MODE template param unused in kernel body)
- **SFPU instructions**: TT_SFPLOADI (x4), TTI_SFPLOAD, TTI_SFPMUL, TTI_SFPMOV (x2, one for PRNG), TTI_SFPSETSGN, TTI_SFPIADD, TTI_SFPENCC, TTI_SFPSTORE
- **PRNG mechanism**: Special SFPMOV mode (instr_mod1=8, lreg_c=9) generates pseudorandom numbers
- **Address mode**: ADDR_MOD_7 with dest.incr=0 (same on WH and BH)
- **WH/BH parity**: Core kernel is byte-for-byte identical on both architectures

## Execution Timeline (dropout)
1. Read unary_op_utils.cpp -- confirmed DROPOUT enum exists but dispatch cases absent (experimental op)
2. Found dropout in experimental/ directory with own program factory
3. Read compute kernel (dropout_kernel.cpp) -- identified dropout_tile() and dropout_kernel_init() calls
4. Read core SFPU kernel (ckernel_sfpu_dropout.h) for both WH and BH -- confirmed identical
5. Read LLK dispatch infrastructure (params.h, unary_sfpu.h, addr_mod config)
6. Read init_prng_seed in ckernel.h (seed write + 600 NOP wait)
7. Read sfpu-hardware-model.md for SFPIADD CC semantics and instruction timing
8. Verified all function names, instruction usage, and file paths via grep
9. Wrote dropout_analysis.md with CC State Machine diagram

## Verification Results (dropout)
- `_calculate_dropout_`: FOUND in wh+bh ckernel_sfpu_dropout.h
- `_init_dropout_`: FOUND in wh+bh ckernel_sfpu_dropout.h
- `init_prng_seed`: FOUND in wh+bh ckernel.h
- All 8 TTI instructions: verified present in ckernel_sfpu_dropout.h
- All file paths: verified existing

## Output (dropout)
- **Analysis file**: `.claude-analysis/rrelu-1/dropout_analysis.md`
- **Commit**: `7730652612`

---

## Session Info (hardtanh)
- **Operation**: hardtanh
- **Agent**: ttnn-unary-sfpu-operation-analyzer
- **Date**: 2026-04-09
- **Status**: SUCCESS

## Summary (hardtanh)
Analyzed the SFPU kernel implementation for the `HARDTANH` unary operation on a deeply-nuked codebase. The core SFPU implementation (`ckernel_sfpu_hardtanh.h`) survives intact in tt_llk for both Wormhole B0 and Blackhole. Upper dispatch layers (compute API header, Metal LLK dispatch, TTNN dispatch case) were deleted by the deep nuke.

## Key Findings (hardtanh)
- **Formula**: `clamp(x, low, high)` with defaults low=-1, high=1
- **API signature**: `hardtanh_tile(uint32_t idst, uint32_t param0, uint32_t param1)` + `hardtanh_tile_init()` (from Doxygen docs)
- **Kernel style**: Pure SFPI abstraction (Style A) using `vFloat`, `dst_reg`, `v_if`/`v_endif`
- **Algorithm**: Additive offset clamping technique -- shifts comparison bounds to zero to exploit efficient LT0/GTE0 condition codes
- **Parameters**: 3 FP16_B uint32_t values encoding `-low`, `low-high`, and `high`
- **Approximation mode**: `false` (APPROXIMATION_MODE template parameter is completely unused -- no branch on it)
- **SFPU instructions per iteration**: 11 (SFPLOAD, 3x SFPMAD, 2x SFPSETCC, 2x SFPLOADI for 0.0, 2x SFPCOMPC, SFPSTORE)
- **Address mode**: ADDR_MOD_7 with dest.incr=0 (standard for most unary SFPU ops)
- **WH/BH parity**: Core SFPU kernel is byte-for-byte identical on both architectures

## Execution Timeline (hardtanh)
1. Read unary_op_utils.hpp/.cpp -- confirmed HARDTANH in enum, is_parametrized_type returns true, but dispatch cases nuked
2. Read DEEP_NUKE_MANIFEST.md -- confirmed hardtanh dispatch/API/LLK/ckernel deleted, tt_llk primitive survives
3. Read ckernel_sfpu_hardtanh.h (WH+BH) -- confirmed identical, analyzed additive offset algorithm
4. Read ckernel_sfpu_clamp.h for comparison -- different approach (direct comparison)
5. Read llk_math_eltwise_unary_sfpu_params.h (WH+BH) -- confirmed face iteration dispatch
6. Read llk_math_eltwise_unary_sfpu.h (WH+BH) -- confirmed ADDR_MOD_7 for hardtanh
7. Read sfpi_fp16.h for s2vFloat16b semantics (uint32_t pass-through for pre-encoded FP16_B)
8. Read sfpi.h for v_if/v_endif instruction mapping (SFPSETCC/SFPCOMPC)
9. Read llk_math_eltwise_unary_sfpu_macros.h -- identified SFPU_UNARY_THREE_PARAM_KERNEL_FN as likely dispatch macro
10. Verified all function names and file paths via grep
11. Wrote hardtanh_analysis.md with algorithm derivation and parameter encoding analysis

## Verification Results (hardtanh)
- `_calculate_hardtanh_`: FOUND in wh+bh ckernel_sfpu_hardtanh.h
- `_llk_math_eltwise_unary_sfpu_params_`: FOUND in wh+bh llk_math_eltwise_unary_sfpu_params.h
- `_llk_math_eltwise_unary_sfpu_init_`: FOUND in wh+bh llk_math_eltwise_unary_sfpu.h
- `eltwise_unary_sfpu_configure_addrmod`: FOUND in wh+bh llk_math_eltwise_unary_sfpu.h
- `SFPU_UNARY_THREE_PARAM_KERNEL_FN`: FOUND in llk_math_eltwise_unary_sfpu_macros.h
- `SfpuType::hardtanh`: FOUND in llk_sfpu_types.h
- All file paths: verified existing

## Output (hardtanh)
- **Analysis file**: `.claude-analysis/rrelu-1/hardtanh_analysis.md`
