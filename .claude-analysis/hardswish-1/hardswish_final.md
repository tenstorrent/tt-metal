# hardswish -- Implementation Report

## Overview
- **Operation**: hardswish
- **Math definition**: x * min(max(x + 3, 0), 6) / 6
- **Date implemented**: 2026-04-04
- **Status**: PASS after 1 iteration
- **Output folder**: `.claude-analysis/hardswish-1/`

## Phase 1: Reference Discovery
- **Duration**: 198s
- **References selected**:
  1. **hardsigmoid** -- Direct building block; hardswish = x * hardsigmoid(x)
  2. **silu** -- Same structural pattern (x * f(x) gating)
  3. **hardtanh** -- Two-sided clamping with v_if/v_endif blocks
  4. **selu** -- Piecewise conditional with scalar multiply/add
  5. **softsign** -- Element-wise gating with scalar operations

## Phase 2: Reference Analysis
- **Duration**: 869s (wall-clock)
- **Agents launched**: 5
- **Results**: 5/5 succeeded (silu was late but completed)

| Reference | Analysis File | Status |
|-----------|---------------|--------|
| hardsigmoid | hardsigmoid_analysis.md | OK |
| silu | silu_analysis.md | OK (late) |
| hardtanh | hardtanh_analysis.md | OK |
| selu | selu_analysis.md | OK |
| softsign | softsign_analysis.md | OK |

## Phase 3: Implementation
- **Duration**: 554s
- **Key design decisions**:
  - Implemented as `x * hardsigmoid(x)` where hardsigmoid = clamp(x/6 + 0.5, 0, 1)
  - Self-contained SFPU kernel with no sub-function calls
  - Uses v_if/v_endif for clamping, same pattern as hardsigmoid
  - No init function needed (no programmable constants)
  - Standard VectorMode::RC dispatch with ADDR_MOD_7

### New Files
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_hardswish.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_hardswish.h`
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_hardswish.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_hardswish.h`
- `tt_metal/hw/inc/api/compute/eltwise_unary/hardswish.h`
- `tests/ttnn/unit_tests/operations/eltwise/test_hardswish.py`

### Modified Files
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h`
- `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h`
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
- `ttnn/cpp/ttnn/operations/eltwise/unary_ng/common/unary_ng_op_utils.cpp`
- `ttnn/ttnn/operations/unary.py`

## Phase 4: Testing & Debugging
- **Total iterations**: 1
- **Final result**: PASS
- **All 4 tests passed**: PCC >= 0.999

### Test Results
| Test | Shape | Result |
|------|-------|--------|
| test_hardswish[bfloat16, 1x1x32x32] | 1x1x32x32 | PASS |
| test_hardswish[bfloat16, 1x1x320x384] | 1x1x320x384 | PASS |
| test_hardswish[bfloat16, 1x3x320x384] | 1x3x320x384 | PASS |
| test_hardswish_piecewise[1x1x32x32] | 1x1x32x32 | PASS |

## Phase 5: Documentation
This file.

## Issues Encountered

| # | Phase | Severity | Description | Resolution |
|---|-------|----------|-------------|------------|
| 1 | 2 | LOW | silu analyzer agent was slow; completed late after other agents | All 5 analyses eventually produced |
| 2 | 4b | LOW | Implementation notes enrichment agent did not persist changes | Proceeded without enrichment |

## Timing Summary
- **Total wall-clock**: ~2204s (~37 min)
- **Phase 1 (Discovery)**: 198s
- **Phase 2 (Analysis)**: 869s
- **Phase 3 (Implementation)**: 554s
- **Phase 4 (Testing)**: 74s
- **Phase 5 (Documentation)**: ~30s
- **Phase 6 (Self-Reflection)**: pending
