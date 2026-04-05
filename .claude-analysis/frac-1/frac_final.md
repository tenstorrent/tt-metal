# frac -- Implementation Report

## Overview
- **Operation**: frac
- **Math definition**: frac(x) = x - trunc(x) (fractional part, preserving sign; matches torch.frac)
- **Date implemented**: 2026-04-05
- **Status**: PASS after 2 iterations
- **Output folder**: `.claude-analysis/frac-1/`

## Phase 1: Reference Discovery
- **Duration**: ~30s
- **References selected**:
  1. **softsign** - Simple unary SFPU op pattern (no params, basic arithmetic)
  2. **hardswish** - Recently generated, clean modern patterns
  3. **softshrink** - Parameterized op, shows v_if/v_endif conditional patterns
  4. **cbrt** - SFPI bit manipulation (reinterpret, shift, exponent extraction)
  5. **selu** - More complex op with conditional branching and constants

## Phase 2: Reference Analysis
- **Duration**: ~2 minutes
- **Agents launched**: 5 (inline analysis)
- **Results**: 5/5 succeeded

| Reference | Analysis File | Status |
|-----------|---------------|--------|
| softsign | softsign_analysis.md | OK |
| hardswish | hardswish_analysis.md | OK |
| softshrink | softshrink_analysis.md | OK |
| cbrt | cbrt_analysis.md | OK |
| selu | selu_analysis.md | OK |

## Phase 3: Implementation
- **Duration**: ~5 minutes
- **Key design decisions**:
  - Used IEEE 754 bit manipulation approach for trunc computation
  - Created dedicated `ckernel_sfpu_frac.h` rather than modifying existing rounding_ops
  - No init callback needed (only uses basic SFPI instructions)
  - Followed hardswish pattern for file structure (simplest no-param unary)
- **Files created**: 6 new files
- **Files modified**: 6 existing files

## Phase 4: Testing & Debugging
- **Total iterations**: 2
- **Final result**: PASS
- **PCC**: 0.999 (all shapes)

### Iteration Log
| # | Action | Test Result | Error | Fix Applied |
|---|--------|------------|-------|-------------|
| 1 | Initial implementation (x - floor(x)) | 3 pass, 1 fail | test_frac_properties: negative values in output | Changed kernel to x - trunc(x) to match torch.frac golden |
| 2 | Fixed kernel + test | 5 pass, 0 fail | - | - |

### Test Details
- `test_frac[bfloat16, 1x1x32x32]`: PASS (PCC=0.999)
- `test_frac[bfloat16, 1x1x320x384]`: PASS (PCC=0.999)
- `test_frac[bfloat16, 1x3x320x384]`: PASS (PCC=0.999)
- `test_frac_properties[1x1x32x32]`: PASS (|frac| < 1)
- `test_frac_integers[1x1x32x32]`: PASS (frac(integer) == 0)

## Phase 5: Documentation
This file.

## Files Created/Modified

### New Files
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_frac.h` -- SFPU kernel
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_frac.h` -- SFPU kernel (blackhole)
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_frac.h` -- LLK dispatch
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_frac.h` -- LLK dispatch (blackhole)
- `tt_metal/hw/inc/api/compute/eltwise_unary/frac.h` -- Compute API header
- `tests/ttnn/unit_tests/operations/eltwise/test_frac.py` -- Test file

### Modified Files
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h` -- Added `frac` to SfpuType enum
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h` -- Added `frac` to SfpuType enum
- `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h` -- Added SFPU_OP_FRAC_INCLUDE guard
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp` -- Registered frac op
- `ttnn/cpp/ttnn/operations/eltwise/unary_ng/common/unary_ng_op_utils.cpp` -- Registered frac in ng path
- `ttnn/ttnn/operations/unary.py` -- Added golden function for torch.frac

## Issues Encountered

| # | Phase | Severity | Description | Resolution |
|---|-------|----------|-------------|------------|
| 1 | 4 | Medium | Initial kernel used x-floor(x) semantics but golden uses torch.frac (x-trunc(x)) | Changed kernel to trunc-based approach, fixed test assertions |
| 2 | 3 | Low | tt_llk submodule empty; floor_tile/rounding_op_tile_init not available | Created standalone frac kernel using raw SFPI bit manipulation |

## Timing Summary
- **Total wall-clock**: ~13 minutes
- **Phase 1 (Discovery)**: ~30s
- **Phase 2 (Analysis)**: ~2 min
- **Phase 3 (Implementation)**: ~5 min
- **Phase 4 (Testing)**: ~5 min (2 iterations)
- **Phase 5 (Documentation)**: ~1 min
