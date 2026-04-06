# sinh -- Implementation Report

## Overview
- **Operation**: sinh
- **Math definition**: sinh(x) = (exp(x) - exp(-x)) / 2
- **Date implemented**: 2026-04-06
- **Status**: PASS after 2 iterations (initial + Taylor fix)
- **Output folder**: `.claude-analysis/sinh-1/`

## Phase 1: Reference Discovery
- **Duration**: ~320s
- **References selected**:
  1. **rpow** -- Contains the exp_21f (2^z) algorithm, directly reusable for exp(x) = 2^(x*log2(e))
  2. **cbrt** -- setsgn/abs sign-handling pattern, addexp exponent manipulation
  3. **hardsigmoid** -- Cleanest minimal SFPU kernel skeleton, canonical loop/init structure
  4. **hardswish** -- Composite subexpression pattern: compute intermediate, combine with input
  5. **softshrink** -- Parameter decoding and constant setup idioms

## Phase 2: Reference Analysis
- **Duration**: ~964s (wall-clock)
- **Agents launched**: 5
- **Results**: 5/5 succeeded

| Reference | Analysis File | Status |
|-----------|---------------|--------|
| rpow | rpow_analysis.md | OK |
| cbrt | cbrt_analysis.md | OK |
| hardsigmoid | hardsigmoid_analysis.md | OK |
| hardswish | hardswish_analysis.md | OK |
| softshrink | softshrink_analysis.md | OK |

## Phase 3: Implementation
- **Duration**: ~1198s
- **Key design decisions**:
  - Extracted exp_21f algorithm from rpow into reusable inline helper
  - Computes sinh(x) = (2^(x*log2e) - 2^(-x*log2e)) / 2
  - Clamps z values to >= -127 to prevent underflow
  - Uses float_to_fp16b for explicit bfloat16 rounding
  - Identical implementation on Wormhole B0 and Blackhole architectures

## Phase 4: Testing and Debugging
- **Total iterations**: 2
- **Final result**: PASS
- **Test file**: `tests/ttnn/unit_tests/operations/eltwise/test_sinh.py`

### Iteration Log
| # | Action | Test Result | Error | Fix Applied |
|---|--------|------------|-------|-------------|
| 1 | Initial implementation | FAIL | Catastrophic cancellation for small |x|: exp(x)-exp(-x) near 0 | Taylor approximation: sinh(x) ~ x + x^3/6 for |x|<0.5 |
| 2 | Taylor approx fix | PASS | - | - |

### Key Discovery
The JIT compiler resolves kernel headers from the runtime install root, not the worktree. The sinh operation dispatches through SFPU_OP_TRIG_FAMILY_INCLUDE -> trigonometry.h -> ckernel_sfpu_trigonometry.h.

## Phase 5: Documentation
This file.

## Files Created/Modified

### New Files
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_sinh.h` -- SFPU kernel (exp_21f-based)
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_sinh.h` -- SFPU kernel (Blackhole)
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_sinh.h` -- LLK dispatch
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_sinh.h` -- LLK dispatch (Blackhole)
- `tt_metal/hw/inc/api/compute/eltwise_unary/sinh.h` -- Compute API header
- `tests/ttnn/unit_tests/operations/eltwise/test_sinh.py` -- Test file
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_trigonometry.h` -- Trig family kernel with Taylor fix
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_trigonometry.h` -- Trig family kernel (Blackhole)

### Modified Files
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h` -- Added SfpuType::sinh
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h` -- Added SfpuType::sinh
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_math_unary_sfpu_api.h` -- Added sinh dispatch case
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_math_unary_sfpu_api.h` -- Added sinh dispatch case
- `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h` -- Added sinh include
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp` -- Registered SINH op
- `ttnn/cpp/ttnn/operations/eltwise/unary_ng/common/unary_ng_op_utils.cpp` -- Registered SINH op (ng)
- `ttnn/ttnn/operations/unary.py` -- Python binding for ttnn.sinh

## Issues Encountered

| # | Phase | Severity | Description | Resolution |
|---|-------|----------|-------------|------------|
| 1 | 2 | MINOR | cbrt and hardswish analyzer agents were slow | Both eventually committed successfully; all 5/5 completed |
| 2 | 4 | MINOR | Catastrophic cancellation for small |x| | Tester added Taylor approx path: sinh(x) ~ x + x^3/6 for |x| < 0.5 |
| 3 | 4 | MINOR | clang-format pre-commit hook failed | Re-staged formatted files and re-committed |

## Timing Summary
- **Total wall-clock**: ~86 min (5116s from start to Phase 5)
- **Phase 1 (Discovery)**: ~320s
- **Phase 2 (Analysis)**: ~964s
- **Phase 3 (Implementation)**: ~1198s
- **Phase 4 (Testing)**: ~2293s
- **Phase 5 (Documentation)**: ~60s
- **Phase 6 (Self-Reflection)**: pending
