# rrelu -- Implementation Report

## Overview
- **Operation**: rrelu (Randomized Leaky ReLU)
- **Math definition**: RReLU(x) = max(0, x) + a * min(0, x), where a = Uniform(lower, upper) in training, a = (lower + upper) / 2 in eval
- **Date implemented**: 2026-04-09
- **Status**: PASS after 1 iteration (2 test-fix cycles within the tester -- SfpuType enum fix)
- **Output folder**: `.claude-analysis/rrelu-1/`

## Phase 1: Reference Discovery
- **Duration**: ~279 seconds
- **References selected**:
  1. **hardshrink** -- Float scalar parameter + conditional logic + multiply pattern. Closest structural match.
  2. **swish** -- `v_if(x < 0.0f)` branching pattern in SFPU kernel.
  3. **hardtanh** -- Two-parameter parametrized op (min_val, max_val) -- shows how to pack/dispatch two float params.
  4. **frac** -- Sign-branch pattern (`v_if(exp < 0)`) -- demonstrates conditional SFPU arithmetic.
  5. **where_tss** -- Three-scalar conditional kernel -- shows two packed scalar runtime arguments.

## Phase 2: Reference Analysis
- **Duration**: ~518 seconds (wall-clock, all 5 in parallel)
- **Agents launched**: 5
- **Results**: 5/5 succeeded

| Reference | Analysis File | Commit | Status |
|-----------|---------------|--------|--------|
| swish | swish_analysis.md | 149b1ab3ee | OK |
| frac | frac_analysis.md | d5044dee1c | OK |
| hardshrink | hardshrink_analysis.md | fb94abf14e | OK |
| hardtanh | hardtanh_analysis.md | de71169a22 (bundled with where_tss) | OK |
| where_tss | where_tss_analysis.md | de71169a22 | OK |

Note: The hardtanh analysis was committed alongside where_tss because the where_tss agent's `git add .claude-analysis/rrelu-1/` picked up the hardtanh file that had been written to disk but not yet committed by its own agent.

## Phase 3: Implementation
- **Duration**: ~815 seconds
- **Commit**: c7221b9dd2
- **Key design decisions**:
  - Eval-mode only: slope is pre-computed on host as `(lower + upper) / 2` and packed as bfloat16 bits into the SFPU_OP_CHAIN_0 macro string
  - Standard parametrized unary SFPU pattern following hardtanh/frac templates
  - No runtime args needed (slope embedded in compute kernel defines)
  - Training mode (stochastic per-element slope) not implemented due to SFPU lacking suitable RNG

### Files Created
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_rrelu.h` -- SFPU kernel
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_rrelu.h` -- SFPU kernel (blackhole)
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_rrelu.h` -- LLK wrapper
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_rrelu.h` -- LLK wrapper (blackhole)
- `tt_metal/hw/inc/api/compute/eltwise_unary/rrelu.h` -- Compute API

### Files Modified
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h` -- Added RRELU to SfpuType enum
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h` -- Added RRELU to SfpuType enum
- `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h` -- Include rrelu.h
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp` -- Added RRELU to UnaryOpType enum
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp` -- Registered in get_block_defines, get_op_init_and_func
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp` -- Registered in is_parametrized_type, get_op_approx_mode
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp` -- Added C++ API function
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp` -- Python bindings
- `ttnn/cpp/ttnn/operations/eltwise/unary_ng/common/unary_ng_op_utils.cpp` -- NG golden function registration
- `ttnn/ttnn/operations/unary.py` -- Python API exposure

## Phase 4: Testing & Debugging
- **Total iterations**: 1 (tester internally fixed SfpuType enum before first successful test)
- **Final result**: PASS
- **Tests passed**: 4/4

### Test Results
| Test | Result |
|------|--------|
| test_rrelu_default[bfloat16] | PASSED |
| test_rrelu_default[fp32] | PASSED |
| test_rrelu_custom_params[bfloat16] | PASSED |
| test_rrelu_custom_params[fp32] | PASSED |

### Debug Notes
The tester identified and fixed a missing `SfpuType` enum issue in `llk_sfpu_types.h` for both wormhole_b0 and blackhole. The "deep nuke" operation that prepared the evaluation repo had stripped out standard comparison/inf/nan enum members needed by the third-party LLK library. The tester restored: `equal_zero`, `not_equal_zero`, `less_than_zero`, `greater_than_equal_zero`, `greater_than_zero`, `less_than_equal_zero`, `unary_ne/eq/gt/lt/ge/le`, `isinf`, `isposinf`, `isneginf`, `isnan`, `isfinite`.

## Phase 5: Documentation
This file.

## Issues Encountered

| # | Phase | Severity | Description | Resolution |
|---|-------|----------|-------------|------------|
| 1 | 2 | LOW | hardtanh analysis committed bundled with where_tss commit | Non-issue -- file was picked up by git add glob |
| 2 | 4 | MEDIUM | SfpuType enum missing standard members due to deep nuke | Tester restored missing enum members in both arch variants |

## Timing Summary
- **Total wall-clock**: ~2700 seconds (~45 minutes)
- **Phase 1 (Discovery)**: ~279s
- **Phase 2 (Analysis)**: ~518s
- **Phase 3 (Implementation)**: ~815s
- **Phase 4 (Testing)**: ~967s
- **Phase 5 (Documentation)**: ~30s
- **Phase 6 (Self-Reflection)**: pending
