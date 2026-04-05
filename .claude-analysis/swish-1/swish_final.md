# swish -- Implementation Report

## Overview
- **Operation**: swish
- **Math definition**: swish(x) = x * sigmoid(x) = x / (1 + exp(-x))
- **Date implemented**: 2026-04-05
- **Status**: PASS after 1 iteration
- **Output folder**: `.claude-analysis/swish-1/`

## Phase 1: Reference Discovery
- **Duration**: ~4 minutes
- **References selected**:
  1. **hardswish** -- Closest structural match: x * f(x) pattern where f is a sigmoid variant
  2. **selu** -- Uses exp() with conditional logic, shows _calculate_exponential_piecewise_ and exp init
  3. **cosh** -- Direct use of _sfpu_exp_21f_bf16_, simple exp pattern
  4. **hardsigmoid** -- Sigmoid approximation, clamping patterns
  5. **softsign** -- x / (1 + |x|) pattern, shows reciprocal usage

## Phase 2: Reference Analysis
- **Duration**: ~40 seconds (inline analysis, no sub-agents)
- **Agents launched**: 0 (performed inline by orchestrator)
- **Results**: 5/5 analyzed successfully

| Reference | Analysis File | Duration (s) | Tokens | Status |
|-----------|---------------|-------------|--------|--------|
| hardswish | hardswish_analysis.md | ~8 | N/A | OK |
| selu | selu_analysis.md | ~8 | N/A | OK |
| cosh | cosh_analysis.md | ~8 | N/A | OK |
| hardsigmoid | hardsigmoid_analysis.md | ~8 | N/A | OK |
| softsign | softsign_analysis.md | ~8 | N/A | OK |

## Phase 3: Implementation
- **Duration**: ~27 minutes (including build)
- **Key design decisions**:
  - Used `_sfpu_exp_21f_bf16_` for exp(-x) computation (good bfloat16 accuracy)
  - Used `_sfpu_reciprocal_<2>` (2 Newton-Raphson iterations) for computing 1/(1+exp(-x))
  - Used `#pragma GCC unroll 0` since exp is computationally complex
  - Both `_init_exponential_` and `_init_sfpu_reciprocal_` called in `swish_init()`
  - Removed old `swish` inline alias for `silu` in unary.hpp, replaced with first-class SFPU operation

### Files Created
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_swish.h` -- SFPU kernel
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_swish.h` -- SFPU kernel (wormhole)
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_swish.h` -- LLK dispatch
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_swish.h` -- LLK dispatch (wormhole)
- `tt_metal/hw/inc/api/compute/eltwise_unary/swish.h` -- Compute API
- `tests/ttnn/unit_tests/operations/eltwise/test_swish.py` -- Test file

### Files Modified
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h` -- Added `swish` to SfpuType enum
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h` -- Added `swish` to SfpuType enum
- `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h` -- Added SFPU_OP_SWISH_INCLUDE guard
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp` -- Added SWISH to UnaryOpType enum
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp` -- Registered in get_macro_definition and get_op_init_and_func_default
- `ttnn/cpp/ttnn/operations/eltwise/unary_ng/common/unary_ng_op_utils.cpp` -- Registered in unary_ng utils
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp` -- Added REGISTER_UNARY_OPERATION(swish, SWISH), removed old alias
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp` -- Added nanobind binding
- `ttnn/ttnn/operations/unary.py` -- Added golden function using torch.nn.functional.silu

## Phase 4: Testing & Debugging
- **Total iterations**: 1
- **Final result**: PASS
- **All 5 tests passed on first try**

### Test Results
| Test | Result | Duration |
|------|--------|----------|
| test_swish[bfloat16, 1x1x32x32] | PASS | 3.45s |
| test_swish[bfloat16, 1x1x320x384] | PASS | 3.93s |
| test_swish[bfloat16, 1x3x320x384] | PASS | 4.23s |
| test_swish_properties[1x1x32x32] | PASS | 0.06s |
| test_swish_wide_range[1x1x32x32] | PASS | 0.08s |

All tests use PCC (Pearson Correlation Coefficient) >= 0.998 threshold.

## Phase 5: Documentation
This file.

## Issues Encountered

| # | Phase | Severity | Description | Resolution |
|---|-------|----------|-------------|------------|
| 1 | 3 | Minor | Name collision: `swish` was already an inline alias for `silu` in unary.hpp | Removed the alias, replaced with REGISTER_UNARY_OPERATION |
| 2 | 3 | Minor | Build failure due to missing submodules in worktree | Initialized submodules with git submodule update --init |
| 3 | 3 | Minor | tt_ops_code_gen submodule failed to fetch (remote ref issue) | Copied from main repo, then cleaned up to avoid git errors |

## Timing Summary
- **Total wall-clock**: ~35 minutes
- **Phase 1 (Discovery)**: ~4 min
- **Phase 2 (Analysis)**: ~1 min
- **Phase 3 (Implementation)**: ~27 min (dominated by build time)
- **Phase 4 (Testing)**: ~1 min
- **Phase 5 (Documentation)**: ~2 min

## Architecture: Implementation Layers

```
Layer 1: SFPU Kernel (ckernel_sfpu_swish.h)
    - calculate_swish<APPROXIMATION_MODE, is_fp32_dest_acc_en, ITERATIONS>()
    - swish_init<APPROXIMATION_MODE>()
    |
Layer 2: LLK Dispatch (llk_math_eltwise_unary_sfpu_swish.h)
    - llk_math_eltwise_unary_sfpu_swish_init<APPROXIMATE>()
    - llk_math_eltwise_unary_sfpu_swish<APPROXIMATE, ITERATIONS>()
    |
Layer 3: Compute API (swish.h)
    - swish_tile(uint32_t idst)
    - swish_tile_init()
    |
Layer 4: Split Includes (sfpu_split_includes.h)
    - #if SFPU_OP_SWISH_INCLUDE -> #include "swish.h"
    |
Layer 5: Op Registration (unary_op_utils.cpp, unary_ng_op_utils.cpp)
    - get_macro_definition(SWISH) -> "SFPU_OP_SWISH_INCLUDE"
    - get_op_init_and_func_default(SWISH) -> swish_tile_init()/swish_tile()
    |
Layer 6: C++ API (unary.hpp)
    - REGISTER_UNARY_OPERATION(swish, SWISH)
    |
Layer 7: Python Binding (unary_nanobind.cpp)
    - bind_unary_operation<"swish", &ttnn::swish>()
    |
Layer 8: Golden Function (unary.py)
    - torch.nn.functional.silu(input_tensor_a)
```
