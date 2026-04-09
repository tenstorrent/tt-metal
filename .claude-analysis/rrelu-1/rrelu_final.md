# rrelu -- Implementation Report

## Overview
- **Operation**: rrelu
- **Math definition**: RReLU(x) = x if x >= 0, a * x if x < 0; Training: a ~ Uniform(lower, upper); Eval: a = (lower + upper) / 2
- **Parameters**: lower (default 1/8), upper (default 1/3), training (default False)
- **Date implemented**: 2026-04-09
- **Status**: PASS after 1 iteration
- **Output folder**: `.claude-analysis/rrelu-1/`

## Phase 1: Reference Discovery
- **Duration**: 401s
- **References selected**:
  1. **leaky_relu** -- structurally identical to rrelu eval mode (conditional multiply with single slope)
  2. **prelu_sfpu** -- demonstrates parameterized negative-branch scaling pattern
  3. **dropout** -- PRNG usage pattern needed for training mode random slope generation
  4. **threshold** -- two-parameter SFPU kernel pattern (uint32_t to float conversion)
  5. **hardtanh** -- dual-parameter bounds handling with v_if/v_endif branching

## Phase 2: Reference Analysis
- **Duration**: 1097s (wall-clock)
- **Agents launched**: 5
- **Results**: 5/5 succeeded

| Reference | Analysis File | Status |
|-----------|---------------|--------|
| leaky_relu | leaky_relu_analysis.md | OK |
| prelu_sfpu | prelu_sfpu_analysis.md | OK |
| dropout | dropout_analysis.md | OK |
| threshold | threshold_analysis.md | OK |
| hardtanh | hardtanh_analysis.md | OK |

## Phase 3: Implementation
- **Duration**: 1143s
- **Key design decisions**:
  - Eval mode uses precomputed slope = (lower + upper) / 2 passed as single uint32_t parameter (identical to leaky_relu pattern)
  - Training mode uses PRNG via TTI_SFPMOV special mode for random number generation, with bit manipulation for uniform float construction
  - Overloaded rrelu_tile() and rrelu_tile_init() for eval (1-param) vs training (2-param) modes
  - Dedicated SFPU_OP_RRELU_INCLUDE macro group for conditional include
  - Registered as parameterized type in is_parametrized_type()

### Files Created
- `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_rrelu.h` -- SFPU kernel (WH)
- `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_rrelu.h` -- SFPU kernel (BH)
- `tt_metal/third_party/tt_llk/tt_llk_quasar/common/inc/sfpu/ckernel_sfpu_rrelu.h` -- SFPU kernel (Quasar, eval only)
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_rrelu.h` -- LLK dispatch (WH)
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_rrelu.h` -- LLK dispatch (BH)
- `tt_metal/hw/inc/api/compute/eltwise_unary/rrelu.h` -- Compute API header
- `tests/ttnn/unit_tests/operations/eltwise/test_rrelu.py` -- Test file

### Files Modified
- `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h` -- Added SFPU_OP_RRELU_INCLUDE
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp` -- Added RRELU to UnaryOpType enum
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp` -- Registered in op utils
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp` -- get_block_defines, get_op_init_and_func, get_op_approx_mode
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp` -- Python API function declaration
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp` -- Python binding
- `ttnn/ttnn/experimental_loader/golden_functions.py` -- Golden function registration
- `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp` -- Stale includes removed

## Phase 4: Testing & Debugging
- **Total iterations**: 1
- **Final result**: PASS
- **Tests**: 18 total, 18 passed

### Test Cases
| Test | Cases | Result |
|------|-------|--------|
| test_rrelu_eval_bfloat16 | 9 (3 shapes x 3 param combos) | PASS |
| test_rrelu_eval_default_params | 2 | PASS |
| test_rrelu_eval_positive_inputs | 4 (identity for x>=0) | PASS |
| test_rrelu_eval_negative_inputs | 2 (slope multiplication for x<0) | PASS |
| test_rrelu_eval_l1_memory | 1 (L1 memory config) | PASS |

### Issues Fixed During Testing
- Removed stale `#include` directives from `eltwise_sfpu.cpp` (references to trigonometry.h, rpow.h, rdiv.h, fill.h, mul_int_sfpu.h that no longer exist in the nuked checkout)

## Phase 5: Documentation
This file.

## Phase 6: Self-Reflection
See `sfpu_reflection.md` in this directory.

## Issues Encountered

| # | Phase | Severity | Description | Resolution |
|---|-------|----------|-------------|------------|
| 1 | 4 | LOW | Stale includes in eltwise_sfpu.cpp from nuked checkout | Tester removed the stale includes |
| 2 | 3 | INFO | Quasar training mode not implemented (different SFPAND/SFPOR instruction signatures) | Only eval mode provided for Quasar platform |
| 3 | 3 | INFO | Training mode PRNG seed is deterministic per (lower, upper) pair | Acceptable for current implementation; production would need per-core seeds |

## Timing Summary
- **Phase 1 (Discovery)**: 401s
- **Phase 2 (Analysis)**: 1097s
- **Phase 3 (Implementation)**: 1143s
- **Phase 4 (Testing)**: 1132s
- **Phase 5 (Documentation)**: ~60s
- **Phase 6 (Self-Reflection)**: 518s
- **Total wall-clock**: ~4762s (~79 min)
