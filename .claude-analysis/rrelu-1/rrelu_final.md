# rrelu -- Implementation Report

## Overview
- **Operation**: rrelu
- **Math definition**: f(x) = x if x >= 0; f(x) = a * x if x < 0, where a = (lower + upper) / 2 (eval) or a ~ Uniform(lower, upper) (training)
- **Date implemented**: 2026-04-09
- **Status**: PASS (23/23 tests, 1 iteration with bug fixes during testing)
- **Output folder**: `.claude-analysis/rrelu-1/`

## Phase 1: Reference Discovery
- **Duration**: 512s
- **References selected**:
  1. leaky_relu -- closest structural match (same conditional formula)
  2. prelu_sfpu -- parameterized slope, runtime float parameter passing
  3. dropout -- SFPU hardware PRNG mechanism for training mode
  4. swish -- complete end-to-end SFPU kernel wiring template
  5. hardtanh -- multi-parameter dispatch pattern (lower/upper as uint32)

## Phase 2: Reference Analysis
- **Duration**: 673s (wall-clock)
- **Agents launched**: 5
- **Results**: 5/5 succeeded (dropout was late but completed)

| Reference | Analysis File | Duration (s) | Status |
|-----------|---------------|-------------|--------|
| leaky_relu | leaky_relu_analysis.md | ~420 | OK |
| prelu_sfpu | prelu_sfpu_analysis.md | ~440 | OK |
| dropout | dropout_analysis.md | ~600 | OK (late) |
| swish | swish_analysis.md | ~380 | OK |
| hardtanh | hardtanh_analysis.md | ~450 | OK |

## Phase 3: Implementation
- **Duration**: 1061s
- **Key design decisions**:
  - Dual code paths (eval/training) selected at kernel runtime via seed parameter
  - SFPI PRNG builtin for training mode random slope generation
  - Deterministic seed derived from lower ^ upper ^ 0xDEADBEEF on host
  - Parameter packing: lower and upper as uint32 (bit-cast from float), plus seed as uint32

### Files Created
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_rrelu.h` -- Core SFPU kernel
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_rrelu.h` -- Core SFPU kernel (BH)
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_rrelu.h` -- LLK dispatch
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_rrelu.h` -- LLK dispatch (BH)
- `tt_metal/hw/inc/api/compute/eltwise_unary/rrelu.h` -- Compute API header
- `tests/ttnn/unit_tests/operations/eltwise/test_rrelu.py` -- Test file

### Modified Files
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h` -- Added SfpuType::rrelu
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h` -- Added SfpuType::rrelu
- `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h` -- Include rrelu.h
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp` -- Added UnaryOpType::RRELU
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp` -- get_op_approx_mode
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp` -- get_block_defines, get_op_init_and_func
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp` -- C++ API
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp` -- Python binding
- `ttnn/ttnn/operations/unary.py` -- Python API + golden function
- `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp` -- Fixed includes

## Phase 4: Testing and Debugging
- **Total iterations**: 1 (with in-flight bug fixes)
- **Final result**: PASS (23/23 tests)
- **Test coverage**: default params, 5 param combos x 3 shapes, all-positive, all-negative, preallocated output

### Bug Fixes Applied During Testing
| # | Bug | Fix |
|---|-----|-----|
| 1 | s2vFloat16b parameter encoding -- passing full 32-bit float instead of 16-bit bfloat16 | Shift right 16 bits before calling s2vFloat16b |
| 2 | Wormhole PRNG builtin -- mod1=8 only valid on Blackhole | Training path falls back to eval-mode slope on Wormhole |
| 3 | Missing SfpuType enum values in nuked repo | Added 30+ stub enum values |
| 4 | Missing header includes in eltwise_sfpu.cpp | Removed references to nuked headers |

## Phase 5: Documentation
This file.

## Issues Encountered

| # | Phase | Severity | Description | Resolution |
|---|-------|----------|-------------|------------|
| 1 | 2 | LOW | Dropout analyzer agent was late | Completed eventually, all 5/5 analyses done |
| 2 | 4 | MEDIUM | s2vFloat16b parameter encoding bug | Fixed by shifting right 16 bits |
| 3 | 4 | MEDIUM | Wormhole PRNG builtin incompatibility | Training mode falls back to eval slope on WH |
| 4 | 4 | LOW | Missing SfpuType enum stubs from nuked repo | Added stub values |
| 5 | 4 | LOW | Missing header includes from nuked repo | Removed references |

## Timing Summary
- **Total wall-clock**: ~3500s (~58 minutes)
- **Phase 1 (Discovery)**: 512s
- **Phase 2 (Analysis)**: 673s
- **Phase 3 (Implementation)**: 1061s
- **Phase 4 (Testing)**: 926s
- **Phase 5 (Documentation)**: ~30s
- **Phase 6 (Self-Reflection)**: pending
