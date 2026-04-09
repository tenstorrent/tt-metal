# rrelu -- Implementation Report

## Overview
- **Operation**: rrelu (Randomized Leaky ReLU)
- **Math definition**: f(x) = x if x >= 0, a*x if x < 0; Eval: a = (lower + upper) / 2; Train: a ~ Uniform(lower, upper)
- **Date implemented**: 2026-04-09
- **Status**: PASS after 1 iteration (6/6 tests passed first try)
- **Output folder**: `.claude-analysis/rrelu-1/`

## Phase 1: Reference Discovery
- **Duration**: ~403s
- **References selected**:
  1. **swish** -- Complete SFPU kernel with v_if conditional pattern; best end-to-end reference
  2. **hardshrink** -- Parameterized op with scalar passing through runtime args
  3. **frac** -- Clean SFPI conditional kernel with v_if/v_endif
  4. **sinh** -- Helper function composition, conditional override, bfloat16 rounding
  5. **atanh** -- Programmable constant registers (vConstFloatPrgm0) for precomputing values in init

## Phase 2: Reference Analysis
- **Duration**: ~463s (wall-clock)
- **Agents launched**: 5
- **Results**: 5/5 succeeded

| Reference | Analysis File | Commit | Status |
|-----------|---------------|--------|--------|
| swish | swish_analysis.md | 852c695ebd | OK |
| hardshrink | hardshrink_analysis.md | c4c1417b48 | OK |
| frac | frac_analysis.md | 606e2cfad0 | OK |
| sinh | sinh_analysis.md | e4f7554a83 | OK |
| atanh | atanh_analysis.md | 7273b8dc66 | OK |

## Phase 3: Implementation
- **Duration**: ~1183s
- **Commit**: 5177e0576a
- **Key design decisions**:
  - Used atanh pattern for programmable constant registers (vConstFloatPrgm0/1/2) to precompute eval-mode midpoint
  - Used swish pattern for v_if sign-based conditional branching
  - Used hardshrink pattern for 3-parameter passing via UnaryWithParam
  - Training mode uses xorshift PRNG with per-lane diversity from input bits
  - Parameters passed as bit-cast hex literals to kernel init/func strings

### Files Created
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_rrelu.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_rrelu.h`
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_rrelu.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_rrelu.h`
- `tt_metal/hw/inc/api/compute/eltwise_unary/rrelu.h`
- `tests/ttnn/unit_tests/operations/eltwise/test_rrelu.py`

### Files Modified
- `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h`
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h`
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp`
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp`
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp`
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary.cpp`
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp`
- `ttnn/ttnn/operations/unary.py`
- `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`

## Phase 4: Testing & Debugging
- **Total iterations**: 1
- **Final result**: PASS
- **Tests**: 6/6 passed in 9.6s
- **Coverage**:
  - 3 parameter combinations: default (0.125, 1/3), wide (0.0, 0.5), constant (0.1, 0.1)
  - 2 dtypes: bfloat16 and fp32
  - Exhaustive input: all 65,536 bfloat16 bit patterns
  - Both allclose and ULP assertions

### Iteration Log
| # | Action | Test Result | Error | Fix Applied |
|---|--------|-------------|-------|-------------|
| 1 | Initial implementation | 6/6 PASSED | - | - |

## Phase 5: Documentation
This file.

## Phase 6: Self-Reflection
See `sfpu_reflection.md` in the output folder.

## Issues Encountered

| # | Phase | Severity | Description | Resolution |
|---|-------|----------|-------------|------------|
| - | - | - | No issues encountered | - |

## Timing Summary
- **Total wall-clock**: ~2400s (~40 min)
- **Phase 1 (Discovery)**: ~403s
- **Phase 2 (Analysis)**: ~463s
- **Phase 3 (Implementation)**: ~1183s
- **Phase 4 (Testing)**: ~108s
- **Phase 5 (Documentation)**: ~30s
- **Phase 6 (Self-Reflection)**: pending
