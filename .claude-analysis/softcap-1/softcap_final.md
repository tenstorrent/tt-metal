# softcap -- Implementation Report

## Overview
- **Operation**: softcap
- **Math definition**: `cap * tanh(x / cap)`, where cap is a positive float scalar (default = 50.0)
- **Date implemented**: 2026-04-09
- **Status**: PASS after 1 iteration (28/28 tests passed)
- **Output folder**: `.claude-analysis/softcap-1/`

## Phase 1: Reference Discovery
- **Duration**: ~366s
- **References selected**:
  1. **atanh** -- Init-with-programmable-constants mechanism for loading the `cap` parameter
  2. **sinh** -- Builds hyperbolic function from exp with scalar multiplies
  3. **swish** -- `x * sigmoid(x)` mirrors the scalar-times-function pattern
  4. **tanhshrink** -- Only kernel that calls `tanh_tile()` directly
  5. **hardshrink** -- Canonical runtime float parameter pattern (`get_arg_val` + `reinterpret_cast`)

## Phase 2: Reference Analysis
- **Duration**: ~798s (wall-clock)
- **Agents launched**: 5
- **Results**: 5/5 succeeded

| Reference | Analysis File | Commit | Status |
|-----------|---------------|--------|--------|
| atanh | atanh_analysis.md | 831e48c797 | OK |
| sinh | sinh_analysis.md | 99e68cb576 | OK |
| swish | swish_analysis.md | 35d2e1ca83 | OK |
| tanhshrink | tanhshrink_analysis.md | 2ac09c13c5 | OK |
| hardshrink | hardshrink_analysis.md | d28196f833 | OK |

## Phase 3: Implementation
- **Duration**: ~1038s (~17 min)
- **Key design decisions**:
  - Piecewise polynomial approximation for tanh (4 regions) instead of exp-based
  - Cap parameter passed as hex-encoded uint32_t through compile-time defines
  - Division by cap replaced with multiplication by precomputed 1/cap
  - Coefficients fitted for bfloat16 accuracy across regions

### Files Created
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_softcap.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_softcap.h`
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_softcap.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_softcap.h`
- `tt_metal/hw/inc/api/compute/eltwise_unary/softcap.h`
- `tests/ttnn/unit_tests/operations/eltwise/test_softcap.py`

### Modified Files
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h`
- `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h`
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp`
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp`
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp`
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp`
- `ttnn/ttnn/operations/unary.py`
- `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`

## Phase 4: Testing & Debugging
- **Total iterations**: 1
- **Final result**: PASS (28/28 tests)
- **Test breakdown**:
  - 16 bfloat16 tests (4 shapes x 4 cap values, ULP threshold = 2)
  - 6 float32 tests (2 shapes x 3 cap values, allclose rtol=1.6e-2, atol=1e-2)
  - 2 bfloat16 allclose tests
  - 4 edge case tests (zeros, large values, small cap=0.5, default cap=50.0)

### Implementation Fixes Applied During Testing
1. **eltwise_sfpu.cpp**: Removed stale includes for nuked headers (trigonometry.h, rpow.h, rdiv.h, fill.h, mul_int_sfpu.h)
2. **llk_sfpu_types.h** (both arches): Added ~35 missing SfpuType enum values needed by third-party LLK code
3. **ckernel_sfpu_softcap.h** (both arches): Replaced `std::bit_cast<float>` with union-based conversion (C++20 not available in SFPU runtime)

## Phase 5: Documentation
This file.

## Issues Encountered

| # | Phase | Severity | Description | Resolution |
|---|-------|----------|-------------|------------|
| 1 | 4 | LOW | Stale includes in eltwise_sfpu.cpp | Removed references to nuked headers |
| 2 | 4 | LOW | Missing SfpuType enum values | Added ~35 enum values to llk_sfpu_types.h |
| 3 | 4 | MEDIUM | std::bit_cast not available in C++17 SFPU runtime | Replaced with union-based float<->uint32 conversion |

## Timing Summary
- **Total wall-clock**: ~50 minutes (Phases 1-5)
- **Phase 1 (Discovery)**: ~366s
- **Phase 2 (Analysis)**: ~798s
- **Phase 3 (Implementation)**: ~1038s
- **Phase 4 (Testing)**: ~817s
- **Phase 5 (Documentation)**: ~30s
- **Phase 6 (Self-Reflection)**: pending
