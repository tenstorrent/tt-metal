# softcap -- Implementation Report

## Overview
- **Operation**: softcap
- **Math definition**: cap * tanh(x / cap)
- **Date implemented**: 2026-04-14
- **Status**: PARTIAL PASS -- 3/6 tests pass (all bfloat16), 3/6 fail (all fp32)
- **Output folder**: `.claude-analysis/softcap-1/`

## Phase 1: Reference Discovery
- **Duration**: ~234 seconds
- **References selected**: hardtanh, swish, sinh, atanh, frac

## Phase 2: Reference Analysis
- **Duration**: ~744 seconds (wall-clock)
- **Agents launched**: 5
- **Results**: 5/5 succeeded

| Reference | Analysis File | Status |
|-----------|---------------|--------|
| hardtanh | hardtanh_analysis.md | OK |
| swish | swish_analysis.md | OK |
| sinh | sinh_analysis.md | OK |
| atanh | atanh_analysis.md | OK |
| frac | frac_analysis.md | OK |

## Phase 3: Implementation
- **Duration**: ~542 seconds
- **Key design decisions**: Used Pade [5,4] rational approximation for tanh instead of exponential-based computation (SFPI lacks exp/div)

### Files Created
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_softcap.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_softcap.h`
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_softcap.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_softcap.h`
- `tt_metal/hw/inc/api/compute/eltwise_unary/softcap.h`

### Files Modified
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp` (added SOFTCAP enum)
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp` (registered parametrized type)
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp` (registered macro, init/func)
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp` (added softcap function)
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp` (Python binding)
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h` (SfpuType enum)
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h` (SfpuType enum)
- `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h` (include directive)

## Phase 4: Testing & Debugging
- **Total iterations**: ~10 kernel revisions by orchestrator after tester agent
- **Final result**: PARTIAL PASS (3/6)

### Test Results
| Test | Result | Max ULP | Notes |
|------|--------|---------|-------|
| bfloat16-cap1 | PASS | <= 2 | All bitpatterns tested |
| bfloat16-cap10 | PASS | <= 2 | All bitpatterns tested |
| bfloat16-cap50 | PASS | <= 2 | All bitpatterns tested |
| fp32-cap1 | FAIL | ~22759 | Pade [5,4] insufficient precision for fp32 transition region |
| fp32-cap10 | FAIL | unknown | Same root cause as fp32-cap1 |
| fp32-cap50 | FAIL | unknown | Same root cause as fp32-cap1 |

### Key Technical Challenges
1. **SFPI register pressure**: The SFPU has extremely limited vector registers (~8). Newton-Raphson reciprocal + Pade coefficients easily exceed this limit, causing "maximum reload insns" compiler errors.
2. **No division in SFPI**: SFPI does not support the `/` operator. Division must be approximated via Newton-Raphson reciprocal using magic number seed.
3. **No unary negation**: `-x` causes compiler bugs in SFPI. Must use `sfpi::setsgn()` or `0.0f - x`.
4. **Tanh precision tradeoff**: The Pade [5,4] is accurate to ~2 ULP in bfloat16 for |u| < 3.5, but has ~22000 ULP error in fp32 for the transition region |u| = 2.5-4.0.
5. **Parameter passing**: Cap value passed as raw IEEE 754 float32 bits via `bit_cast<uint32_t>`, interpreted in kernel via `sfpi::reinterpret<sfpi::vFloat>(sfpi::vInt(param0))`.

### Root Cause of FP32 Failure
The Pade [5,4] rational approximation `tanh(t) = t*(945 + 105*t^2 + t^4)/(945 + 420*t^2 + 15*t^4)` is exact through O(t^11) but diverges from true tanh for |t| > 2.5. In this transition region, the approximation error exceeds fp32 ULP tolerance (2 ULP = ~1.2e-7 at values near 1.0). Achieving fp32 2-ULP accuracy would require either:
- Pade [7,6] or higher (exceeds SFPI register limit of ~8 vector registers)
- Exponential-based tanh (requires `exp` which is not available in SFPI)
- Multi-pass computation (not straightforward in SFPU architecture)

## Phase 5: Documentation
This file.

## Issues Encountered

| # | Phase | Severity | Description | Resolution |
|---|-------|----------|-------------|------------|
| 1 | 3 | HIGH | Initial kernel used _float_to_int32_positive_ and division (not in SFPI) | Rewrote using polynomial/Pade approach |
| 2 | 4 | HIGH | Unary negation (-val) causes SFPI compiler ICE | Used sfpi::setsgn() instead |
| 3 | 4 | HIGH | Newton-Raphson reciprocal inside loop exceeds register limit | Moved cap reciprocal outside loop, used Pade reformulation |
| 4 | 4 | HIGH | Pade [3,2] had 18 ULP error at boundary | Upgraded to Pade [5,4] |
| 5 | 4 | MEDIUM | FP32 precision insufficient with Pade [5,4] | Known limitation -- documented |
| 6 | 4 | LOW | sfpi::setsgn doesn't accept vConst1 directly | Assigned to local vFloat first |

## Timing Summary
- **Pipeline start**: 1776208246
- **Phase 1 (Discovery)**: ~234s
- **Phase 2 (Analysis)**: ~744s (wall-clock, parallel agents)
- **Phase 3 (Implementation)**: ~542s
- **Phase 4 (Testing)**: ~1636s (multiple iterations)
- **Phase 5 (Documentation)**: ~60s
