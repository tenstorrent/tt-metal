# frac -- Implementation Report

## Overview
- **Operation**: frac
- **Math definition**: frac(x) = x - trunc(x) (sign-preserving fractional part)
- **Date implemented**: 2026-04-06
- **Status**: PASS after 2 iterations
- **Output folder**: `.claude-analysis/frac-1/`

## Phase 1: Reference Discovery
- **Duration**: ~278 seconds
- **References selected**:
  1. **cbrt** - IEEE 754 bit manipulation with exexp/reinterpret
  2. **hardtanh** - Standard ckernel_sfpu boilerplate and v_if/v_endif
  3. **hardsigmoid** - Parameterless unary op with no init constants
  4. **hardswish** - Intermediate computation + subtraction pattern
  5. **softshrink** - Three-case conditional with default result

## Phase 2: Reference Analysis
- **Duration**: ~970 seconds (wall-clock)
- **Agents launched**: 5
- **Results**: 5/5 succeeded (4 committed by orchestrator, 1 by agent)

| Reference | Analysis File | Status |
|-----------|---------------|--------|
| hardswish | hardswish_analysis.md | OK |
| hardsigmoid | hardsigmoid_analysis.md | OK |
| softshrink | softshrink_sfpu_analysis.md | OK |
| hardtanh | hardtanh_analysis.md | OK |
| cbrt | cbrt_analysis.md | OK |

## Phase 3: Implementation
- **Duration**: ~517 seconds
- **Key design decisions**:
  - Used IEEE 754 bit manipulation for exact trunc computation (no approximation needed)
  - Three-case branching based on debiased exponent E from sfpi::exexp()
  - E < 0: result = x (entire value is fractional)
  - E >= 23: result = 0 (exact integer, no fractional bits)
  - 0 <= E < 23: mask mantissa bits to get trunc(x), compute x - trunc(x)

### Files Created
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_frac.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_frac.h`
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_frac.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_frac.h`
- `tt_metal/hw/inc/api/compute/eltwise_unary/frac.h`
- `tests/ttnn/unit_tests/operations/eltwise/test_frac.py`

### Files Modified
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h`
- `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h`
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
- `ttnn/cpp/ttnn/operations/eltwise/unary_ng/common/unary_ng_op_utils.cpp`
- `ttnn/ttnn/operations/unary.py`
- `tt_metal/hw/sources.cmake`

## Phase 4: Testing & Debugging
- **Total iterations**: 2
- **Final result**: PASS
- **PCC**: > 0.999
- **allclose**: PASS (rtol=1.6e-2, atol=1e-2)

### Iteration Log
| # | Action | Test Result | Error | Fix Applied |
|---|--------|------------|-------|-------------|
| 1 | Initial implementation (floor semantics) | FAIL | Wrong sign for negative inputs | Changed from x-floor(x) to x-trunc(x) to match torch.frac() |
| 2 | Fixed trunc semantics | PASS | - | - |

## Phase 5: Documentation
This file.

## Issues Encountered

| # | Phase | Severity | Description | Resolution |
|---|-------|----------|-------------|------------|
| 1 | 2 | Low | cbrt analyzer slow to complete | Proceeded with 4/5, cbrt committed later |
| 2 | 4 | Medium | Initial kernel used floor(x) semantics instead of trunc(x) | Tester fixed by changing mantissa mask to preserve sign |
| 3 | 4 | Low | Tester ran find / searching entire filesystem | Killed the process, tester recovered |

## Timing Summary
- **Total wall-clock**: ~67 minutes
- **Phase 1 (Discovery)**: ~278s
- **Phase 2 (Analysis)**: ~970s
- **Phase 3 (Implementation)**: ~517s
- **Phase 4 (Testing)**: ~2385s
- **Phase 5 (Documentation)**: ~60s
- **Phase 6 (Self-Reflection)**: pending
