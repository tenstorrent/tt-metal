# softcap -- Implementation Report

## Overview
- **Operation**: softcap
- **Math definition**: cap * tanh(x / cap)
- **Date implemented**: 2026-04-15
- **Status**: PARTIAL PASS -- 4/6 tests pass, 2 fp32 parametrized tests fail at 3 ULP (threshold 2)
- **Output folder**: `.claude-analysis/softcap-1/`

## Phase 1: Reference Discovery
- **Duration**: ~5 min
- **References selected**: swish, tanhshrink, atanh, hardtanh, sinh

## Phase 2: Reference Analysis
- **Duration**: ~13 min (wall-clock)
- **Agents launched**: 5
- **Results**: 5/5 succeeded

| Reference | Analysis File | Status |
|-----------|---------------|--------|
| swish | swish_analysis.md | OK |
| tanhshrink | tanhshrink_analysis.md | OK |
| atanh | atanh_analysis.md | OK |
| hardtanh | hardtanh_analysis.md | OK |
| sinh | sinh_analysis.md | OK |

## Phase 3: Implementation
- **Duration**: ~35 min (direct implementation by orchestrator after subagent timeout)
- **Key design decisions**: SFPU-only implementation using piecewise degree-7 centered polynomials for tanh approximation; float32 parameter encoding; custom SfpuType enum extension for base LLK compatibility

## Phase 4: Testing & Debugging
- **Total iterations**: 5+ debug cycles
- **Final result**: 4/6 PASS

### Test Results
| Test | Result | Max ULP |
|------|--------|---------|
| bfloat16-cap1 | PASS | < 2 |
| bfloat16-cap10 | PASS | < 2 |
| bfloat16-cap50 | PASS | < 2 |
| fp32-cap1 | PASS | < 2 |
| fp32-cap10 | FAIL | 3 |
| fp32-cap50 | FAIL | 3 |

### Root Cause of fp32 Failures
The 3 ULP error on fp32 parametrized tests (cap=10, cap=50) is caused by float32 rounding in the `cap * (x * inv_cap)` computation path. Since `1/10` and `1/50` are not exactly representable in float32, the roundtrip `cap * inv_cap` introduces ~1-3 ULP error for small input values where `tanh(t) ~ t`. This is a fundamental limitation of the decomposition approach when cap is not a power of 2.

## Files Created/Modified

### New Files
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_softcap.h` -- SFPU kernel
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_softcap.h` -- LLK dispatch
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_softcap.h` -- SFPU kernel (blackhole)
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_softcap.h` -- LLK dispatch (blackhole)
- `tt_metal/hw/inc/api/compute/eltwise_unary/softcap.h` -- Compute API header

### Modified Files
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h` -- Added softcap + base LLK compat entries
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h` -- Added softcap + base LLK compat entries
- `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h` -- Added SFPU_OP_SOFTCAP_INCLUDE
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp` -- Added SOFTCAP enum
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp` -- Registered softcap
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp` -- Added to is_parametrized_type
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp` -- Added ttnn::softcap function
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp` -- Python binding
- `ttnn/ttnn/operations/unary.py` -- Golden function
- `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp` -- Fixed broken includes

## Issues Encountered
| # | Phase | Severity | Description | Resolution |
|---|-------|----------|-------------|------------|
| 1 | 3 | HIGH | Implementor subagent timed out after 22+ min | Switched to direct implementation |
| 2 | 4 | HIGH | Hardware tanh (LUT) not accessible from TTNN kernel path | Implemented tanh as piecewise polynomial |
| 3 | 4 | HIGH | TTNN eltwise_sfpu.cpp has broken includes (trigonometry.h) | Removed dead includes |
| 4 | 4 | HIGH | SfpuType enum mismatch between metal and base LLK layers | Added compatibility entries |
| 5 | 4 | HIGH | Register pressure: SFPU compiler rejects complex kernels | Used helper function and centered polynomials |
| 6 | 4 | MEDIUM | BFloat16 parameter encoding loses precision for 1/cap | Changed to float32 bit pattern encoding |
| 7 | 4 | LOW | fp32 cap * inv_cap rounding causes 3 ULP at small values | Unfixable due to compiler register limit |

## Timing Summary
- **Total wall-clock**: ~100 min
- **Phase 1 (Discovery)**: ~5 min
- **Phase 2 (Analysis)**: ~13 min
- **Phase 3 (Implementation)**: ~35 min
- **Phase 4 (Testing)**: ~45 min
