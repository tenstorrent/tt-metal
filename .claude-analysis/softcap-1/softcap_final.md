# softcap -- Implementation Report

## Overview
- **Operation**: softcap
- **Math definition**: cap * tanh(x / cap)
- **Parameters**: cap (positive float, default 50.0)
- **Date implemented**: 2026-04-09
- **Status**: PASS after 1 iteration
- **Output folder**: `.claude-analysis/softcap-1/`

## Phase 1: Reference Discovery
- **Duration**: 353s
- **References selected**:
  1. **swish** -- Composite unary SFPU op with nonlinear + scale; complete layer-by-layer template
  2. **sinh** -- Scalar multiply bracketing nonlinear computation; exact computation shape match
  3. **atanh** -- Programmable constant registers (vConstFloatPrgm0/1/2) for scalar parameter passing
  4. **tanhshrink** -- Direct tanh_tile() invocation pattern in compute kernel
  5. **hardtanh** -- Parametrized SFPU operation infrastructure (is_parametrized_type, pack_scalar_runtime_arg)

## Phase 2: Reference Analysis
- **Duration**: 816s (wall-clock)
- **Agents launched**: 5
- **Results**: 5/5 succeeded

| Reference | Analysis File | Commit | Status |
|-----------|---------------|--------|--------|
| swish | swish_analysis.md | 2693f7dd41 | OK |
| tanhshrink | tanhshrink_analysis.md | 1395594d14 | OK |
| atanh | atanh_analysis.md | 7afda464b6 | OK |
| sinh | sinh_analysis.md | 3fa5053f64 | OK |
| hardtanh | hardtanh_analysis.md | f4fa0d07a4 | OK (committed by orchestrator) |

Note: The hardtanh analyzer agent completed its analysis file but did not commit it before finishing. The orchestrator committed on its behalf.

## Phase 3: Implementation
- **Duration**: 1070s
- **Commit**: 10855c7ce1
- **Key design decisions**:
  - Piecewise tanh approximation with 4 segments:
    - |u| <= 1.0: 9th-degree Taylor series in Horner form
    - 1.0 < |u| <= 2.0: Quadratic Lagrange interpolation
    - 2.0 < |u| <= 3.0: Quadratic Lagrange interpolation
    - |u| > 3.0: Exact saturation to +/-cap
  - Two programmable constant registers: vConstFloatPrgm0 = cap, vConstFloatPrgm1 = 1/cap
  - Sign handling: Compute on |u|, apply sign of x at the end

## Phase 4: Testing & Debugging
- **Total iterations**: 1
- **Final result**: PASS
- **Test configurations**: 6 tests (3 cap values x 2 dtypes)
  - bfloat16 with cap=1.0, 10.0, 50.0
  - fp32 with cap=1.0, 10.0, 50.0

### Iteration Log
| # | Action | Test Result | Error | Fix Applied |
|---|--------|------------|-------|-------------|
| 1 | Initial run with test creation | PASS (6/6) | - | Fixed kernel JIT includes (removed nuked op headers), added placeholder SfpuType enum values |

### Fixes Applied During Testing
1. Removed unconditional includes of nuked operation headers in eltwise_sfpu.cpp (trigonometry, mul_int_sfpu, rpow, rdiv, fill)
2. Added placeholder SfpuType enum values in llk_sfpu_types.h for both wormhole_b0 and blackhole

## Phase 5: Documentation
This file.

## Files Created/Modified

### New Files
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_softcap.h` -- Core SFPU kernel with piecewise tanh approximation
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_softcap.h` -- Blackhole variant (identical)
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_softcap.h` -- LLK dispatch bridge
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_softcap.h` -- Blackhole variant (identical)
- `tt_metal/hw/inc/api/compute/eltwise_unary/softcap.h` -- Compute API (softcap_tile_init, softcap_tile)
- `tests/ttnn/unit_tests/operations/eltwise/test_softcap.py` -- Test file

### Modified Files
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h` -- Added SOFTCAP to SfpuType enum + placeholder types
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h` -- Added SOFTCAP to SfpuType enum + placeholder types
- `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h` -- Added softcap.h include
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp` -- Added SOFTCAP to UnaryOpType enum
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp` -- Added is_parametrized_type for SOFTCAP
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp` -- Added get_block_defines, get_op_init_and_func, get_op_approx_mode for SOFTCAP
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp` -- Added softcap function declaration
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp` -- Added Python binding
- `ttnn/ttnn/operations/unary.py` -- Added golden function and Python-side registration
- `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp` -- Fixed JIT compilation includes

## Issues Encountered

| # | Phase | Severity | Description | Resolution |
|---|-------|----------|-------------|------------|
| 1 | 2 | Low | Hardtanh analyzer agent did not commit before completing | Orchestrator committed on its behalf |
| 2 | 4 | Medium | Kernel JIT compilation failed due to unconditional includes of nuked operation headers | Removed includes of trigonometry, mul_int_sfpu, rpow, rdiv, fill from eltwise_sfpu.cpp |
| 3 | 4 | Medium | Missing SfpuType enum values referenced by third_party tt_llk headers | Added placeholder enum values in llk_sfpu_types.h |

## Timing Summary
- **Total wall-clock**: ~3411s (~57 min)
- **Phase 1 (Discovery)**: 353s
- **Phase 2 (Analysis)**: 816s
- **Phase 3 (Implementation)**: 1070s
- **Phase 4 (Testing)**: 328s
- **Phase 5 (Documentation)**: ~30s
- **Phase 6 (Self-Reflection)**: 478s
