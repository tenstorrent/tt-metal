# softcap -- Implementation Report

## Overview
- **Operation**: softcap
- **Math definition**: cap * tanh(x / cap)
- **Date implemented**: 2026-04-09
- **Status**: PASS after 1 iteration (all 6 tests pass)
- **Output folder**: `.claude-analysis/softcap-1/`

## Phase 1: Reference Discovery
- **Duration**: 360s
- **References selected**:
  1. **atanh** -- Full 3-layer custom SFPU kernel pattern, programmable constant registers
  2. **tanhshrink** -- Uses tanh_tile() as building block, DST multi-slot arithmetic
  3. **swish** -- Composite x * sigmoid(x) pattern, cleanest self-contained SFPU kernel
  4. **hardshrink** -- Runtime scalar parameter passing from host to kernel
  5. **sinh** -- Non-trivial custom kernel with exp_21f helper, two-regime computation

## Phase 2: Reference Analysis
- **Duration**: 762s (wall-clock)
- **Agents launched**: 5
- **Results**: 5/5 succeeded

| Reference | Analysis File | Status |
|-----------|---------------|--------|
| atanh | atanh_analysis.md | OK |
| tanhshrink | tanhshrink_analysis.md | OK |
| swish | swish_analysis.md | OK |
| hardshrink | hardshrink_analysis.md | OK (orchestrator committed on behalf) |
| sinh | sinh_analysis.md | OK |

## Phase 3: Implementation
- **Duration**: 947s
- **Key design decisions**:
  - Two-regime tanh approximation: degree-9 Taylor polynomial for |u| < 1.0, exp-based formula for |u| >= 1.0
  - Cap parameter passed as compile-time constant embedded in init/func strings
  - softcap_init(cap) stores 1/cap in vConstFloatPrgm0 and cap in vConstFloatPrgm1
  - Self-contained exp_21f_softcap helper (copied from sinh to avoid cross-kernel dependencies)

### Files Created
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_softcap.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_softcap.h`
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_softcap.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_softcap.h`
- `tt_metal/hw/inc/api/compute/eltwise_unary/softcap.h`
- `tests/ttnn/unit_tests/operations/eltwise/test_softcap.py`
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_conversions.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_conversions.h`

### Files Modified
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
- **Total iterations**: 1 (with compilation fixes)
- **Final result**: PASS
- **Tests**: 6/6 pass
  - bfloat16 cap=1.0: PASS
  - bfloat16 cap=10.0: PASS
  - bfloat16 cap=50.0: PASS
  - fp32 cap=1.0: PASS
  - fp32 cap=10.0: PASS
  - fp32 cap=50.0: PASS

### Compilation Fixes Applied by Tester
1. **eltwise_sfpu.cpp**: Removed includes for nuked operations (trigonometry.h, mul_int_sfpu.h, rpow.h, rdiv.h, fill.h)
2. **llk_sfpu_types.h** (both arches): Restored full SfpuType enum needed by third-party LLK code
3. **ckernel_sfpu_softcap.h** (both arches): Fixed RISC-V GCC ICE by:
   - Replacing _float_to_int32_positive_ (which has v_if/v_elseif/v_else branching) with simplified float_to_int32_pos_simple_
   - Flattening nested v_if blocks (3 levels deep) to flat, non-nested blocks to avoid GCC LTO segfault

## Phase 5: Documentation
This file.

## Issues Encountered

| # | Phase | Severity | Description | Resolution |
|---|-------|----------|-------------|------------|
| 1 | 2 | LOW | Hardshrink analyzer did not commit | Orchestrator committed on behalf |
| 2 | 4 | MEDIUM | RISC-V GCC ICE with nested v_if blocks | Tester flattened v_if nesting |
| 3 | 4 | LOW | eltwise_sfpu.cpp had includes for nuked operations | Tester removed broken includes |
| 4 | 4 | LOW | SfpuType enum needed restoration | Tester restored full enum entries |

## Timing Summary
- **Total wall-clock**: ~70 min
- **Phase 1 (Discovery)**: 360s (~6 min)
- **Phase 2 (Analysis)**: 762s (~12.7 min)
- **Phase 3 (Implementation)**: 947s (~15.8 min)
- **Phase 4 (Testing)**: 1929s (~32.2 min)
- **Phase 5 (Documentation)**: ~30s
- **Phase 6 (Self-Reflection)**: pending
