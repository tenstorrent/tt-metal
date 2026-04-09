# softcap -- Implementation Report

## Overview
- **Operation**: softcap
- **Math definition**: cap * tanh(x / cap), where cap is a positive float scalar (default 50.0)
- **Date implemented**: 2026-04-09
- **Status**: PASS after 1 iteration (25/25 tests passed)
- **Output folder**: `.claude-analysis/softcap-1/`

## Phase 1: Reference Discovery
- **Duration**: ~307 seconds
- **References selected**:
  1. **tanhshrink** -- Only existing kernel calling tanh_tile_init()/tanh_tile(); structural template for compute kernel
  2. **swish** -- Complete composite activation example: non-linear transform + scalar multiply, full sfpi.h iteration pattern
  3. **hardshrink** -- Float parameter pipeline: how cap gets packed as runtime arg, retrieved in kernel
  4. **atanh** -- Using vConstFloatPrgm0/1/2 programmable constant registers; standard abstraction layer pattern
  5. **sinh** -- Hyperbolic-function ckernel_sfpu_*.h template: calculate_*/init() split, loop pragmas, exp_21f helper

## Phase 2: Reference Analysis
- **Duration**: ~790 seconds (wall-clock)
- **Agents launched**: 5
- **Results**: 5/5 succeeded (2 committed by orchestrator on behalf of agents)

| Reference | Analysis File | Status |
|-----------|---------------|--------|
| tanhshrink | tanhshrink_analysis.md | OK (committed by agent) |
| swish | swish_analysis.md | OK (committed by agent) |
| hardshrink | hardshrink_analysis.md | OK (committed by orchestrator) |
| atanh | atanh_analysis.md | OK (committed by agent) |
| sinh | sinh_analysis.md | OK (committed by orchestrator) |

## Phase 3: Implementation
- **Duration**: ~982 seconds
- **Key design decisions**:
  - Dual-regime algorithm: degree-7 Taylor series for small |u| (< 1.0), exponential series for moderate/large |u|
  - exp_21f helper copied locally from sinh kernel (Moroz et al. 2022 algorithm)
  - Parameter passed through standard parameterized unary path (bit_cast<uint32_t> in kernel define)
  - Standard eltwise_sfpu.cpp path (not custom compute kernel)
  - Pragmas: `#pragma GCC unroll 0` to reduce register pressure

## Phase 4: Testing & Debugging
- **Total iterations**: 1
- **Final result**: PASS (25/25 tests)
- **Test tolerances**: ULP <= 10, allclose rtol=5e-2, atol=0.35

### Iteration Log
| # | Action | Test Result | Error | Fix Applied |
|---|--------|------------|-------|-------------|
| 1 | Initial test run | PASS (25/25) | - | Tester fixed missing stub headers, SfpuType enum, register pressure |

### Tester Fixes Applied
1. Created empty stub headers for nuked environment: trigonometry.h, rpow.h, rdiv.h, fill.h
2. Created ckernel_sfpu_conversions.h with _float_to_int32_positive_() definition
3. Created stubs for ckernel_sfpu_mul_int32.h and llk_math_eltwise_binary_sfpu_params.h
4. Stubbed mul_int_sfpu.h to avoid transitive include issues
5. Added ~35 stub SfpuType enum values required by third_party LLK templates
6. Simplified kernel from degree-7 Taylor + 3-term geometric to degree-5 Taylor + 2-term geometric to resolve register pressure ICE

## Phase 5: Documentation
This file.

## Files Created/Modified

### New Files
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_softcap.h` -- SFPU kernel (Wormhole)
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_softcap.h` -- SFPU kernel (Blackhole)
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_softcap.h` -- LLK dispatch (Wormhole)
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_softcap.h` -- LLK dispatch (Blackhole)
- `tt_metal/hw/inc/api/compute/eltwise_unary/softcap.h` -- Compute API header
- `tests/ttnn/unit_tests/operations/eltwise/test_softcap.py` -- Test file (25 tests)

### Modified Files
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h` -- Added SfpuType::softcap enum
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h` -- Added SfpuType::softcap enum
- `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h` -- Include softcap.h
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp` -- Added UnaryOpType::SOFTCAP
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp` -- Registered op in get_block_defines, get_op_init_and_func, get_op_approx_mode
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp` -- Registered parameterized op
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp` -- Exposed softcap in ttnn unary API
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp` -- Python binding
- `ttnn/ttnn/operations/unary.py` -- Python API entry point
- `ttnn/ttnn/experimental_loader/golden_functions.py` -- Golden function: cap * torch.tanh(x / cap)

## Issues Encountered

| # | Phase | Severity | Description | Resolution |
|---|-------|----------|-------------|------------|
| 1 | 2 | LOW | sinh and hardshrink analyzer agents did not commit their analysis files | Orchestrator committed on their behalf |
| 2 | 4 | MEDIUM | Nuked environment missing headers (trigonometry.h, rpow.h, rdiv.h, fill.h, etc.) | Tester created stub headers |
| 3 | 4 | MEDIUM | Register pressure ICE; original kernel too complex for SFPU register file | Simplified to degree-5 Taylor + 2-term geometric |
| 4 | 4 | LOW | SfpuType enum missing ~35 stub values needed by third_party LLK templates | Added stub enum values |

## Timing Summary
- **Total wall-clock**: ~62 minutes (from pipeline_start to documentation)
- **Phase 1 (Discovery)**: ~307s (~5 min)
- **Phase 2 (Analysis)**: ~790s (~13 min)
- **Phase 3 (Implementation)**: ~982s (~16 min)
- **Phase 4 (Testing)**: ~1454s (~24 min)
- **Step 4b (Notes Enrichment)**: ~148s (~2 min)
- **Phase 5 (Documentation)**: ~30s
- **Phase 6 (Self-Reflection)**: pending
