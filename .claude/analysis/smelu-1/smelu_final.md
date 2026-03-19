# smelu -- Implementation Report

## Overview
- **Operation**: smelu (Smooth ReLU)
- **Math definition**: SmeLU(x, beta) = x if x >= beta; (x + beta)^2 / (4*beta) if |x| <= beta; 0 if x < -beta
- **Paper**: [Real World Large Scale Recommendation Systems Reproducibility and Smooth Activations (arXiv:2202.06499)](https://arxiv.org/abs/2202.06499)
- **Date implemented**: 2026-03-19
- **Status**: PASS (9/9 tests, 1st iteration)
- **Output folder**: `.claude/analysis/smelu-1/`

## Phase 1: Reference Discovery
- **Duration**: ~99s
- **References selected**:
  1. **softshrink** -- 3-region piecewise with parameter, SFPI v_if/v_elseif pattern (primary structural reference)
  2. **xielu** -- Multi-branch piecewise with quadratic computation, two parameters
  3. **leaky_relu** -- Conditional multiply with parameter loading via raw TTI
  4. **celu** -- Parametrized smooth activation with host-side precomputation pattern
  5. **hardmish** -- Piecewise polynomial activation

## Phase 2: Reference Analysis
- **Duration**: ~942s (wall-clock, 5 agents in parallel)
- **Agents launched**: 5
- **Results**: 5/5 succeeded

| Reference | Analysis File | Duration (s) | Tokens | Status |
|-----------|---------------|-------------|--------|--------|
| softshrink | [softshrink_analysis.md](./softshrink_analysis.md) | 402 | 74,894 | OK |
| xielu | [xielu_analysis.md](./xielu_analysis.md) | 436 | 73,012 | OK |
| leaky_relu | [leaky_relu_analysis.md](./leaky_relu_analysis.md) | 382 | 108,169 | OK |
| celu | [celu_analysis.md](./celu_analysis.md) | 942 | 137,543 | OK |
| hardmish | [hardmish_analysis.md](./hardmish_analysis.md) | 367 | 77,176 | OK |

## Phase 3: Implementation
- **Duration**: ~386s
- **Key design decisions**:
  - **softshrink as primary reference**: Both are 3-region piecewise functions with a single float parameter, using SFPI v_if/v_elseif/v_endif
  - **celu-style parameter precomputation**: beta and 1/(4*beta) computed on host, passed as two uint32 params to avoid device division
  - **ACTIVATIONS family**: Added to existing SFPU_OP_ACTIVATIONS_INCLUDE group (same as softshrink, celu, hardsigmoid)
  - **SFPU kernel**: Computes quadratic for middle region, then conditionally overwrites with identity or zero

## Phase 4: Testing & Debugging
- **Total iterations**: 1 (passed on first real test run after environment fixes)
- **Final result**: PASS
- **PCC achieved**: 0.999990 - 1.000000 (all well above 0.999 threshold)

### Test Matrix

| Shape | beta=0.5 | beta=1.0 | beta=2.0 |
|-------|----------|----------|----------|
| (32, 32) | 1.000000 | 0.999998 | 0.999990 |
| (64, 64) | 1.000000 | 0.999997 | 0.999991 |
| (128, 128) | 1.000000 | 0.999999 | 0.999992 |

### Iteration Log
| # | Action | Test Result | Error | Fix Applied |
|---|--------|------------|-------|-------------|
| 1 | Initial implementation + test | PASS (9/9) | - (env issues only) | keyword arg fix, tt_llk submodule update |

## Phase 5: Documentation
This file.

## Files Created/Modified

### New Files
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_smelu.h` -- SFPU kernel (Wormhole)
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_smelu.h` -- SFPU kernel (Blackhole)
- `tests/ttnn/unit_tests/operations/eltwise/test_smelu.py` -- Unit tests

### Modified Files
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_activations.h` -- LLK dispatch
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_activations.h` -- LLK dispatch
- `tt_metal/hw/inc/api/compute/eltwise_unary/activations.h` -- Compute API
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h` -- SfpuType enum
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h` -- SfpuType enum
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp` -- UnaryOpType enum
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp` -- Block defines, init/func registration
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp` -- Parametrized type registration
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp` -- C++ operation registration
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp` -- Python binding (beta default=2.0)
- `ttnn/ttnn/operations/unary.py` -- Golden function

## Issues Encountered

| # | Phase | Severity | Description | Resolution |
|---|-------|----------|-------------|------------|
| 1 | 4 | LOW | Root conftest.py import failure (pre-existing, not smelu-related) | Self-contained test with --noconftest |
| 2 | 4 | LOW | beta keyword-only arg in nanobind | Changed call to ttnn.smelu(tt_input, beta=beta) |
| 3 | 4 | LOW | Stale tt_llk submodule missing tensor_shape.h | Updated submodule to commit 59ea0128 |

## Timing Summary
- **Total wall-clock**: ~2,930s (~49 min)
- **Phase 1 (Discovery)**: ~99s
- **Phase 2 (Analysis)**: ~942s
- **Phase 3 (Implementation)**: ~386s
- **Phase 4 (Testing)**: ~1,124s
- **Step 4b (Enrich Notes)**: ~76s
- **Phase 5 (Documentation)**: ~10s
