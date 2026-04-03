# rrelu - Implementation Report

## Overview
- **Operation**: rrelu (Randomized Leaky ReLU)
- **Math definition**: RReLU(x) = x if x >= 0, a*x if x < 0. Training: a ~ Uniform(lower, upper). Eval: a = (lower+upper)/2. Defaults: lower=1/8, upper=1/3
- **Date implemented**: 2026-04-03
- **Status**: PASS after 2 iterations
- **Output folder**: `.claude-analysis/rrelu-1/`

## Phase 1: Reference Discovery
- **References selected**: prelu_sfpu, leaky_relu, rand, dropout, selu
- **Rationale**: prelu_sfpu/leaky_relu for piecewise conditional multiply; rand for PRNG generation; dropout for PRNG seeding; selu for multi-parameter registration

## Phase 2: Reference Analysis
- **Agents launched**: 5
- **Results**: 5/5 succeeded

| Reference | Analysis File | Status |
|-----------|---------------|--------|
| prelu_sfpu | prelu_sfpu_analysis.md | OK |
| leaky_relu | leaky_relu_analysis.md | OK |
| rand | rand_analysis.md | OK |
| dropout | dropout_analysis.md | OK |
| selu | selu_analysis.md | OK |

## Phase 3: Implementation
- **Key design decisions**: Dual-path kernel (eval uses SFPI abstractions, training uses raw TTI for PRNG), precomputed range parameter, fixed PRNG seed=0
- **Files created**: 6 new files (SFPU kernels for WH/BH, LLK headers, compute API, test)
- **Files modified**: 10 existing files (types, utils, Python binding, nanobind)

## Phase 4: Testing & Debugging
- **Total iterations**: 2
- **Final result**: PASS
- **Max ULP (bfloat16)**: 1
- **Max ULP (fp32)**: 0
- **allclose**: PASS

### Iteration Log
| # | Action | Test Result | Error | Fix Applied |
|---|--------|------------|-------|-------------|
| 1 | Initial test | 2/3 PASS, 1 FAIL | Training test torch.equal too strict on subnormals/-0 | Flush subnormals, use >0 mask |
| 2 | Fixed test assertions | 3/3 PASS | - | - |

## Files Created/Modified

### New Files
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_rrelu.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_rrelu.h`
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_rrelu.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_rrelu.h`
- `tt_metal/hw/inc/api/compute/eltwise_unary/rrelu.h`
- `tests/ttnn/unit_tests/operations/eltwise/test_rrelu.py`

### Modified Files
- `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h`
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h`
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp`
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp`
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp`
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary.cpp`
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp`
- `ttnn/ttnn/operations/unary.py`

## Issues Encountered

| # | Phase | Severity | Description | Resolution |
|---|-------|----------|-------------|------------|
| 1 | 4 | LOW | Training test assertion too strict (torch.equal fails on subnormals and -0 vs 0) | Flush subnormals, use strictly positive mask |

## Architecture Notes
- **Eval mode**: Uses SFPI abstractions (vFloat, v_if/v_endif) for clean conditional multiply
- **Training mode**: Uses raw TTI instructions for PRNG access (SFPMOV from PRNG counter) + CC-guarded multiply
- **Parameters**: 3 params passed to kernel: lower, range=(upper-lower), training_flag
- **PRNG**: Hardware LFSR with fixed seed=0, different per-core sequences
