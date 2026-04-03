# rrelu - Implementation Report

## Overview
- **Operation**: rrelu (Randomized Leaky ReLU)
- **Math definition**: RReLU(x) = x if x >= 0; a*x if x < 0. Training: a ~ Uniform(lower, upper). Eval: a = (lower+upper)/2. Default lower=1/8, upper=1/3.
- **Date implemented**: 2026-04-03
- **Status**: PASS after 7 iterations
- **Output folder**: `.claude-analysis/rrelu-1/`

## Phase 1: Reference Discovery
- **References selected**: leaky_relu, prelu, rand, dropout, selu
- leaky_relu: conditional branch pattern (x >= 0 vs x < 0)
- prelu: parametric slope for negative values
- rand: PRNG random number generation on SFPU
- dropout: PRNG + conditional apply
- selu: two-parameter conditional activation

## Phase 2: Reference Analysis
- **Agents launched**: 5
- **Results**: 5/5 succeeded

| Reference | Analysis File | Status |
|-----------|---------------|--------|
| leaky_relu | leaky_relu_analysis.md | OK |
| prelu | prelu_analysis.md | OK |
| rand | rand_analysis.md | OK |
| dropout | dropout_analysis.md | OK |
| selu | selu_analysis.md | OK |

## Phase 3: Implementation
- **Key design decisions**:
  - Raw TTI kernel (not SFPI) to avoid register allocation conflicts
  - 3-parameter design: lower, upper, seed (uint32 bitcast to float)
  - PRNG seeding via rrelu_tile_init(seed)
  - Subtraction via SFPMAD with -1.0 factor
  - Architecture-specific NOPs for Wormhole B0 vs Blackhole

## Phase 4: Testing & Debugging
- **Total iterations**: 7
- **Final result**: PASS
- **Max ULP bfloat16**: 1.0
- **Max ULP fp32**: 0.0
- **allclose**: PASS

### Iteration Log
| # | Action | Test Result | Error | Fix Applied |
|---|--------|------------|-------|-------------|
| 1 | Initial test | FAIL | TypeError: stale API mismatch | Attempted to match stale binary API |
| 2 | Rebuild | FAIL | Wrong arg count in stale binary | Ran build_metal.sh, reverted test |
| 3 | Numerical check | FAIL | Max ULP 14483456.0 | Added subnormal flush (wrong fix) |
| 4 | Root cause analysis | FAIL | Max ULP 14745600.0 | Fixed unary.cpp: pass midpoint as lower/upper for eval mode |
| 5 | ULP precision | FAIL | Max ULP 65536.0 (=2^16 bf16/fp32 ratio) | Compare ULP as bfloat16 dtype |
| 6 | Training test | FAIL (partial) | Subnormal passthrough mismatch | Flush subnormals in positive passthrough check |
| 7 | Final | PASS | All 8 tests pass | - |

## Phase 5: Documentation
This file.

## Files Created/Modified

### New Files
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_rrelu.h` - SFPU kernel (Wormhole B0)
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_rrelu.h` - SFPU kernel (Blackhole)
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_rrelu.h` - LLK dispatch (WH)
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_rrelu.h` - LLK dispatch (BH)
- `tt_metal/hw/inc/api/compute/eltwise_unary/rrelu.h` - Compute API
- `tests/ttnn/unit_tests/operations/eltwise/test_rrelu.py` - Test file

### Modified Files
- `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h` - Include rrelu header
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h` - Added RRELU enum
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h` - Added RRELU enum
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp` - Added RRELU UnaryOpType
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp` - Registered op
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp` - Registered op
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp` - Exposed rrelu function
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary.cpp` - Implemented rrelu + eval mode midpoint fix
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp` - Python binding
- `ttnn/ttnn/operations/unary.py` - Python API + golden function

## Issues Encountered

| # | Phase | Severity | Description | Resolution |
|---|-------|----------|-------------|------------|
| 1 | 4 | Medium | Stale compiled binary had different API than source | Rebuilt with build_metal.sh |
| 2 | 4 | High | Eval mode kernel used random slopes instead of fixed midpoint | Fixed unary.cpp to pass midpoint as both lower/upper when seed=0 |
| 3 | 4 | Medium | ULP comparison incorrect: float32 ULPs on bfloat16 data | Changed to compare as bfloat16 dtype |
| 4 | 4 | Low | Hardware flushes subnormal positive inputs to zero | Added subnormal flush in test |
