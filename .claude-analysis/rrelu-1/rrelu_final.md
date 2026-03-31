# rrelu -- Implementation Report

## Overview
- **Operation**: rrelu (Randomized Leaky ReLU)
- **Math definition**: RReLU(x) = x if x >= 0; ((lower + upper) / 2) * x if x < 0 (eval mode). Training: a ~ U(lower, upper) per negative element.
- **Parameters**: lower=0.125 (default), upper=1/3 (default)
- **Date implemented**: 2026-03-31
- **Status**: PASS after 1 iteration
- **Output folder**: `.claude-analysis/rrelu-1/`

## Phase 1: Reference Discovery
- **Duration**: ~271s
- **References selected**:
  1. **leaky_relu** -- Exact eval-mode pattern: conditional branch on sign, multiply negative elements by fixed slope using raw SFPU instructions
  2. **prelu** -- Same conditional multiply but using sfpi high-level DSL for cleaner code reference
  3. **dropout** -- Complete PRNG init + per-element random generation infrastructure needed for training mode
  4. **rand** -- Shows the [lower, upper] uniform scaling idiom using SFPMAD for random slope sampling
  5. **selu** -- Demonstrates two-parameter (scale, alpha) registration and dual-param loading

## Phase 2: Reference Analysis
- **Duration**: ~777s (wall-clock)
- **Agents launched**: 5
- **Results**: 5/5 succeeded

| Reference | Analysis File | Status |
|-----------|---------------|--------|
| leaky_relu | [leaky_relu_analysis.md](./leaky_relu_analysis.md) | OK |
| prelu | [prelu_analysis.md](./prelu_analysis.md) | OK |
| dropout | [dropout_analysis.md](./dropout_analysis.md) | OK |
| rand | [rand_analysis.md](./rand_analysis.md) | OK |
| selu | [selu_analysis.md](./selu_analysis.md) | OK |

## Phase 3: Implementation
- **Duration**: ~901s
- **Key design decisions**:
  - Dual-mode SFPU kernel: eval mode uses SFPI abstractions (vFloat, v_if, v_endif); training mode uses raw TTI instructions for PRNG
  - PRNG seeding: init_prng_seed() called once via static bool guard (takes 600 SFPNOP cycles)
  - 3 float params passed through unary pipeline: lower, upper, seed (seed=0 for eval, non-zero for training)
  - Address modes: WH uses ADDR_MOD_3, BH uses ADDR_MOD_7 for training mode
- **Reference operations most useful**: leaky_relu (eval mode pattern), dropout/rand (PRNG infrastructure), selu (multi-param registration)

## Phase 4: Testing & Debugging
- **Total iterations**: 1
- **Final result**: PASS
- **ULP threshold**: 2 (passed)
- **allclose**: PASS (rtol=1.6e-2, atol=1e-2)

### Iteration Log
| # | Action | Test Result | Error | Fix Applied |
|---|--------|------------|-------|-------------|
| 1 | Initial implementation | PASS | - | - |

Note: The C++ host code required rebuilding (`./build_metal.sh`) since source files had been modified but not yet compiled. After rebuild, all 65536 bfloat16 bit patterns passed evaluation mode testing.

## Phase 5: Documentation
This file.

## Files Created/Modified

### New Files
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_rrelu.h` -- SFPU kernel (Wormhole)
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_rrelu.h` -- SFPU kernel (Blackhole)
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_rrelu.h` -- LLK dispatch (Wormhole)
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_rrelu.h` -- LLK dispatch (Blackhole)
- `tt_metal/hw/inc/api/compute/eltwise_unary/rrelu.h` -- Compute API header
- `tests/ttnn/unit_tests/operations/eltwise/test_rrelu.py` -- Test file

### Modified Files
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h` -- Added rrelu to SfpuType enum
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h` -- Added rrelu to SfpuType enum
- `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h` -- Added SFPU_OP_RRELU_INCLUDE guard
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp` -- Added RRELU to UnaryOpType enum
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp` -- Added RRELU to is_parametrized_type
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp` -- Registered op in get_macro_definition, get_op_init_and_func, string_to_unary_with_param
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp` -- Added rrelu function declaration
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary.cpp` -- Added rrelu function implementation
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp` -- Added Python binding
- `ttnn/ttnn/operations/unary.py` -- Added golden function for rrelu

## Issues Encountered

| # | Phase | Severity | Description | Resolution |
|---|-------|----------|-------------|------------|
| - | - | - | No issues encountered | - |

The implementation completed cleanly with tests passing on the first iteration.

## Timing Summary
- **Total wall-clock**: ~2409s (~40 minutes)
- **Phase 1 (Discovery)**: ~271s
- **Phase 2 (Analysis)**: ~777s
- **Phase 3 (Implementation)**: ~901s
- **Phase 4 (Testing)**: ~460s
- **Phase 5 (Documentation)**: <60s
