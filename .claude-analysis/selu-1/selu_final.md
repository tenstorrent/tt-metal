# selu -- Implementation Report

## Overview
- **Operation**: selu
- **Math definition**: SELU(x) = scale * (max(0,x) + min(0, alpha*(exp(x)-1))), scale=1.0507009873554804934193349852946, alpha=1.6732632423543772848170429916717
- **Date implemented**: 2026-04-04
- **Status**: IMPLEMENTATION COMPLETE, TESTS PENDING
- **Output folder**: `.claude-analysis/selu-1/`

## Phase 1: Reference Discovery
- **Duration**: ~60s
- **References selected**:
  1. **celu** -- Structurally similar activation (conditional + exponential pattern)
  2. **elu** -- Closest reference (SELU is effectively scaled ELU with fixed alpha)
  3. **prelu_sfpu** -- Conditional branch pattern reference
  4. **rrelu** -- Randomized conditional pattern reference
  5. **expm1** -- Full wiring pattern reference for LLK dispatch and compute API

## Phase 2: Reference Analysis
- **Duration**: ~15 min (wall-clock)
- **Agents launched**: 5
- **Results**: 5/5 succeeded (expm1 was late but completed)

| Reference | Analysis File | Status |
|-----------|---------------|--------|
| rrelu | rrelu_analysis.md | OK |
| elu | elu_analysis.md | OK |
| prelu_sfpu | prelu_sfpu_analysis.md | OK |
| celu | celu_analysis.md | OK |
| expm1 | expm1_analysis.md | OK (late) |

## Phase 3: Implementation
- **Duration**: ~8 min
- **Key design decisions**:
  - No-parameter operation: scale and alpha are fixed FP32 hex constants baked into the SFPU kernel
  - Single conditional + unconditional multiply pattern (more efficient than two conditionals)
  - FP32 hex constants: alpha = 0x3FD63840, scale = 0x3F868640
  - New SfpuType::selu entry for LLK dispatch
  - Custom init callback following ELU pattern exactly

## Phase 4: Testing and Debugging
- **Total iterations**: 1 (test execution in progress)
- **Final result**: PENDING
- **Test file**: tests/ttnn/unit_tests/operations/eltwise/test_selu.py
- **Notes**: Test was created with bfloat16 and fp32 parametrizations. The tester agent modified sources.cmake to include the new kernel files. Tests are running via scripts/run_safe_pytest.sh.

## Phase 5: Documentation
This file.

## Files Created/Modified

### New Files
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_selu.h` -- SFPU kernel (WH)
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_selu.h` -- SFPU kernel (BH)
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_selu.h` -- LLK dispatch (WH)
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_selu.h` -- LLK dispatch (BH)
- `tt_metal/hw/inc/api/compute/eltwise_unary/selu.h` -- Compute API header
- `tests/ttnn/unit_tests/operations/eltwise/test_selu.py` -- Test file

### Modified Files
- `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h` -- Added selu include
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h` -- Added SfpuType::selu
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h` -- Added SfpuType::selu
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp` -- Registered op in get_block_defines, get_op_init_and_func, get_op_approx_mode
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp` -- Added selu C++ API
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp` -- Python binding
- `ttnn/ttnn/operations/unary.py` -- Python-level exposure
- `tt_metal/hw/sources.cmake` -- Added new kernel files to build

## Issues Encountered

| # | Phase | Severity | Description | Resolution |
|---|-------|----------|-------------|------------|
| 1 | 2 | LOW | expm1 analyzer agent was late (completed after proceeding) | Eventually completed successfully |
| 2 | 3 | LOW | Implementor agent failed to commit; orchestrator committed on behalf | Pre-commit hooks required 3 retries (clang-format, validate-metalium-includes) |
| 3 | 4 | MEDIUM | Another agent (cbrt) running concurrently in same worktree | May cause device contention |
| 4 | 4 | MEDIUM | Test execution taking extended time (>25 min) | Runtime kernel compilation is inherently slow |

## Timing Summary
- **Total wall-clock**: ~45 min (phases 1-5, test still pending)
- **Phase 1 (Discovery)**: ~60s
- **Phase 2 (Analysis)**: ~15 min
- **Phase 3 (Implementation)**: ~8 min
- **Phase 4 (Testing)**: >25 min (still in progress)
- **Phase 5 (Documentation)**: ~2 min
- **Phase 6 (Self-Reflection)**: pending

## Implementation Architecture

The SELU implementation follows the standard 7-layer unary SFPU operation pattern:

```
Layer 1: SFPU Kernel (ckernel_sfpu_selu.h)
  |
Layer 2: LLK Dispatch (llk_math_eltwise_unary_sfpu_selu.h)
  |
Layer 3: Compute API (selu.h)
  |
Layer 4: Split Includes (sfpu_split_includes.h)
  |
Layer 5: Op Utils (unary_op_utils.cpp) -- get_block_defines, get_op_init_and_func
  |
Layer 6: C++ API (unary.hpp) + Python Binding (unary_nanobind.cpp)
  |
Layer 7: Python API (unary.py)
```

The SFPU kernel implements the piecewise formula:
- For x >= 0: result = scale * x
- For x < 0: result = scale * alpha * (exp(x) - 1)

Using SFPI vector instructions with conditional masking (`v_if`/`v_endif`).
