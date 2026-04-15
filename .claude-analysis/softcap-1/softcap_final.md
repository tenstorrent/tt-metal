# softcap -- Implementation Report

## Overview
- **Operation**: softcap
- **Math definition**: cap * tanh(x / cap), where cap is a positive float parameter
- **Date implemented**: 2026-04-15
- **Status**: FAIL after multiple iterations (sigmoid polynomial cancellation at near-zero values)
- **Output folder**: `.claude-analysis/softcap-1/`

## Phase 1: Reference Discovery
- **Duration**: 200s
- **References selected**: swish, tanhshrink, hardtanh, atanh, softshrink
  - swish: multiplicative composition pattern, polynomial sigmoid on SFPU
  - tanhshrink: tanh implementation patterns
  - hardtanh: parameterized operation infrastructure
  - atanh: custom SFPU implementation structure
  - softshrink: parameterized type system integration

## Phase 2: Reference Analysis
- **Duration**: 770s (wall-clock)
- **Agents launched**: 5
- **Results**: 5/5 succeeded (4 completed on time, tanhshrink was late)

| Reference | Analysis File | Status |
|-----------|---------------|--------|
| swish | swish_analysis.md | OK |
| tanhshrink | tanhshrink_analysis.md | OK (late) |
| hardtanh | hardtanh_analysis.md | OK |
| atanh | atanh_analysis.md | OK |
| softshrink | softshrink_analysis.md | OK |

## Phase 3: Implementation
- **Duration**: 554s
- **Key design decisions**:
  - Parameterized unary op passing both `cap` and `1/cap` to SFPU kernel
  - `s2vFloat16b` for parameter unpacking on SFPU
  - tanh computed via `2*sigmoid(2u) - 1` identity using polynomial sigmoid

## Phase 4: Testing & Debugging
- **Total iterations**: ~24 kernel revisions by orchestrator
- **Final result**: FAIL
- **Root cause**: Sigmoid polynomial `0.5 + tiny*coefficient` suffers catastrophic cancellation when computing `tanh = 2*(0.5+offset) - 1 = 2*offset`. For very small arguments, the offset is subnormal and gets lost when added to 0.5.

### Key debugging findings:
1. **SFPU has no built-in tanh**: Must use polynomial approximation
2. **SFPU v_if with `<` comparison**: Behaves unexpectedly for values that were modified inside previous v_if blocks. All attempts to add `v_if(au < threshold)` to handle small values caused ALL lanes (including large values) to take the small-value path.
3. **SfpuType enum missing values**: Third-party LLK references enum values not present in the stripped-down evaluation codebase
4. **Host C++ rebuild required**: Changes to `unary_op_utils.cpp` require `./build_metal.sh` rebuild

### Attempted approaches:
- Pure sigmoid polynomial: works for large values, fails at near-zero (255 ULP)
- Taylor series tanh(u) ~ u - u^3/3 + 2u^5/15: same near-zero issue
- v_if branching (small-value linear override): breaks large values due to SFPU predication
- Offset-based sigmoid (compute sig-0.5 directly): same issue -- SFPU internal computation rounds tiny values to 0
- Nested v_if (at > threshold for sigmoid, default linear): same large-value breakage
- ax clamping before au computation: same v_if issue

## Phase 5: Documentation
This file.

## Files Created/Modified

### New Files
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_softcap.h`
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_softcap.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_softcap.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_softcap.h`
- `tt_metal/hw/inc/api/compute/eltwise_unary/softcap.h`

### Modified Files
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp` -- Added SOFTCAP to UnaryOpType enum
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp` -- Registered softcap in get_block_defines, get_op_init_and_func, get_op_approx_mode
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp` -- Added is_parametrized_type support
- `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp` -- Parameter packing
- `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp` -- Include softcap.h
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp` -- Python API binding
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h` -- Added SfpuType::softcap + missing enum values
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h` -- Same

## Issues Encountered

| # | Phase | Severity | Description | Resolution |
|---|-------|----------|-------------|------------|
| 1 | 2 | LOW | tanhshrink analyzer agent was slow | Proceeded with 4/5 analyses |
| 2 | 4 | CRITICAL | SFPU has no built-in tanh function | Used polynomial sigmoid approximation |
| 3 | 4 | CRITICAL | sfpi::vFloat does not support `/` operator | Pre-compute 1/cap on host, pass as param |
| 4 | 4 | HIGH | SfpuType enum missing values from third-party LLK | Added stub values to suppress compiler errors |
| 5 | 4 | CRITICAL | Sigmoid polynomial cancellation at near-zero | Unsolved: 2*(0.5+tiny)-1 loses tiny precision |
| 6 | 4 | CRITICAL | SFPU v_if with `<` comparison breaks large values | Unsolved: all branching approaches fail |
| 7 | 4 | MEDIUM | Host C++ not rebuilt after op_utils change | Ran `./build_metal.sh` |

## Timing Summary
- **Total wall-clock**: ~2700s (~45 min)
- **Phase 1 (Discovery)**: 200s
- **Phase 2 (Analysis)**: 770s
- **Phase 3 (Implementation)**: 554s
- **Phase 4 (Testing)**: ~1200s (extensive debugging)
- **Phase 5 (Documentation)**: ~60s
- **Phase 6 (Self-Reflection)**: pending
