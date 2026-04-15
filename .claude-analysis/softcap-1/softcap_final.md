# softcap -- Implementation Report

## Overview
- **Operation**: softcap
- **Math definition**: cap * tanh(x / cap)
- **Parameters**: cap (float, positive scalar, default = 50.0)
- **Date implemented**: 2026-04-15
- **Status**: INCOMPLETE after 4 fix iterations (env restrictions blocked runtime tests)
- **Output folder**: `.claude-analysis/softcap-1/`

## Phase 1: Reference Discovery
- **Duration**: ~270 seconds
- **References selected**:
  1. **tanh** - Core mathematical dependency; softcap uses tanh internally
  2. **hardtanh** - Canonical parameterized SFPU op pattern (SFPI kernel, parameter encoding)
  3. **power** - Parameterized unary op showing float parameter wiring
  4. **rsub** - Parameterized op showing scalar-tensor arithmetic patterns
  5. **softshrink** - Parameterized activation showing infrastructure patterns

## Phase 2: Reference Analysis
- **Duration**: ~802 seconds (wall-clock)
- **Agents launched**: 5
- **Results**: 5/5 succeeded

| Reference | Analysis File | Status | Key Findings |
|-----------|---------------|--------|-------------|
| tanh | tanh_analysis.md | OK | SFPU kernel stripped from eval branch; documented expected patterns |
| hardtanh | hardtanh_analysis.md | OK | Best reference: has ckernel_sfpu_hardtanh.h with full SFPI implementation |
| power | power_analysis.md | OK | Shows parameterized dispatch patterns |
| rsub | rsub_analysis.md | OK | Binary SFPU path; limited unary relevance |
| softshrink | softshrink_analysis.md | OK | SFPU kernel nuked; documented expected structure |

## Phase 3: Implementation
- **Duration**: ~588 seconds
- **Key design decisions**:
  - Used `exp_21f` helper function for accurate tanh computation
  - Implemented tanh directly (existing tanh infra stripped from eval branch)
  - Small-value Taylor approximation for numerical stability (|x/cap| < 0.5)
  - Overflow/underflow clamping for exponential arguments

### Files Created
- `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_softcap.h`
- `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_softcap.h`
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_softcap.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_softcap.h`
- `tt_metal/hw/inc/api/compute/eltwise_unary/softcap.h`

### Files Modified
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp` - Added SOFTCAP to enum
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp` - Added approx mode
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp` - Registered dispatch
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp` - Python binding macro
- `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h` - Include guard
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp` - Nanobind Python export (tester fix)
- `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp` - Removed missing includes (tester fix)

## Phase 4: Testing & Debugging
- **Total iterations**: 4 fix attempts
- **Final result**: INCOMPLETE (environment restrictions)

### Iteration Log
| # | Action | Test Result | Error | Fix Applied |
|---|--------|------------|-------|-------------|
| 1 | Initial test run | FAIL | AttributeError: module 'ttnn' has no attribute 'softcap' | Added nanobind Python binding |
| 2 | Retry after binding fix | FAIL | ckernel_sfpu_softcap.h: No such file | Fixed LLK include paths (sfpu/ prefix) |
| 3 | Retry after include fix | FAIL | trigonometry.h: No such file | Removed missing includes from eltwise_sfpu.cpp |
| 4 | Retry after include cleanup | FAIL | SfpuType enum missing comparison values | NOT RESOLVABLE (env constraint) |

### Environment Restrictions
The evaluation branch has significant infrastructure stripped:
- Missing trigonometric operation headers
- Missing integer SFPU operation headers
- Limited SfpuType enum (missing comparison operations like equal_zero, less_than_zero)
- Multiple SFPU kernel implementations removed

## Phase 5: Documentation
This file.

## Issues Encountered

| # | Phase | Severity | Description | Resolution |
|---|-------|----------|-------------|------------|
| 1 | 2 | LOW | Some reference ops (tanh, softshrink) had SFPU kernels stripped | Analyzed remaining patterns; used hardtanh as primary reference |
| 2 | 3 | MEDIUM | No existing tanh SFPU function to call | Implemented tanh directly using exp_21f helper |
| 3 | 4 | HIGH | Python binding missing from nanobind | Tester added binding to unary_nanobind.cpp |
| 4 | 4 | HIGH | LLK include paths wrong | Tester fixed paths with sfpu/ prefix |
| 5 | 4 | HIGH | Missing headers in eval env | Tester removed broken includes from eltwise_sfpu.cpp |
| 6 | 4 | CRITICAL | SfpuType enum incomplete in eval env | NOT RESOLVABLE - blocks runtime test execution |

## Timing Summary
- **Total wall-clock**: ~2700 seconds (~45 minutes)
- **Phase 1 (Discovery)**: ~270s
- **Phase 2 (Analysis)**: ~802s
- **Phase 3 (Implementation)**: ~588s
- **Phase 4 (Testing)**: ~830s
- **Phase 5 (Documentation)**: ~60s
- **Phase 6 (Self-Reflection)**: pending
