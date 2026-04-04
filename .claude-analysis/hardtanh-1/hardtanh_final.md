# hardtanh -- Implementation Report

## Overview
- **Operation**: hardtanh
- **Math definition**: max(min_val, min(max_val, x)) where min_val=-1.0 (default), max_val=1.0 (default)
- **Date implemented**: 2026-04-04
- **Status**: PASS after 1 test iteration (2 implementation fixes applied during testing)
- **Output folder**: `.claude-analysis/hardtanh-1/`

## Phase 1: Reference Discovery
- **Duration**: 779s
- **References selected**:
  1. **logit** -- Two-parameter clamp_tile pattern, program factory two-scalar packing. The only existing kernel that directly calls clamp_tile(0, packed_scalar1, packed_scalar2).
  2. **relu6** -- Special case of hardtanh (min=0, max=6), relu_max_tile pattern. Shows SFPU_OP_CHAIN dispatch path for two-sided clamp operations.
  3. **hardsigmoid** -- Clamp-based operation (relu6(x+3)/6), SFPU_OP_CHAIN dispatch. Shows hardsigmoid_tile pattern registration.
  4. **hardshrink** -- One-parameter conditional SFPU with custom kernel. Demonstrates parameter-packing infrastructure.
  5. **where_tss** -- Two separate float scalars (packed_scalar1 + packed_scalar2) in unary program factory. Exact two-parameter runtime arg pattern.

## Phase 2: Reference Analysis
- **Duration**: 1474s (wall-clock)
- **Agents launched**: 5
- **Results**: 5/5 succeeded

| Reference | Analysis File | Status |
|-----------|---------------|--------|
| logit | [logit_analysis.md](./logit_analysis.md) | OK |
| relu6 | [relu6_analysis.md](./relu6_analysis.md) | OK |
| hardsigmoid | [hardsigmoid_analysis.md](./hardsigmoid_analysis.md) | OK |
| hardshrink | [hardshrink_analysis.md](./hardshrink_analysis.md) | OK |
| where_tss | [where_tss_analysis.md](./where_tss_analysis.md) | OK |

## Phase 3: Implementation
- **Duration**: 1149s
- **Key design decisions**:
  - **SFPU kernel approach**: Clean two-comparison clamp using direct `v_if(val < min_val)` and `v_if(val > max_val)` comparisons, rather than existing 3-parameter arithmetic trick.
  - **Parameter passing**: Parameters (min_val, max_val) bitcast from float to uint32_t (IEEE 754) and baked into SFPU_OP_CHAIN compile-time define string as hex literals. Follows relu_max pattern.
  - **Two-parameter registration**: Custom inline function in unary.hpp with nanobind wrapper bridging 6-param C++ function to 5-param binding.
  - **Old unary path only**: Registered in old unary_op_utils.cpp path (not unary_ng) because ng path cannot embed parameters.

## Phase 4: Testing & Debugging
- **Total iterations**: 1 test run (with 2 implementation fixes applied before tests ran)
- **Final result**: PASS
- **Max ULP**: 0 (hardtanh is exact -- pure clamping)
- **allclose**: PASS (both bfloat16 and fp32)

### Bugs Fixed During Testing

| # | Error Type | Description | Fix |
|---|-----------|-------------|-----|
| 1 | build_error | SFPU kernel signature mismatch -- `calculate_hardtanh` took `iterations` as runtime param but `_llk_math_eltwise_unary_sfpu_params_` passes only `(param0, param1)` | Removed `iterations` parameter, use `ITERATIONS` template param instead |
| 2 | build_error | fmt::format interpreted `{min_val}` and `{max_val}` as named format args in nanobind docstring | Escaped as `{{min_val}}` and `{{max_val}}` |

### Test Results

| Parametrization | bfloat16 | fp32 |
|----------------|----------|------|
| default (-1, 1) | PASS | PASS |
| narrow (-0.5, 0.5) | PASS | PASS |
| relu6-like (0, 6) | PASS | PASS |
| wide (-2, 2) | PASS | PASS |

## Phase 5: Documentation
This file.

## Files Created/Modified

### New Files
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_hardtanh.h` -- SFPU kernel (Wormhole B0)
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_hardtanh.h` -- SFPU kernel (Blackhole)
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_hardtanh.h` -- LLK dispatch (Wormhole B0)
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_hardtanh.h` -- LLK dispatch (Blackhole)
- `tt_metal/hw/inc/api/compute/eltwise_unary/hardtanh.h` -- Compute API header
- `tests/ttnn/unit_tests/operations/eltwise/test_hardtanh.py` -- Test file (8 parametrized tests)

### Modified Files
- `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h` -- Added hardtanh include
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h` -- Added HARDTANH to SfpuType enum
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h` -- Added HARDTANH to SfpuType enum
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp` -- Registered in get_block_defines, get_op_init_and_func, get_op_approx_mode
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp` -- Added string-to-op mapping
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp` -- Added hardtanh inline function with two float params
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp` -- Added Python binding with docstring
- `ttnn/ttnn/operations/unary.py` -- Exposed ttnn.hardtanh in Python API

## Issues Encountered

| # | Phase | Severity | Description | Resolution |
|---|-------|----------|-------------|------------|
| 1 | 0 | LOW | Broken symlinks for .claude/scripts, .claude/agents, .claude/skills in worktree (submodule not initialized) | Fixed by pointing symlinks to main repo |
| 2 | 4 | MEDIUM | SFPU kernel `calculate_hardtanh` signature mismatch with LLK template caller | Removed `iterations` runtime parameter, use `ITERATIONS` template param |
| 3 | 4 | MEDIUM | fmt::format interprets `{min_val}` and `{max_val}` in nanobind docstring as format args | Escaped braces as `{{min_val}}` and `{{max_val}}` |

## Known Limitations
- No INT32 support (hardtanh is float-only)
- Parameters baked as compile-time defines: each unique (min_val, max_val) pair triggers kernel recompile
- Not registered in unary_ng dispatch path

## Timing Summary
- **Total wall-clock**: ~120 minutes
- **Phase 1 (Discovery)**: 779s (~13 min)
- **Phase 2 (Analysis)**: 1474s (~25 min)
- **Phase 3 (Implementation)**: 1149s (~19 min)
- **Phase 4 (Testing)**: 3024s (~50 min)
- **Phase 5 (Documentation)**: 40s (~1 min)
- **Phase 6 (Self-Reflection)**: 756s (~13 min)
