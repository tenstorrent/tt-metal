# Issues Log: swish

## Configuration
- **Operation**: swish
- **Math definition**: swish(x) = x * sigmoid(x) = x / (1 + exp(-x))
- **Source**: direct formula
- **Output folder**: `.claude-analysis/swish-1/`
- **Date**: 2026-04-05

## Phase Timeline
| Phase | Description | Status | Duration | Issues |
|-------|-------------|--------|----------|--------|
| 1 | Reference Discovery | ok | ~4 min | 0 |
| 2 | Reference Analysis | ok | ~1 min | 0 |
| 3 | Implementation | ok | ~27 min | 3 (all minor) |
| 4 | Testing & Debugging | ok | ~1 min | 0 |
| 5 | Documentation | ok | ~2 min | 0 |
| 6 | Self-Reflection | skipped | - | - |

## Issues

### Issue 1: Name Collision
- **Phase**: 3 (Implementation)
- **Severity**: Minor
- **Description**: The name `swish` was already defined as an inline alias for `silu` in `unary.hpp`. Adding `REGISTER_UNARY_OPERATION(swish, SWISH)` caused a redefinition error.
- **Resolution**: Removed the old `swish` inline alias and replaced it with the `REGISTER_UNARY_OPERATION` macro, making swish a first-class SFPU operation.

### Issue 2: Missing Submodules
- **Phase**: 3 (Implementation)
- **Severity**: Minor
- **Description**: The worktree did not have initialized git submodules, causing CMake to fail with "Missing submodules" error.
- **Resolution**: Ran `git submodule update --init` for `tt_llk`, `tracy`, and `umd` submodules.

### Issue 3: tt_ops_code_gen Submodule
- **Phase**: 3 (Implementation)
- **Severity**: Minor
- **Description**: The `tt_ops_code_gen` submodule failed to fetch (remote ref not found). Also caused `git status` failures.
- **Resolution**: Copied contents from the main repo, then cleaned up the directory to avoid git errors.

## File Manifest

### New Files
- tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_swish.h
- tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_swish.h
- tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_swish.h
- tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_swish.h
- tt_metal/hw/inc/api/compute/eltwise_unary/swish.h
- tests/ttnn/unit_tests/operations/eltwise/test_swish.py

### Modified Files
- tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h
- tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h
- tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h
- ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp
- ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp
- ttnn/cpp/ttnn/operations/eltwise/unary_ng/common/unary_ng_op_utils.cpp
- ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp
- ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp
- ttnn/ttnn/operations/unary.py
