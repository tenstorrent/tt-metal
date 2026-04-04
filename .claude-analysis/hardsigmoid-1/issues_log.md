# Issues Log: hardsigmoid

## Configuration
- **Operation**: hardsigmoid
- **Math definition**: max(0, min(1, x/6 + 0.5))
- **Source**: direct formula
- **Output folder**: `.claude-analysis/hardsigmoid-1/`
- **Date**: 2026-04-04

## Phase Timeline
| Phase | Description | Status | Duration | Issues |
|-------|-------------|--------|----------|--------|
| 1 | Reference Discovery | ok | ~5 min | 0 |
| 2 | Reference Analysis | ok | ~10 min | 0 |
| 3 | Implementation | ok | ~30 min | 4 (nuke fallout) |
| 4 | Testing & Debugging | blocked | - | 2 (build infra broken) |
| 5 | Documentation | ok | ~5 min | 0 |
| 6 | Self-Reflection | skipped | - | - |

## Issues

### Issue 1: Batch nuke left sources.cmake broken
- **Phase**: 3
- **Severity**: HIGH
- **Description**: sources.cmake listed ~50 header files that were deleted by the batch nuke
- **Resolution**: Cleaned sources.cmake to only list files that actually exist

### Issue 2: Corrupted switch statements in unary_ng_op_utils.cpp
- **Phase**: 3
- **Severity**: HIGH
- **Description**: The nuke removed case labels but left code bodies, resulting in orphaned if-blocks inside switch statements
- **Resolution**: Removed dangling code blocks

### Issue 3: Cascading missing operation stubs
- **Phase**: 3
- **Severity**: HIGH
- **Description**: ~100 nuked operations still referenced from binary, ternary, backward, and composite op files
- **Resolution**: Added comprehensive REGISTER_UNARY_OPERATION stubs for all referenced operations

### Issue 4: Signature mismatches for binary-like SFPU ops
- **Phase**: 3
- **Severity**: MEDIUM
- **Description**: sub_sfpu, power, etc. have different calling conventions (Tensor+Tensor vs Tensor+float vs ScalarVariant)
- **Resolution**: Added overloads and used UNARY_OP_SCALAR_VARIANT where appropriate

### Issue 5: Pre-built _ttnn.so incompatible with current Python code
- **Phase**: 4
- **Severity**: CRITICAL
- **Description**: ttnn/__init__.py references get_fabric_config which doesn't exist in the pre-nuke _ttnn.so
- **Resolution**: UNRESOLVED - requires full C++ rebuild from a state where both the Python code and C++ code are in sync

### Issue 6: Worktree venv incompatibility
- **Phase**: 4
- **Severity**: HIGH
- **Description**: Worktree has no python_env; symlinked from main repo but ttnn package resolution conflicts
- **Resolution**: Partial - built worktree's _ttnn.so successfully but couldn't run tests due to Issue 5

## File Manifest

### New Files
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_hardsigmoid.h`
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_hardsigmoid.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_hardsigmoid.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_hardsigmoid.h`
- `tt_metal/hw/inc/api/compute/eltwise_unary/hardsigmoid.h`
- `tt_metal/hw/inc/api/compute/eltwise_unary/activations.h`
- `tt_metal/hw/inc/api/compute/eltwise_unary/README.md`
- `tests/ttnn/unit_tests/operations/eltwise/test_hardsigmoid.py`
