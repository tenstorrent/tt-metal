# Issues Log: frac

## Configuration
- **Operation**: frac
- **Math definition**: x - floor(x) (fractional part of x)
- **Source**: direct formula
- **Output folder**: `.claude-analysis/frac-1/`
- **Date**: 2026-04-04

## Phase Timeline
| Phase | Description | Status | Duration | Issues |
|-------|-------------|--------|----------|--------|
| 1 | Reference Discovery | ok | ~2 min | 0 |
| 2 | Reference Analysis | ok | ~3 min | 0 |
| 3 | Implementation | ok | ~15 min | 2 (LOW severity) |
| 4 | Testing & Debugging | ok | ~1 min | 0 |
| 5 | Documentation | ok | ~2 min | 0 |
| 6 | Self-Reflection | ok | ~2 min | 0 |

## Issues

### Issue 1 (Phase 3, LOW)
**Description**: LLK submodule not checked out in worktree, causing build to fail initially.
**Resolution**: Ran `git submodule update --init` for required submodules.

### Issue 2 (Phase 3, LOW)
**Description**: clang-format pre-commit hook modified `unary_nanobind.cpp` during commit.
**Resolution**: Re-staged the formatted file and committed successfully on second attempt.

### Issue 3 (Phase 0, INFO)
**Description**: `.claude/scripts/logging` symlink broken in worktree (points to non-existent `tt_ops_code_gen/scripts`).
**Resolution**: Used absolute path to main repo's logging scripts instead.

## File Manifest

### New Files
- `tests/ttnn/unit_tests/operations/eltwise/test_frac.py`

### Modified Files
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp`
- `ttnn/ttnn/operations/unary.py`
