# Issues Log: cbrt

## Configuration
- **Operation**: cbrt
- **Math definition**: x^(1/3) (cube root)
- **Source**: direct formula + pre-nuke git history recovery
- **Output folder**: `.claude-analysis/cbrt-1/`
- **Date**: 2026-04-03

## Phase Timeline
| Phase | Description | Status | Duration | Issues |
|-------|-------------|--------|----------|--------|
| 1 | Reference Discovery | ok | ~2min | Pre-nuke code recovered from git history |
| 2 | Reference Analysis | ok | ~3min | Analyzed 5 reference ops via git history |
| 3 | Implementation | ok | ~10min | All 12 layers implemented; fixed nuke infrastructure damage |
| 4 | Testing & Debugging | blocked | - | Build fails due to cascading nuke damage |
| 5 | Documentation | ok | ~2min | Final report created |
| 6 | Self-Reflection | skipped | - | Build-blocked, no test results to reflect on |

## Issues

| # | Phase | Severity | Description | Resolution |
|---|-------|----------|-------------|------------|
| 1 | 3 | HIGH | llk_math_unary_sfpu_api.h included 20+ missing headers from nuked operations | Removed all missing includes, kept only cbrt + infrastructure |
| 2 | 3 | HIGH | sources.cmake listed 50+ deleted eltwise_unary header files | Trimmed to only 3 existing files (cbrt.h, eltwise_unary.h, sfpu_split_includes.h) |
| 3 | 3 | HIGH | eltwise_unary.h (shared infrastructure with init_sfpu) was deleted by nuke | Restored from pre-nuke git history |
| 4 | 3 | MEDIUM | REGISTER_UNARY_OPERATION vs DECLARE_UNARY_NG_OP conflict | cbrt uses unary_ng path only (matching pre-nuke pattern) |
| 5 | 4 | CRITICAL | Full build fails - 7 compilation units reference nuked operations | Fixed complex_binary, complex_unary, creation, nanobind; binary/ternary/backward/quantization still broken |
| 6 | 4 | CRITICAL | Tests cannot run without a linkable _ttnn.so | Blocked until all nuke damage is repaired |

## File Manifest

### New Files
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_cbrt.h`
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_cbrt.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_cbrt.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_cbrt.h`
- `tt_metal/hw/inc/api/compute/eltwise_unary/cbrt.h`
- `tt_metal/hw/inc/api/compute/eltwise_unary/eltwise_unary.h` (restored shared infrastructure)
- `tests/ttnn/unit_tests/operations/eltwise/test_cbrt.py`

### Modified Files
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h`
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_math_unary_sfpu_api.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_math_unary_sfpu_api.h`
- `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h`
- `tt_metal/hw/sources.cmake`
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
- `ttnn/cpp/ttnn/operations/eltwise/unary_ng/common/unary_ng_op_utils.cpp`
- `ttnn/cpp/ttnn/operations/eltwise/unary_ng/unary_ng.hpp`
- `ttnn/cpp/ttnn/operations/eltwise/unary_ng/unary_ng.cpp`
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp`
- `ttnn/ttnn/operations/unary.py`
- `ttnn/cpp/ttnn/operations/eltwise/complex_binary/device/complex_binary_op.cpp` (nuke fix)
- `ttnn/cpp/ttnn/operations/eltwise/complex_unary/device/complex_unary_op.cpp` (nuke fix)
- `ttnn/cpp/ttnn/operations/creation/creation.cpp` (nuke fix)
