# Issues Log: rrelu

## Configuration
- **Operation**: rrelu
- **Math definition**: RReLU(x) = max(0, x) + a * min(0, x), where a = Uniform(lower, upper) in training, a = (lower + upper) / 2 in eval
- **Source**: direct formula (PyTorch reference: torch.nn.functional.rrelu)
- **Output folder**: `.claude-analysis/rrelu-1/`
- **Date**: 2026-04-09

## Phase Timeline
| Phase | Description | Status | Duration | Issues |
|-------|-------------|--------|----------|--------|
| 1 | Reference Discovery | ok | ~279s | 0 |
| 2 | Reference Analysis | ok | ~518s | 1 (bundled commit) |
| 3 | Implementation | ok | ~815s | 0 |
| 4 | Testing & Debugging | ok | ~967s | 1 (SfpuType enum fix) |
| 5 | Documentation | ok | ~30s | 0 |
| 6 | Self-Reflection | pending | - | - |

## Issues

### Issue 1: hardtanh analysis committed bundled with where_tss
- **Phase**: 2
- **Severity**: LOW
- **Description**: The where_tss analyzer agent ran `git add .claude-analysis/rrelu-1/` which picked up the hardtanh analysis file that was on disk but not yet committed by its own agent.
- **Resolution**: Non-issue -- both files were committed correctly. The hardtanh commit never appeared separately in git log.

### Issue 2: SfpuType enum missing standard members
- **Phase**: 4
- **Severity**: MEDIUM
- **Description**: The "deep nuke" repo preparation had stripped standard SfpuType enum members (comparison ops, inf/nan checks) from `llk_sfpu_types.h`. This caused build failures when the test attempted JIT compilation.
- **Resolution**: The tester agent restored the missing enum members in both wormhole_b0 and blackhole variants of `llk_sfpu_types.h`.

## File Manifest

### New Files
- tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_rrelu.h
- tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_rrelu.h
- tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_rrelu.h
- tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_rrelu.h
- tt_metal/hw/inc/api/compute/eltwise_unary/rrelu.h
- tests/ttnn/unit_tests/operations/eltwise/test_rrelu.py

### Modified Files
- tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h
- tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h
- tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h
- ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp
- ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp
- ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp
- ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp
- ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp
- ttnn/cpp/ttnn/operations/eltwise/unary_ng/common/unary_ng_op_utils.cpp
- ttnn/ttnn/operations/unary.py
