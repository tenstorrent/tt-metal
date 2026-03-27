# Issues Log: rrelu

## Configuration
- **Operation**: rrelu
- **Math definition**: RReLU(x) = x if x >= 0; RReLU(x) = a * x if x < 0; where in training mode: a ~ Uniform(lower, upper) (random per element), in eval mode: a = (lower + upper) / 2. Default: lower = 1/8 (0.125), upper = 1/3 (~0.333).
- **Source**: https://docs.pytorch.org/docs/stable/generated/torch.nn.RReLU.html
- **Output folder**: `.claude-analysis/rrelu-1/`
- **Date**: 2026-03-27

## Phase Timeline
| Phase | Description | Status | Duration | Issues |
|-------|-------------|--------|----------|--------|
| 1 | Reference Discovery | ok | ~120s | 0 |
| 2 | Reference Analysis | failed | ~300s | All 5 agents failed to write analysis files |
| 3 | Implementation | ok | ~1700s | Training mode added manually |
| 4 | Testing & Debugging | ok | ~600s | 1 test fix (subnormal handling) |
| 5 | Documentation | ok | ~300s | 0 |

## Issues

### Issue 1 -- Phase 2: All 5 analyzer agents failed to produce analysis files
- **Severity**: MEDIUM
- **Description**: All 5 `ttnn-unary-sfpu-operation-analyzer` agents launched in parallel but none produced their `*_analysis.md` files.
- **Impact**: No pre-analyzed reference documentation available for Phase 3.
- **Resolution**: Proceeded without analyses. Implementor agent read source files directly.

### Issue 2 -- Phase 3: Initial implementation was eval-mode only
- **Severity**: LOW
- **Description**: Implementor agent only implemented eval mode, citing PRNG incompatibility with standard unary factory.
- **Impact**: Training mode was missing (user required ALL functionalities).
- **Resolution**: Training mode added manually by orchestrator using raw TTI instructions for PRNG, following dropout/rand patterns.

### Issue 3 -- Phase 4: Subnormal comparison failure in training test
- **Severity**: LOW
- **Description**: Training mode test failed because hardware flushes subnormal positive values to zero.
- **Impact**: Test failure on first run.
- **Resolution**: Fixed test to call `flush_subnormal_values_to_zero()` before comparing positive inputs.

## File Manifest

### New Files
- tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_rrelu.h
- tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_rrelu.h
- tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_rrelu.h
- tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_rrelu.h
- tt_metal/hw/inc/api/compute/eltwise_unary/rrelu.h
- tests/ttnn/unit_tests/operations/eltwise/test_rrelu.py

### Modified Files
- tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h
- tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h
- tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h
- ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp
- ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp
- ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp
- ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp
- ttnn/cpp/ttnn/operations/eltwise/unary/unary.cpp
- ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp
- ttnn/ttnn/operations/unary.py
