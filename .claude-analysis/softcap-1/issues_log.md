# Issues Log: softcap

## Configuration
- **Operation**: softcap
- **Math definition**: cap * tanh(x / cap)
- **Parameters**: cap (float, positive scalar, default = 50.0)
- **Source**: direct formula
- **Output folder**: `.claude-analysis/softcap-1/`
- **Date**: 2026-04-09

## Phase Timeline
| Phase | Description | Status | Duration | Issues |
|-------|-------------|--------|----------|--------|
| 1 | Reference Discovery | ok | ~307s | None |
| 2 | Reference Analysis | ok | ~790s | 2 agents failed to commit; orchestrator committed on their behalf |
| 3 | Implementation | ok | ~982s | None |
| 4 | Testing & Debugging | ok | ~1454s | Tester needed to create stub headers for nuked env; kernel simplified from degree-7 to degree-5 Taylor |
| 5 | Documentation | pending | - | - |
| 6 | Self-Reflection | pending | - | - |

## Issues

| # | Phase | Severity | Description | Resolution |
|---|-------|----------|-------------|------------|
| 1 | 2 | LOW | sinh and hardshrink analyzer agents did not commit their analysis files | Orchestrator committed on their behalf |
| 2 | 4 | MEDIUM | Nuked environment missing headers (trigonometry.h, rpow.h, rdiv.h, fill.h, ckernel_sfpu_conversions.h, ckernel_sfpu_mul_int32.h) | Tester created stub headers |
| 3 | 4 | MEDIUM | Register pressure caused ICE; degree-7 Taylor + 3-term geometric too complex for SFPU | Tester simplified to degree-5 Taylor + 2-term geometric |
| 4 | 4 | LOW | SfpuType enum missing ~35 stub values needed by third_party LLK templates | Tester added stub enum values |

## File Manifest

### New Files
- tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_softcap.h
- tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_softcap.h
- tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_softcap.h
- tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_softcap.h
- tt_metal/hw/inc/api/compute/eltwise_unary/softcap.h
- tests/ttnn/unit_tests/operations/eltwise/test_softcap.py

### Modified Files
- tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h
- tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h
- tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h
- ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp
- ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp
- ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp
- ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp
- ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp
- ttnn/ttnn/operations/unary.py
- ttnn/ttnn/experimental_loader/golden_functions.py
