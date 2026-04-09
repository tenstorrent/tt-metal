# Issues Log: softcap

## Configuration
- **Operation**: softcap
- **Math definition**: cap * tanh(x / cap)
- **Source**: direct formula
- **Output folder**: `.claude-analysis/softcap-1/`
- **Date**: 2026-04-09

## Phase Timeline
| Phase | Description | Status | Duration | Issues |
|-------|-------------|--------|----------|--------|
| 1 | Reference Discovery | ok | 360s | none |
| 2 | Reference Analysis | ok | 762s | hardshrink agent did not commit, orchestrator committed on behalf |
| 3 | Implementation | ok | 947s | none |
| 4 | Testing & Debugging | ok | 1929s | RISC-V GCC ICE required v_if flattening; eltwise_sfpu.cpp includes needed fixing for nuked clone |
| 5 | Documentation | pending | - | - |
| 6 | Self-Reflection | pending | - | - |

## File Manifest
### New Files
- tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_softcap.h
- tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_softcap.h
- tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_softcap.h
- tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_softcap.h
- tt_metal/hw/inc/api/compute/eltwise_unary/softcap.h
- tests/ttnn/unit_tests/operations/eltwise/test_softcap.py
- tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_conversions.h
- tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_conversions.h

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
- ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp

## Issues
| # | Phase | Severity | Description | Resolution |
|---|-------|----------|-------------|------------|
| 1 | 2 | LOW | Hardshrink analyzer did not commit; orchestrator committed on behalf | Committed manually |
| 2 | 4 | MEDIUM | RISC-V GCC ICE with nested v_if blocks in ckernel_sfpu_softcap.h | Tester flattened v_if nesting to avoid GCC LTO segfault |
| 3 | 4 | LOW | eltwise_sfpu.cpp had includes for nuked operations | Tester removed broken includes |
| 4 | 4 | LOW | SfpuType enum needed restoration for third-party LLK code | Tester restored full enum entries |
