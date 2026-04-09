# Issues Log: softcap

## Configuration
- **Operation**: softcap
- **Math definition**: cap * tanh(x / cap)
- **Source**: direct formula + key_notes (docs/sfpu_operations/key_notes/softcap_key_notes.md)
- **Output folder**: `.claude-analysis/softcap-1/`
- **Date**: 2026-04-09

## Phase Timeline
| Phase | Description | Status | Duration | Issues |
|-------|-------------|--------|----------|--------|
| 1 | Reference Discovery | ok | 353s | 0 |
| 2 | Reference Analysis | ok | 816s | 1 (hardtanh agent did not commit) |
| 3 | Implementation | ok | 1070s | 0 |
| 4 | Testing & Debugging | ok | 328s | 2 (kernel JIT includes, missing SfpuType enums) |
| 5 | Documentation | ok | ~30s | 0 |
| 6 | Self-Reflection | ok | 478s | 0 |

## Issues

| # | Phase | Severity | Description | Resolution |
|---|-------|----------|-------------|------------|
| 1 | 2 | Low | Hardtanh analyzer agent completed analysis file but did not commit before finishing | Orchestrator staged and committed on its behalf |
| 2 | 4 | Medium | Kernel JIT compilation failed due to unconditional includes of nuked operation headers (trigonometry, mul_int_sfpu, rpow, rdiv, fill) in eltwise_sfpu.cpp | Tester agent removed the problematic includes |
| 3 | 4 | Medium | Missing SfpuType enum values (Trigonometry, Mul_int_SFPU, Rpow, Rdiv, Fill) referenced by third_party tt_llk headers that were stripped during the nuke operation | Tester agent added placeholder enum values in llk_sfpu_types.h |

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
- ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp
