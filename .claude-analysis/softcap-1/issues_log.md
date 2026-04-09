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
| 1 | Reference Discovery | ok | ~366s | 0 |
| 2 | Reference Analysis | ok | ~798s | 0 |
| 3 | Implementation | ok | ~1038s | 0 |
| 4 | Testing & Debugging | ok | ~817s | 3 (build fixes) |
| 5 | Documentation | ok | ~30s | 0 |
| 6 | Self-Reflection | ok | ~300s | 0 |

## Issues

| # | Phase | Severity | Description | Resolution |
|---|-------|----------|-------------|------------|
| 1 | 4 | LOW | Stale includes in eltwise_sfpu.cpp referencing nuked headers | Removed references to trigonometry.h, rpow.h, rdiv.h, fill.h, mul_int_sfpu.h |
| 2 | 4 | LOW | Missing SfpuType enum values in llk_sfpu_types.h | Added ~35 enum values for both wormhole_b0 and blackhole |
| 3 | 4 | MEDIUM | std::bit_cast<float> not available in C++17 SFPU runtime | Replaced with union-based float<->uint32 conversion |

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
