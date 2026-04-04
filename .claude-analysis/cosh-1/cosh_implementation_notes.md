# Implementation Notes: cosh

## Math Definition
cosh(x) = (e^x + e^(-x)) / 2

### New Files
- tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_cosh.h
- tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_cosh.h
- tt_metal/hw/inc/api/compute/eltwise_unary/cosh.h

### Modified Files
- tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h
- tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h
- tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h
- ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp
- ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp
- ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp
- ttnn/ttnn/operations/unary.py
- ttnn/cpp/ttnn/operations/eltwise/unary_ng/common/unary_ng_op_utils.cpp
- ttnn/cpp/ttnn/operations/eltwise/unary_ng/unary_ng.hpp
- ttnn/cpp/ttnn/operations/eltwise/unary_ng/unary_ng.cpp

## Design Decisions
- **Reference operations used**: sinh (closest match -- identical structure with subtraction instead of addition), exp (core building block -- cosh directly uses _sfpu_exp_21f_bf16_)
- **SFPU kernel**: Uses `_sfpu_exp_21f_bf16_<is_fp32_dest_acc_en>` which is the shared 21-term bfloat16 exponential approximation from the tt_llk submodule. Called twice per element: once for exp(x) and once for exp(-x), then summed and multiplied by 0.5.
- **Init function**: Uses `_init_exponential_<APPROXIMATION_MODE, false, p_sfpu::kCONST_1_FP16B>()` which sets up programmable constants for the exponential computation. Named `cosh_init` (separate from the pre-nuke `init_hyperbolic_trig` to keep the kernel self-contained).
- **Macro selection**: Uses `SFPU_THREE_PARAM_KERNEL_FP32_FIRST` for dispatch (same as pre-nuke cosh/sinh), and `SFPU_INIT_KERNEL_CALL` for init (same as pre-nuke).
- **Include guard**: Uses dedicated `SFPU_OP_COSH_INCLUDE` macro (not the family `SFPU_OP_TRIG_FAMILY_INCLUDE` since we're implementing each op individually).
- **SfpuType**: Added `cosh` entry to the enum (required for the `SFPU_INIT_KERNEL_CALL` macro).
- **UnaryOpType**: `COSH` already existed (preserved during nuke).
- **No parameterized type**: cosh takes no parameters, so no changes to `is_parametrized_type()`.
- **Both old and ng paths**: Registered cosh in both the legacy unary path (unary_op_utils.cpp, unary.hpp, unary_nanobind.cpp) and the next-gen path (unary_ng_op_utils.cpp, unary_ng.hpp, unary_ng.cpp).
- **approx_mode**: Falls through to `default: return false` -- cosh does not use approximation mode.

## Known Limitations
- **Input range**: Supported range approximately -9 to 9. For |x| > ~9, exp(x) overflows in bfloat16, causing incorrect results. This matches the pre-nuke behavior.
- **Precision**: The `_sfpu_exp_21f_bf16_` approximation introduces some error compared to the mathematically exact cosh. For bfloat16, expect ULP errors of 1-3.
