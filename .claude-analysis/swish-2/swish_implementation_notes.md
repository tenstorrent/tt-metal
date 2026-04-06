# Implementation Notes: swish

## Math Definition
swish(x) = x / (1 + exp(-x)) = x * sigmoid(x)

### New Files
tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_swish.h
tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_swish.h
tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_swish.h
tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_swish.h
tt_metal/hw/inc/api/compute/eltwise_unary/swish.h

### Modified Files
tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h
tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h
tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h
tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_math_unary_sfpu_api.h
tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_math_unary_sfpu_api.h
ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp
ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp
ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp
ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp
ttnn/ttnn/operations/unary.py

## Design Decisions

### Sigmoid Approximation Strategy
The exp and sigmoid hardware primitives were intentionally removed from the codebase. The `approx_exp()` and `approx_recip()` SFPI functions (which map to SFPNONLINEAR) are only available on Blackhole (`__riscv_xtttensixbh`), not Wormhole. Since both architectures must use identical source files, we could not use these hardware-accelerated functions.

Instead, we implemented a hybrid polynomial + piecewise-linear approximation of sigmoid:

1. **Polynomial segment (|x| <= 2.5)**: Degree-3 polynomial fitted at x = 0.5, 1.0, 2.5:
   `sigmoid(t) ≈ 0.5 + t * (0.2533 + t * (-0.01479 + t * (-0.00747)))`
   Max error ≈ 0.007 (at t ≈ 2.0)

2. **Linear segment (2.5 < |x| <= 5.0)**: Linear interpolation through sigmoid(2.5) and sigmoid(5.0):
   `sigmoid(t) ≈ 0.0276 * t + 0.855`
   Max error ≈ 0.017 (at t ≈ 4.0)

3. **Saturation (|x| > 5.0)**: sigmoid = 1.0
   Max error ≈ 0.007 (at |x| = 5.0)

For negative x: `sigmoid(x) = 1 - sigmoid(|x|)`

### Reference Operations Used
- **hardswish**: Most useful reference. Very similar structure (x * hardsigmoid(x)). Used as template for all dispatch layers (LLK, API header, registration). The SFPU kernel pattern (pure SFPI with v_if clamping) was directly adapted.
- **hardsigmoid**: Pattern for no-parameter unary op registration. Verified the dispatch chain structure.
- **rpow**: Referenced for understanding SFPI primitives (abs, addexp, exexp, etc.) and the exp_21f algorithm concept. The `_float_to_int32_positive_()` was undefined, confirming that the exp algorithm approach wouldn't work directly.
- **cbrt**: Useful for understanding programmable constant registers and `is_fp32_dest_acc_en` patterns. Not directly used since swish doesn't need programmable constants.
- **softsign**: Verified dispatch wiring pattern for stubbed operations.

### PyTorch Golden Function
Used `torch.nn.functional.silu()` as the golden function since PyTorch's SiLU is mathematically identical to swish (SiLU = Sigmoid Linear Unit = swish).

## Known Limitations
- **Approximation accuracy**: The polynomial+piecewise approach has max sigmoid error of ~0.017 (in the linear segment around |x| ≈ 4). For swish, this translates to max absolute error of ~0.07 at x ≈ 4.
- **No hardware exp acceleration on Wormhole**: Blackhole's SFPNONLINEAR instruction supports exp() and reciprocal() which would enable a more accurate implementation, but we use the polynomial approach for architecture compatibility.
- **bfloat16 only**: The implementation is optimized for bfloat16 precision. FP32 accumulation mode is not specifically handled (no `is_fp32_dest_acc_en` branching).
