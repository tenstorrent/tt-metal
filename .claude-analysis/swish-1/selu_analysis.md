# Reference Analysis: selu

## Math Definition
SELU(x) = scale * (max(0,x) + min(0, alpha*(exp(x)-1)))
- scale = 1.0507009873554804934193349852946
- alpha = 1.6732632423543772848170429916717

## SFPU Kernel Pattern
- File: `ckernel_sfpu_selu.h`
- Namespace: `ckernel::sfpu`
- Includes: `ckernel_sfpu_converter.h`, `ckernel_sfpu_exp.h`, `sfpi.h`, `sfpi_fp16.h`
- Template: `<bool APPROXIMATION_MODE, int ITERATIONS = 8>`
- Function: `calculate_selu()`
- Uses `#pragma GCC unroll 0` (no unrolling due to exp complexity)
- Uses `_calculate_exponential_piecewise_<APPROXIMATION_MODE, SCALE_EN, SKIP_POSITIVE_CHECK>()` for exp
- Constants via `Converter::as_float(hex_value)` for exact FP32 representations
- Has `selu_init()` function that calls `_init_exponential_<APPROXIMATION_MODE, FAST_APPROX, EXP_BASE_SCALE_FACTOR>()`

## LLK Dispatch
- File: `llk_math_eltwise_unary_sfpu_selu.h`
- Init: `llk_math_eltwise_unary_sfpu_init<SfpuType::selu, APPROXIMATE>(ckernel::sfpu::selu_init<APPROXIMATE>)`
- Execute: `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::calculate_selu<APPROXIMATE>, dst_index, vector_mode)`

## Key Insight for Swish
- Shows how to use exp() in custom SFPU kernels
- Shows the init pattern for exp-dependent operations
- The `_calculate_exponential_piecewise_` function is the preferred way to compute exp()
- Init calls `_init_exponential_<APPROXIMATION_MODE, false, p_sfpu::kCONST_1_FP16B>()`
