# SFPU Operation Analysis: cosh

## Overview
- **Operation**: cosh (hyperbolic cosine)
- **Math**: cosh(x) = (exp(x) + exp(-x)) / 2
- **Parameters**: None
- **Approx mode**: false (no special approximation)

## SFPU Kernel (ckernel_sfpu_cosh.h)
Located at: `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_cosh.h`

Both architectures have identical implementations:
- Template: `<bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>`
- Uses `_sfpu_exp_21f_bf16_<is_fp32_dest_acc_en>(v)` for exp(x) and `_sfpu_exp_21f_bf16_<is_fp32_dest_acc_en>(-v)` for exp(-x)
- Sum of both exps multiplied by 0.5f
- Init: `_init_exponential_<APPROXIMATION_MODE, false, p_sfpu::kCONST_1_FP16B>()`
- Uses `#pragma GCC unroll 8` for the loop

## Compute API (cosh.h)
Located at: `tt_metal/hw/inc/api/compute/eltwise_unary/cosh.h`

- Includes: `ckernel_sfpu_cosh.h`, `llk_math_eltwise_unary_sfpu_macros.h`
- `cosh_tile_init()`: Uses `SFPU_INIT_KERNEL_CALL(cosh, ckernel::sfpu::cosh_init, APPROX)`
- `cosh_tile(uint32_t idst)`: Uses `SFPU_THREE_PARAM_KERNEL_FP32_FIRST(calculate_cosh, APPROX, DST_ACCUM_MODE, 8, idst, (int)VectorMode::RC)`

## Registration in unary_op_utils.cpp
- `get_macro_definition`: returns `"SFPU_OP_COSH_INCLUDE"`
- `get_op_init_and_func_default`: returns `{"cosh_tile_init();", "cosh_tile({idst});"}`
- `string_to_unary_with_param`: `"cosh"` -> `UnaryWithParam(UnaryOpType::COSH)`

## Registration in unary_ng_op_utils.cpp
- `get_macro_definition`: returns `"SFPU_OP_COSH_INCLUDE"`
- `get_op_init_and_func`: returns `{"cosh_tile_init();", "cosh_tile({idst});"}`

## Split Includes (sfpu_split_includes.h)
```cpp
#if SFPU_OP_COSH_INCLUDE
#include "api/compute/eltwise_unary/cosh.h"
#endif
```

## Enum (unary_op_types.hpp)
`COSH` is in the `UnaryOpType` enum.

## Python binding (unary.hpp)
`REGISTER_UNARY_OPERATION(cosh, COSH)`

## Golden function (unary.py)
Separate golden function outside the TTNN_ELTWISE_UNARY_CPP_FUNCTIONS list:
```python
def _golden_function_cosh(input_tensor_a, *args, **kwargs):
    return torch.cosh(input_tensor_a)
ttnn.attach_golden_function(ttnn.cosh, golden_function=_golden_function_cosh)
```

## Key Takeaway for sinh
sinh is nearly identical to cosh. The only difference is subtraction instead of addition in the kernel:
`(exp(x) - exp(-x)) * 0.5f` instead of `(exp(x) + exp(-x)) * 0.5f`
