# SFPU Analysis: cosh

## Overview
cosh is the hyperbolic cosine operation: cosh(x) = (exp(x) + exp(-x)) / 2.

## SFPU Kernel
- **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_cosh.h`
- **Function**: `calculate_cosh<APPROXIMATION_MODE, is_fp32_dest_acc_en, ITERATIONS>()`
- **Init**: `cosh_init()` calls `_init_exponential_<APPROXIMATION_MODE, false, p_sfpu::kCONST_1_FP16B>()`

### Key Implementation Pattern
```cpp
for (int d = 0; d < ITERATIONS; d++) {
    sfpi::vFloat v = sfpi::dst_reg[0];
    sfpi::vFloat result =
        (_sfpu_exp_21f_bf16_<is_fp32_dest_acc_en>(v) + _sfpu_exp_21f_bf16_<is_fp32_dest_acc_en>(-v)) * 0.5f;
    sfpi::dst_reg[0] = result;
    sfpi::dst_reg++;
}
```

Uses `_sfpu_exp_21f_bf16_` for computing exp(x) and exp(-x), then averages.

## Compute API
- **File**: `tt_metal/hw/inc/api/compute/eltwise_unary/cosh.h`
- **Init**: `cosh_tile_init()` -> `SFPU_INIT_KERNEL_CALL(cosh, ckernel::sfpu::cosh_init, APPROX)`
- **Compute**: `cosh_tile(idst)` -> `SFPU_THREE_PARAM_KERNEL_FP32_FIRST(calculate_cosh, APPROX, DST_ACCUM_MODE, 8, idst, (int)VectorMode::RC)`

## Split Includes
In `sfpu_split_includes.h`:
```cpp
#if SFPU_OP_COSH_INCLUDE
#include "api/compute/eltwise_unary/cosh.h"
#endif
```

## Op Registration
- **Enum**: `UnaryOpType::COSH` in `unary_op_types.hpp`
- **Block defines**: `case UnaryOpType::COSH: return "SFPU_OP_COSH_INCLUDE";`
- **Init/func**: `case UnaryOpType::COSH: return {"cosh_tile_init();", fmt::format("cosh_tile({});", idst)};`
- **String parse**: `if (name == "cosh") { return UnaryWithParam(UnaryOpType::COSH); }`

## Python Binding
- `unary.hpp`: `REGISTER_UNARY_OPERATION(cosh, COSH)`
- `unary_nanobind.cpp`: `bind_unary_operation<"cosh", &ttnn::cosh>(...)`
- Golden: `ttnn.attach_golden_function(ttnn.cosh, golden_function=_golden_function_cosh)` using `torch.cosh`

## Relevance to sinh
Extremely high - identical structure, just change `+` to `-` in the formula. Same exp-based approach, same init.
