# Reference Analysis: hardswish

## Math Definition
hardswish(x) = x * min(max(x + 3, 0), 6) / 6 = x * hardsigmoid(x)

## SFPU Kernel Pattern
- File: `ckernel_sfpu_hardswish.h` (both blackhole and wormhole_b0)
- Namespace: `ckernel::sfpu`
- Template: `<bool APPROXIMATION_MODE, int ITERATIONS = 8>`
- Function: `calculate_hardswish()`
- No init function needed (no exp dependency)
- Simple loop with `#pragma GCC unroll 8`
- Uses `sfpi::dst_reg[0]` for input, writes result back to `sfpi::dst_reg[0]`
- Increments `sfpi::dst_reg++` each iteration
- Uses `sfpi::vFloat` for vector operations
- Uses `v_if`/`v_endif` for conditional assignment
- Uses `sfpi::vConst1` for constant 1.0f

## LLK Dispatch
- File: `llk_math_eltwise_unary_sfpu_hardswish.h`
- Includes: `llk_math_eltwise_unary_sfpu_init.h`, `llk_math_eltwise_unary_sfpu_params.h`, `ckernel_sfpu_hardswish.h`
- Init: `llk_math_eltwise_unary_sfpu_init<SfpuType::hardswish, APPROXIMATE>()`
- Execute: `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::calculate_hardswish<APPROXIMATE, ITERATIONS>, dst_index, vector_mode)`

## Compute API
- File: `tt_metal/hw/inc/api/compute/eltwise_unary/hardswish.h`
- `hardswish_tile(uint32_t idst)` calls `MATH((llk_math_eltwise_unary_sfpu_hardswish<APPROX>(idst)))`
- `hardswish_tile_init()` calls `MATH((llk_math_eltwise_unary_sfpu_hardswish_init<APPROX>()))`

## Registration Points
- SfpuType enum: `SfpuType::hardswish`
- UnaryOpType enum: `HARDSWISH`
- Split includes: `SFPU_OP_HARDSWISH_INCLUDE`
- unary_op_utils.cpp: get_macro_definition, get_op_init_and_func_default
- unary.hpp: `REGISTER_UNARY_OPERATION(hardswish, HARDSWISH)`
- Python golden: `torch.nn.functional.hardswish(input)`
