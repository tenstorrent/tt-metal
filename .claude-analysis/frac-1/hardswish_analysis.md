# Analysis: hardswish (reference for frac)

## SFPU Kernel Pattern
- **File**: `ckernel_sfpu_hardswish.h`
- **Namespace**: `ckernel::sfpu`
- **Template**: `template <bool APPROXIMATION_MODE, int ITERATIONS = 8>`
- **Function**: `calculate_hardswish()` - no extra params
- **No init function needed** (no special init)
- **Includes**: `ckernel.h`, `ckernel_defs.h`

## Key SFPI Patterns
- Uses `sfpi::vFloat`, arithmetic `*`, `+`, constants like `0.5f`, `0.0f`
- Conditional: `v_if(hsigmoid < 0.0f) { ... } v_endif;`
- No special init function (simplest pattern)

## LLK Dispatch Pattern
- Init: `llk_math_eltwise_unary_sfpu_init<SfpuType::hardswish, APPROXIMATE>()` - no callback!
- Compute: `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::calculate_hardswish<APPROXIMATE, ITERATIONS>, dst_index, vector_mode)`

## Registration
- `get_macro_definition`: returns `"SFPU_OP_HARDSWISH_INCLUDE"`
- `get_op_init_and_func_default`: `{"hardswish_tile_init();", fmt::format("hardswish_tile({});", idst)}`
- Both `unary_op_utils.cpp` and `unary_ng_op_utils.cpp` have entries

## Key Takeaway
- Simplest pattern: no init callback, no params, just uses SFPI basic arithmetic and conditionals
- This is the ideal pattern for `frac` to follow since frac also needs no parameters and no special initialization
