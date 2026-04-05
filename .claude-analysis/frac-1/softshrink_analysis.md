# Analysis: softshrink (reference for frac)

## SFPU Kernel Pattern
- **File**: `ckernel_sfpu_softshrink.h`
- **Namespace**: `ckernel::sfpu`
- **Template**: `template <bool APPROXIMATION_MODE, int ITERATIONS = 8>`
- **Function**: `calculate_softshrink(std::uint32_t param0)` - takes one parameter
- **Includes**: `ckernel_sfpu_converter.h`, `sfpi.h`

## Key SFPI Patterns
- Parameter conversion: `sfpi::vFloat lambda_val = Converter::as_float(param0);`
- Conditional branching: `v_if(val > lambda_val) { ... } v_endif;`
- Default result initialization: `sfpi::vFloat result = 0.0f;`

## LLK Dispatch (parameterized)
- Extra param in llk function: `llk_math_eltwise_unary_sfpu_softshrink(uint dst_index, uint param0, ...)`
- Passes param0 to `_llk_math_eltwise_unary_sfpu_params_`

## Registration (parameterized)
- In `get_op_init_and_func_parameterized`:
  ```cpp
  case UnaryOpType::SOFTSHRINK:
      return {"softshrink_tile_init();", fmt::format("softshrink_tile({}, {:#010x}u);", idst, std::bit_cast<uint32_t>(lambda_val))};
  ```

## Key Takeaway
- Shows how parameterized ops work, but frac doesn't need params
- Shows clean v_if/v_endif pattern for conditional logic
