# Analysis: selu (reference for frac)

## SFPU Kernel Pattern
- **File**: `ckernel_sfpu_selu.h`
- **Namespace**: `ckernel::sfpu`
- **Template**: `template <bool APPROXIMATION_MODE, int ITERATIONS = 8>`
- **Function**: `calculate_selu()`
- **Init function**: `selu_init()` calls `_init_exponential_`
- **Includes**: `ckernel_sfpu_converter.h`, `ckernel_sfpu_exp.h`, `sfpi.h`, `sfpi_fp16.h`

## Key SFPI Patterns
- Uses `Converter::as_float(0x3FD63840)` to create float constants from IEEE754 bits
- Uses `v_if(v < 0.0f) { ... } v_endif;` for conditional branching
- Calls other SFPU helper functions: `_calculate_exponential_piecewise_`

## Compute API Pattern (Macro-based)
- `selu_tile(uint32_t idst)` uses `MATH(SFPU_UNARY_NO_PARAM_KERNEL_FN(calculate_selu, RC, APPROX, idst))`
- `selu_tile_init()` uses `MATH(SFPU_INIT_KERNEL_CALL(selu, ckernel::sfpu::selu_init, APPROX))`
- This is the macro-based API pattern (alternative to the direct llk call pattern)

## Key Takeaway for frac
- Shows IEEE754 bit constant pattern: `Converter::as_float(hex_value)`
- Shows the macro-based API header pattern (alternative to direct llk call)
- frac should use the simpler direct pattern (like softsign/hardswish) since it's not complex
