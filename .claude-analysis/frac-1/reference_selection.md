# Reference Selection for frac

## Operation
- **Name**: frac
- **Math**: x - floor(x) (fractional part of x)

## Analysis
The `frac` operation is a rounding-family operation. It already has:
- SFPU kernel implementation in LLK (`ckernel_sfpu_rounding_ops.h` -> `_calculate_frac_()`)
- API header (`rounding.h` -> `frac_tile()`, `rounding_op_tile_init()`)
- Enum entry (`UnaryOpType::FRAC`)
- C++ function registration (`REGISTER_UNARY_OPERATION(frac, FRAC)`)
- Composite fallback (`unary_composite_op.cpp`)
- Python binding (auto-generated `frac_t` type)

## Missing Pieces
1. `unary_op_utils.cpp`: FRAC not in `get_op_init_and_func_default()` -- will throw at dispatch
2. `unary_nanobind.cpp`: frac not explicitly bound with docs
3. `unary.py`: no golden function registered

## Selected References
The following 5 operations serve as patterns for the missing integration layers:

1. **softsign** - Simple parameterless unary SFPU op, recently added, similar pattern
2. **silu** - Simple parameterless unary SFPU op in both old and new paths
3. **floor** - Same rounding family, uses `rounding_op_tile_init()`, same dispatch pattern
4. **ceil** - Same rounding family, same init function
5. **trunc** - Same rounding family, `frac` is defined as `x - trunc(x)` in SFPU kernel

SELECTED_REFERENCES: softsign, silu, floor, ceil, trunc
