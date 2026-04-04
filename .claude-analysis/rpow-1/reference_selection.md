# Reference Selection: rpow

## Operation
- **Name**: rpow
- **Math**: base^x where base is a float parameter
- **Type**: Parameterized unary (float parameter)

## Selected References

### 1. power (POWER)
- **Rationale**: Direct inverse of rpow. power computes x^exponent, rpow computes base^x. The _sfpu_unary_power_21f_ algorithm (Moroz et al. 2022) can be reused with swapped operands.
- **Key patterns**: Parameter passing as uint32_t IEEE 754 bits, exp/log2 based computation, sfpu_unary_pow_init constants.

### 2. hardtanh (HARDTANH)
- **Rationale**: Recently implemented parameterized operation on this branch. Shows the exact pattern for passing uint32_t parameters through all layers (compute API -> LLK -> SFPU kernel).
- **Key patterns**: get_op_init_and_func_parameterized, is_parametrized_type, Converter::as_float for parameter decoding.

### 3. selu (SELU)
- **Rationale**: Recently implemented operation using exponential computation. Shows how to use _calculate_exponential_piecewise_ and exponential init patterns.
- **Key patterns**: ckernel_sfpu_exp.h inclusion, _init_exponential_ for init function, exp-based SFPU computation.

### 4. cbrt (CBRT)
- **Rationale**: Simple non-parameterized operation on this branch. Shows the complete 12-layer pattern for adding a new SFPU operation.
- **Key patterns**: Full file creation pattern, sfpu_split_includes.h, nanobind registration.

### 5. cosh (COSH)
- **Rationale**: Another simple operation on this branch showing compute API -> LLK -> SFPU kernel chain.
- **Key patterns**: string_to_unary_with_param registration, golden function attachment pattern.

## SELECTED_REFERENCES: power, hardtanh, selu, cbrt, cosh
