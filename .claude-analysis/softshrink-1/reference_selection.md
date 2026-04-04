# Reference Selection: softshrink

## Target Operation
- **Name**: softshrink
- **Math**: x - lambda if x > lambda; x + lambda if x < -lambda; 0 otherwise
- **Parameters**: lambda (float, default=0.5)
- **Category**: Piecewise activation with threshold parameter

## Selected References (ranked by relevance)

### 1. hardtanh (BEST MATCH)
- **Rationale**: Piecewise function with two float parameters passed via `param0`/`param1`. Uses `v_if`/`v_endif` branching in SFPU kernel. Identical structure to softshrink (comparison-based branching with parameter-dependent output).
- **Key files**: `ckernel_sfpu_hardtanh.h`, `llk_math_eltwise_unary_sfpu_hardtanh.h`, `hardtanh.h` (compute API)

### 2. hardsigmoid
- **Rationale**: Piecewise function with clamping. Shows the pattern for `v_if`/`v_endif` branching and constant usage in SFPU kernels. No parameters (constants hardcoded).
- **Key files**: `ckernel_sfpu_hardsigmoid.h`, `llk_math_eltwise_unary_sfpu_hardsigmoid.h`, `hardsigmoid.h`

### 3. rpow
- **Rationale**: Parameterized SFPU operation that takes a single float parameter via `uint32_t` bitcast. Shows the `get_op_init_and_func_parameterized` pattern.
- **Key files**: `ckernel_sfpu_rpow.h`, `llk_math_eltwise_unary_sfpu_rpow.h`, `rpow.h`

### 4. selu
- **Rationale**: Piecewise activation (positive/negative branches). Shows `v_if(v < 0.0f)` pattern and init function.
- **Key files**: `ckernel_sfpu_selu.h`, `llk_math_eltwise_unary_sfpu_selu.h`, `selu.h`

### 5. softsign
- **Rationale**: Recently added SFPU operation showing the complete 12-layer integration pattern (kernel + LLK + compute API + op_utils + unary.hpp).
- **Key files**: `ckernel_sfpu_softsign.h`, `llk_math_eltwise_unary_sfpu_softsign.h`, `softsign.h`

## SELECTED_REFERENCES: hardtanh, hardsigmoid, rpow, selu, softsign
