# Reference Selection: swish

## Operation
- **Name**: swish
- **Math**: swish(x) = x * sigmoid(x) = x / (1 + exp(-x))

## Selected References (Top 5)

### 1. hardswish
- **Rationale**: Structurally identical pattern - x * f(x) where f is a sigmoid variant. hardswish = x * hardsigmoid(x), swish = x * sigmoid(x). Closest structural match.
- **Files**: ckernel_sfpu_hardswish.h, llk_math_eltwise_unary_sfpu_hardswish.h, hardswish.h (compute API)

### 2. selu
- **Rationale**: Uses exp() function with conditional logic. Shows _calculate_exponential_piecewise_ usage and exp init pattern. Swish needs exp for sigmoid computation.
- **Files**: ckernel_sfpu_selu.h, llk_math_eltwise_unary_sfpu_selu.h, selu.h (compute API)

### 3. cosh
- **Rationale**: Uses _sfpu_exp_21f_bf16_ function directly. Shows exp init pattern and how to use exponential in a simple formula.
- **Files**: ckernel_sfpu_cosh.h, cosh.h (compute API)

### 4. hardsigmoid
- **Rationale**: Sigmoid approximation reference. Shows clamping patterns and the structural template for sigmoid-like functions.
- **Files**: ckernel_sfpu_hardsigmoid.h, llk_math_eltwise_unary_sfpu_hardsigmoid.h, hardsigmoid.h (compute API)

### 5. softsign
- **Rationale**: x / (1 + |x|) pattern is similar to x / (1 + exp(-x)). Shows how to combine division (via reciprocal) with x.
- **Files**: ckernel_sfpu_softsign.h, llk_math_eltwise_unary_sfpu_softsign.h, softsign.h (compute API)

SELECTED_REFERENCES: hardswish, selu, cosh, hardsigmoid, softsign
