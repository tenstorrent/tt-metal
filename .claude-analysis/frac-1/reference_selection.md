# Reference Selection: frac

## Operation
- **Name**: frac
- **Math definition**: frac(x) = x - floor(x)
- **Category**: Rounding/truncation family

## Selected References (Top 5)

### 1. softsign
- **Rationale**: Simple unary SFPU operation with no parameters. Clean pattern for ckernel_sfpu, LLK dispatch, and API header. Uses basic SFPI arithmetic.
- **Similarity**: Both are simple unary ops with straightforward math and no extra parameters.

### 2. hardswish
- **Rationale**: Recently generated operation with clean, modern code patterns. Shows the complete end-to-end pattern including unary_op_utils.cpp registration.
- **Similarity**: Both use basic SFPI arithmetic operations (multiply, add, subtract) and conditional logic.

### 3. softshrink
- **Rationale**: Shows how parameterized operations work (with lambda). Also demonstrates conditional branching in SFPI (v_if/v_endif).
- **Similarity**: frac may need conditional handling for edge cases; softshrink shows the pattern.

### 4. cbrt
- **Rationale**: Another simple parameterless unary operation. Shows the minimal implementation pattern.
- **Similarity**: Both are simple math operations applied element-wise.

### 5. selu
- **Rationale**: Shows a slightly more complex operation with constants and conditional logic. Useful as a reference for SFPI register manipulation.
- **Similarity**: Both involve basic SFPI operations; selu shows how to use multiple SFPI operations in sequence.

## Selection Criteria
1. **Structural similarity**: Operations that follow the same file creation/modification pattern
2. **Mathematical similarity**: Operations using basic SFPI arithmetic (add, subtract, multiply)
3. **Complexity match**: frac is a simple operation, so simple references are preferred
4. **Recency**: Prefer recently generated/maintained code for up-to-date patterns

SELECTED_REFERENCES: softsign, hardswish, softshrink, cbrt, selu
