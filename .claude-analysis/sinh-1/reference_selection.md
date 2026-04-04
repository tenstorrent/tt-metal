# Reference Selection: sinh

## Operation
- **Name**: sinh
- **Math definition**: (exp(x) - exp(-x)) / 2

## Selected References

### 1. cosh (PRIMARY)
- **Rationale**: Nearly identical mathematical structure. cosh(x) = (exp(x) + exp(-x)) / 2, sinh differs only by subtraction instead of addition. Same exp-based computation, same init function, same template parameters. This is the single most important reference.
- **Similarity**: 95% - identical structure, only the arithmetic operator differs

### 2. selu
- **Rationale**: Uses exp-based computation with exponential initialization. Demonstrates conditional branches and the _calculate_exponential_piecewise_ helper. Shows how to handle the exp init pattern.
- **Similarity**: 60% - shared exp dependency

### 3. atanh
- **Rationale**: Recently added operation following the modern split-include pattern. Good reference for the full registration stack (sfpu_split_includes.h, unary_op_utils.cpp, unary_ng_op_utils.cpp).
- **Similarity**: 40% - same registration pattern

### 4. cbrt
- **Rationale**: Simple unary operation with no parameters. Good structural reference for the registration layers without parameterized complexity.
- **Similarity**: 35% - simple no-param pattern

### 5. lgamma
- **Rationale**: Recently added operation with full modern stack. Shows the complete end-to-end pattern for both wormhole and blackhole architectures.
- **Similarity**: 30% - full stack reference

## SELECTED_REFERENCES: cosh, selu, atanh, cbrt, lgamma
