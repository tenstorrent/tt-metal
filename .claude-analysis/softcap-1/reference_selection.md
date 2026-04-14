# Reference Operation Selection for softcap

## Target Operation
- **Name**: softcap
- **Definition**: cap * tanh(x / cap)
- **Component operations identified**: scalar division (x / cap), tanh operation, scalar multiplication (cap * result)

## Selected References (ranked by relevance)

### 1. HARDTANH
- **Why selected**: Parametrized activation function that demonstrates how to handle scalar parameters in SFPU operations. Shows parameter passing, FP16_B format handling, and conditional logic patterns needed for softcap's cap parameter.
- **Relevance**: high — provides the structural template for parametrized unary operations with scalar values

### 2. SWISH
- **Why selected**: Composite activation function (x * sigmoid(x)) that combines input transformation with complex mathematical operations. Uses polynomial approximations and piecewise-linear segments, similar complexity to tanh implementations.
- **Relevance**: high — demonstrates composite activation pattern and polynomial approximation techniques that would be needed for tanh

### 3. SINH
- **Why selected**: Hyperbolic function implementation that uses exp-based computation (sinh(x) = (exp(x) - exp(-x)) / 2). Shows how to implement hyperbolic functions using exponential primitives and handles numerical stability issues.
- **Relevance**: high — tanh can be implemented as tanh(x) = sinh(x) / cosh(x) or using similar exp-based methods

### 4. ATANH
- **Why selected**: Inverse hyperbolic tangent that uses logarithmic decomposition and polynomial approximations. Demonstrates IEEE 754 floating-point manipulation and minimax polynomial coefficients for complex transcendental functions.
- **Relevance**: medium — provides insights into hyperbolic function implementation patterns and numerical techniques

### 5. FRAC
- **Why selected**: Simple unary operation that demonstrates bit manipulation techniques and IEEE 754 exponent handling. Shows clean SFPU kernel structure without complex approximations, good reference for basic operation patterns.
- **Relevance**: medium — provides structural template for straightforward unary operations and bit-level float manipulation techniques
