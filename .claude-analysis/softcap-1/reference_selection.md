# Reference Operation Selection for softcap

## Target Operation
- **Name**: softcap
- **Definition**: cap * tanh(x / cap)
- **Component operations identified**: division by parameter (x / cap), hyperbolic tangent (tanh), multiplication by parameter (cap * result)
- **Parameter**: float `cap` that must be wired through all abstraction layers

## Selected References (ranked by relevance)

### 1. tanh
- **Why selected**: Direct dependency - softcap uses tanh(x / cap) internally as its core non-linearity
- **Relevance**: high - the exact SFPU operation needed for softcap's mathematical definition
- **Implementation**: Standard unary operation in UnaryOpType enum (TANH), has torch golden function mapping

### 2. hardtanh
- **Why selected**: Parameterized activation with similar structure to softcap (clamping with parameters)
- **Relevance**: high - shows complete parameter wiring through all layers: marked in is_parametrized_type(), has SFPU kernel with 3 float parameters (ckernel_sfpu_hardtanh.h), demonstrates parameter handling patterns
- **Implementation**: Fully implemented parameterized operation with SFPU kernel showing parameter conversion and conditional logic

### 3. power
- **Why selected**: Standard parameterized unary operation that takes a float exponent parameter
- **Relevance**: high - listed in UnaryOpType enum (POWER, POWER_ITERATIVE), shows pattern for single-parameter float operations
- **Implementation**: Shows how mathematical operations with parameters are structured, though SFPU kernel implementation needs to be analyzed

### 4. rsub (reverse subtract)
- **Why selected**: Parameterized operation that performs scalar-tensor arithmetic (param - x)
- **Relevance**: medium - demonstrates parameter-first arithmetic operations, shows pattern for operations where parameter interacts directly with input values
- **Implementation**: Listed in UnaryOpType enum, represents pattern for operations like (cap - x) which is structurally similar to softcap's parameter usage

### 5. softshrink
- **Why selected**: Parameterized activation function marked in is_parametrized_type() infrastructure
- **Relevance**: medium - another parameterized activation that shows the overall infrastructure for parameter handling in unary SFPU operations
- **Implementation**: Marked as parameterized in the type system, provides pattern for how parameters flow through the unary operation framework
