# Reference Operation Selection for softcap

## Target Operation
- **Name**: softcap
- **Definition**: cap * tanh(x / cap)
- **Component operations identified**: division by parameter, tanh, multiplication by parameter

## Selected References (ranked by relevance)

### 1. swish
- **Why selected**: swish(x) = x * sigmoid(x) implements the exact same pattern as softcap: input multiplication by a function result. The SFPU kernel shows sophisticated approximation techniques for transcendental functions using polynomial and piecewise-linear approaches. Uses conditional branches and vectorized SFPI operations.
- **Relevance**: high — structural template for multiplicative composition with transcendental function, SFPU implementation patterns, approximation strategies

### 2. tanhshrink
- **Why selected**: tanhshrink(x) = x - tanh(x) directly uses the tanh operation that softcap requires. The kernel implementation shows how to call tanh_tile_init() and tanh_tile(), which are the exact primitives needed for softcap's tanh component.
- **Relevance**: high — provides direct tanh implementation pattern, tile operation sequencing, shows how to combine tanh with arithmetic operations

### 3. hardtanh
- **Why selected**: hardtanh is a parameterized operation that takes min_val and max_val parameters, demonstrating the complete parameter handling pipeline from API definition to kernel parameter passing. Uses the same UnaryWithParam infrastructure that softcap will need.
- **Relevance**: high — parameterized operation template, shows parameter registration, validation, and runtime argument packing

### 4. atanh
- **Why selected**: atanh is implemented as a custom SFPU operation (has dedicated ckernel_sfpu_atanh.h), showing the complete SFPU implementation structure including custom init and compute functions. Related to tanh mathematically (inverse hyperbolic tangent).
- **Relevance**: medium — SFPU implementation structure, custom operation patterns, hyperbolic function family

### 5. softshrink
- **Why selected**: softshrink is a parameterized unary operation (listed in is_parametrized_type) that demonstrates how parameterized operations are registered and handled in the type system. Provides reference for parameter-dependent conditional logic.
- **Relevance**: medium — parameterized operation infrastructure, type system integration, conditional parameter-based computation
