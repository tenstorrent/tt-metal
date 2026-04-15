# Reference Operation Selection for softcap

## Target Operation
- **Name**: softcap
- **Definition**: cap * tanh(x / cap)
- **Component operations identified**: division by parameter, tanh transcendental function, multiplication by parameter

## Selected References (ranked by relevance)

### 1. tanh
- **Why selected**: Core mathematical component - softcap directly uses tanh(x/cap). The tanhshrink implementation shows tanh_tile_init() and tanh_tile() SFPU functions are available.
- **Relevance**: high — tanh is the fundamental transcendental function used in softcap. Provides direct reference for SFPU tanh implementation patterns.

### 2. hardtanh
- **Why selected**: Parameterized operation that takes min/max values and performs conditional clamping. Shows how to handle float parameters in unary operations and demonstrates parameter validation patterns.
- **Relevance**: high — demonstrates parameter handling infrastructure for unary operations with float parameters, which softcap needs for the 'cap' parameter.

### 3. softshrink
- **Why selected**: Parameterized operation with lambda threshold parameter. Identified in is_parametrized_type() function and shows single-parameter operation structure.
- **Relevance**: medium — provides template for single float parameter operations similar to softcap's cap parameter handling.

### 4. xielu
- **Why selected**: Parameterized activation function taking two float parameters (alpha_p, alpha_n). Shows multi-parameter unary operation implementation pattern and parameter passing through EltwiseUnaryWithParam.
- **Relevance**: medium — demonstrates advanced parameter handling for unary operations, useful for understanding the parameter infrastructure softcap will use.

### 5. tanhshrink
- **Why selected**: Composite operation that performs input - tanh(input), directly using tanh SFPU function in kernel implementation. Shows how to combine tanh with other operations in SFPU kernels.
- **Relevance**: medium — shows practical tanh usage in composite operations and demonstrates the SFPU kernel pattern that softcap will need (tanh combined with arithmetic operations).
