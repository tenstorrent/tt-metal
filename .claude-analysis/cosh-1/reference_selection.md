# Reference Operation Selection for cosh

## Target Operation
- **Name**: cosh
- **Definition**: (e^x + e^(-x)) / 2
- **Component operations identified**: exp, negation, addition, scalar multiplication (0.5)

## Selected References (ranked by relevance)

### 1. sinh
- **Why selected**: Nearly identical formula structure: sinh = (e^x - e^(-x)) / 2. The only difference is subtraction vs addition. Same trig family (SFPU_OP_TRIG_FAMILY_INCLUDE), same non-parametrized pattern, same exp-based implementation.
- **Relevance**: high -- the cosh implementation will be almost line-for-line identical to sinh

### 2. exp
- **Why selected**: Core building block. cosh directly calls `_sfpu_exp_21f_bf16_` twice (once for x, once for -x). Understanding exp's SFPU kernel implementation is critical for understanding how cosh computes its values.
- **Relevance**: high -- exp is a parametrized operation (fast_and_approximate_mode), providing a template for the init/func registration pattern

### 3. expm1
- **Why selected**: Another exp-based composition (e^x - 1). Shows how to compose exp with simple arithmetic in the SFPU pipeline. Non-parametrized, similar complexity to cosh.
- **Relevance**: medium -- demonstrates exp composition pattern in the SFPU_OP_COMPUTE_KERNEL_API_INCLUDE group

### 4. cos
- **Why selected**: Same trig family (SFPU_OP_TRIG_FAMILY_INCLUDE). Non-parametrized. Shows how trigonometric operations register their init and compute functions.
- **Relevance**: medium -- structural template for trig family registration

### 5. acosh
- **Why selected**: Inverse of cosh, same trig family. Non-parametrized. Demonstrates a more complex trig function that uses multiple SFPU primitives (log, sqrt, multiply).
- **Relevance**: medium -- shows trig family registration pattern with a different compute kernel
