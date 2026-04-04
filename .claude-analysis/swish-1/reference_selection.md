# Reference Operation Selection for swish

## Target Operation
- **Name**: swish
- **Definition**: x * sigmoid(x) = x / (1 + exp(-x))
- **Component operations identified**: sigmoid (LUT-based approximation), exp (exponential used inside sigmoid), multiply (x * sigmoid(x)), negate (-x inside sigmoid denominator)

## Selected References (ranked by relevance)

### 1. silu
- **Why selected**: silu (Sigmoid Linear Unit) is mathematically identical to swish: `f(x) = x * sigmoid(x)`. The kernel at `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_silu.h` implements exactly this computation using a piecewise-linear sigmoid approximation (`_sigmoid_piecewise_linear_positive_`) followed by the `val * result` multiply. This is the direct structural blueprint — the swish implementation should either reuse or closely mirror this kernel.
- **Relevance**: high — directly implements the same mathematical formula; provides the exact `dst_reg` loop body, the piecewise-positive sigmoid helper, the sign-based reflection for negative inputs, and the final multiply pattern that swish needs.

### 2. sigmoid
- **Why selected**: sigmoid is the central sub-expression of swish. The kernel at `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_sigmoid.h` shows the full LUT-based sigmoid implementation: `_init_sigmoid_()` loads six-entry piecewise-linear coefficients into LRegs, and `_calculate_sigmoid_()` applies `lut2(...) + 0.5f`. Swish must call sigmoid (or an equivalent inline helper) as an inner computation, and this file is the canonical reference for how sigmoid is initialised and computed on wormhole hardware.
- **Relevance**: high — swish is defined as x multiplied by sigmoid(x); the init pattern (LReg loading) and the lut2 compute call are directly reused in a swish implementation that delegates to sigmoid internally.

### 3. hardsigmoid
- **Why selected**: hardsigmoid is the project's existing in-tree sigmoid-approximation kernel (`tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_hardsigmoid.h`). It demonstrates the preferred file layout, header guards, namespace conventions (`ckernel::sfpu`), and `calculate_hardsigmoid` / `hardsigmoid_tile_init` naming conventions used in the local (non-LLK) ckernel layer — the same layer where a new `ckernel_sfpu_swish.h` would be placed. It is also a sigmoid variant, making its structure directly transferable.
- **Relevance**: high — primary template for file structure, namespace, and `unary_op_utils.cpp` registration pattern; its piecewise-linear sigmoid formula is the approximation swish will rely on when not using the LUT.

### 4. selu
- **Why selected**: selu (`tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_selu.h`) is the most complete exp-based activation already in the project's local ckernel directory. It uses `_calculate_exponential_piecewise_` and `_init_exponential_`, shows the `ckernel_sfpu_exp.h` include chain, and demonstrates the pattern of calling an exp helper inside a loop and then multiplying the result by a scalar. Swish's alternative formulation `x / (1 + exp(-x))` would follow the same init and exp-call pattern.
- **Relevance**: medium — informs the exp-based code path; the `selu_init()` / `calculate_selu()` pairing and `SFPU_OP_SELU_INCLUDE` macro registration directly show how to wire a new exp-dependent op into `unary_op_utils.cpp` and `get_macro_definition()`.

### 5. elu
- **Why selected**: elu (`tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_elu.h`) is a parameterised activation that calls `_calculate_exponential_piecewise_` inside a conditional branch, then multiplies by a slope. It is the simplest clean example of: (a) passing a float parameter through as a `vFloat` scalar, (b) using `_init_elu_` / `_calculate_elu_` split, and (c) the `SCALE_EN` / `SKIP_POSITIVE_CHECK` flags passed to the exp helper. These patterns appear in the non-LUT swish implementation path that computes exp(-x), adds 1, and divides.
- **Relevance**: medium — reference for the exp-helper interface (`_calculate_exponential_piecewise_`, `_init_exponential_`), parameter-passing idiom, and the init/compute function split; easier to follow than selu because it has no two-constant coupling.
