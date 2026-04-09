# Reference Operation Selection for rrelu

## Target Operation
- **Name**: rrelu
- **Definition**: x if x >= 0, a*x if x < 0, where a = (lower+upper)/2 in eval mode, a ~ Uniform(lower, upper) in training mode. Parameters: lower (default 0.125), upper (default 1/3).
- **Component operations identified**:
  - Conditional branch on sign of x (x >= 0 vs x < 0)
  - Scalar multiply for the negative branch (a * x)
  - In eval mode: deterministic scalar a = (lower + upper) / 2
  - In training mode: stochastic a sampled from Uniform(lower, upper) via PRNG
  - Two float parameters passed at dispatch time: lower, upper

## Selected References (ranked by relevance)

### 1. leaky_relu
- **Why selected**: leaky_relu is mathematically identical to rrelu in eval mode. Both implement `x if x >= 0, slope*x if x < 0`. The only structural difference is that rrelu takes two params (lower, upper) and computes slope = (lower+upper)/2 at kernel dispatch, while leaky_relu takes slope directly. The SFPU kernel for leaky_relu uses the same v_if conditional branch pattern and a single float multiplier on the negative branch. This is the closest structural match available.
- **Relevance**: high — the core kernel logic (`v_if(x < 0) { dst = x * slope; } v_endif`) maps directly to rrelu's eval-mode kernel body.

### 2. prelu_sfpu
- **Why selected**: prelu_sfpu has the same formula as leaky_relu (`max(0,x) + weight*min(0,x)`) but uses a learnable parameter. Like rrelu, the slope parameter is a runtime value rather than a compile-time constant. The prelu implementation demonstrates how to receive a float weight parameter from the C++ dispatch layer and apply it inside the SFPU kernel loop. Since rrelu also needs to receive `lower` and `upper` as float parameters at runtime, prelu shows the two-float-param pattern at the dispatch layer.
- **Relevance**: high — parameterized slope multiply pattern maps directly to rrelu's negative-branch multiply.

### 3. dropout
- **Why selected**: rrelu's training mode requires sampling a ~ Uniform(lower, upper) per element. The dropout kernel is the canonical SFPU implementation that uses the hardware PRNG (TTI_SFPMOV with instr_mod1=8, lreg_c=9 to generate a pseudorandom uint32). The init_prng_seed() function from dropout's `_init_dropout_` is the same mechanism rrelu will need to seed the RNG for training-mode operation. Without this reference, the implementor would need to discover the PRNG interface independently.
- **Relevance**: high — the PRNG generation pattern (TTI_SFPMOV for random number, SETSGN to clear sign bit, comparison logic) is essential for rrelu training mode.

### 4. swish
- **Why selected**: swish is the best fully-implemented SFPU kernel in the repository following the Wave-2/Wave-3 kernel template pattern. It demonstrates the complete five-layer stack: `ckernel_sfpu_swish.h` (core math), `llk_math_eltwise_unary_sfpu_swish.h` (LLK glue), `api/compute/eltwise_unary/swish.h` (compute API header), registration in `sfpu_split_includes.h`, `SfpuType::swish` enum entry, and `unary_op_utils.cpp` dispatch. It also uses the same `v_if(condition) { ... } v_endif` sfpi conditional branch structure that rrelu requires. As the most complete working example of the new-style kernel, it provides the best structural template to follow.
- **Relevance**: high — end-to-end implementation template for all five layers an rrelu implementation must touch.

### 5. hardtanh
- **Why selected**: hardtanh takes three uint32_t parameters (packed as FP16_B) and dispatches them via `_calculate_hardtanh_(iterations, param0, param1, param2)`. This demonstrates how to pack multiple float parameters from C++ into uint32 arguments that the SFPU kernel receives, then unpack them inside the kernel using `s2vFloat16b()`. For rrelu, the two parameters `lower` and `upper` will need to be passed to the kernel in the same way. The `SFPU_UNARY_TWO_PARAM_KERNEL_FN` or similar macro in `llk_math_eltwise_unary_sfpu_macros.h` follows this pattern. hardtanh is simpler than softplus (beta+threshold) and shows the minimal two-param case.
- **Relevance**: medium — parameter packing/unpacking pattern for passing lower and upper as uint32 args to the SFPU calculate function.
