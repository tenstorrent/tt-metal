# Reference Operation Selection for rrelu

## Target Operation
- **Name**: rrelu
- **Definition**: RReLU(x) = x if x >= 0; RReLU(x) = a * x if x < 0; where in training mode: a ~ Uniform(lower, upper) (random per element), in eval mode: a = (lower + upper) / 2. Default: lower = 1/8 (0.125), upper = 1/3 (~0.333).
- **Parameters**: lower (float), upper (float)
- **Component operations identified**:
  - Conditional branch on sign of x (x >= 0 vs x < 0)
  - Identity pass-through for non-negative inputs
  - Scalar multiply for negative inputs (a * x)
  - Eval mode: precomputed scalar a = (lower + upper) / 2 (exactly leaky relu)
  - Training mode: per-element uniform random float generation in [lower, upper) used as the scalar multiplier

## Selected References (ranked by relevance)

### 1. LEAKY_RELU
- **Why selected**: The eval mode of rrelu is mathematically identical to leaky relu: apply a fixed scalar `a` to negative inputs and pass positive inputs through unchanged. The SFPU kernel `_calculate_lrelu_` in `ckernel_sfpu_relu.h` (backed by `tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_relu.h`) uses `SFPSETCC` to conditionally multiply by a loaded slope for x < 0. For eval mode, rrelu can be implemented by computing `a = (lower + upper) / 2` on the host and forwarding it as the slope parameter using this exact same pattern.
- **Relevance**: High — the eval-mode implementation is a direct reuse of this kernel with a precomputed parameter; the conditional multiply-by-slope pattern is the structural core of both operations.

### 2. PRELU_SFPU
- **Why selected**: PReLU (`ckernel_sfpu_prelu.h`) implements the same `v_if(a < 0.0f) { a = a * init; }` pattern using the higher-level sfpi DSL (`vFloat`, `v_if`, `dst_reg` loop) rather than raw SFPU instructions. It is the simplest parametrized conditional-multiply activation in the codebase and provides the cleanest structural template for rrelu's negative-branch multiply logic.
- **Relevance**: High — same single-parameter conditional multiply structure; easier to follow than leaky_relu's raw-instruction implementation; directly applicable to the eval-mode case and as the negative-branch template for training mode.

### 3. DROPOUT
- **Why selected**: Training-mode rrelu requires generating a per-element uniform random float in [lower, upper) and using it as the slope. The dropout kernel (`ckernel_sfpu_dropout.h` / `_calculate_dropout_`) is the primary reference for PRNG usage on the SFPU: it calls `init_prng_seed`, uses `TTI_SFPMOV(0, 9, lreg, 8)` to extract a pseudorandom uint32 from the hardware RNG, then manipulates sign/exponent bits to produce a float in a desired range. The `rand` kernel in `ckernel_sfpu_rand.h` demonstrates the complete pipeline: generate raw random bits, force exponent to 127 to get [1,2), subtract 1 to get [0,1), then scale with `SFPMAD(rand, scale, from)` to reach [lower, lower+range). Both files are essential reading for the training-mode PRNG path.
- **Relevance**: High — supplies the exact PRNG instruction sequence and the [lower, upper) scaling arithmetic needed for training-mode rrelu.

### 4. ELU
- **Why selected**: ELU (`ckernel_sfpu_elu.h`) uses the same `v_if(v < 0.0f) { ... } v_endif; dst_reg++` per-element loop pattern with a loaded scalar parameter, but applies a more complex expression inside the negative branch. It demonstrates the standard pattern for a parametrized activation that modifies only negative inputs, including the `is_fp32_dest_acc_en` template flag and the `float_to_fp16b` rounding path — both of which rrelu will need to replicate for correct dtype handling.
- **Relevance**: Medium — shares the conditional negative-branch loop structure and the fp16b rounding guard; good model for the overall function signature and dtype handling.

### 5. SELU
- **Why selected**: SELU (`ckernel_sfpu_unary_selu.h`) takes two float parameters (`scale` and `alpha`), loads both via `Converter::as_float`, and applies a two-branch conditional: `v_if(v >= 0.0f)` scales by one parameter, `v_else` applies a more complex expression with both parameters. This two-parameter structure is the closest match to rrelu's two-parameter interface (`lower`, `upper`). It also demonstrates `TT_FATAL(params.size() == 2, ...)` validation and the `selu_tile({}, {:#x}u, {:#x}u)` two-argument tile call registration pattern in `unary_op_utils.cpp`, which rrelu should replicate.
- **Relevance**: Medium — provides the two-parameter registration pattern (init string, tile-call format string with two hex args) and the dual-branch parametrized activation structure.
