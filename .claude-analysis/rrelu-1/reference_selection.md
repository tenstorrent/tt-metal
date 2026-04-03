# Reference Operation Selection for rrelu

## Target Operation
- **Name**: rrelu
- **Definition**: `x if x >= 0; a*x if x < 0` where in training mode `a ~ Uniform(lower, upper)` (default lower=1/8, upper=1/3), and in eval mode `a = (lower + upper) / 2`
- **Component operations identified**:
  - Conditional branch on sign of x (same as relu/leaky_relu/prelu)
  - Identity pass-through for non-negative values
  - Scalar multiply by slope `a` for negative values
  - Uniform random number generation in range `[lower, upper)` for training mode
  - PRNG seed initialization
  - Two float parameters (lower, upper) encoded as uint32 bit-casts
  - Fixed-slope eval mode (structurally identical to leaky_relu)

## Selected References (ranked by relevance)

### 1. leaky_relu
- **Why selected**: The eval mode of rrelu is structurally identical to leaky_relu: `x if x >= 0, slope*x if x < 0` with a single precomputed float slope. The LLK inner implementation (`_calculate_lrelu_` in `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_relu.h`) shows the exact low-level SFPU pattern to use: `TTI_SFPLOAD` → `TTI_SFPSETCC` (branch on sign) → `TTI_SFPMUL` (multiply by slope in LREG2) → `TTI_SFPENCC` → `TTI_SFPSTORE`. The `unary_op_utils.cpp` registration (`leaky_relu_tile_init()` + `leaky_relu_tile(idst, slope_as_uint32)`) is the direct template for rrelu's eval-mode registration.
- **Relevance**: High — provides the complete SFPU compute kernel pattern and the `unary_op_utils.cpp` single-float-parameter registration idiom for the conditional slope-multiply branch.

### 2. prelu
- **Why selected**: `calculate_prelu` in `ckernel_sfpu_prelu.h` uses the high-level sfpi `v_if(a < 0.0f) { a = a * init; } v_endif` pattern (vs leaky_relu's low-level TTI instructions), making it a cleaner, more readable template for implementing the same conditional slope multiplication. The two implementations together show both coding styles available for the same mathematical operation, allowing the implementor to choose the appropriate abstraction level.
- **Relevance**: High — identical mathematical structure to rrelu's eval mode; the sfpi high-level conditional style is directly adaptable for the `x < 0` branch of rrelu.

### 3. rand
- **Why selected**: The `rand` SFPU kernel (`ckernel_sfpu_rand.h`) implements exactly the `Uniform(from, to)` distribution that rrelu training mode requires per element. It shows: (1) `init_prng_seed(seed)` for PRNG initialization, (2) `TTI_SFPMOV(0, 9, LREG0, 8)` as the PRNG instruction that generates a pseudorandom uint32, (3) sign-bit clearing and exponent forcing to get a float in `[1,2)`, (4) subtraction of 1.0 to get `[0,1)`, (5) scaling via `TTI_SFPMAD(LREG0, scale, from, LREG0)` to map to `[from, from+scale)`. The `rand_program_factory.cpp` and `compute_uniform.cpp` show how `from` and `to` are passed as bit-cast uint32 runtime args and how per-core seeds are assigned.
- **Relevance**: High — provides the complete Uniform(lower, upper) generation pattern that is the core novelty of rrelu vs. leaky_relu/prelu.

### 4. dropout
- **Why selected**: Dropout is the only other existing operation in the codebase that combines PRNG generation with a conditional value transformation on the same element. The LLK inner implementation (`sfpu/ckernel_sfpu_dropout.h`) shows the interleaved pattern of: load data → scale → generate PRNG → compare against threshold → conditional zero-out, all within a single tile loop. `_init_dropout_` calls `init_prng_seed(seed)` identically to `rand_init`. The kernel-level `dropout_kernel.cpp` shows the `dropout_kernel_init(seed)` + `dropout_tile(idst, probability, scale)` API shape, which is analogous to what rrelu's training-mode compute kernel will need (rrelu_init(seed) + rrelu_tile(idst, lower, upper)).
- **Relevance**: High — the only existing operation that combines PRNG with in-place conditional modification of a data tile; shows the structural template for rrelu's training mode at both LLK and compute kernel API level.

### 5. selu
- **Why selected**: SELU (`calculate_selu` in `ckernel_sfpu_unary_selu.h`) is a two-parameter conditional activation that uses `v_if(v >= 0.0f) { ... } v_else { ... } v_endif` with two preloaded float parameters (`scale` and `alpha`) both passed as `uint32_t` bit-casts. The `unary_op_utils.cpp` registration shows exactly how to implement the two-parameter init/compute pair: `selu_tile_init()` and `selu_tile(idst, param0_as_uint32, param1_as_uint32)`. For rrelu's eval mode, this is the registration template to follow when `lower` and `upper` are both needed as parameters to compute `(lower+upper)/2` on-device, or to store them directly as `lower` and `upper` for use in training mode.
- **Relevance**: Medium-High — the two-parameter float activation registration pattern in `unary_op_utils.cpp` is directly applicable to rrelu's eval mode (two params: lower, upper), and the `v_if >= 0 / v_else` branch structure mirrors rrelu's conditional logic.
