# Reference Operation Selection for rrelu

## Target Operation
- **Name**: rrelu
- **Definition**: x if x >= 0, a * x if x < 0. In training: a ~ Uniform(lower, upper) per element. In eval: a = (lower + upper) / 2.
- **Parameters**: lower (default 1/8), upper (default 1/3), training (default False)
- **Component operations identified**: sign-conditional branch (x >= 0 vs x < 0), scalar multiply on negative branch, two float parameters (lower, upper), uniform random number generation (training mode), fixed scale factor (eval mode)

## Selected References (ranked by relevance)

### 1. leaky_relu
- **Why selected**: Structurally identical to rrelu in evaluation mode. Formula `max(0, x) + negative_slope * min(0, x)` maps exactly to rrelu eval formula with `a = (lower + upper) / 2` substituted for `negative_slope`. Both belong to `SFPU_OP_RELU_FAMILY_INCLUDE`, share the same sign-conditional branching pattern, and take a single float parameter passed to the SFPU kernel.
- **Relevance**: high — the eval mode SFPU kernel body can be derived directly from leaky_relu's `_calculate_leaky_relu_` implementation by substituting the precomputed `(lower + upper) / 2` value. The init/compute function naming, macro group, and parameter-passing convention are all directly reusable.

### 2. prelu_sfpu
- **Why selected**: Same formula as leaky_relu (`max(0, x) + weight * min(0, x)`) and same relu-family pattern but demonstrates the idiom where the slope parameter is a learned weight rather than a fixed constant. Shows how the relu family handles per-op float parameters. Also uses `SFPU_OP_PRELU_INCLUDE` as a standalone op, which is the same structural pattern rrelu needs (its own `SFPU_OP_RRELU_INCLUDE` macro group).
- **Relevance**: high — confirms the implementation pattern for sign-conditional multiply ops that do not share the leaky_relu kernel directly but follow the same structure. The parameter-as-float convention and dispatch through `get_op_init_and_func_parameterized` is directly applicable to rrelu's eval mode.

### 3. dropout
- **Why selected**: The only remaining SFPU kernel that demonstrates PRNG usage inside the SFPU pipeline. It initializes the hardware PRNG with `init_prng_seed(seed)` and generates random uint32 values using `TTI_SFPMOV(0, 9, LREG, 8)`. Training mode of rrelu requires sampling from Uniform(lower, upper) per element, which must use the same PRNG instruction to generate the random slope `a`. The dropout kernel also shows how probability/scale parameters are loaded into SFPU local registers via `TT_SFPLOADI`.
- **Relevance**: high — training mode implementation requires the PRNG pattern. The `_init_dropout_` calling `init_prng_seed` and the `TTI_SFPMOV(0, 9, ...)` random generation instruction in `_calculate_dropout_` are the exact primitives needed for rrelu training mode uniform sampling.

### 4. threshold
- **Why selected**: Takes two parameters (threshold value and replacement value) and performs a simple conditional branch (`v_if (in <= v_threshold) { dst = v_value; }`). This demonstrates the two-parameter SFPU kernel pattern where both parameters are passed as `uint32_t` and converted with `Converter::as_float`. The signature `_calculate_threshold_<bool, int, T>(T threshold, T value)` shows the cleanest two-parameter conditional kernel template.
- **Relevance**: medium — rrelu needs to pass two float parameters (lower, upper) to the SFPU kernel for training mode. The threshold kernel's parameter-passing approach (both params as `uint32_t`, converted to float inside the kernel) is the exact pattern to follow for passing lower/upper bounds.

### 5. hardtanh
- **Why selected**: Takes three uint32 parameters (encoded as FP16_B floats) and performs two conditional branches using `v_if`/`v_endif`. The three-parameter signature `_calculate_hardtanh_(const int iterations, uint32_t param0, uint32_t param1, uint32_t param2)` shows how to extend beyond two parameters when needed (e.g., lower, upper, plus a packed training-mode flag). Also shows the `s2vFloat16b(param)` pattern for converting packed float params inside the SFPU kernel body.
- **Relevance**: medium — the multi-parameter passing structure and `v_if` branching idiom are directly applicable. If rrelu passes lower, upper, and a training flag as three packed params, the hardtanh three-parameter pattern is the exact template to follow. Also demonstrates `SFPU_OP_HARDTANH_INCLUDE` as a standalone macro group, consistent with rrelu needing its own include group.
