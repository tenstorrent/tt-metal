# Reference Operation Selection for rrelu

## Target Operation
- **Name**: rrelu
- **Definition**: f(x) = x if x >= 0, a * x if x < 0; eval mode: a = (lower + upper) / 2, train mode: a ~ Uniform(lower, upper)
- **Component operations identified**: sign-based conditional branch (x < 0 check), scalar multiply by slope parameter `a`, uniform random number generation (train mode), fixed parameter evaluation (eval mode), two-bound parameter passing (lower, upper)

## Selected References (ranked by relevance)

### 1. swish
- **Why selected**: `ckernel_sfpu_swish.h` contains exactly the structural pattern rrelu needs: `v_if(x < 0.0f)` sign check followed by multiplication in the negative branch. The swish implementation loads x, computes a conditional based on the sign of x, and applies different computation for negative vs non-negative values. Rrelu eval mode is structurally `v_if(x < 0) { result = a * x; } v_endif;` — a direct simplification of this pattern.
- **Relevance**: high — the sign-conditional multiplication loop in swish is the direct template for rrelu's core compute loop; the ITERATIONS template parameter, `dst_reg` iteration pattern, and `v_if`/`v_endif` structure will be reused verbatim.

### 2. dropout
- **Why selected**: `ckernel_sfpu_dropout.h` is the only existing operation that uses the hardware PRNG via `TTI_SFPMOV(0, 9, p_sfpu::LREG3, 8)` and `init_prng_seed(seed)`. Rrelu train mode requires sampling `a ~ Uniform(lower, upper)` independently per element, which requires generating a random number per element — exactly the pattern dropout uses. The PRNG seed initialization via `_init_dropout_` → `init_prng_seed` is the infrastructure needed for rrelu train mode.
- **Relevance**: high — the `TTI_SFPMOV` PRNG instruction and `init_prng_seed` call are essential for rrelu train mode; the probability comparison pattern `TTI_SFPIADD(0, p_sfpu::LREG2, p_sfpu::LREG3, 10)` shows how to use the raw PRNG value in comparisons/scaling to implement Uniform(lower, upper) sampling.

### 3. hardtanh
- **Why selected**: `ckernel_sfpu_hardtanh.h` is the canonical example of a multi-parameter SFPU operation that takes runtime float parameters via `s2vFloat16b(param)` conversion. It shows how to convert packed uint32 parameters to vFloat, load them before the compute loop, and use them inside `v_if`/`v_endif` blocks. Rrelu also needs two float parameters (lower and upper) that must be similarly loaded and used inside the compute loop.
- **Relevance**: high — the parameter loading pattern `sfpi::vFloat p0 = sfpi::s2vFloat16b(param0);` before the loop, and usage inside conditionals, is directly applicable to rrelu's lower/upper parameter handling in both eval and train mode.

### 4. threshold
- **Why selected**: `ckernel_sfpu_threshold.h` provides the simplest and cleanest conditional replacement pattern in the codebase: a single `v_if (in <= v_threshold) { dst_reg[0] = v_value; } v_endif;` loop. This is the minimal template for rrelu's negative-region behavior. It also shows the template parameter pattern `<bool APPROXIMATION_MODE, int ITERATIONS, typename T>` and how float/uint32 parameters are converted via `Converter::as_float()`.
- **Relevance**: medium — the clean single-conditional structure with typed parameters shows the minimal form of parameterized conditional operations; the pattern for a "check sign, replace/scale" loop closely mirrors rrelu's core logic.

### 5. clamp_tss
- **Why selected**: `ckernel_sfpu_clamp.h` is the primary example of an operation that takes two float bounds (min and max) as parameters loaded via `s2vFloat16a`, directly mirroring rrelu's lower and upper bounds. It shows how to use two separate float parameters inside conditional branches (`v_if (val < min)`, `v_elseif (val >= max)`) and how to store parameter values as `sfpi::vFloat` variables before the compute loop.
- **Relevance**: medium — the two-float-parameter loading pattern and multi-branch conditional structure provides a direct template for rrelu's eval mode parameter calculation `a = (lower + upper) / 2`, and train mode bound-clamping of the random sample to [lower, upper].
