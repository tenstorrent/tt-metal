# Reference Operation Selection for rrelu

## Target Operation
- **Name**: rrelu
- **Definition**: `RReLU(x) = x if x >= 0; ((lower + upper) / 2) * x if x < 0` (eval mode). Training mode: `a ~ U(lower, upper)` per negative element, then `a * x`. Parameters: `lower=0.125`, `upper=1/3`.
- **Component operations identified**:
  - Conditional branch on sign of `x` (positive pass-through, negative scaling)
  - Scalar multiply for the negative branch (`slope * x` where `slope = (lower+upper)/2` in eval mode)
  - Two float parameters (`lower`, `upper`) passed as bit-cast `uint32_t`
  - Training mode: uniform PRNG to sample `a ~ U(lower, upper)`, then element-wise multiply `a * x`
  - PRNG seed initialization pattern

## Selected References (ranked by relevance)

### 1. leaky_relu
- **Why selected**: RReLU in eval mode is structurally identical to leaky_relu: pass through positive values unchanged, and for negative values multiply by a fixed scalar slope. The actual SFPU kernel `_calculate_lrelu_` in `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_relu.h` uses `TTI_SFPSETCC` to branch on sign, then `TTI_SFPMUL` to apply the slope — exactly the pattern needed for rrelu eval mode. The `unary_op_utils.cpp` registration with `leaky_relu_tile_init()` and `leaky_relu_tile(idst, slope_as_uint)` is the direct template for the new op's registration.
- **Relevance**: high — the eval-mode kernel body can be adapted almost verbatim; the slope just needs to be precomputed as `(lower+upper)/2` on the host side before being passed as a uint32_t bit-cast.

### 2. prelu
- **Why selected**: PReLU (`a * x` for `x < 0`, `x` otherwise) with a scalar parameter is the same mathematical structure as RReLU eval mode, but implemented in the sfpi high-level DSL (`v_if / v_endif`, `Converter::as_float`, `vFloat`) rather than raw TT instructions. The kernel in `ckernel_sfpu_prelu.h` is a clean, readable template showing how to load a float parameter, do the conditional branch in sfpi, and apply the multiply. The `unary_op_utils.cpp` entry for `PRELU_SFPU` shows the two-step `prelu_tile_init()` + `prelu_tile(idst, param)` registration pattern.
- **Relevance**: high — provides the sfpi-style alternative implementation pattern for the same conditional-multiply operation; useful if the implementor prefers the higher-level sfpi API over raw TT instructions.

### 3. dropout
- **Why selected**: RReLU training mode requires per-element pseudorandom number generation (one random slope `a` per negative element). Dropout uses exactly this PRNG infrastructure: `_init_dropout_` calls `init_prng_seed(seed)`, and `_calculate_dropout_` uses `TTI_SFPMOV(0, 9, LREG3, 8)` to generate a pseudorandom uint32, then `TTI_SFPSETSGN` to clear the sign bit before comparison. This is the only SFPU kernel that demonstrates PRNG-driven conditional scalar application, which is the core of rrelu training mode. The `dropout_init` wrapper in `ckernel_sfpu_dropout.h` also shows how to expose a seed-based init function.
- **Relevance**: high — the PRNG generation sequence (`SFPMOV` with `instr_mod1=8` + sign clear) is required for training mode; the conditional apply pattern (if random < threshold, apply scale) is analogous to the rrelu training flow.

### 4. rand
- **Why selected**: RReLU training mode must generate `a ~ U(lower, upper)`, not just a raw uint. The `rand` kernel in `ckernel_sfpu_rand.h` shows exactly how to convert the raw PRNG output to a bounded uniform float: set exponent to 127 to get `[1,2)`, subtract 1 to get `[0,1)`, then scale/shift via `SFPMAD(lreg0, scale, from, lreg0)` to reach `[from, from+scale)`. For rrelu, `from=lower` and `scale=upper-lower`. The kernel also demonstrates loading two float parameters into LREG1 and LREG2 via `TT_SFPLOADI` — the same pattern needed to have `lower` and `upper` available inside the loop.
- **Relevance**: high — the uniform-to-bounded-range conversion idiom is directly reusable; combining this with dropout's conditional application gives the complete training mode implementation.

### 5. selu
- **Why selected**: SELU takes two float parameters (`scale` and `alpha`) both passed as bit-cast `uint32_t` and loaded via `Converter::as_float` before the loop, and applies them in a conditional branch (`if x >= 0: scale*x; else: alpha*exp(x)*scale`). RReLU also takes two parameters (`lower`, `upper`). The `calculate_selu` signature `(uint scale, uint alpha)` and the `unary_op_utils.cpp` registration entry for `SELU` — `"selu_tile_init();"` + `fmt::format("selu_tile({}, {:#x}u, {:#x}u);", idst, bit_cast(param0), bit_cast(param1))` — is the direct template for a two-parameter rrelu registration.
- **Relevance**: medium — the structural pattern of a two-parameter conditional activation with both params loaded outside the inner loop is directly applicable; the exp-based negative branch is not needed but the framework for dual-param handling is.
