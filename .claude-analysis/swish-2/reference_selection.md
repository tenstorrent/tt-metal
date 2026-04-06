# Reference Operation Selection for swish

## Target Operation
- **Name**: swish
- **Definition**: `x / (1 + exp(-x))`
- **Equivalent forms**: `x * sigmoid(x)` (since `sigmoid(x) = 1 / (1 + exp(-x))`)
- **Component operations identified**:
  - `exp(-x)` — exponential of negated input
  - `1 + exp(-x)` — scalar addition
  - `1 / (1 + exp(-x))` — reciprocal (sigmoid)
  - `x * sigmoid(x)` — element-wise multiply of input with its sigmoid

## Analysis Notes

Swish is mathematically identical to `silu(x) = x * sigmoid(x)`, already present in
`UnaryOpType::SILU`. The new swish operation needs a **custom SFPI kernel** implementing
`sigmoid(x)` via SFPI arithmetic (not the hardware LLK sigmoid primitive), following the
same pattern used by all custom ckernel_sfpu_*.h operations in this worktree.

The key challenge is: implementing `exp(-x)` from scratch using SFPI integer bit
manipulation (as there is no direct SFPI `exp()` call in the custom kernel layer).

## Selected References (ranked by relevance)

### 1. hardswish
- **Path**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_hardswish.h`
- **Math**: `x * clamp(x/6 + 0.5, 0, 1)` = `x * hardsigmoid(x)`
- **Why selected**: This is the **exact structural template** for swish. Both hardswish and
  swish compute `x * f(x)`, where `f(x)` is some form of sigmoid. The entire loop body,
  `calculate_hardswish<>()` function signature, `sfpi::dst_reg[0] = x * hsigmoid` multiply
  pattern, and the `v_if`/`v_endif` clamping structure can be adapted. The implementor
  replaces `hardsigmoid(x)` with true `sigmoid(x)`.
- **Relevance**: **high** — outer loop, init pattern, x * activation(x) multiply are
  all directly reused. Only the inner activation computation changes.

### 2. hardsigmoid
- **Path**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_hardsigmoid.h`
- **Math**: `max(0, min(1, x/6 + 0.5))`
- **Why selected**: This is the "activation sub-function" embedded inside hardswish. Its
  `calculate_hardsigmoid<>()` is what gets called before the multiply in hardswish — the
  direct precursor to calling `sigmoid(x)` in swish. It shows how to compute the clamped
  output that gets multiplied by `x`, including the `v_if(result < 0.0f)` / `v_if(result > 1.0f)`
  range-bounding pattern that mirrors sigmoid's output range.
- **Relevance**: **high** — models the inner sigmoid computation and its interface into
  the surrounding `x * f(x)` compute structure.

### 3. rpow
- **Path**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_rpow.h`
- **Math**: `base^x = 2^(x * log2(base))`
- **Why selected**: Contains a complete, tested SFPI implementation of `2^z` using the
  **exp_21f algorithm** (Moroz et al. 2022). This algorithm is directly reusable for
  computing `exp(-x) = 2^(-x * log2(e))` with `log2(e) ≈ 1.4426950408`. The implementation
  uses `sfpi::addexp()`, `sfpi::exexp()`, `sfpi::exman9()`, `sfpi::int32_to_float()`,
  `_float_to_int32_positive_()`, and Horner-form polynomial evaluation — all required
  for implementing sigmoid's exponential component without hardware primitives.
- **Relevance**: **high** — the `2^z` kernel body is the core implementation needed for
  `exp(-x)` in sigmoid. The precomputed `log2(e)` constant substitution is the only
  adaptation required.

### 4. softsign
- **Path**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_softsign.h`
- **Math**: `x / (1 + |x|)`
- **Why selected**: Softsign is the **closest structural analog** to swish's alternate form
  `x / (1 + exp(-x))`. Both divide `x` by `(1 + something)`. The numerator pattern is
  identical, and both produce a smooth squashing activation bounded in a finite range.
  Although the current implementation is a stub (removed because it depends on the `recip`
  primitive), the function signature `calculate_softsign<APPROX, ITERATIONS>()` and
  `softsign_init<APPROX>()` show the exact pattern swish should follow. Also informs that
  `recip` (1/x) is needed and should be implemented via the exp_21f approach or algebraic
  manipulation to avoid calling the hardware recip LLK.
- **Relevance**: **medium** — mathematical structural match reveals the implementation
  shape; stub indicates known dependency on `recip`, guiding the swish implementor to
  fold recip into the sigmoid computation (compute `1 / (1 + exp(-x))` in one step).

### 5. cbrt
- **Path**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_cbrt.h`
- **Math**: `x^(1/3)` via magic-constant initial guess + Newton-Raphson refinement
- **Why selected**: Shows the **polynomial approximation + Newton-Raphson refinement**
  pattern using `sfpi::vConstFloatPrgm0/1/2` programmable constant registers. The
  `cube_root_init<>()` loads constants into `vConstFloatPrgm0/1/2`, which are then used
  in `calculate_cube_root<>()` for Horner-form polynomial evaluation. Swish may benefit
  from a similar pattern if using a polynomial approximation for sigmoid instead of the
  exp_21f approach — e.g., a 3rd/4th order polynomial approximation of sigmoid that can
  be evaluated using `vConstFloatPrgm` registers loaded in `swish_init()`.
- **Relevance**: **medium** — the init() → calculate() constant-register pattern and
  polynomial evaluation template are valuable if the implementor chooses a polynomial
  sigmoid approximation; also models the fp32 vs bfloat16 destination accumulation
  branching (`if constexpr (is_fp32_dest_acc_en)`).
