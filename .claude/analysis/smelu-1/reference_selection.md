# Reference Operation Selection for smelu

## Target Operation
- **Name**: smelu
- **Definition**: SmeLU(x, β) = x if x ≥ β; (x + β)² / (4β) if |x| ≤ β; 0 if x < -β
- **Component operations identified**:
  - Three-region piecewise conditional structure (v_if / v_elseif / v_else)
  - Comparison against a scalar parameter β (and -β)
  - Addition with scalar: (x + β)
  - Squaring: (x + β)²
  - Multiplication by scalar reciprocal: × 1/(4β)
  - Identity passthrough for x ≥ β
  - Zero output for x < -β
  - Parametrized operation (β is a runtime scalar)

## Selected References (ranked by relevance)

### 1. softshrink
- **Why selected**: The most structurally identical operation in the codebase. Softshrink is a three-region piecewise function: output is `x - λ` if `x > λ`, `x + λ` if `x < -λ`, and `0` otherwise — the same three-region skeleton as SmeLU. The middle region in softshrink writes `0` while SmeLU writes a quadratic expression, but the surrounding `v_if / v_elseif / v_endif` scaffold, the parameter loading via `Converter::as_float`, and the use of both `+lambda` and `-lambda` thresholds are directly reusable. SmeLU's implementation is essentially softshrink with the middle branch body replaced by the quadratic formula.
- **Relevance**: High — the complete three-branch control flow template (including the negated-lambda comparison) can be adapted verbatim; only the middle branch body changes.
- **Kernel file**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_softshrink.h`

### 2. xielu
- **Why selected**: xielu is the only existing operation that contains an explicit quadratic expression of the form `alpha_p * x * x + beta * x` in its positive branch. SmeLU's middle region `(x + β)² / (4β)` expands to `x² / (4β) + x/2 + β/4`, a similar polynomial in `x` computed from the raw input. The xielu kernel also demonstrates the correct pattern for passing two float parameters (`alpha_p`, `alpha_n`) into the kernel and using `Converter::as_float`, which maps directly to how β and `1/(4β)` would be passed for SmeLU. It also shows the `_xielu_mad_` helper for fused multiply-add that is useful for the quadratic computation.
- **Relevance**: High — the quadratic branch body and two-parameter passing convention are the most relevant structural elements; the three-region piecewise structure is also a multi-branch v_if/v_elseif/v_else chain.
- **Kernel file**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_xielu.h`

### 3. leaky_relu (relu family)
- **Why selected**: The relu family (relu, relu_min, relu_max, leaky_relu) are the canonical references for ReLU-like piecewise activations that pass input through for positive values and apply a transformation for non-positive values. SmeLU is explicitly a smooth approximation of ReLU — for x ≥ β it is the identity (just like ReLU's positive region), and for x < -β it outputs 0 (just like ReLU's negative region). The `relu_max` implementation shows how to apply a threshold clamp with a runtime parameter, and `relu_min` shows the min-threshold pattern. The `calculate_lrelu` function in the same file shows how a slope parameter is passed to a parametrized relu variant, which is the same pattern SmeLU uses for β.
- **Relevance**: High — the identity passthrough for x ≥ β and zero output for x < -β directly match relu's positive and negative branches; the parametrized slope pattern in leaky_relu is the template for passing β.
- **Kernel file**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_relu.h`

### 4. celu
- **Why selected**: CELU has the form `x` for `x ≥ 0` and `α * (exp(x/α) - 1)` for `x < 0`, making it a parametrized two-region piecewise activation that passes input through in one region and applies a scaled nonlinear transformation in the other — the same high-level architecture as SmeLU. Crucially, the CELU kernel demonstrates how to precompute a reciprocal outside the loop (`alpha_recip = 1/alpha`) and pass it alongside the main parameter to avoid repeated division in the hot path. SmeLU should use the same trick: pass β and `1/(4β)` so the division becomes a multiply. The `calculate_celu` signature `(uint32_t param0, uint32_t param1)` with `param1 = alpha_recip` is a direct precedent for SmeLU's `(param0 = beta, param1 = recip_4beta)` interface.
- **Relevance**: Medium-high — the precomputed-reciprocal pattern and the parametrized piecewise structure directly inform SmeLU's parameter interface and loop body structure.
- **Kernel file**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_celu.h`

### 5. hardmish
- **Why selected**: hardmish computes `x * clamp(x + 2.8, 0, 5) * 0.2`, which is a piecewise polynomial activation using arithmetic on shifted inputs — structurally similar to SmeLU's quadratic middle branch `(x + β)² / (4β)`. The hardmish kernel demonstrates the `sfpi::vec_min_max` approach for clamping, which can serve as an alternative implementation path for SmeLU's threshold conditions. More importantly, it shows a clean, concise SFPU kernel with no `v_if` branches, instead relying on hardware min/max for the conditional logic — a useful contrast with the branch-heavy approach. The `x * a * 0.2f` pattern also demonstrates precomputed scale constants (equivalent to `1/(4β)`) applied as a final multiply.
- **Relevance**: Medium — the arithmetic-on-shifted-input pattern and the use of a precomputed scale factor inform the middle-branch computation; `vec_min_max` is a possible implementation strategy for the threshold guards.
- **Kernel file**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_hardmish.h`
