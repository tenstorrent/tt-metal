# Leftover Ideas from Floating-Point / ASA References

Sources:

- `/home/ttuser/978-3-319-76526-6.pdf`: *Handbook of Floating-Point Arithmetic*, 2nd ed. Most relevant: Ch. 5, Ch. 10, Ch. 13.
- `/home/ttuser/SpringerASA.pdf`: *Application-Specific Arithmetic*. Most relevant: Ch. 16, Ch. 17, Ch. 18, Ch. 20, Ch. 22.

This is not a general literature summary. It is a backlog of ideas that still fit the project's core template: activation JSON and coefficient metadata drive range reduction, polynomial/rational/basis fitting, and generic tt-metal kernel lowering. Avoid per-activation hardcoding unless the activation JSON describes the identity or lowering explicitly.

## Highest Priority

### 1. ULP-weighted minimax fitting

Reference anchors: Handbook Ch. 10.3, especially the minimax/relative-error construction around printed pp. 389-395.

The Handbook notes that relative-error minimax can be expressed by changing the approximation basis, e.g. minimizing `|p(x)-f(x)| / |f(x)|`. We should do the same for ULP directly:

```text
minimize |p(x) - f(x)| / ulp(f(x))
```

For functions crossing zero, do not use raw relative error. Use a piecewise weight based on output binades or sampled BF16/FP32 output ULPs. This is especially relevant for GELU, swish/silu, mish, erf-like tails, logsigmoid, and functions where the current fit looks good in absolute/relative error but loses on max ULP.

Implementation hook:

- Add `fit_error_mode = ulp_weighted` in activation JSON or sweep config.
- Generate a Sollya weight expression when feasible.
- Fall back to sampled weighted fitting / local search when Sollya cannot represent the ULP staircase cleanly.
- Persist `weight_mode`, `target_ulp`, and measured `max_ulp` in CSV metadata.

Status note, 2026-06-27:

- A generic post-fit coefficient-neighbor search now lives in the fitter repo as
  `scripts/local_coeff_nudge.py`. It keeps the existing CSV shape and kernel
  schedule, tries nearby float32 coefficient encodings, and ranks candidates with
  the canonical `ttpoly.stages.s40_eval` max-ULP metric.
- A bounded GELU P6/S1 affine-even test did not reduce max ULP below 0.25, so no
  GELU coefficient churn was promoted. This supports using the tool as a gate:
  only commit changed coefficients when the headline max-ULP metric improves.

### 2. Fit the emitted coefficients, not ideal coefficients

Reference anchors: Handbook Ch. 10.3.3 and ASA Ch. 18.1.5 / Algorithm 18.1.

Both books stress that rounding a real-valued Remez polynomial after the fact is often materially worse than fitting under coefficient-format constraints. We already use constrained `fpminimax` in places, but the selection loop should rank candidates after coefficients are represented exactly as the kernel will see them.

Implementation hook:

- For every candidate, quantize coefficients to the emitted format before scoring.
- Recompute max ULP with the same bit-exact evaluator used by silicon prefiltering.
- Add a small local search over adjacent coefficient encodings for sensitive coefficients.
- Record `coefficient_format`, `coefficient_lsb`, `coefficient_rounding`, and `post_quantized_score = true` in CSV metadata.

This is a generic route to close small ULP losses without changing the kernel.

### 3. First-class centered segment residuals

Reference anchors: ASA Ch. 18.2.1.2, printed pp. 540-541.

For uniform segmentation, ASA shows that centering the reduced argument on each subinterval (`Ys in [-1, 1)`) reduces coefficient magnitude compared to unsigned local coordinates (`Y in [0, 1)`). Smaller coefficients can reduce quantization error and may simplify generated code.

Implementation hook:

```text
basis_kind = centered_segment
segment_coordinate = signed_centered
```

Kernel lowering computes a local residual around the segment center and evaluates the emitted coefficients in that basis. This must be metadata-driven; no activation-name checks.

### 4. Segment-kind lowering for mixed piecewise functions

Reference anchors: ASA Ch. 16.2 and Ch. 18.2.1.

Several activations are not "one polynomial everywhere"; they contain constant, identity, affine, clamped, saturated, or polynomial regions. We already saw this pay off for ReLU-family clamp rows. Make it a first-class representation instead of an incidental method tag.

Suggested metadata:

```text
segment_kind = constant | identity | affine | clamped_affine | polynomial | rational
segment_domain = [lo, hi]
segment_value / segment_coefficients / clamp_bounds
```

Kernel lowering should bypass Horner for constant/identity/affine segments and only run the expensive evaluator where required. This remains faithful to the fitter flow because the JSON describes the piecewise function and the coefficient CSV records the selected lowering.

### 5. Explicit error-budget metadata

Reference anchors: ASA Ch. 18.4.1, Handbook Ch. 5.1 / Ch. 10.4.2.

Current sweeps measure final ULP, which is necessary but not enough for pruning and debugging. Store the components of error separately:

```text
target_ulp
approx_abs_error
approx_ulp_error
coefficient_quant_error
range_reduction_error
eval_model_error
final_round_error
measured_max_ulp
```

This lets the generator reject candidates that cannot meet target before silicon, and it tells us whether a miss is a bad fit, bad coefficient quantization, reciprocal/range-reduction amplification, or kernel evaluation error.

## Performance-Oriented Kernel / Codegen Ideas

### 6. Evaluation-plan metadata: Horner, Estrin, x2-Horner, developed form

Reference anchors: Handbook Ch. 10.4, ASA Ch. 18.3.

Horner minimizes operations and registers, but is serial. Estrin and even/odd decomposition can reduce latency for higher-degree polynomials if SFPU scheduling and register pressure cooperate.

Implementation hook:

```text
eval_scheme = horner | estrin | x2_horner | even_odd | developed
```

Use `x2_horner` automatically when parity metadata exists:

```text
odd:  x * Horner(c1, c3, c5, ..., x*x)
even: Horner(c0, c2, c4, ..., x*x)
```

Good targets: trig residuals, erf/tanh-like odd cores, cos/cosh/I0-like even cores. Validate with disassembly; do not assume Estrin wins without measuring register pressure and instruction count.

### 7. Cost-aware degree/segment/rational selection

Reference anchors: Handbook Ch. 10.2.3 / Table 10.2, ASA Ch. 18.2.1.1.

Interval width, degree, and segment count trade memory for arithmetic. Rational approximations must also pay for denominator evaluation and reciprocal/division. The selector should rank the Pareto frontier by measured or predicted cost:

```text
score = silicon_runtime + compile_risk + LUT_bytes + reciprocal_cost + ULP_penalty
```

Do not select by max ULP alone. For rational rows, require:

```text
den_min_abs
has_pole_in_domain = false
reciprocal_iters
reciprocal_error_sensitivity
```

This is directly relevant to tanh/sigmoid/softsign/atanh-style candidates.

### 8. Table-assisted second range reduction

Reference anchors: Handbook Ch. 10.2.3-10.2.4, ASA Ch. 18.2.6 and Ch. 22.

For log/exp families, a small table can reduce the polynomial domain enough to cut degree and runtime. The Handbook describes Tang-style reductions for exp/log with table constants; ASA frames this as a memory/arithmetic tradeoff.

Implementation hook:

```text
range_reduction.family = table_assisted
range_reduction.table = r_i / correction constants
reconstruction = exp_scale | log_offset | affine
```

Likely targets: `log`, `log1p`, `log2`, `exp`, `exp2`, `softplus`, `logsigmoid`, sigmoid-family transforms. This should be benchmarked against higher-degree one-shot polynomials and current exponent-ALU paths.

### 9. Coefficient-table compression

Reference anchors: ASA Ch. 17.2.8 and Ch. 18.2.1.4.

Per-segment coefficient tables are often smooth as a function of segment index. ASA describes lossless differential / multipartite-style compression. For L1/CB LUT modes this may reduce footprint:

```text
coeff[idx] = coarse[idx >> s] + delta[idx]
```

Only use when table footprint or loads are a bottleneck. The extra add can lose if embedded constants are already cheap.

### 10. Pre-bias final rounding where valid

Reference anchors: ASA Ch. 18.4.1.

Fixed-point generators sometimes fold the final rounding bias into `C0` or table entries, so truncation implements rounding. For our flow this is only safe if metadata states clearly whether the coefficient set is already biased.

Implementation hook:

```text
final_rounding_bias = none | folded_into_c0 | folded_into_table
```

Risk: double-applying this would silently corrupt ULP. Treat as an explicit lowering mode, not a default.

## Correctness / Audit Ideas

### 11. Range-reduction exactness audits

Reference anchors: Handbook Ch. 10.2.1-10.2.2.

Cody-Waite and related reductions rely on split constants and exactness properties. Add metadata and tests for:

```text
C = C1 + C2 (+ C3)
k*C1 exact over supported input range
max_abs_reduction_error
max_relative_reduction_error
worst_reduced_argument
```

Good targets: trig, exp/log, tan, and any method with cancellation-prone reconstruction.

### 12. Alternation diagnostics for minimax quality

Reference anchors: Handbook Ch. 10.3.2.

Minimax polynomial residuals should equioscillate with alternating extrema. Add an optional diagnostic that samples each segment and reports:

```text
alternation_count
expected_alternation_count
clustered_extrema
endpoint_dominated_error
denominator_instability
```

This is not a proof, but it catches bad fits before silicon runs.

### 13. Monotonicity and endpoint validation

Reference anchors: ASA Ch. 16.2.2 / Ch. 17.2.10, Handbook Ch. 10.5.

Faithful approximations can be non-monotone by 1 ULP. For ML this may be acceptable, but monotone activations should at least record whether monotonicity holds.

Implementation hook:

```text
monotone_expected = true
monotone_verified = true | false
endpoint_exact = true | false
hardest_point_tested = true | false
```

Use exhaustive BF16 tests where possible and sampled FP32 tests otherwise.

### 14. Certified bounds for selected winners

Reference anchors: Handbook Ch. 10.3.4, Ch. 10.4.2, Ch. 13.3.

For paper-quality winners, keep measured silicon ULP as the primary result, but also emit offline proof artifacts:

- Sollya `supnorm` / `infnorm` approximation bound.
- Gappa-style or local model bound for the actual Horner/Estrin evaluation schedule.
- Denominator safety proof for rational candidates.

This can live outside the hot sweep path and run only for final manifests.

## Lower Priority / Use Carefully

### 15. CORDIC and shift-add families

Reference anchors: Handbook Ch. 10.1.2.2.

CORDIC is versatile and hardware-friendly historically, but it is a different execution model from our current SFPU polynomial/rational flow. Keep it as background unless an SFPU-specific path shows it beats polynomial/range-reduction kernels cleanly.

### 16. Payne-Hanek full-range trig

Reference anchors: Handbook Ch. 10.2.1.2.

Essential for libm over huge domains; probably not worth bringing into activation sweeps unless input ranges expand. Current trig activation ranges are narrow enough that simpler reductions should dominate.

### 17. Ziv-style runtime retries / Table Maker's Dilemma

Reference anchors: Handbook Ch. 10.5.

Useful for correctly rounded libraries; awkward for tt-metal kernels. Keep as offline validation inspiration, not a runtime strategy.

## Suggested Execution Order

Status note, 2026-06-27 overnight:

- Landed the constant-placement idea for active standalone exponent-ALU paths:
  `exp2` was already fixed, and `log/log1p/log2/log10` now call the preloaded
  log evaluator when emitted metadata enables `HW_PRELOAD`.
- Added a metadata-gated log-domain cut: when the generated CSV range proves
  the decomposed log input is strictly positive (`input + offset > 0`), the
  kernel skips negative/zero special-case branches. This is a range-reduction
  domain property, not an activation-name exception.
- Landed first-class coefficient hoisting for standalone trig residual kernels.
  The residual basis and Cody-Waite reduction are unchanged; the polynomial
  coefficients are materialized once outside the per-element replay body.
- Controlled BF16 replay now shows 49/60 activations beating TTNN on both ULP
  and runtime where TTNN refs exist. New flips from this pass: log, log1p, log2,
  log10, sin, and cos. Remaining TTNN-ref misses are tiny exp/sigmoid runtime
  margins, GELU's explicit 0.25-ULP waiver, and multigammaln's native-op floor.
- Still-open high-value items below remain valid: post-quantized scoring as a
  normal pipeline stage, true ULP-weighted fitting, centered segment residuals,
  richer segment-kind lowering, denominator-safety metadata, and eval-scheme
  selection beyond the current hand-shaped fast paths.

1. Add post-quantized scoring and coefficient-format metadata.
2. Add ULP-weighted fitting mode.
3. Add centered segment residual metadata and kernel lowering.
4. Generalize `segment_kind` lowering and audit hard/clamp/threshold activations.
5. Add denominator-safety gates for rational candidates.
6. Add `eval_scheme` metadata and benchmark `x2_horner` / Estrin with disassembly.
7. Prototype table-assisted range reduction for one log/exp-family loser.
8. Add monotonicity, alternation, and range-reduction audit reports.
