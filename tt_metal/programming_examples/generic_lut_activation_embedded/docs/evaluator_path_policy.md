# Generic LUT Evaluator Path Policy

Dispatch must come from explicit CSV/schema fields and algebraic row structure.
Do not dispatch from activation name, filename tokens, or activation-specific
allowlists.

## Selection Order

1. Validate the record first. Required sources are data rows plus `METADATA`:
   approximation type, coefficient columns, `lo`/`hi`, range-reduction fields,
   basis fields, asymptotic fields, and rational parity/reciprocal metadata when
   present. Missing method constants mean unsupported config, not permission to
   guess from activation name.

2. Use whole-function collapses only when CSV algebra proves the full fit:
   identity, affine, or clamped affine. These bypass segment selection entirely.

3. Use basis transforms when `basis_kind` metadata is present. The basis wrapper
   owns input remapping and postprocessing; do not collapse its inner polynomial
   as if it were over raw `x`.

4. Use range-reduced or standalone methods when metadata declares them:
   reduced-poly/rational for `exp`, `trig`, `tan`, `log`, `cbrt`;
   exponent-ALU for `exp2`/`log2`/`pow`; Newton-root for declared root methods.
   Required constants must come from metadata.

5. Use rational cascade when rational columns or schema declare rational
   approximation and no higher-priority wrapper owns the evaluation.

6. Use polynomial cascade as the default. Future `segment_kind` lowering belongs
   inside this family, not as activation-name branches.

## Current Collapses

`affine_collapse` requires a non-rational, non-basis, non-range-reduced,
single-segment fit with effective degree <= 1. It emits `y = c1*x + c0`, or a
pure copy for identity.

`clamped_affine_collapse` requires a non-rational, non-basis,
non-range-reduced fit where every segment is constant or affine and the whole
piecewise function matches:

```text
y = min(max(c0 + c1*x, y_min), y_max)
```

Either clamp bound may be absent. Bounds are derived from constant rows and
segment order, then validated against segment endpoints and midpoints.

## Next Step

Add first-class `segment_kind` metadata for partial lowering:

```text
constant -> constant select/fill
affine -> c1*x + c0
clamped_affine -> min/max composition
polynomial -> Horner
rational -> P/Q
```

This should improve mixed piecewise functions without pretending the whole
function is clamped affine. If a segment kind is unsupported, fail loudly or
fall back only when a complete polynomial/rational representation is present.

## Hardcoding Ban

- No activation-name switches for evaluator choice.
- No filename parsing for degree, rational/poly, segment count, or method.
- No implicit range-reduction, Newton, exponent, basis, parity, or asymptotic
  constants when metadata is missing.
- No hidden activation-specific allowlists in the emitter or kernel dispatcher.
