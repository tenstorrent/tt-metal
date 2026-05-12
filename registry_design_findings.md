# Per-Axis Registry Design — Findings

## Proposed model

Two registries per op, both as **per-axis lists** (not cell tuples):

```python
# TARGET registry — authored upfront, ambition
TARGET = {
    "layout":    [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
    "dtype":     [ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b],
    "alignment": ["tile_aligned", "non_tile_aligned"],
    # ...op-specific axes
}

# CURRENT registry — lives in the op next to validate()
CURRENT = {
    "layout":    [ttnn.TILE_LAYOUT],
    "dtype":     [ttnn.float32],
    "alignment": ["tile_aligned"],
}
```

All axes — including shape-derived ones like `alignment` — are per-axis lists.
Shape itself is not in the registry; instead, op-defined `tag_*` functions
project each shape onto one or more categorical axes (see §"Shapes" below),
and those axes look just like any other.

A combination is supported iff (a) every axis value (including shape-derived
tags) appears in its CURRENT list and (b) the combination isn't in EXCEPTIONS.

## Supported set = cartesian(CURRENT) minus EXCEPTIONS

Per-axis lists define the *intended* rectangle of supported cells.
A small **EXCEPTIONS** list subtracts specific cells the op author has decided
not to combine in practice — even though both axes individually support those
values.

```python
CURRENT = {
    "layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
    "dtype":  [ttnn.float32, ttnn.bfloat16],
    "shape":  lambda s: s[-1] % 32 == 0 and s[-2] % 32 == 0,
    "dim":    [-1, -2],
}

EXCEPTIONS = [
    # "Theoretically valid but not worth combining."
    {"layout": ttnn.ROW_MAJOR_LAYOUT, "dim": -2},
    # ...
]

def is_supported(shape, **kwargs):
    for k, v in CURRENT.items():
        if k == "shape":
            if not v(shape):
                return False
        elif kwargs[k] not in v:
            return False
    return not any(_matches(e, shape=shape, **kwargs) for e in EXCEPTIONS)
```

Properties:

- **Per-axis lists carry most of the weight.** Adding a new dtype is one list
  edit; the rectangle expands across every other axis automatically.
- **EXCEPTIONS handle the genuinely L-shaped cases** — "row_major + dim=-2 is
  too painful, don't support it" — without polluting the per-axis lists with
  ambiguity.
- **The EXCEPTIONS list stays small** if axes are mostly orthogonal in practice.
  If it grows large, that's a signal that two axes aren't really orthogonal
  and should be re-thought.
- **Same function powers `validate()` and the xfail decision.** Single source
  of truth; no algorithm has to derive cells from observations.

## Iteration mechanism

Standard `pytest.mark.xfail(strict=True)`:

1. Test file auto-parameterizes over cartesian product of **TARGET** lists.
2. For each generated case, mark xfail if it's not in the cartesian product
   of **CURRENT** lists.
3. Implementer adds a feature → previously-xfail cases pass → XPASS → CI
   fails → implementer updates CURRENT.
4. With CURRENT updated, those cases are no longer xfail. Green.

`pytest -rx` is the mechanical to-do list (all unimplemented cells, derived
from `TARGET - CURRENT`).

## The four-tier sieve, in per-axis form

| Tier | How expressed |
|---|---|
| **INVALID** (math-impossible) | Not in TARGET (lists are pre-filtered) |
| **WON'T SUPPORT** (could work, choose not to) | In TARGET, never going into CURRENT; tagged separately, paired with `NotImplementedError` guard in op code, verified by `xfail(raises=NotImplementedError, strict=True)` |
| **NOT YET** (refinement queue) | In TARGET, not yet in CURRENT; `xfail(strict=True)` |
| **SUPPORTED** | In CURRENT; normal test |

WON'T_SUPPORT needs its own per-axis (or per-cell) list because per-axis-only
representation can't express "everything works except this corner." Pragmatic
fix: a small set of explicit `WONT_SUPPORT` cells alongside the per-axis
TARGET/CURRENT lists. Expected to be small.

## Shapes: tag, then treat the tag as a normal axis

Shapes are effectively infinite. Listing them is ridiculous. Categorizing
by generic metrics (`prod(shape) < N`) is too coarse — it hides
implementation-specific failure modes.

**Each shape passes through one or more `tag_*` functions that produce
categorical labels.** Those labels are then normal CURRENT axes — same shape
as `dtype`, `layout`, etc.

```python
# Op-author writes the tag functions — they encode what the op
# implementation actually cares about.
def tag_alignment(shape):
    if shape[-1] % 32 == 0 and shape[-2] % 32 == 0:
        return "tile_aligned"
    return "non_tile_aligned"

SHAPE_TAGGERS = {
    "alignment": tag_alignment,
    # Other ops can add more facets if their implementation cares about
    # them, e.g. `tag_aspect(shape) -> "wide" | "tall" | "square"`.
}
```

Then the rest of the registry looks uniform — *all* axes (finite enums and
shape-derived tags alike) are per-axis lists:

```python
TARGET = {
    "layout":    [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
    "dtype":     [ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b],
    "alignment": ["tile_aligned", "non_tile_aligned"],
}

CURRENT = {
    "layout":    [ttnn.TILE_LAYOUT],
    "dtype":     [ttnn.float32],
    "alignment": ["tile_aligned"],
}
```

**For each shape in the test universe** the test harness runs every
`tag_*` function to produce a dict of `{axis: tag_value}`, combines with
the cartesian of finite axes, and checks every value (including shape tags)
against CURRENT.

**The implementer writes the tag functions** as part of implementing the op
— they're the ones who know which shape facets the kernel actually cares
about. The verifier's role is narrower: confirm the tags + CURRENT lists
correctly classify the observed pass/fail, fix the lists if they
under/over-claim, and only adjust tag functions when the implementer
genuinely missed a facet.

**TARGET ships an exhaustive representative shape set** (tile-aligned and
not, "compact" and "wide"). Shapes whose tags fall outside CURRENT are
xfail, not omitted.

**Why this beats a single shape predicate:**
- Each facet of shape is a named axis, not buried in a boolean.
- Tag values are categorical strings, comparable across runs and visible
  in test ids and dashboards.
- Adding a new facet (some op cares about width-divisibility, say) is one
  new tag function and one new CURRENT axis — same mechanism as adding
  any other axis.
- Validate() reads from CURRENT uniformly, doesn't need a special
  "shape predicate" branch.

**Tag functions are op-local — no shared helpers.** Even when two ops both
have an `alignment` axis, the underlying tagger is different:

- Channel-last conv2d only cares about the C-dimension being tile-aligned.
  Other dims can be anything.
- Channel-last groupnorm cares about both total C *and* C / num_groups
  being tile-aligned.
- A pure elementwise op might just care about the last two dims.

A shared `tag_alignment(shape)` would need to be a thin dispatch that
delegates to per-op logic anyway. Better to skip the indirection: every
op writes its own taggers, and axis names happen to overlap when the
concept is similar but the rule isn't.

This also keeps the registry honest. If two ops share an axis name but
mean different things, a reader can't accidentally assume cross-op
compatibility — they have to read each op's tag function to know what
`alignment="tile_aligned"` means for that op.

## What disappears from the current system

- `compute_passing_signatures` — registry is the truth, not pass rates.
- `_project_signature` — no marker projection; tests get their cell identity
  from parametrize ids.
- `build_subset_filter` — subset = non-xfail tests; no substitution algebra.
- Complete-signature gate — no joint signatures to gate.
- Marginal capability view — no inferred per-marker pass rates.
- Smoke filter `None`-return edge case — smoke becomes a plain `@pytest.mark.smoke`.
- Phase 0 fallback in `build_subset_filter` — registry handles bootstrap
  trivially (empty CURRENT lists = all xfail).
- `**Subset**: --dim KEY=VAL` parsing in op_requirements.md — refinements
  just update CURRENT.
- The "stale marker / typo" filtering in `_project_signature` — there are
  no markers to be stale; you edit Python lists.

Probably 70–80% of `eval/golden_capabilities.py`.

## What stays

- `eval/eval_test_runner.sh` (simpler — no `--dim` substitution machinery; just
  pass-through of pytest args).
- `eval/dim_vocab.py` becomes much smaller — could be just a list of axis names
  shared across ops, or deleted entirely if axes are op-local.
- Regression / numerical stability tests (`0.01 * randn` etc) live alongside
  the parameterized cross-product as separate, non-parameterized tests.
  Tagged `@pytest.mark.regression` (or `@pytest.mark.numerics`), run in the
  full suite but not in the per-refinement xfail check.

## CURRENT lives in the op's `validate()`

The Python op gets a `validate()` function — the equivalent of the C++ TTNN
op's `validate()`. It actively rejects unsupported inputs before any kernel
runs. CURRENT and EXCEPTIONS are defined right next to it (same file), so
"what does this op support?" and "what does this op reject?" are one piece of
code.

```python
# ttnn/ttnn/operations/<op>/<op>.py

def tag_alignment(shape):
    if shape[-1] % 32 == 0 and shape[-2] % 32 == 0:
        return "tile_aligned"
    return "non_tile_aligned"

SHAPE_TAGGERS = {"alignment": tag_alignment}

CURRENT = {
    "layout":    [ttnn.TILE_LAYOUT],
    "dtype":     [ttnn.float32],
    "alignment": ["tile_aligned"],
    # ...op-specific axes
}
EXCEPTIONS = []

def _axis_values(input_tensor, **kwargs):
    values = {"layout": input_tensor.layout, "dtype": input_tensor.dtype}
    for axis, tagger in SHAPE_TAGGERS.items():
        values[axis] = tagger(input_tensor.shape)
    values.update(kwargs)
    return values

def validate(input_tensor, **kwargs):
    values = _axis_values(input_tensor, **kwargs)
    for axis, allowed in CURRENT.items():
        if values[axis] not in allowed:
            raise NotImplementedError(f"{axis}={values[axis]} not supported")
    for exc in EXCEPTIONS:
        if _matches(exc, values):
            raise NotImplementedError(f"unsupported combination: {exc}")

def <op>(input_tensor, **kwargs):
    validate(input_tensor, **kwargs)
    # ...program descriptor + actual op invocation
```

Tests don't import a separate registry — they import `validate` (or `CURRENT`
+ `EXCEPTIONS` directly) from the op module. The xfail decision uses the
same source of truth that runtime validation uses. There is no possibility of
drift between "what tests expect" and "what the op accepts."

## Pipeline-level integration

- **Planner**: writes TARGET registry as part of op_design.md.
- **Implementer (Phase 0)**: builds baseline. Writes:
  - `SHAPE_TAGGERS` (the tag functions — the implementer knows which shape
    facets the kernel cares about).
  - Initial `CURRENT` + `EXCEPTIONS`.
  - `validate()` inside the op file.
- **Verifier (Phase 0 only)**: runs the test suite. Confirms CURRENT matches
  reality. Specifically:
  - No unexpected XPASSes (CURRENT doesn't under-claim).
  - No unexpected FAILs in cells CURRENT claims (CURRENT doesn't over-claim).
  - Fixes the lists if either condition fails.
  - Adjusts tag functions only when the implementer genuinely missed a facet
    (e.g. the op also fails on some not-yet-tagged property — verifier flags
    it, may add a new tag axis).
  - Writes human-readable capabilities.md summarizing the registry.
- **Implementer (Refinement N)**: implements the next feature. Updates CURRENT
  (and validate(), and tag functions if the refinement introduces a new
  axis facet). No verifier between refinements is needed — XPASS-strict is
  the mechanical signal that CURRENT lags behind reality.

## Open questions to resolve

1. **How to handle "soft" partial support?** E.g. bf16 works on most cells
   but 1 cell out of 30 has higher numerical error than threshold. Today's
   0.8 pass-rate threshold tolerates this. xfail-strict doesn't. Options:
   - Add the corner to EXCEPTIONS (with a `NotImplementedError` guard via
     validate()) — clean and explicit.
   - Add a per-cell tolerance override (more flexible, but a new mechanism).
   - Accept that you don't have "soft" support — either the rectangle holds
     or it doesn't.

2. **Op-specific axes** (like layer_norm's `affine`). These should live in
   the per-op TARGET/CURRENT, not in a universal vocab. The cross-product
   generator handles them naturally — it's just another key in the dict.

3. **Multi-value axis additions in one refinement.** "Add bf16 and bf8b at
   the same time" — clean: add both to CURRENT["dtype"] in one edit. XPASS
   on all newly-supported cells, no edits required to other axes.

4. **Test discovery / runner.** Replace `eval_test_runner.sh`'s filter
   plumbing with a single pytest invocation. The test runner doesn't need
   to know about subsets — `pytest -rx` reports unimplemented cells naturally.

5. **Tag-function evolution.** Most refinements just add a value to an
   existing shape-axis list (e.g. `CURRENT["alignment"] += ["non_tile_aligned"]`).
   When a refinement uncovers a *new* shape facet the implementer didn't
   anticipate (say, the op fails specifically on shapes where W is not a
   multiple of the tile size in some new way), the implementer adds a new
   `tag_*` function + a new CURRENT axis. The XPASS-strict mechanism handles
   the simple case automatically; the new-facet case requires implementer
   judgment, which is appropriate.

## Migration strategy (if pursuing)

Smallest credible slice:

1. Start from `main` of `tt_ops_code_gen` (clean slate, no current eval logic).
2. Build the registry primitives: `TARGET`, `CURRENT`, cartesian product
   generator, xfail decision function (~50 lines total).
3. Pick one op (layer_norm_rm is a good candidate — has the partial-marker
   problem starkly today).
4. Write the test file as a single auto-parameterized cross-product driven
   by the registries.
5. Run through a Phase 0 + 1 refinement loop manually to validate the
   workflow before agent-automating.
6. Decide on how the verifier writes the shape rule (literal list of passing
   shapes, predicate like `prod(shape) < N`, or both).
7. Port the pipeline driver (`run_eval.py`, `run_refinements.py`) to the new
   model — much smaller surface to maintain.

Net change vs. current: drop ~70% of `golden_capabilities.py`, simplify
`eval_test_runner.sh` significantly, replace three test files per op
(cross_product / shapes / modes) with one cross-product test file plus a
small regression-tests file, and have the implementer agent maintain a
registry instead of the verifier inferring capabilities post-hoc.
