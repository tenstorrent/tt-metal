# Registry Migration Plan

Starting point: `main` of `tt_ops_code_gen` (already much simpler than
`mstaletovic/RefinementsTesting`). Goal: build the per-op
TARGET/CURRENT/SHAPE_TAGGERS/EXCEPTIONS/validate() model from
`registry_design_findings.md` on top of main, with **significantly less
code** than what's on the branch.

## What main already gives us (don't re-build)

These work as-is or near-as-is:

- `eval/db.py`, `eval/ingest.py`, `eval/update_run.py` — DB layer.
- `eval/dashboard.py`, `eval/detail_server.py` — presentation.
- `eval/list_runs.py`, `eval/annotate.py`, `eval/rescore.py`,
  `eval/diagnose_run.py`, `eval/quick_ingest.py` — CLIs.
- `eval/score.py`, `eval/classify_failures.py`, `eval/validate_contract.py`.
- `eval/hang_plugin.py` — pytest hang detection.
- `eval/superset_views.sql`.
- `eval/run_eval.py`, `eval/pipeline.py` in their main shape — orchestration
  shell stays. Inner phases change semantics, not interface.

`eval_test_runner.sh` on main is also already simple. Likely no changes
needed (we just don't add `--dim`/`-m` substitution plumbing back in).

## What's brand new (build these)

### Naming

Two registry-shaped things, distinct purposes:

- **Helper math (shared, in eval/)**: `eval/feature_matrix.py` — pure
  functions over dicts and lists. The math that builds cell combinations,
  decides xfail vs pass, matches exclusion entries. ~50–80 lines.
- **Per-op spec (in eval/, per op)**: `eval/golden_tests/<op>/feature_spec.py`
  — the TARGET universe + the shape list + tag functions. Pure data + tag
  funcs. ~30–80 lines per op.
- **Per-op support (with the op)**: `SUPPORTED` + `EXCLUSIONS` + `validate()`
  living alongside the op (where exactly — see Open decision #1, still open).

Renamed registry vocabulary (was CURRENT/EXCEPTIONS):
- `SUPPORTED` — the per-axis dict of what the op accepts right now.
- `EXCLUSIONS` — list of cell-level entries inside SUPPORTED that the op
  refuses anyway.

### 1. `eval/feature_matrix.py` — the helper math

One small module, ~50–80 lines. Provides:

```python
def cartesian(target, shape_taggers, inputs):
    """Yield {axis: value} dicts for one set of input shapes × all finite-axis
    combos in TARGET. `inputs` is a tuple of shapes (single-shape ops pass a
    1-tuple; matmul-style ops pass (lhs_shape, rhs_shape, ...))."""

def is_supported(values, supported, exclusions):
    """Check a {axis: value} dict against SUPPORTED per-axis lists + EXCLUSIONS."""

def xfail_reason(values, supported, exclusions):
    """None if supported, else a string for pytest.mark.xfail(reason=...).
    Always paired with strict=True, raises=NotImplementedError."""
```

No I/O, no markers, no JSON. Easy to unit-test. This is the only "shared"
eval-side code touching the registry model.

### 2. `eval/golden_tests/<op>/feature_spec.py` — per-op test universe

One file per op. Pure data + tag functions:

```python
# Op-author writes the tag functions — encode what the kernel cares about.
def tag_alignment(inputs):
    """inputs is a 1-tuple here (single-input op)."""
    shape = inputs[0]
    if shape[-1] % 32 == 0 and shape[-2] % 32 == 0:
        return "tile_aligned"
    return "non_tile_aligned"

SHAPE_TAGGERS = {"alignment": tag_alignment}

TARGET = {
    "layout":    [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
    "dtype":     [ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b],
    "alignment": ["tile_aligned", "non_tile_aligned"],
    # ...op-specific finite axes
}

# For multi-input ops (matmul, etc.), each entry is a tuple of shapes.
INPUTS = [
    ((1,1,32,32),),
    ((1,1,64,128),),
    ...,
]
```

Multi-input ops handle by making `INPUTS` a list of tuples-of-shapes, and
having taggers accept that tuple. e.g.:

```python
def tag_inner_dim_match(inputs):
    lhs, rhs = inputs
    return "match" if lhs[-1] == rhs[-2] else "mismatch"
```

### 3. `SUPPORTED` + `EXCLUSIONS` + `validate()` lives with the op

Living next to the op. Tests import these directly — single source of truth
for runtime gate AND xfail decisions.

```python
SUPPORTED = {
    "layout":    [ttnn.TILE_LAYOUT],
    "dtype":     [ttnn.float32],
    "alignment": ["tile_aligned"],
}
EXCLUSIONS = []

def validate(*inputs, **kwargs):
    # ...uses SUPPORTED, EXCLUSIONS, tag funcs to gate inputs
    # raises NotImplementedError for anything outside SUPPORTED or in EXCLUSIONS
```

**Open decision #1 (still open, to discuss):** where exactly this lives —
inline in the op file `ttnn/ttnn/operations/<op>/<op>.py`, or in a sibling
file `ttnn/ttnn/operations/<op>/feature_support.py`. See bottom of doc.

### 4. New `test_golden.py` template (replaces three files per op)

One file per op, replaces `test_golden_cross_product.py`,
`test_golden_shapes.py`, `test_golden_modes.py`:

```python
# test_golden.py
import pytest
from eval.feature_matrix import cartesian, xfail_reason
from eval.golden_tests.<op>.feature_spec import (
    TARGET, INPUTS, SHAPE_TAGGERS,
)
# (Import path depends on Open decision #1 — for now, hand-wave it.)
from <op_module> import SUPPORTED, EXCLUSIONS, <op>

def _parametrize_cases():
    cases = []
    for inputs in INPUTS:
        for axes in cartesian(TARGET, SHAPE_TAGGERS, inputs):
            marks = []
            reason = xfail_reason(axes, SUPPORTED, EXCLUSIONS)
            if reason is not None:
                marks.append(pytest.mark.xfail(reason=reason, strict=True,
                                               raises=NotImplementedError))
            cases.append(pytest.param(inputs, axes, marks=marks, id=_case_id(inputs, axes)))
    return cases

@pytest.mark.parametrize("inputs,axes", _parametrize_cases())
def test_op(inputs, axes, device):
    # ...build tensors from inputs+axes, call run_<op>, assert against golden
```

`cartesian()` runs every `tag_*` over `inputs` to produce the shape-derived
axis values, then mixes with the finite-axis cartesian. Each parametrize
case carries the full {axis: value} dict — no projection, no marker
substitution.

Regression / numerical-stability tests (the `0.01 * randn` family) live in
a separate `test_regression.py` with their own non-parameterized cases,
tagged `@pytest.mark.numerics`. Not driven by the registry.

**Per-op `run_<op>(inputs, axes, ...)` helper.** Each op's test file imports
or defines a `run_<op>(inputs, axes, device)` function that builds the
appropriate tensors from the axis dict (dtype, layout from `axes`) and
calls the op, with per-axis tolerances baked in. This is the natural place
for "bf16 cells get tolerance X, fp32 cells get tolerance Y" — no separate
tolerance mechanism needed.

### 5. Updated `/golden-tests` skill

Rewrite the skill to emit:
- `eval/golden_tests/<op>/feature_spec.py` (TARGET, INPUTS, SHAPE_TAGGERS)
- `eval/golden_tests/<op>/test_golden.py` (single auto-parameterized file)
- `eval/golden_tests/<op>/test_regression.py` (numerical edge cases)
- `eval/golden_tests/<op>/helpers.py` (op-specific torch reference,
  `run_<op>` wrapper, tolerance map)

**No `api_contract.md` anymore.** The signature lives in the op file
itself; the skill's job is to emit a clearer agent-facing prompt instead.
That removes the entire `validate_contract.py` flow as user-visible.

### 6. Updated agents

Three agent prompt files (under `agents/`):

- **`incremental-planner`**: writes `op_design.md` (architecture, CB layout,
  helpers). Also writes the initial TARGET registry (the ambition — what
  the op should eventually support) as part of `feature_spec.py`.
  **Does NOT write op_requirements.md** — that's the verifier's job.
  Slightly shorter prompt than today.

- **`incremental-implementer`** (Phase 0):
  - Builds the baseline op from `op_design.md`.
  - Writes initial `SUPPORTED` + `EXCLUSIONS` + tag functions +
    `validate()` next to the op (location per Open decision #1).
  - **Does NOT see the golden tests.** Same boundary as today —
    implementer codes the op, doesn't peek at test expectations.

- **`incremental-verifier`** (Phase 0):
  - Runs the golden tests.
  - For each cell that's outside SUPPORTED: confirms it actually does
    fail (xfail-strict gives this for free; verifier just reports).
  - For each cell that's in SUPPORTED: confirms it passes; if not, flags
    a Refinement 1 entry for "fix this regression."
  - Writes `op_requirements.md` as a sequence of refinements. **Each
    refinement is one or more `add X to SUPPORTED["axis"]` instructions,
    not a `**Subset**:` filter.** Example:
    ```
    ## Refinement 1 — bfloat16 support
    Goal: add `ttnn.bfloat16` to SUPPORTED["dtype"].

    ## Refinement 2 — row-major layout
    Goal: add `ttnn.ROW_MAJOR_LAYOUT` to SUPPORTED["layout"].
    ```
    The verifier's bigger job is figuring out the right *order* of
    refinements — what unblocks the most tests, what dependencies exist
    between axis additions.
  - Final refinement target: every cell in TARGET is in SUPPORTED (modulo
    EXCLUSIONS) and the test suite is all green, no xfails.

- **`refinement-implementer`** (Refinement N — same agent identity as
  implementer, different prompt):
  - Reads the refinement's "add X to Y" goal.
  - Edits SUPPORTED (and validate()) to claim X is supported.
  - Iterates on the kernel until the newly-not-xfail tests pass.
  - The test harness automatically picks them up — SUPPORTED was the
    source of truth, and the xfail decoration re-evaluates next run.
  - If something's harder than expected: add a cell to EXCLUSIONS rather
    than fight it, and note in changelog.

### 7. Simplified `run_refinements.py`

Today on branch: 752 lines, mostly subset-filter machinery. On main: simpler
but still parses `**Subset**:` lines. After migration:

- No more `**Subset**:` parsing.
- No more `golden_capabilities` import.
- Refinement loop becomes: invoke refinement-implementer → run pytest →
  check for unexpected XPASS/FAIL → record snapshot → repeat.
- Probably ~150 lines.

The refinement queue lives in op_requirements.md (written by the verifier).
Each entry is "add X to SUPPORTED[axis]" — direct, declarative, no
substitution rules.

### 8. Failure classification — expanded

Today's `eval/classify_failures.py` distinguishes: hang, OOM, compilation,
signature, numerical, other. Expand to:

| Category | Definition | Detection |
|---|---|---|
| `hang` | Dispatch timeout / triage marker | Existing logic (unchanged) |
| `oom` | L1 / DRAM allocation failure | Existing regex (unchanged) |
| `compilation` | Kernel build failure | Existing regex (unchanged) |
| **`validation`** *(new)* | `validate()` raised `NotImplementedError` on a cell that was supposed to be supported | Failure type is `NotImplementedError` AND test was not xfail-strict (xfail-strict cells with NotImplementedError = expected, not a failure) |
| **`numerical-precision`** *(new)* | Off but not catastrophic — PCC > 0.9, or allclose passes within 3× the test's nominal rtol/atol | Parsed from `check_output`'s structured failure message |
| **`numerical-bug`** *(new)* | Catastrophic — PCC ≤ 0.9, OR Inf/NaN in output, OR allclose fails by >3× rtol/atol | Parsed from `check_output`'s structured failure message |
| `other` | Anything not matching above | Default bucket |

**This requires `check_output` (per-op helper) to emit structured failure
messages** so the classifier can pick precision vs bug apart without
re-running the comparison. Suggested format:

```python
class CheckOutputError(AssertionError):
    def __init__(self, *, pcc, rtol_actual, rtol_target, atol_actual,
                 atol_target, has_inf, has_nan, severity):
        self.severity = severity   # "precision" | "bug"
        msg = (f"CheckOutputError severity={severity} "
               f"pcc={pcc:.4f} "
               f"rtol_actual={rtol_actual:.4f} (target {rtol_target}) "
               f"atol_actual={atol_actual:.4f} (target {atol_target}) "
               f"inf={has_inf} nan={has_nan}")
        super().__init__(msg)
```

`check_output` computes the severity itself based on the thresholds (PCC,
3× rtol/atol band, Inf/NaN presence). The classifier just regex-scans for
`severity=precision` vs `severity=bug` in the JUnit XML.

Why bake severity into the message rather than re-derive it in the
classifier:

- The check function already has the raw values; re-parsing them out of
  human-readable assertion text would be fragile.
- A single source of truth for "what counts as catastrophic" lives next
  to the comparison — easy to tune per op if needed.
- The classifier stays simple (regex on a known marker string).

`validation` detection uses the failure type tag in JUnit (`type=` attribute
on `<failure>`). If the test had an `xfail(raises=NotImplementedError,
strict=True)` decoration and raised NotImplementedError, pytest records
that as `xfail`, not `failure` — so xfail cells never trigger `validation`.
A non-xfail cell raising NotImplementedError is a real bug:
SUPPORTED claimed it works, validate() rejected it.

## What does NOT come from the branch

Explicitly leave behind:

- `golden_capabilities.py` — substitution algebra, joint passing signatures,
  projection, complete-signature gate, marginal view, smoke filter "no
  signal" logic. ~579 lines deleted.
- `dim_vocab.py` (top-level) and per-op `dim_vocab.py` files — ~170 + 4×~20
  lines deleted.
- `markers_plugin.py` — 57 lines deleted.
- `conftest.py` marker registration (per-op + shared) — ~80–100 lines deleted.
- `--dim`/`-m` plumbing in `eval_test_runner.sh` — ~80 lines deleted.
- Subset-filter machinery in `run_refinements.py` — ~600 lines deleted.
- `test_golden_capabilities.py`, `test_classify_capabilities_roundtrip.py`,
  `test_markers_plugin.py` — ~800 lines of tests for code we're deleting.

Rough total deletion vs. branch: ~2200+ lines. Replacement: ~300 lines
across `registry.py` + per-op registry files. Net: ~1900 lines lighter
than the branch, even though `main` is already lighter than the branch.

## What's worth cherry-picking from the branch

A few branch commits are unrelated to the marker system and are pure
improvements. Check each before discarding:

- Device-lock contention profiling in `eval_test_runner.sh` and
  `run_safe_pytest.sh` (commit `d9b6a0b9746` and friends in the parent
  repo, plus the submodule-side timing emit).
- Failure classification refinements in `classify_failures.py` (note: we're
  also expanding it ourselves — see §8 — so cherry-pick before adding new
  categories to avoid conflicts).
- DB column additions in `db.py` (refinement_snapshots, phases, etc.) —
  some of these may still be useful for tracking per-refinement progress.
- Dashboard tabs for per-phase cost/turns (`_html_phases_panel` etc.).
- The "harness-failure" surfacing in `eval_test_runner.sh`.

Treat these as cherry-picks, one at a time, with judgment. Do NOT
mass-merge.

## Order of work

**Phase 1 — primitives (no agents, no migration yet):**

1. Branch off `main` of the submodule (e.g. `mstaletovic/registry-model`).
2. Write `eval/feature_matrix.py` + unit tests. ~50–80 lines + ~120 lines tests.
3. Hand-write `feature_spec.py` and `test_golden.py` for **one** op
   (recommend `softmax` — already exists on main and has a clean cross-
   product structure, plus enough axes to exercise the model).
4. Add `SUPPORTED`/`EXCLUSIONS`/`validate()` next to the op (location per
   Open decision #1).
5. Run the test suite manually. Verify:
   - Baseline SUPPORTED yields the expected xfail set.
   - Supported cells pass.
   - EXCLUSIONS work (xfail-strict catches them as NotImplementedError).
   - Tag-driven axes (alignment) work.
   - validate() raises NotImplementedError for cells outside SUPPORTED.

**Phase 2 — pipeline wiring:**

6. Strip `run_refinements.py` down to the new simpler form (~150 lines).
7. Update `pipeline.py` so phase 0 calls planner + implementer + verifier
   with the new responsibilities.
8. Update `run_eval.py` to drop any subset-filter assumptions.
9. Run an end-to-end Phase 0 → 1 refinement on softmax manually (driving
   the agents by hand if needed). Confirm: implementer adds a feature,
   XPASS-strict signals, implementer updates CURRENT, green.

**Phase 3 — agents:**

10. Rewrite the three agent prompts (`incremental-planner`,
    `incremental-implementer`, `incremental-verifier`). The verifier prompt
    shrinks most — its big responsibility (figuring out capabilities) is
    now a one-liner in `eval/registry.py`.
11. Rewrite `/golden-tests` skill to emit the new file layout.
12. Run a full agent-driven pipeline on softmax. Iterate on prompts.

**Phase 4 — port other ops:**

13. For each existing op (rms_norm, layer_norm_rm, conv2d_nhwc, etc.):
    - Write per-op registry.py and test_golden.py from scratch using the
      new format (don't try to convert old files).
    - Add CURRENT/EXCEPTIONS/validate() to the op file.
    - Run the tests. If a real bug shows up that the old marker system
      missed (likely some partial-coverage cases), fix the op.

**Phase 5 — cleanup:**

14. Delete the old eval files definitively (after a release tag for
    archive).
15. Cherry-pick the unrelated improvements from the branch.
16. Update CLAUDE.md / README in the submodule to reflect the new model.

## How we know it works

- Softmax Phase 0 → Refinement 1 (e.g. add bf16) → green, with CURRENT
  updated by the implementer agent and verified by XPASS-strict.
- Old failure modes from `RefinementsTesting` that should no longer be
  possible:
  - Partial-marker tests being invisible to refinement runs (no more
    markers).
  - Silent capability over-claim from marginal view (no more marginal
    view).
  - Distribution tests not running in the inner loop (they're either in
    `test_golden.py` parameterized like everything else, or in
    `test_regression.py` always running).

## Risk / fallback

- `RefinementsTesting` stays on its branch as the working "old world"
  until softmax + at least one other op (rms_norm, since it has the
  partial-marker problem starkly) work end-to-end on the new model.
- If something fundamental about XPASS-strict + the registry model
  doesn't pan out (e.g. agents struggle to maintain CURRENT
  consistently), the branch is still available.
- Estimate: 2–3 days for primitives + softmax end-to-end manual driving,
  1 week with agents, another week for the full op port if going at it
  full-time.

## Open decisions

### Settled

1. **Location of `SUPPORTED` + `EXCLUSIONS` + `SHAPE_TAGGERS` + `validate()`:
   inline in `ttnn/ttnn/operations/<op>/<op>.py`.** Read the op file, see the
   support contract. validate() right at the dispatch site. One file per op.
   Test-side picks up the (slightly heavier) op import — acceptable cost
   for the "everything about the op in one place" ergonomic.

2. **Tag functions are op-local.** No shared helpers. Axis names may
   overlap across ops (e.g. `alignment` in conv2d_nhwc vs groupnorm) but
   mean different things per op.
3. **Regression-test marker name: `numerics`.** Used as
   `@pytest.mark.numerics` on `test_regression.py` cases.
4. **Soft partial support: use EXCLUSIONS.** Per-axis tolerances live in
   the per-op `run_<op>` helper — bf16 cells get one tolerance, fp32 gets
   another. No separate tolerance-override mechanism.
5. **DB schema: not in scope.** Leave it alone.
6. **`api_contract.md` and `validate_contract.py` flow: dropped.** The
   `/golden-tests` skill emits better prompts instead.
7. **Expanded failure classification (see §"Failure classification" below).**
