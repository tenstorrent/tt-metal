# Registry Migration — Review Guide

This is the Phase 1 PR: primitives + softmax port. No agent prompts, no
pipeline rewiring, no other ops touched. Designed to be reviewable as a
self-contained slice.

## Branches

| Repo | Branch | Based on | Head |
|---|---|---|---|
| `tt-metal` (parent) | `mstaletovic/registry-model` | `mstaletovic/agent_eval` | `ced0d733d64` |
| `tt_ops_code_gen` (submodule) | `mstaletovic/registry-model` | `main` | `1cb3b72` |

Submodule commit: `Registry-model migration (Phase 1): primitives + softmax port`.
Parent commit: `Bump tt_ops_code_gen to registry-model migration (Phase 1)`.

## What changed at a glance

```
 eval/REGISTRY_MODEL.md                              | 170 +++ (new)
 eval/feature_matrix.py                              | 163 +++ (new)
 eval/tests/test_feature_matrix.py                   | 273 ++++ (new)
 eval/golden_tests/softmax/feature_spec.py           |  51 ++ (new)
 eval/golden_tests/softmax/test_golden.py            |  65 ++ (new)
 eval/golden_tests/softmax/test_regression.py        |  82 ++ (new)
 eval/classify_failures.py                           |  49 +++ (modified)
 eval/tests/test_classify.py                         |  46 +++ (modified)
 eval/golden_tests/softmax/conftest.py               |  18 +-- (replaced)
 eval/golden_tests/softmax/helpers.py                | 212 +++ (rewritten)
 eval/golden_tests/softmax/test_golden_cross_product.py | (deleted)
 eval/golden_tests/softmax/test_golden_modes.py      | (deleted)
 eval/golden_tests/softmax/test_golden_shapes.py     | (deleted)
 eval/golden_tests/softmax/api_contract.md           | (deleted)
 eval/golden_tests/softmax/op_requirements.md        | (deleted)
 eval/golden_tests/softmax/changelog.md              | (deleted)
```

Net: ~800 lines added, ~700 deleted. Doesn't yet show the full simplification —
this PR is the **foundation**. The big deletions (`golden_capabilities.py`,
`dim_vocab.py`, `markers_plugin.py`, the marker substitution in
`run_refinements.py`) happen in Phase 2 because main of the submodule doesn't
have those files (they exist only on `mstaletovic/RefinementsTesting`).

## Where to start reviewing

In this order:

### 1. `registry_design_findings.md` + `registry_migration_plan.md` (top-level of tt-metal)

The conversation that produced this PR. Skim if you want the rationale;
skip if you just want to see what changed.

### 2. `eval/REGISTRY_MODEL.md` (submodule)

The reference doc. Shows what the agent-produced op file must contain:
`SHAPE_TAGGERS`, `SUPPORTED`, `EXCLUSIONS`, `validate()`. Walks through how
the test harness uses these, what the three xfail outcomes mean, the
refinement flow, the failure classification expansion.

### 3. `eval/feature_matrix.py` (submodule)

The whole primitive in one file. Four functions: `apply_shape_taggers`,
`cartesian`, `is_supported`, `unsupported_reason`, plus `case_id` and
`matches_exclusion`. Pure functions over dicts/lists. No I/O, no pytest,
no ttnn.

Key design point: when an axis appears in both `target` (universe
declaration) and `shape_taggers` (computation), the tagger wins. This is
the "alignment is computed from shape, not iterated" rule.

### 4. `eval/tests/test_feature_matrix.py` (submodule)

29 unit tests. Most useful to read:
- `test_cartesian_yields_full_combos` — basic cartesian behavior.
- `test_cartesian_tagger_wins_over_target_for_shared_axis` — the
  axis-overlap rule.
- `test_end_to_end_xfail_decisions` — the loop test_golden.py runs at
  parametrize time. Spec for the parameterization contract.

Run with: `pytest .claude/eval/tests/test_feature_matrix.py -v`.
68 total tests pass (29 here + 39 in test_classify.py).

### 5. `eval/classify_failures.py` + `eval/tests/test_classify.py` (submodule)

The diff is small. Three new categories added: `validation`, `numerical-bug`,
`numerical-precision`. Old `numerical` kept as a backwards-compat fallback.

The new categories key off:
- `NotImplementedError` text → `validation`
- `severity=bug` text → `numerical-bug`
- `severity=precision` text → `numerical-precision`

The `severity=*` markers are baked into `CheckOutputError` messages by the
per-op `helpers.py`. The classifier doesn't recompute severity.

Read the new tests in `test_classify.py` (search for `# --- New categories ---`):
- `test_validation_not_implemented`
- `test_numerical_bug_severity`
- `test_numerical_precision_severity`
- Several precedence tests confirming `hang > validation` and
  `severity=bug > legacy numerical`.

### 6. `eval/golden_tests/softmax/` (submodule) — the first ported op

Worth reading the *whole* directory. It's small now:

```
__init__.py
conftest.py          # 18 lines — registers only 'numerics' marker
feature_spec.py      # 51 lines — TARGET + INPUTS (planner output)
helpers.py           # 212 lines — pytorch reference, run_softmax,
                     #              CheckOutputError with severity tagging
test_golden.py       # 65 lines — the parameterized cross-product test
test_regression.py   # 82 lines — @pytest.mark.numerics regression cases
```

Compare with the pre-PR softmax directory (still on
`mstaletovic/RefinementsTesting`): three test files, plus `api_contract.md`,
`op_requirements.md`, `changelog.md`, `dim_vocab.py`, much heavier
`conftest.py`.

Key reads:
- **`test_golden.py`** — note how short the parameterize logic is.
  The xfail decision is `unsupported_reason(axes, SUPPORTED, EXCLUSIONS)`
  paired with `pytest.mark.xfail(strict=True, raises=NotImplementedError)`.
  No marker substitution.
- **`feature_spec.py`** — the planner's output. Two top-level objects:
  TARGET (per-axis lists) and INPUTS (list of shape tuples).
- **`helpers.py`** — most of the size is `check_output` with severity
  detection. The `_classify_severity` function encodes the
  catastrophic-vs-precision rule.

### 7. `eval/golden_tests/softmax/helpers.py` — the CheckOutputError pattern

This is the single load-bearing pattern that makes the new
`classify_failures` categories work. Severity is computed at the comparison
site (where PCC and RMS are already in hand), not re-derived from string
parsing.

```python
def _classify_severity(pcc, rms, pcc_target, rms_target, has_inf, has_nan):
    if has_inf or has_nan: return "bug"
    if pcc <= BUG_PCC_FLOOR: return "bug"       # 0.9
    if rms > BUG_FACTOR * rms_target: return "bug"  # 3x
    return "precision"
```

Per-dtype tolerances live in `TOLERANCES` at the top of helpers.py. To
adjust a cell's tolerance, edit the dict.

## What this PR doesn't do (deferred)

These are out of scope for this PR but called out in
`registry_migration_plan.md`:

- Rewrite `run_refinements.py` to drop `**Subset**:` parsing. On
  main the file already lacks the heavy substitution machinery
  (that was added on the branch), so this is a smaller cleanup
  rather than a deletion fest.
- Rewrite the three agent prompts (`incremental-planner`,
  `incremental-implementer`, `incremental-verifier`) for the new
  responsibilities.
- Rewrite the `/golden-tests` skill to emit `feature_spec.py` +
  `test_golden.py` + `test_regression.py` instead of the old three
  files.
- Port the other ops (`rms_norm`, `layer_norm_rm`, `conv2d_nhwc`,
  `groupnorm_*`, `interpolate*`, `scaled_dot_product_attention`,
  `tilize`). Each is its own small migration.
- Wire failure-classification severity through the dashboard and
  Superset views.

## How to validate locally

```bash
# Submodule unit tests (no device required)
cd /localdev/mstaletovic/tt-metal
source python_env/bin/activate
python3 -m pytest .claude/eval/tests/test_feature_matrix.py -v
python3 -m pytest .claude/eval/tests/test_classify.py -v

# All 68 should pass.
```

Note: `test_golden.py` and `test_regression.py` for softmax import from
`ttnn.operations.softmax`, which **doesn't exist in the base repo** — it's
agent-generated by the pipeline at run time. Collecting these tests
locally will fail at import. That's expected; once the agent produces a
softmax op with the inline registry block (per `eval/REGISTRY_MODEL.md`),
they'll collect and run.

To exercise the parameterization mechanism without a real op, the unit
tests in `test_feature_matrix.py::test_end_to_end_xfail_decisions`
already cover the contract.

## Open questions for review

1. **Should `SHAPE_TAGGERS` really be in the op file?** I followed
   Open decision #1 (Option A inline). If the op file gets long, you may
   prefer Option B (sibling `feature_support.py`). The model itself works
   either way.

2. **`TARGET` lives in `feature_spec.py` on the test side.** Alternative:
   put it in the op file too, so the entire support story is in one
   place. Counter-argument: TARGET is the planner's ambition (testing
   universe), SUPPORTED is the implementer's current claim — separating
   them keeps responsibilities clear.

3. **`numeric_stable` and `dim` are softmax-specific axes.** They live
   in `TARGET` (and would be in `SUPPORTED`) like any other axis. Are we
   OK with treating op-specific kwargs as just more axes? Today's softmax
   `feature_spec.py` does this; the alternative is keeping them out of
   the registry and only parameterizing on universal axes.

4. **EXCLUSIONS is empty in the current softmax port.** Once we hand-
   port more ops, expect this list to grow for cases like
   `{"layout": ROW_MAJOR, "dim": -2}` if they prove not worth supporting.

## Next steps after this PR merges

Per `registry_migration_plan.md`:

1. Phase 2: simplify `run_refinements.py` + `pipeline.py` + `run_eval.py`.
2. Phase 3: rewrite agents and the `/golden-tests` skill.
3. End-to-end manual run of softmax Phase 0 → Refinement 1 to validate
   the agent flow.
4. Phase 4: port remaining ops one at a time.
