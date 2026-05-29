# Phase 2 plan — tackling the persistent skip-list

Phase 1 (current) focuses on graduating components that have working PCC tests.
Phase 2 (future) addresses the 10-ish components per model that get
**skip-listed** because the test harness can't even invoke them — no PCC
signal means the auto-iterate loop has nothing to converge on.

**Do not implement this until Phase 1 graduates all currently-tested
components.** Phase 1 will likely reshape some of the failure-class
assumptions below, and building the orchestrator before then risks
designing for the wrong inputs.

---

## Skip-list root causes (from facebook/sam2-hiera-tiny v14)

Skip-list lives at `scripts/tt_hw_planner/overlays/<safe_model_id>/skipped_components.json`.

| Count | Reason category | Components (SAM2 example) | Root cause |
|---|---|---|---|
| 5 | `ModuleList no forward` | `multi_scale_block`, `video_mask_down_sampler_layer`, `video_memory_attention_layer`, `video_memory_fuser_c_x_block`, `video_two_way_attention_block` | Class wraps / subclasses `nn.ModuleList`; calling its forward raises NotImplementedError. ModuleList is a container — it doesn't have a forward by design. |
| 2 | `permute(sparse_coo) dim mismatch` | `video_layer_norm`, `video_mask_down_sampler` | Test scaffold synthesizes wrong input shape (sparse/dense handling). |
| 2 | `missing required arg` | `video_memory_encoder`, `video_position_embedding_sine` | Forward needs more than just `pixel_values` (e.g. `(hidden, attn_mask)`). |
| 1 | `groups=256 weight shape mismatch` | `video_memory_fuser` | Grouped-conv input shape inference incorrect. |

All 10 are **harness-layer** failures, not agent-layer. The agent never gets
a chance because pytest can't even run the component.

---

## What already exists (re-use, don't rebuild)

| Capability | Location |
|---|---|
| Skip-list persist/load/clear | `overlay_manager.py:115,132,152` |
| `overlay-clear-skips` CLI | `cli.py:8804` (manual only) |
| Auto-onboard capture driver (LLM-drafted) | `auto_capture_driver_onboard.py` |
| 3-layer capture framework | `capture_drivers.py` |
| Captured-input persistence + test wiring | `capture_inputs.py` |
| ModuleList awareness in clustering | `module_tree.py:63,385` |
| Per-component capture CLI (`--component`) | `cli.py:8649` |

---

## What's missing — the orchestrator layer

1. **No Phase 2 entry-point CLI command**: nothing knows "Phase 1 done, now
   process skip-list".
2. **No per-skip-reason routing**: the skip-list has 4 different root causes
   but no logic that says "ModuleList → drop, missing-arg → re-capture".
3. **No scaffold-level ModuleList filter**: `module_tree.py` knows about
   containers (line 63) but `scaffold.py` still emits standalone PCC tests
   for ModuleList-wrapping classes.
4. **No re-test loop after unskip**: once captured, components need to flow
   back into the normal PCC pipeline.
5. **No Phase 2 graduation reporting**: separate counter from Phase 1 so
   progress is legible.

---

## Sketch — `tackle-skipped` CLI command

New file: `scripts/tt_hw_planner/commands/tackle_skipped.py` (~150 LOC)

```python
def cmd_tackle_skipped(args):
    """Phase 2 entry point. Walks the skip-list and routes each entry
    to the appropriate unblock strategy. Designed to run AFTER Phase 1
    graduates all currently-tested components."""

    skips = load_persistent_skips(args.model_id)
    if not skips:
        print("no persistent skips — nothing to do")
        return 0

    # 1. Classify each entry by reason
    by_reason: Dict[str, List[str]] = defaultdict(list)
    for comp, info in skips.items():
        reason = _classify_skip_reason(info["reason"])
        # → MODULELIST | MISSING_ARG | SHAPE_MISMATCH | DRIVER_FAILURE | UNKNOWN
        by_reason[reason].append(comp)

    # 2a. ModuleList components → drop (they have no testable forward;
    #     their behavior is verified via the parent's PCC test).
    for comp in by_reason["MODULELIST"]:
        drop_from_scaffold(args.model_id, comp)
        remove_from_skip_list(args.model_id, comp)
        print(f"  {comp}: dropped (ModuleList — tested via parent)")

    # 2b. Missing-arg + shape-mismatch → retry capture with auto-onboard.
    #     Real captured inputs replace the broken scaffold synthesis.
    needs_capture = (
        by_reason["MISSING_ARG"]
        + by_reason["SHAPE_MISMATCH"]
        + by_reason["DRIVER_FAILURE"]
    )
    if needs_capture:
        os.environ["TT_PLANNER_AUTO_ONBOARD_DRIVER"] = "1"
        for comp in needs_capture:
            cmd_capture_inputs(_make_args(model_id=args.model_id, component=comp))

    # 3. Re-attempt PCC tests on now-unblocked components
    post_skips = load_persistent_skips(args.model_id)
    unblocked = [c for c in skips if c not in post_skips]
    graduated, still_stuck = [], []
    if unblocked:
        for comp in unblocked:
            rc = run_pcc_test_for_component(args.model_id, comp)
            (graduated if rc == 0 else still_stuck).append(comp)

    # 4. Report
    print_phase2_summary(
        dropped=by_reason["MODULELIST"],
        graduated=graduated,
        still_stuck=still_stuck,
        unknown_reason=by_reason["UNKNOWN"],
    )
    return 0 if not still_stuck else 1
```

Helper functions needed in the same file:

```python
_REASON_PATTERNS = {
    "MODULELIST":     [r"ModuleList no forward"],
    "MISSING_ARG":    [r"missing.*required.*positional"],
    "SHAPE_MISMATCH": [r"permute.*dim mismatch", r"weight shape mismatch",
                       r"groups=\d+"],
    "DRIVER_FAILURE": [r"capture driver", r"no driver matched"],
}

def _classify_skip_reason(reason: str) -> str:
    for cat, patterns in _REASON_PATTERNS.items():
        if any(re.search(p, reason) for p in patterns):
            return cat
    return "UNKNOWN"

def drop_from_scaffold(model_id: str, comp: str) -> None:
    """Remove the component's stub + test files so future runs don't
    re-emit them. Verified safe because the parent's PCC test covers it."""
    demo_dir = _resolve_demo_dir(model_id)
    safe = _safe_id(comp)
    (demo_dir / "_stubs" / f"{safe}.py").unlink(missing_ok=True)
    (demo_dir / "tests" / "pcc" / f"test_{safe}.py").unlink(missing_ok=True)
```

---

## Plus one small fix to scaffold (~10 LOC)

`scripts/tt_hw_planner/scaffold.py` — at the point where per-component
test scaffolds are emitted, before writing the file:

```python
from .module_tree import _CONTAINER_CLASS_NAMES

# ...

if class_name in _CONTAINER_CLASS_NAMES:
    log(f"skipping standalone test emission for {comp}: "
        f"ModuleList/container — tested via parent")
    continue
```

This prevents the 5 ModuleList components from getting standalone test
scaffolds on FUTURE runs. (For SAM2 today they're already emitted, so the
`tackle-skipped` command handles cleanup.)

---

## Unit tests to add

`scripts/tt_hw_planner/tests/test_tackle_skipped.py` (~80 LOC)

- `test_classify_skip_reason_modulelist`
- `test_classify_skip_reason_missing_arg`
- `test_classify_skip_reason_shape_mismatch`
- `test_classify_skip_reason_unknown_falls_back`
- `test_tackle_modulelist_drops_and_clears_skip`
- `test_tackle_missing_arg_enables_auto_onboard_env`
- `test_tackle_skipped_reports_graduated_and_stuck`
- `test_scaffold_skips_emission_for_modulelist`

---

## CLI hookup

`scripts/tt_hw_planner/cli.py` — add subparser ~15 LOC:

```python
pts = sub.add_parser(
    "tackle-skipped",
    help=(
        "Phase 2: walk the persistent skip-list and route each entry "
        "to the appropriate unblock strategy. Run AFTER Phase 1 "
        "graduates all currently-tested components."
    ),
)
pts.add_argument("model_id", help="HuggingFace model id")
pts.add_argument("--dry-run", action="store_true",
                 help="classify and report only; do not modify state")
pts.set_defaults(func=cmd_tackle_skipped)
```

---

## Total framework cost

| Piece | Estimate |
|---|---|
| `tackle_skipped.py` command | ~150 LOC |
| `scaffold.py` ModuleList filter | ~10 LOC |
| `test_tackle_skipped.py` | ~80 LOC |
| `cli.py` parser hookup | ~15 LOC |
| **Total** | **~255 LOC** |

All framework, no LLM cost. Implementable in a half-day after Phase 1
passes.

---

## When to implement

**Do not implement until Phase 1 hits its convergence criterion.**

Phase 1 = all currently-tested components graduate to PCC ≥ 0.99.

After that:

1. Run `tackle-skipped <model_id> --dry-run` to see what each skip
   would route to.
2. Decide whether the routing makes sense before letting it run for real.
3. Iterate on the reason-classifier if Phase 1 surfaced new skip
   patterns we haven't seen yet.

The reason-classifier needs to be re-audited against fresh Phase-1 data
because the failure modes the scaffold hits today may shift once
extras-enrichment + activation_diff trigger expansion + auto-onboard
driver are all in play.
