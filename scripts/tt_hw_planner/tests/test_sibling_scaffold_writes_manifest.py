"""Pin: the sibling ("closest already-ported model") scaffold route must
emit a bring-up manifest, like the other two routes already do.

The Qwen-Image-Edit run on 2026-09-22 failed its `text_encoder` component
with:

    ERROR: No scaffolded demo folder found for '.../qwen_image_edit_text_encoder'.
    Run `python -m scripts.tt_hw_planner scaffold ... --apply` first.

one step AFTER scaffold printed "APPLIED scaffold". The component is a VLM,
so it passed the category escape in `plan_scaffold`, and its routed backend
is a specific (non-generic) entry with `use_module_tree=False`, so it passed
the module-tree escape too. It landed in the sibling branch, which edited the
tuning tables and copied model_params but returned with `new_demo_dir=None`
and no `bringup_status.json`. Everything downstream locates a component only
via `bringup_loop.find_demo_dir`, which matches on that manifest — so the
component was invisible and Step 4 autofill hard-failed.
"""

from __future__ import annotations

import inspect

from scripts.tt_hw_planner import scaffold as scaffold_mod

MANIFEST_NAME = "bringup_status.json"


def _sibling_branch_source() -> str:
    """Source of plan_scaffold from its sibling branch to its return."""
    src = inspect.getsource(scaffold_mod.plan_scaffold)
    marker = src.find("sibling_id = compat.similar_supported_model")
    assert marker >= 0, "sibling branch missing — test invariant out of date"
    return src[marker:]


def test_sibling_branch_emits_a_bringup_manifest():
    block = _sibling_branch_source()
    assert "_bringup_plan_changes" in block, (
        "the sibling scaffold branch must emit the bring-up plan files "
        "(BRING_UP_PLAN.md + bringup_status.json); without them "
        "bringup_loop.find_demo_dir cannot locate the component and Step 4 "
        "autofill fails right after scaffold reports success"
    )


def test_sibling_branch_reports_its_demo_dir():
    block = _sibling_branch_source()
    assert "new_demo_dir=new_demo_dir" in block, (
        "the sibling branch must return the demo dir it wrote the manifest "
        "into, so callers agree on one dir per model"
    )


def test_manifest_is_among_the_collected_plan_files():
    """The helper the branch relies on must actually produce the manifest
    the downstream lookup keys on."""
    from scripts.tt_hw_planner.bringup_plan import collect_bringup_plan_files

    sig = inspect.signature(collect_bringup_plan_files)
    assert "new_demo_dir_rel" in sig.parameters
    body = inspect.getsource(collect_bringup_plan_files)
    assert MANIFEST_NAME in body, f"{MANIFEST_NAME} is no longer emitted by collect_bringup_plan_files"


def test_all_three_scaffold_routes_share_one_demo_dir_resolver():
    """One model -> one demo dir. Each route must go through the shared
    resolver rather than deriving its own path."""
    src = inspect.getsource(scaffold_mod)
    assert src.count("_resolve_demo_dir_rel(") >= 4, (
        "expected the shared resolver to be defined once and used by all "
        "three scaffold routes (escalation, sibling, demo-folder)"
    )


def test_manifest_is_generated_before_the_nothing_to_scaffold_guard():
    """Re-running an already-scaffolded model must not abort at Step 2.

    The tuning-table row and the model_params copy are both create-once: on a
    second run (or when an overlay restores them) each is skipped, so the
    change set is empty and the `nothing to scaffold` guard fired -- even
    though the scaffold was complete and usable. The manifest is rewritten
    unconditionally, so generating it BEFORE the guard keeps the change set
    non-empty and lets the run continue. Order is the whole fix; pin it.
    """
    block = _sibling_branch_source()
    manifest_at = block.find("_bringup_plan_changes")
    guard_at = block.find("if not changes:")
    assert manifest_at >= 0, "sibling branch no longer emits the manifest"
    assert guard_at >= 0, "nothing-to-scaffold guard missing — invariant out of date"
    assert manifest_at < guard_at, (
        "the manifest must be generated BEFORE the `nothing to scaffold` guard; "
        "otherwise re-running an already-scaffolded model aborts at Step 2 with "
        "a complete scaffold already on disk"
    )


def test_guard_message_names_the_manifest_as_the_third_condition():
    """The guard now means 'nothing at all was produced', including no
    manifest. Its message must say so, or the next person reads a stale
    explanation that points only at the sibling."""
    block = _sibling_branch_source()
    guard_at = block.find("if not changes:")
    msg = block[guard_at : guard_at + 1400]
    assert "manifest" in msg, "guard message must mention the manifest condition"


def test_an_already_scaffolded_model_is_a_no_op_not_an_error():
    """Re-running scaffold on a component that already has a manifest must
    return a plan with no changes, NOT raise.

    Both `already supported — no scaffolding needed` and `nothing to
    scaffold` used to fire on a complete, usable scaffold, aborting Step 2 on
    every retry. The presence of the manifest is the tool's own definition of
    'scaffolded' (find_demo_dir matches on it), so it short-circuits first."""
    src = inspect.getsource(scaffold_mod.plan_scaffold)
    early = src.find("ALREADY SCAFFOLDED")
    assert early >= 0, "the idempotent early-return is gone"
    # Anchor on the raise SITES, not prose — the explanatory comments quote
    # these same messages.
    for raise_msg in (
        'f"{new_model_id} is already supported',
        'raise ScaffoldError(\n            "nothing to scaffold',
    ):
        at = src.find(raise_msg)
        assert at >= 0, f"expected raise {raise_msg!r} — invariant out of date"
        assert early < at, (
            f"the already-scaffolded short-circuit must come before {raise_msg!r}, "
            "or a retry aborts on a scaffold that is already complete"
        )


def test_escalation_still_rescaffolds_on_purpose():
    """The escalation hook re-scaffolds deliberately (REUSE -> ADAPT demotion),
    so it must be exempt from the no-op short-circuit."""
    src = inspect.getsource(scaffold_mod.plan_scaffold)
    early = src.find("ALREADY SCAFFOLDED")
    guard = src.rfind("if not force_already_supported:", 0, early)
    assert guard >= 0, "the short-circuit must be gated on force_already_supported"
