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
