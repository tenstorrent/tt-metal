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


def test_emitter_lookup_and_short_circuit_agree_on_the_manifest_name():
    """The manifest filename is a contract between three places: the emitter
    (collect_bringup_plan_files), the lookup that decides whether a component
    exists (bringup_loop.find_demo_dir) and scaffold's already-scaffolded
    short-circuit. They must all read it from one constant, or a rename
    silently splits them and components go missing again."""
    from scripts.tt_hw_planner import bringup_loop, bringup_plan

    assert bringup_plan.BRINGUP_STATUS_FILENAME == MANIFEST_NAME

    emitter = inspect.getsource(bringup_plan.collect_bringup_plan_files)
    lookup = inspect.getsource(bringup_loop.find_demo_dir)
    short_circuit = inspect.getsource(scaffold_mod.plan_scaffold)
    for name, body in (("emitter", emitter), ("lookup", lookup), ("short-circuit", short_circuit)):
        assert "BRINGUP_STATUS_FILENAME" in body, f"{name} does not use the shared constant"
        assert f'"{MANIFEST_NAME}"' not in body, f"{name} still hardcodes the manifest filename"


def test_sibling_and_demo_folder_routes_share_one_demo_dir_resolver():
    """One model -> one demo dir.

    The sibling and demo-folder routes both go through the shared resolver
    instead of each deriving a path. The escalation branch deliberately keeps
    its own inline derivation: test_scaffold_escalation_demo_dir pins that
    branch's source for the slug + `.parent` pattern, since that is how the
    Phi-3.5 [Errno 17] regression reached only that branch."""
    src = inspect.getsource(scaffold_mod)
    # one definition + one call per non-escalation route
    assert src.count("_resolve_demo_dir_rel(") >= 3, (
        "expected the shared resolver to be defined once and used by the " "sibling and demo-folder routes"
    )
    sibling = _sibling_branch_source()
    assert "_resolve_demo_dir_rel(" in sibling, "sibling route must use the shared resolver"


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


def test_every_mesh_label_the_planner_emits_resolves_in_the_demo_it_invokes():
    """A label the planner exports must exist in the demo's table.

    The tables read MESH_DEVICE with `.get(label, <default>)`, so an unknown
    label does not error — it silently falls back to a DIFFERENT mesh. A
    planner that emits a label the demo has never heard of therefore runs the
    model on the wrong topology and reports success. Pin that every Wormhole
    label in MESH_DEVICE_MAP resolves, to the same shape, in the demo the
    planner actually invokes."""
    import ast
    import re
    from pathlib import Path

    from scripts.tt_hw_planner.bringup import MESH_DEVICE_MAP

    consumers = [
        Path("models/tt_transformers/demo/simple_text_demo.py"),
        Path("models/tt_transformers/conftest.py"),
    ]
    for path in consumers:
        src = path.read_text()
        match = re.search(r'\{\s*"N150".*?\}', src, re.S)
        assert match, f"no mesh-label table found in {path}"
        table = ast.literal_eval(match.group(0))
        # Some labels are deliberately aliased to both orientations of one
        # shape (Galaxy's (4,8)/(8,4) -> "TG"). For those, orientation is the
        # authors' choice; a CHIP-COUNT difference never is.
        shapes_per_label = {}
        for (arch, shape), label in MESH_DEVICE_MAP.items():
            if arch == "Wormhole":
                shapes_per_label.setdefault(label, set()).add(shape)

        for label, shapes in shapes_per_label.items():
            if label not in table:
                continue
            got = table[label]
            assert got[0] * got[1] in {s[0] * s[1] for s in shapes}, (
                f"{path}: label {label!r} resolves to {got} "
                f"({got[0] * got[1]} chips) but the planner emits it for "
                f"{sorted(shapes)} — the demo would run on a different number of chips"
            )
            if len(shapes) == 1:
                assert got == next(iter(shapes)), (
                    f"{path}: label {label!r} resolves to {got} but the planner "
                    f"emits it for {next(iter(shapes))} — the demo would run on the wrong mesh"
                )
