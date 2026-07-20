"""Pin: `_refresh_plan` and `apply_scaffold` must refuse to silently
overwrite a valid `bringup_status.json` (N>0 components) with an empty
plan (0 components).

Background: the 2026-06-04 seamless-m4t promote corruption — a torch-less
scaffold subprocess produced an empty plan, scaffold wrote it over the
canonical manifest, and the iter loop's apply step silently rejected
every LLM solution because the components dict was empty.

Three layers of defense covered by these tests:
  1. `_refresh_plan` surfaces subprocess stderr + corruption-guard print
     when components drop from N>0 to 0.
  2. `apply_scaffold` REFUSES to overwrite a valid manifest with a
     0-component plan (preserve existing on N→0 transition).
  3. `apply_all_responses` raises `LLMError` when the manifest has 0
     NEW/ADAPT components but `_synth_responses/` has files to apply.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_refresh_plan_has_corruption_guard() -> None:
    """The `_refresh_plan` function must compare pre/post component
    counts and surface a guard message when post drops to 0."""
    from scripts.tt_hw_planner import bringup_loop

    src = Path(bringup_loop.__file__).read_text(encoding="utf-8")
    body_start = src.find("def _refresh_plan(")
    body_end = src.find("\ndef ", body_start + 1)
    body = src[body_start:body_end]
    assert "CORRUPTION GUARD" in body, (
        "_refresh_plan must include the pre/post components-count guard "
        "that surfaces when scaffold subprocess silently drops components."
    )
    assert "pre_counts" in body and "post_counts" in body, (
        "_refresh_plan must capture pre_counts before subprocess and "
        "post_counts after, so the corruption-guard has values to compare."
    )


def test_apply_scaffold_refuses_empty_overwrite(tmp_path) -> None:
    """`apply_scaffold` must NOT overwrite an existing valid
    `bringup_status.json` with one that has 0 components."""
    from scripts.tt_hw_planner.scaffold import ScaffoldChange, ScaffoldPlan, apply_scaffold

    # Pre-existing valid manifest with 25 components.
    demo_dir = tmp_path / "models" / "demos" / "test_model"
    demo_dir.mkdir(parents=True)
    existing = {
        "new_model_id": "test/test-model",
        "counts": {"REUSE": 2, "ADAPT": 0, "NEW": 23},
        "components": [{"name": f"comp_{i}", "status": "NEW"} for i in range(25)],
    }
    status_path = demo_dir / "bringup_status.json"
    status_path.write_text(json.dumps(existing, indent=2))

    # New plan: empty.
    empty_plan_bytes = json.dumps({"counts": {"REUSE": 0, "ADAPT": 0, "NEW": 0}, "components": []}).encode("utf-8")

    # Build a ScaffoldPlan that would overwrite this file.
    rel_path = status_path.relative_to(tmp_path)
    plan = ScaffoldPlan(
        new_model_id="test/test-model",
        new_base_name="test-model",
        new_tail="test-model",
        sibling_model_id="test/test-model",
        sibling_base_name="test-model",
        sibling_tail="test-model",
        compat_overall="OK",
        compat_summary="",
        changes=[
            ScaffoldChange(
                kind="create",
                path=str(rel_path),
                new_content=empty_plan_bytes,
                source=None,
                added_lines=0,
                preserve_if_exists=False,
            )
        ],
        skipped=[],
        warnings=[],
    )

    import scripts.tt_hw_planner.discovery as _disc

    _orig = _disc.BRINGUP_ROOT
    _disc.BRINGUP_ROOT = lambda: tmp_path  # type: ignore[assignment]
    try:
        applied = apply_scaffold(plan)
    finally:
        _disc.BRINGUP_ROOT = _orig

    # The existing file must remain unchanged.
    after = json.loads(status_path.read_text())
    assert len(after.get("components", [])) == 25, (
        "apply_scaffold OVERWROTE a valid 25-component manifest with an "
        "empty plan. This is the seamless-m4t corruption signature — "
        "Fix 3 must refuse the write."
    )
    # The applied list must mention the refusal.
    assert any(
        "REFUSED" in a for a in applied
    ), "apply_scaffold must report the refusal in its result list; got: " + str(applied)


def test_apply_all_responses_refuses_empty_manifest(tmp_path) -> None:
    """`apply_all_responses` must raise LLMError when the manifest has
    0 NEW/ADAPT components but the responses dir has files to apply."""
    from scripts.tt_hw_planner.llm_synth import LLMError, apply_all_responses

    demo_dir = tmp_path / "models" / "demos" / "test_model"
    demo_dir.mkdir(parents=True)
    # Empty manifest.
    (demo_dir / "bringup_status.json").write_text(
        json.dumps(
            {
                "new_model_id": "test/test-model",
                "counts": {"REUSE": 0, "ADAPT": 0, "NEW": 0},
                "components": [],
            },
            indent=2,
        )
    )
    # But responses dir has a file to apply.
    (demo_dir / "_synth_responses").mkdir()
    (demo_dir / "_synth_responses" / "some_component.py").write_text("def __call__(): pass\n")

    # Patch find_demo_dir to return our tmp_path one.
    import scripts.tt_hw_planner.llm_synth as _ls

    _orig_fdd = _ls.find_demo_dir
    _ls.find_demo_dir = lambda model_id, repo_root=None: demo_dir  # type: ignore[assignment]
    try:
        with pytest.raises(LLMError, match="0 NEW.*0 ADAPT"):
            apply_all_responses(model_id="test/test-model")
    finally:
        _ls.find_demo_dir = _orig_fdd
