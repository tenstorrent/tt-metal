"""Pin: find_demo_dir must search backend-demo parent dirs, not just
models/demos/.

The Phi-3.5 escalation run on 2026-06-02 scaffolded to
``models/tt_transformers/demo/phi_3_5_mini_instruct/`` (the
ALREADY-SUPPORTED escalation path writes there because the backend's
demo_path lives under tt_transformers). Step 4's autofill then
called find_demo_dir() which only searched ``models/demos/`` and
returned None — producing the confusing "No scaffolded demo folder
found" error right after scaffold had successfully created it.

This test pins that find_demo_dir scans backend-demo parents too.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.tt_hw_planner.bringup_loop import find_demo_dir


def _write_status(dir_path: Path, model_id: str) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / "bringup_status.json").write_text(json.dumps({"new_model_id": model_id, "components": []}))


def test_finds_in_models_demos(tmp_path: Path) -> None:
    """Original behaviour: cold-start path under models/demos/."""
    target = tmp_path / "models" / "demos" / "phi_3_5_mini_instruct"
    _write_status(target, "microsoft/Phi-3.5-mini-instruct")
    found = find_demo_dir("microsoft/Phi-3.5-mini-instruct", repo_root=tmp_path)
    assert found == target


def test_finds_in_tt_transformers_demo(tmp_path: Path) -> None:
    """NEW: ALREADY-SUPPORTED escalation path writes scaffolds under
    models/tt_transformers/demo/<slug>/ — must be searched."""
    target = tmp_path / "models" / "tt_transformers" / "demo" / "phi_3_5_mini_instruct"
    _write_status(target, "microsoft/Phi-3.5-mini-instruct")
    found = find_demo_dir("microsoft/Phi-3.5-mini-instruct", repo_root=tmp_path)
    assert found == target


def test_returns_none_when_no_match(tmp_path: Path) -> None:
    _write_status(tmp_path / "models" / "demos" / "other", "some/other-model")
    found = find_demo_dir("microsoft/Phi-3.5-mini-instruct", repo_root=tmp_path)
    assert found is None


def test_returns_none_when_no_demo_tree(tmp_path: Path) -> None:
    found = find_demo_dir("microsoft/Phi-3.5-mini-instruct", repo_root=tmp_path)
    assert found is None


def test_matches_first_hit_when_present_in_both_trees(tmp_path: Path) -> None:
    """If somehow both trees contain a status for the same model
    (shouldn't happen in practice but guard the order), the
    ``models/demos/`` location should win — that's the canonical
    location and matches pre-fix behaviour."""
    canonical = tmp_path / "models" / "demos" / "phi_3_5_mini_instruct"
    sibling = tmp_path / "models" / "tt_transformers" / "demo" / "phi_3_5_mini_instruct"
    _write_status(canonical, "microsoft/Phi-3.5-mini-instruct")
    _write_status(sibling, "microsoft/Phi-3.5-mini-instruct")
    found = find_demo_dir("microsoft/Phi-3.5-mini-instruct", repo_root=tmp_path)
    assert found == canonical
