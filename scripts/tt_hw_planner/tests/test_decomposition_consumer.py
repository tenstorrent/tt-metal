"""Unit tests for ``decomposition_consumer.consume_decomposition_plan``.

Pins: the consumer mutates bringup_status.json by adding decomposed
children as NEW components and marks the parent no_emit. Idempotent
on re-runs (archives consumed plan, re-apply is no-op).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.tt_hw_planner.decomposition_consumer import consume_decomposition_plan  # noqa: E402


def _make_demo(tmp_path: Path, components: list) -> Path:
    demo_dir = tmp_path / "demo"
    demo_dir.mkdir()
    status = {"components": components, "new_model_id": "test/m"}
    (demo_dir / "bringup_status.json").write_text(json.dumps(status))
    return demo_dir


def test_no_plan_means_no_changes(tmp_path: Path) -> None:
    """No decomposition_plan.json → consumer is a no-op."""
    demo_dir = _make_demo(tmp_path, [{"name": "a", "status": "NEW"}])
    added, notes = consume_decomposition_plan(model_id="test/m", demo_dir=demo_dir)
    assert added == 0
    # bringup_status should be unchanged
    status = json.loads((demo_dir / "bringup_status.json").read_text())
    assert len(status["components"]) == 1


def test_consumer_adds_children_to_bringup_status(tmp_path: Path, monkeypatch) -> None:
    """The consumer reads decomposition_plan.json and adds each child as
    a NEW component to bringup_status.json."""
    from scripts.tt_hw_planner import overlay_manager as om

    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path / "overlays")

    demo_dir = _make_demo(tmp_path, [{"name": "parent_a", "status": "NEW", "submodule_path": "encoder"}])
    plan = [
        {
            "parent_name": "parent_a",
            "parent_path": "encoder",
            "children": [
                {"name": "child1", "submodule_path": "encoder.layers.0", "class_name": "Block"},
                {"name": "child2", "submodule_path": "encoder.layers.1", "class_name": "Block"},
            ],
        }
    ]
    (demo_dir / "decomposition_plan.json").write_text(json.dumps(plan))

    added, notes = consume_decomposition_plan(model_id="test/m", demo_dir=demo_dir)
    assert added == 2

    status = json.loads((demo_dir / "bringup_status.json").read_text())
    names = [c["name"] for c in status["components"]]
    assert "child1" in names
    assert "child2" in names
    # Each child has its submodule_path
    by_name = {c["name"]: c for c in status["components"]}
    assert by_name["child1"]["submodule_path"] == "encoder.layers.0"


def test_consumer_marks_parent_no_emit(tmp_path: Path, monkeypatch) -> None:
    """After applying the plan, the parent component is marked no_emit
    so its standalone PCC doesn't conflict with the children."""
    from scripts.tt_hw_planner import overlay_manager as om

    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path / "overlays")

    demo_dir = _make_demo(tmp_path, [{"name": "parent_a", "status": "NEW", "submodule_path": "encoder"}])
    plan = [
        {
            "parent_name": "parent_a",
            "children": [{"name": "child1", "submodule_path": "encoder.layers.0", "class_name": "Block"}],
        }
    ]
    (demo_dir / "decomposition_plan.json").write_text(json.dumps(plan))

    consume_decomposition_plan(model_id="test/m", demo_dir=demo_dir)

    no_emit = om.load_no_emit_tests("test/m")
    assert "parent_a" in no_emit


def test_consumer_is_idempotent(tmp_path: Path, monkeypatch) -> None:
    """Re-running the consumer on the same plan is a no-op (the plan
    file was archived after the first apply)."""
    from scripts.tt_hw_planner import overlay_manager as om

    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path / "overlays")

    demo_dir = _make_demo(tmp_path, [{"name": "parent_a", "status": "NEW", "submodule_path": "encoder"}])
    plan = [
        {
            "parent_name": "parent_a",
            "children": [{"name": "child1", "submodule_path": "encoder.layers.0", "class_name": "Block"}],
        }
    ]
    (demo_dir / "decomposition_plan.json").write_text(json.dumps(plan))

    first_added, _ = consume_decomposition_plan(model_id="test/m", demo_dir=demo_dir)
    assert first_added == 1
    # The plan file should have been archived
    assert not (demo_dir / "decomposition_plan.json").is_file()
    assert (demo_dir / "decomposition_plan.applied").is_dir()

    # Re-running has no plan to apply
    second_added, _ = consume_decomposition_plan(model_id="test/m", demo_dir=demo_dir)
    assert second_added == 0


def test_consumer_skips_already_added_children(tmp_path: Path, monkeypatch) -> None:
    """If a child name is already in bringup_status (e.g. partial apply
    state from a crashed run), skip re-adding it."""
    from scripts.tt_hw_planner import overlay_manager as om

    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path / "overlays")

    demo_dir = _make_demo(
        tmp_path,
        [
            {"name": "parent_a", "status": "NEW", "submodule_path": "encoder"},
            {"name": "child1", "status": "NEW", "submodule_path": "encoder.layers.0"},
        ],
    )
    plan = [
        {
            "parent_name": "parent_a",
            "children": [
                {"name": "child1", "submodule_path": "encoder.layers.0", "class_name": "Block"},
                {"name": "child2", "submodule_path": "encoder.layers.1", "class_name": "Block"},
            ],
        }
    ]
    (demo_dir / "decomposition_plan.json").write_text(json.dumps(plan))

    added, _ = consume_decomposition_plan(model_id="test/m", demo_dir=demo_dir)
    assert added == 1  # only child2 is new

    status = json.loads((demo_dir / "bringup_status.json").read_text())
    names = [c["name"] for c in status["components"]]
    assert names.count("child1") == 1  # not duplicated
    assert "child2" in names
