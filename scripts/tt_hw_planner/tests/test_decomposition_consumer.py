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

from scripts.tt_hw_planner.decomposition_consumer import (  # noqa: E402
    consume_decomposition_plan,
    reinject_missing_decomposition_children,
)


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
    assert "encoder_layers_0" in names
    assert "encoder_layers_1" in names
    by_name = {c["name"]: c for c in status["components"]}
    assert by_name["encoder_layers_0"]["submodule_path"] == "encoder.layers.0"


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
            {"name": "encoder_layers_0", "status": "NEW", "submodule_path": "encoder.layers.0"},
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
    assert added == 1  # only encoder_layers_1 is new

    status = json.loads((demo_dir / "bringup_status.json").read_text())
    names = [c["name"] for c in status["components"]]
    assert names.count("encoder_layers_0") == 1  # not duplicated
    assert "encoder_layers_1" in names


def test_consumer_skips_locked_parent(tmp_path: Path, monkeypatch) -> None:
    """A locked (recomposed) parent must NEVER be re-decomposed, even if a
    stale plan names it — its children stay frozen and it is not un-graduated."""
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
    om.persist_locked_module("test/m", "parent_a", reason="recomposed")

    added, _ = consume_decomposition_plan(model_id="test/m", demo_dir=demo_dir)
    assert added == 0
    assert "parent_a" not in om.load_no_emit_tests("test/m")


def test_consumer_skips_parent_already_passing(tmp_path: Path, monkeypatch) -> None:
    """A stale plan must NOT decompose a parent that's already passing this
    run (per seed pytest) — otherwise it un-graduates working components and
    marks them no_emit."""
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

    added, _ = consume_decomposition_plan(model_id="test/m", demo_dir=demo_dir, passed_components={"parent_a"})
    assert added == 0  # passing parent is not decomposed
    assert "parent_a" not in om.load_no_emit_tests("test/m")  # and not un-graduated


def test_emit_repo_root_is_models_parent_not_models_dir() -> None:
    """The child-test emitter must compute the import relative to the repo root
    (the dir CONTAINING models/), not demo_dir.parent.parent.parent (the models/
    dir), else the import comes out `demos.<...>` (unimportable) instead of
    `models.demos.<...>`. Robust to model nesting depth."""
    from scripts.tt_hw_planner.decomposition_consumer import _emit_repo_root

    assert _emit_repo_root(Path("/wt/models/demos/audio/seamless")) == Path("/wt")  # depth-4
    assert _emit_repo_root(Path("/repo/models/demos/foo")) == Path("/repo")  # depth-3
    assert _emit_repo_root(Path("/wt/models/demos/audio/seamless")) != Path("/wt/models")


def test_reinject_re_adds_wiped_child_from_archived_plan(tmp_path: Path, monkeypatch) -> None:
    """A decomposition child wiped from bringup_status (e.g. by a re-scaffold)
    is re-added from the archived plan so its parent can recompose."""
    from scripts.tt_hw_planner import overlay_manager as om

    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path / "overlays")

    demo_dir = _make_demo(tmp_path, [{"name": "parent_a", "status": "NEW", "submodule_path": "t2u_model.model"}])
    arch = demo_dir / "decomposition_plan.applied"
    arch.mkdir(parents=True)
    (arch / "plan_20260101_000000.json").write_text(
        json.dumps(
            [
                {
                    "parent_name": "parent_a",
                    "children": [
                        {"name": "encoder", "submodule_path": "t2u_model.model.encoder", "class_name": "Encoder"},
                        {"name": "decoder", "submodule_path": "t2u_model.model.decoder", "class_name": "Decoder"},
                    ],
                }
            ]
        )
    )

    added, notes = reinject_missing_decomposition_children(model_id="test/m", demo_dir=demo_dir)
    assert added == 2
    status = json.loads((demo_dir / "bringup_status.json").read_text())
    by_path = {c.get("submodule_path"): c for c in status["components"]}
    assert "t2u_model.model.encoder" in by_path
    assert by_path["t2u_model.model.encoder"]["name"] == "t2u_model_model_encoder"
    assert by_path["t2u_model.model.encoder"]["_added_by_decomposition_of"] == "parent_a"

    again, _ = reinject_missing_decomposition_children(model_id="test/m", demo_dir=demo_dir)
    assert again == 0


def test_reinject_skips_intermediate_and_no_emit(tmp_path: Path, monkeypatch) -> None:
    """Reinject must skip a child that was itself further decomposed (a
    descendant child path exists) and a child on the no_emit list."""
    from scripts.tt_hw_planner import overlay_manager as om

    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path / "overlays")

    demo_dir = _make_demo(tmp_path, [{"name": "parent_a", "status": "NEW", "submodule_path": "blk"}])
    arch = demo_dir / "decomposition_plan.applied"
    arch.mkdir(parents=True)
    (arch / "plan_20260101_000000.json").write_text(
        json.dumps(
            [
                {
                    "parent_name": "parent_a",
                    "children": [
                        {"name": "mid", "submodule_path": "blk.mid", "class_name": "Mid"},
                        {"name": "leaf", "submodule_path": "blk.leaf", "class_name": "Leaf"},
                    ],
                },
                {
                    "parent_name": "mid",
                    "children": [{"name": "g", "submodule_path": "blk.mid.g", "class_name": "G"}],
                },
            ]
        )
    )
    om.persist_no_emit_test("test/m", "blk_leaf", reason="ModuleList drop")

    added, _ = reinject_missing_decomposition_children(model_id="test/m", demo_dir=demo_dir)
    status = json.loads((demo_dir / "bringup_status.json").read_text())
    paths = {c.get("submodule_path") for c in status["components"]}
    assert "blk.mid.g" in paths  # leaf grandchild re-added
    assert "blk.mid" not in paths  # intermediate (has descendant) skipped
    assert "blk.leaf" not in paths  # no_emit child skipped
    assert added == 1
