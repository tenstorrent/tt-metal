"""A composite repo is brought up by running the normal pipeline once per part.

A composite (diffusers pipeline, or any repo whose parts each carry their own
config) has no single root model, so the single-root pipeline had nothing to work
on and scaffold refused it outright. Its parts ARE ordinary models, so the fan-out
re-enters cmd_up per part rather than adding a second pipeline.
"""

from __future__ import annotations

import argparse
import json
import os

from scripts.tt_hw_planner import cli
from scripts.tt_hw_planner.probe import component_targets, detect_composite_repo


def _make_composite(tmp_path, parts, with_root_config=False):
    """A repo laid out like a composite: an index plus per-part configs."""
    root = tmp_path / "repo"
    root.mkdir()
    (root / "model_index.json").write_text(json.dumps({"_class_name": "SomePipeline"}))
    if with_root_config:
        (root / "config.json").write_text(json.dumps({"model_type": "something"}))
    for name, cfg in parts.items():
        d = root / name
        d.mkdir()
        (d / "config.json").write_text(json.dumps(cfg))
    return str(root)


# ─── detection ───────────────────────────────────────────────────


def test_local_composite_is_detected_without_a_root_config(tmp_path) -> None:
    root = _make_composite(tmp_path, {"partA": {"_class_name": "A"}, "partB": {"_class_name": "B"}})
    is_comp, parts = detect_composite_repo(root)
    assert is_comp
    assert sorted(parts) == ["partA", "partB"]


def test_single_root_model_is_not_composite(tmp_path) -> None:
    d = tmp_path / "solo"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": "llama"}))
    is_comp, parts = detect_composite_repo(str(d))
    assert not is_comp and parts == []


def test_detection_never_raises_on_garbage() -> None:
    assert detect_composite_repo("") == (False, [])
    assert detect_composite_repo("not a valid id at all !!") == (False, [])


# ─── target resolution ───────────────────────────────────────────


def test_only_parts_with_a_readable_config_become_targets(tmp_path) -> None:
    root = _make_composite(tmp_path, {"good": {"_class_name": "G"}})
    os.mkdir(os.path.join(root, "no_config"))  # e.g. a scheduler/tokenizer dir
    names = [n for n, _ in component_targets(root, ["good", "no_config", "absent"])]
    assert names == ["good"]


def test_component_paths_are_usable_model_dirs(tmp_path) -> None:
    root = _make_composite(tmp_path, {"partA": {"_class_name": "A"}})
    for _, path in component_targets(root, ["partA"]):
        assert os.path.isfile(os.path.join(path, "config.json"))


# ─── fan-out ─────────────────────────────────────────────────────


def _args(model_id):
    return argparse.Namespace(model_id=model_id, isolation="none", box="T3K", mesh=None)


def test_non_composite_returns_none_so_normal_path_runs(tmp_path, monkeypatch) -> None:
    d = tmp_path / "solo"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": "llama"}))
    assert cli._fan_out_composite(_args(str(d))) is None


def test_each_part_is_run_through_the_standard_pipeline(tmp_path, monkeypatch) -> None:
    root = _make_composite(tmp_path, {"partA": {"_class_name": "A"}, "partB": {"_class_name": "B"}})
    seen = []

    def fake_cmd_up(a):
        seen.append(os.path.basename(a.model_id))
        return 0

    monkeypatch.setattr(cli, "cmd_up", fake_cmd_up)
    rc = cli._fan_out_composite(_args(root))
    assert rc == 0
    assert sorted(seen) == ["partA", "partB"]


def test_one_failing_part_does_not_hide_the_others(tmp_path, monkeypatch) -> None:
    root = _make_composite(tmp_path, {"partA": {"_class_name": "A"}, "partB": {"_class_name": "B"}})
    seen = []

    def fake_cmd_up(a):
        seen.append(os.path.basename(a.model_id))
        if a.model_id.endswith("partA"):
            raise RuntimeError("boom")
        return 0

    monkeypatch.setattr(cli, "cmd_up", fake_cmd_up)
    rc = cli._fan_out_composite(_args(root))
    assert sorted(seen) == ["partA", "partB"], "a failure must not abort the remaining parts"
    assert rc != 0, "the run must report failure"


def test_recursion_is_bounded(tmp_path, monkeypatch) -> None:
    """A part that is itself composite must not fan out forever."""
    root = _make_composite(tmp_path, {"partA": {"_class_name": "A"}, "partB": {"_class_name": "B"}})
    a = _args(root)
    a._composite_depth = cli._COMPOSITE_FANOUT_DEPTH_CAP
    assert cli._fan_out_composite(a) is None


def test_parts_inherit_the_parent_run_settings(tmp_path, monkeypatch) -> None:
    root = _make_composite(tmp_path, {"partA": {"_class_name": "A"}})
    captured = {}

    def fake_cmd_up(a):
        captured["box"] = getattr(a, "box", None)
        captured["depth"] = getattr(a, "_composite_depth", None)
        return 0

    monkeypatch.setattr(cli, "cmd_up", fake_cmd_up)
    cli._fan_out_composite(_args(root))
    assert captured["box"] == "T3K", "component runs must keep the parent's target box"
    assert captured["depth"] == 1
