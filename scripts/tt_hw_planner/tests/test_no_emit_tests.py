"""Unit tests for Layer 4 of Phase 2 — persistent no-emit-tests list.

ModuleList components have no testable forward as a standalone unit
(they're containers; the parent's PCC test exercises them indirectly).
Phase 2's MODULELIST routing drops their stub + test files. Without
Layer 4, the next scaffold run would re-emit the test files, capture
would fail again ("no forward"), and the components would re-enter the
skip-list -- losing all the Phase 2 progress.

Layer 4 fixes this by adding a persistent no-emit-tests list that
``_emit_pcc_template`` honors. Once a component is on the list, its
test scaffold is never re-emitted, regardless of how many `up` runs
happen.

The list is keyed per-model so dropping `multi_scale_block` in SAM2
doesn't affect a different model with the same component name."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _om():
    return importlib.import_module("scripts.tt_hw_planner.overlay_manager")


def test_load_no_emit_returns_empty_dict_when_missing(tmp_path, monkeypatch) -> None:
    om = _om()
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path)
    assert om.load_no_emit_tests("facebook/nothing-here") == {}


def test_persist_and_load_roundtrip(tmp_path, monkeypatch) -> None:
    om = _om()
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path)
    model_id = "facebook/test"
    om.persist_no_emit_test(model_id, "multi_scale_block", reason="ModuleList drop")
    listing = om.load_no_emit_tests(model_id)
    assert "multi_scale_block" in listing
    assert "ModuleList" in listing["multi_scale_block"]["reason"]
    assert "captured_ts" in listing["multi_scale_block"]


def test_persist_is_idempotent(tmp_path, monkeypatch) -> None:
    """Re-adding an existing entry must not overwrite the captured_ts
    (so we don't lose the ORIGINAL drop timestamp)."""
    om = _om()
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path)
    model_id = "facebook/test"
    om.persist_no_emit_test(model_id, "comp_a", reason="first")
    ts_first = om.load_no_emit_tests(model_id)["comp_a"]["captured_ts"]
    om.persist_no_emit_test(model_id, "comp_a", reason="second")
    ts_second = om.load_no_emit_tests(model_id)["comp_a"]["captured_ts"]
    assert ts_first == ts_second, "re-persist must preserve original timestamp"


def test_is_no_emit_test_fast_path(tmp_path, monkeypatch) -> None:
    om = _om()
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path)
    model_id = "facebook/test"
    assert om.is_no_emit_test(model_id, "comp_x") is False
    om.persist_no_emit_test(model_id, "comp_x", reason="drop")
    assert om.is_no_emit_test(model_id, "comp_x") is True


def test_remove_no_emit_test(tmp_path, monkeypatch) -> None:
    om = _om()
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path)
    model_id = "facebook/test"
    om.persist_no_emit_test(model_id, "comp_a", reason="drop")
    om.persist_no_emit_test(model_id, "comp_b", reason="drop")
    assert om.remove_no_emit_test(model_id, "comp_a") is True
    assert om.load_no_emit_tests(model_id) == {"comp_b": om.load_no_emit_tests(model_id)["comp_b"]}
    # Removing missing entry is False (idempotent)
    assert om.remove_no_emit_test(model_id, "comp_a") is False
    # Removing the last entry deletes the file
    om.remove_no_emit_test(model_id, "comp_b")
    assert om.load_no_emit_tests(model_id) == {}


def test_is_keyed_per_model(tmp_path, monkeypatch) -> None:
    """Same component name on two different models must not collide.
    A `multi_scale_block` dropped from SAM2 should not affect a
    different model's `multi_scale_block`."""
    om = _om()
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path)
    om.persist_no_emit_test("facebook/sam2", "multi_scale_block", reason="drop")
    assert om.is_no_emit_test("facebook/sam2", "multi_scale_block") is True
    assert om.is_no_emit_test("facebook/other-model", "multi_scale_block") is False


def test_locked_modules_roundtrip(tmp_path, monkeypatch) -> None:
    """Durable lock store: persist/load/is_locked/remove, keyed per model,
    idempotent (preserves first locked_ts), and survives a fresh load."""
    om = _om()
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path)
    model_id = "facebook/test"

    assert om.is_locked_module(model_id, "parent") is False
    om.persist_locked_module(model_id, "parent", reason="recomposed")
    assert om.is_locked_module(model_id, "parent") is True

    ts_first = om.load_locked_modules(model_id)["parent"]["locked_ts"]
    om.persist_locked_module(model_id, "parent", reason="ignored second time")
    assert om.load_locked_modules(model_id)["parent"]["locked_ts"] == ts_first

    assert om.is_locked_module("facebook/other", "parent") is False

    assert om.remove_locked_module(model_id, "parent") is True
    assert om.is_locked_module(model_id, "parent") is False
    assert om.remove_locked_module(model_id, "parent") is False


def test_locked_modules_malformed_file_returns_empty(tmp_path, monkeypatch) -> None:
    om = _om()
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path)
    p = om._locked_modules_path("facebook/test")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{ not json")
    assert om.load_locked_modules("facebook/test") == {}


def test_malformed_file_returns_empty_not_raise(tmp_path, monkeypatch) -> None:
    """A malformed no-emit-tests.json must not crash the scaffold path."""
    om = _om()
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path)
    model_id = "facebook/test"
    p = om._no_emit_tests_path(model_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("not valid json {{{")
    assert om.load_no_emit_tests(model_id) == {}
    assert om.is_no_emit_test(model_id, "anything") is False


def test_emit_pcc_template_respects_no_emit_list(tmp_path, monkeypatch) -> None:
    """The scaffold-time integration: when a component is on the no-emit
    list, ``_emit_pcc_template`` must return without writing a test file.
    Without this, Layer 4 is decorative -- the list exists but scaffold
    ignores it."""
    om = _om()
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path)

    from scripts.tt_hw_planner import bringup_loop as bl

    model_id = "facebook/sam2"
    om.persist_no_emit_test(model_id, "multi_scale_block", reason="drop")

    demo_dir = tmp_path / "demo"
    test_path, generated, already = bl._emit_pcc_template(
        demo_dir=demo_dir,
        component_name="multi_scale_block",
        model_id=model_id,
        hf_reference="",
        new_shape={},
        repo_root=tmp_path,
        overwrite=True,  # explicitly request overwrite -- no-emit must still win
    )
    assert generated is False, "scaffold must NOT generate a test when comp is no-emit"
    assert not test_path.is_file(), "scaffold must NOT write the file"


def test_emit_pcc_template_emits_for_other_components(tmp_path, monkeypatch) -> None:
    """Sanity check: putting `comp_a` on the no-emit list must NOT block
    `comp_b`'s test emission. The list is per-component."""
    om = _om()
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path)

    from scripts.tt_hw_planner import bringup_loop as bl

    model_id = "facebook/sam2"
    om.persist_no_emit_test(model_id, "multi_scale_block", reason="drop")

    demo_dir = tmp_path / "demo"
    test_path, generated, _already = bl._emit_pcc_template(
        demo_dir=demo_dir,
        component_name="some_other_component",
        model_id=model_id,
        hf_reference="",
        new_shape={"x": (1, 3, 4)},
        repo_root=tmp_path,
        overwrite=False,
    )
    assert generated is True, "other components must still get their test emitted"
    assert test_path.is_file()


def test_phase2_modulelist_drop_persists_to_no_emit_list(tmp_path, monkeypatch) -> None:
    """End-to-end: Phase 2's MODULELIST routing must add components to
    the no-emit list. Without this, the next scaffold would re-emit
    the test we just dropped and the component would re-enter the
    skip-list."""
    om = _om()
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path)

    from scripts.tt_hw_planner.commands import tackle_skipped as ts

    demo = tmp_path / "demo"
    (demo / "_stubs").mkdir(parents=True)
    (demo / "tests" / "pcc").mkdir(parents=True)
    (demo / "_stubs" / "ml_comp.py").write_text("# stub")
    (demo / "tests" / "pcc" / "test_ml_comp.py").write_text("# test")

    model_id = "facebook/test"
    om.persist_skip(model_id, "ml_comp", reason="harness: ModuleList no forward")

    rc, dropped, recaptured = ts.run_phase2_stage(model_id, demo)
    assert "ml_comp" in dropped
    assert om.is_no_emit_test(model_id, "ml_comp") is True, (
        "After Phase 2 drops a ModuleList component, it must appear on the "
        "no-emit-tests list so the next scaffold doesn't re-emit it"
    )


def test_emit_pcc_template_handles_missing_overlay_manager_gracefully(tmp_path, monkeypatch) -> None:
    """If `is_no_emit_test` raises for any reason (network FS issue,
    Python import error), the scaffold path must continue, not crash.
    The except clause swallows + proceeds with the standard emission."""
    om = _om()
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path)

    from scripts.tt_hw_planner import bringup_loop as bl

    def _raise(_a, _b):
        raise RuntimeError("simulated overlay manager failure")

    monkeypatch.setattr(om, "is_no_emit_test", _raise)

    demo_dir = tmp_path / "demo"
    # Must not raise even though is_no_emit_test does
    test_path, generated, _already = bl._emit_pcc_template(
        demo_dir=demo_dir,
        component_name="comp_x",
        model_id="facebook/test",
        hf_reference="",
        new_shape={"x": (1, 3, 4)},
        repo_root=tmp_path,
        overwrite=False,
    )
    assert generated is True
