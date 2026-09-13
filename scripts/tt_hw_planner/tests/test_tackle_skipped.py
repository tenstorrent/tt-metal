"""Unit tests for the Phase 2 tackle-skipped orchestrator.

Covers the classifier + the dry-run path + the ModuleList drop path.
The auto-onboard / capture-inputs path is mocked since invoking real
HF + LLM would defeat unit-test cost discipline.

The classifier patterns are pinned against SAM2's actual v14 skip-list
reasons -- if any of those patterns regress (e.g., a refactor narrows
a regex), the test catches it before SAM2 silently re-stuck."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import tempfile
from pathlib import Path
from unittest import mock


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _import_module():
    from scripts.tt_hw_planner.commands import tackle_skipped as ts

    return ts


def _import_overlay_manager():
    from scripts.tt_hw_planner import overlay_manager as om

    return om


def test_classify_modulelist_reasons() -> None:
    ts = _import_module()
    assert ts._classify_skip_reason("harness: ModuleList no forward (v13)") == "MODULELIST"
    assert (
        ts._classify_skip_reason(
            "HF reference forward([]) raised NotImplementedError ... Module [ModuleList] is missing th"
        )
        == "MODULELIST"
    )


def test_classify_missing_arg_reasons() -> None:
    ts = _import_module()
    assert ts._classify_skip_reason("harness: missing 2 required positional args (v13)") == "MISSING_ARG"
    assert ts._classify_skip_reason("harness: missing positional arg (v13)") == "MISSING_ARG"


def test_classify_shape_mismatch_reasons() -> None:
    ts = _import_module()
    assert ts._classify_skip_reason("harness: permute(sparse_coo) dim mismatch (v13)") == "SHAPE_MISMATCH"
    assert ts._classify_skip_reason("harness: groups=256 weight shape mismatch (v13)") == "SHAPE_MISMATCH"
    assert ts._classify_skip_reason("RuntimeError: shape '[0, 11, 11]' is invalid") == "SHAPE_MISMATCH"


def test_classify_driver_failure_reasons() -> None:
    ts = _import_module()
    assert ts._classify_skip_reason("no driver matched for class FooBar") == "DRIVER_FAILURE"
    assert ts._classify_skip_reason("capture driver raised: TypeError") == "DRIVER_FAILURE"


def test_classify_unknown_falls_through() -> None:
    ts = _import_module()
    assert ts._classify_skip_reason("") == "UNKNOWN"
    assert ts._classify_skip_reason("some completely opaque message") == "UNKNOWN"


def test_classify_is_case_insensitive() -> None:
    """Reason strings come from various components and capitalization is
    not consistent -- the classifier must not depend on it."""
    ts = _import_module()
    assert ts._classify_skip_reason("modulelist NO FORWARD") == "MODULELIST"
    assert ts._classify_skip_reason("MISSING REQUIRED POSITIONAL ARG") == "MISSING_ARG"


def test_classify_pins_all_sam2_v14_reasons() -> None:
    """Regression check: every actual SAM2 v14 skip-list reason must
    map to a known category (not UNKNOWN). If a refactor narrows the
    patterns and one of these slides to UNKNOWN, the orchestrator would
    silently leave a component stuck."""
    ts = _import_module()
    sam2_reasons = {
        "multi_scale_block": "harness: ModuleList no forward (v13)",
        "video_layer_norm": "harness: permute(sparse_coo) dim mismatch (v13)",
        "video_mask_down_sampler": "harness: permute(sparse_coo) dim mismatch (v13)",
        "video_mask_down_sampler_layer": "harness: ModuleList no forward (v13)",
        "video_memory_attention_layer": "harness: ModuleList no forward (v13)",
        "video_memory_encoder": "harness: missing 2 required positional args (v13)",
        "video_memory_fuser": "harness: groups=256 weight shape mismatch (v13)",
        "video_memory_fuser_c_x_block": "harness: ModuleList no forward (v13)",
        "video_position_embedding_sine": "harness: missing positional arg (v13)",
        "video_two_way_attention_block": "harness: ModuleList no forward (v13)",
    }
    for comp, reason in sam2_reasons.items():
        cat = ts._classify_skip_reason(reason)
        assert cat != "UNKNOWN", f"{comp}: {reason!r} fell to UNKNOWN (would leave component stuck)"


def test_remove_persistent_skip_keyed_per_entry(tmp_path, monkeypatch) -> None:
    """`remove_persistent_skip` must remove exactly one entry without
    touching the others. The all-or-nothing `clear_persistent_skips`
    would be too coarse for Phase 2 -- some components may still be
    genuinely untestable after one round."""
    om = _import_overlay_manager()
    # Point overlay_manager at tmp_path so the test doesn't touch the real
    # scripts/tt_hw_planner/overlays/ tree.
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path)
    model_id = "facebook/test-model"
    om.persist_skip(model_id, "comp_a", reason="harness: ModuleList no forward")
    om.persist_skip(model_id, "comp_b", reason="harness: missing positional arg")
    om.persist_skip(model_id, "comp_c", reason="harness: groups=8 weight shape mismatch")
    assert len(om.load_persistent_skips(model_id)) == 3

    removed = om.remove_persistent_skip(model_id, "comp_b")
    assert removed is True
    remaining = om.load_persistent_skips(model_id)
    assert set(remaining.keys()) == {"comp_a", "comp_c"}, "other entries must be preserved"

    # Removing a missing entry returns False (idempotent)
    assert om.remove_persistent_skip(model_id, "comp_b") is False

    # Removing the last entry deletes the file
    om.remove_persistent_skip(model_id, "comp_a")
    om.remove_persistent_skip(model_id, "comp_c")
    assert om.load_persistent_skips(model_id) == {}


def test_drop_component_files_backup_path(tmp_path) -> None:
    """ModuleList drop must move stub + test to _phase2_dropped/ rather
    than deleting them, so the action is reversible if the user changes
    their mind or the routing was wrong."""
    ts = _import_module()
    demo = tmp_path / "demo"
    (demo / "_stubs").mkdir(parents=True)
    (demo / "tests" / "pcc").mkdir(parents=True)
    stub = demo / "_stubs" / "multi_scale_block.py"
    test = demo / "tests" / "pcc" / "test_multi_scale_block.py"
    stub.write_text("# stub")
    test.write_text("# test")

    msgs = ts._drop_component_files(demo, "multi_scale_block", dry_run=False)
    assert not stub.is_file(), "stub should be moved"
    assert not test.is_file(), "test should be moved"
    backup = demo / "_phase2_dropped"
    assert (backup / "multi_scale_block.py.bak").is_file()
    assert (backup / "test_multi_scale_block.py.bak").is_file()
    assert len(msgs) == 2


def test_drop_component_files_dry_run_changes_nothing(tmp_path) -> None:
    ts = _import_module()
    demo = tmp_path / "demo"
    (demo / "_stubs").mkdir(parents=True)
    stub = demo / "_stubs" / "x.py"
    stub.write_text("# stub")

    msgs = ts._drop_component_files(demo, "x", dry_run=True)
    assert stub.is_file(), "dry-run must not move files"
    assert any("would back up" in m for m in msgs)


def test_drop_component_files_missing_files_noop(tmp_path) -> None:
    """If the per-component files were already deleted (manual cleanup,
    or the scaffold never emitted them), drop must not raise."""
    ts = _import_module()
    demo = tmp_path / "demo"
    demo.mkdir()
    msgs = ts._drop_component_files(demo, "phantom_component", dry_run=False)
    assert msgs == []


def test_cmd_tackle_skipped_empty_skip_list_returns_zero(tmp_path, monkeypatch, capsys) -> None:
    """With no skip-list, the command is a no-op success."""
    ts = _import_module()
    om = _import_overlay_manager()
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path)

    monkeypatch.setattr(
        "scripts.tt_hw_planner.bringup_loop.find_demo_dir",
        lambda _model_id: tmp_path / "demo",
    )

    args = argparse.Namespace(
        model_id="facebook/empty",
        dry_run=False,
        only_modulelist=False,
        only_capture=False,
    )
    rc = ts.cmd_tackle_skipped(args)
    assert rc == 0
    captured = capsys.readouterr()
    assert "nothing to tackle" in captured.out


def test_cmd_tackle_skipped_dry_run_does_not_modify(tmp_path, monkeypatch, capsys) -> None:
    """Dry-run must print the routing decisions without changing the
    skip-list or any files."""
    ts = _import_module()
    om = _import_overlay_manager()
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path)

    demo = tmp_path / "demo"
    (demo / "_stubs").mkdir(parents=True)
    stub = demo / "_stubs" / "multi_scale_block.py"
    stub.write_text("# stub")

    monkeypatch.setattr(
        "scripts.tt_hw_planner.bringup_loop.find_demo_dir",
        lambda _model_id: demo,
    )

    model_id = "facebook/dry"
    om.persist_skip(model_id, "multi_scale_block", reason="harness: ModuleList no forward (v13)")
    om.persist_skip(model_id, "video_memory_encoder", reason="harness: missing 2 required positional args (v13)")

    args = argparse.Namespace(
        model_id=model_id,
        dry_run=True,
        only_modulelist=False,
        only_capture=False,
    )
    rc = ts.cmd_tackle_skipped(args)
    captured = capsys.readouterr()
    assert "DRY RUN" in captured.out

    # Skip-list must be intact
    assert set(om.load_persistent_skips(model_id).keys()) == {"multi_scale_block", "video_memory_encoder"}
    # Files must be untouched
    assert stub.is_file()


def test_cmd_tackle_skipped_modulelist_drop(tmp_path, monkeypatch, capsys) -> None:
    """ModuleList components must get their files dropped and their
    skip-list entry removed -- without touching other-category entries."""
    ts = _import_module()
    om = _import_overlay_manager()
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path)

    demo = tmp_path / "demo"
    (demo / "_stubs").mkdir(parents=True)
    (demo / "tests" / "pcc").mkdir(parents=True)
    ml_stub = demo / "_stubs" / "multi_scale_block.py"
    ml_test = demo / "tests" / "pcc" / "test_multi_scale_block.py"
    other_stub = demo / "_stubs" / "video_memory_encoder.py"
    ml_stub.write_text("# ml")
    ml_test.write_text("# ml test")
    other_stub.write_text("# other")

    monkeypatch.setattr(
        "scripts.tt_hw_planner.bringup_loop.find_demo_dir",
        lambda _model_id: demo,
    )

    model_id = "facebook/test"
    om.persist_skip(model_id, "multi_scale_block", reason="harness: ModuleList no forward (v13)")
    om.persist_skip(model_id, "video_memory_encoder", reason="harness: missing 2 required positional args (v13)")

    args = argparse.Namespace(
        model_id=model_id,
        dry_run=False,
        only_modulelist=True,  # don't engage capture path for unit test
        only_capture=False,
    )
    rc = ts.cmd_tackle_skipped(args)

    # ModuleList side: dropped + removed from skip-list
    assert not ml_stub.is_file()
    assert not ml_test.is_file()
    assert (demo / "_phase2_dropped" / "multi_scale_block.py.bak").is_file()
    assert (demo / "_phase2_dropped" / "test_multi_scale_block.py.bak").is_file()

    # Other category side: untouched, still on skip-list
    assert other_stub.is_file()
    remaining = om.load_persistent_skips(model_id)
    assert "multi_scale_block" not in remaining
    assert "video_memory_encoder" in remaining


def test_cmd_tackle_skipped_only_capture_skips_modulelist(tmp_path, monkeypatch) -> None:
    """--only-capture flag must SKIP the ModuleList drop and only attempt
    the capture path. Symmetric with --only-modulelist."""
    ts = _import_module()
    om = _import_overlay_manager()
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path)
    demo = tmp_path / "demo"
    (demo / "_stubs").mkdir(parents=True)
    ml_stub = demo / "_stubs" / "multi_scale_block.py"
    ml_stub.write_text("# ml")
    monkeypatch.setattr(
        "scripts.tt_hw_planner.bringup_loop.find_demo_dir",
        lambda _model_id: demo,
    )
    model_id = "facebook/test"
    om.persist_skip(model_id, "multi_scale_block", reason="harness: ModuleList no forward")

    # Stub out the capture path so it doesn't actually try to invoke HF
    with mock.patch.object(ts, "_retry_capture", return_value=(False, "stub")):
        args = argparse.Namespace(
            model_id=model_id,
            dry_run=False,
            only_modulelist=False,
            only_capture=True,
        )
        rc = ts.cmd_tackle_skipped(args)

    # ModuleList side must be untouched because --only-capture was set
    assert ml_stub.is_file()
    assert "multi_scale_block" in om.load_persistent_skips(model_id)


def test_cmd_tackle_skipped_capture_success_removes_from_skip(tmp_path, monkeypatch) -> None:
    """If _retry_capture returns True for a MISSING_ARG entry, that
    component must be removed from the skip-list."""
    ts = _import_module()
    om = _import_overlay_manager()
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path)
    demo = tmp_path / "demo"
    demo.mkdir()
    monkeypatch.setattr(
        "scripts.tt_hw_planner.bringup_loop.find_demo_dir",
        lambda _model_id: demo,
    )

    model_id = "facebook/cap"
    om.persist_skip(model_id, "video_memory_encoder", reason="harness: missing 2 required positional args")

    with mock.patch.object(ts, "_retry_capture", return_value=(True, "capture-inputs rc=0")):
        args = argparse.Namespace(
            model_id=model_id,
            dry_run=False,
            only_modulelist=False,
            only_capture=True,
        )
        rc = ts.cmd_tackle_skipped(args)
    assert rc == 0, "all entries unblocked -> rc 0"
    assert om.load_persistent_skips(model_id) == {}


def test_run_phase2_stage_empty_skip_list(tmp_path, monkeypatch, capsys) -> None:
    """The Phase 2 stage callable returns (0, [], []) when no skips exist
    -- this is the success case after a Phase 1 run that captured everything."""
    ts = _import_module()
    om = _import_overlay_manager()
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path)
    rc, dropped, recaptured = ts.run_phase2_stage("facebook/empty", tmp_path / "demo")
    assert rc == 0
    assert dropped == []
    assert recaptured == []


def test_run_phase2_stage_modulelist_drop(tmp_path, monkeypatch) -> None:
    """Phase 2 stage must drop ModuleList files for real (not dry-run)
    when invoked from `up --phase2`. Distinguishes the CLI dry-run path
    from the stage callable."""
    ts = _import_module()
    om = _import_overlay_manager()
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path)
    demo = tmp_path / "demo"
    (demo / "_stubs").mkdir(parents=True)
    (demo / "tests" / "pcc").mkdir(parents=True)
    stub = demo / "_stubs" / "ml_comp.py"
    stub.write_text("# stub")
    test = demo / "tests" / "pcc" / "test_ml_comp.py"
    test.write_text("# test")

    model_id = "facebook/stage"
    om.persist_skip(model_id, "ml_comp", reason="harness: ModuleList no forward")

    rc, dropped, recaptured = ts.run_phase2_stage(model_id, demo)
    assert rc == 0, "stage should report success when all skips were handled"
    assert dropped == ["ml_comp"]
    assert recaptured == []
    assert not stub.is_file()
    assert (demo / "_phase2_dropped" / "ml_comp.py.bak").is_file()
    assert om.load_persistent_skips(model_id) == {}


def test_run_phase2_stage_recaptured_ok(tmp_path, monkeypatch) -> None:
    """When _retry_capture succeeds, the component appears in
    recaptured_ok (caller will then re-run PCC on it)."""
    ts = _import_module()
    om = _import_overlay_manager()
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path)
    demo = tmp_path / "demo"
    demo.mkdir()

    model_id = "facebook/stage2"
    om.persist_skip(model_id, "video_memory_encoder", reason="harness: missing 2 required positional args")

    with mock.patch.object(ts, "_retry_capture", return_value=(True, "ok")):
        rc, dropped, recaptured = ts.run_phase2_stage(model_id, demo)
    assert rc == 0
    assert recaptured == ["video_memory_encoder"]
    assert om.load_persistent_skips(model_id) == {}


def test_run_phase2_stage_failed_capture_nonzero_rc(tmp_path, monkeypatch) -> None:
    ts = _import_module()
    om = _import_overlay_manager()
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path)
    demo = tmp_path / "demo"
    demo.mkdir()

    model_id = "facebook/stage3"
    om.persist_skip(model_id, "video_memory_encoder", reason="harness: missing 2 required positional args")
    with mock.patch.object(ts, "_retry_capture", return_value=(False, "still broken")):
        rc, dropped, recaptured = ts.run_phase2_stage(model_id, demo)
    assert rc == 1, "stage rc must be non-zero when entries remain stuck"
    assert recaptured == []
    assert "video_memory_encoder" in om.load_persistent_skips(model_id)


def test_run_phase2_stage_unknown_category_nonzero_rc(tmp_path, monkeypatch) -> None:
    """Entries we can't classify must surface as non-zero rc -- not
    silently masked. Otherwise a new skip-reason pattern could silently
    accumulate stuck components."""
    ts = _import_module()
    om = _import_overlay_manager()
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path)
    demo = tmp_path / "demo"
    demo.mkdir()

    model_id = "facebook/stage_unknown"
    om.persist_skip(model_id, "weird_comp", reason="some entirely unrecognized failure pattern")
    rc, dropped, recaptured = ts.run_phase2_stage(model_id, demo)
    assert rc == 1
    assert dropped == []
    assert recaptured == []


def test_cmd_tackle_skipped_capture_failure_keeps_skip(tmp_path, monkeypatch) -> None:
    """If _retry_capture returns False, the component stays on the
    skip-list and the command exits non-zero (genuine remaining work)."""
    ts = _import_module()
    om = _import_overlay_manager()
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path)
    demo = tmp_path / "demo"
    demo.mkdir()
    monkeypatch.setattr(
        "scripts.tt_hw_planner.bringup_loop.find_demo_dir",
        lambda _model_id: demo,
    )

    model_id = "facebook/cap2"
    om.persist_skip(model_id, "video_memory_encoder", reason="harness: missing 2 required positional args")

    with mock.patch.object(ts, "_retry_capture", return_value=(False, "capture-inputs rc=1")):
        args = argparse.Namespace(
            model_id=model_id,
            dry_run=False,
            only_modulelist=False,
            only_capture=True,
        )
        rc = ts.cmd_tackle_skipped(args)
    assert rc != 0, "stuck entry -> non-zero rc"
    assert "video_memory_encoder" in om.load_persistent_skips(model_id)
