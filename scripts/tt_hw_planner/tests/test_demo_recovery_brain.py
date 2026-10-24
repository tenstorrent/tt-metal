"""Tests for the brain's end-to-end demo recovery primitive.

When the framework's demo verification (pytest demo.py::test_demo)
fails, the brain decides whether to retry, clean up interference, or
give up. The brain owns the recovery decision — the caller executes it."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.tt_hw_planner.agentic.demo_recovery import (
    DemoRecoveryVerdict,
    archive_demo_files,
    decide_demo_recovery,
    detect_stale_demo_sibling,
    parse_failing_wired_component,
    remove_component_from_wiring,
)


# ---------------------------------------------------------------------------
# Stale-sibling detection
# ---------------------------------------------------------------------------


def test_detects_stale_root_level_demo(tmp_path: Path) -> None:
    """The exact SAM2 case: canonical demo lives at demo/demo.py, but
    a stale demo.py is at the model demo dir root. The brain detects it."""
    demo_dir = tmp_path / "model_x"
    (demo_dir / "demo").mkdir(parents=True)
    canonical = demo_dir / "demo" / "demo.py"
    canonical.write_text("# canonical\n")
    stale = demo_dir / "demo.py"
    stale.write_text("# stale orphan\n")

    found = detect_stale_demo_sibling(demo_dir=demo_dir, canonical_demo=canonical)

    assert found == stale


def test_no_stale_sibling_when_only_canonical_exists(tmp_path: Path) -> None:
    demo_dir = tmp_path / "model_x"
    (demo_dir / "demo").mkdir(parents=True)
    canonical = demo_dir / "demo" / "demo.py"
    canonical.write_text("# canonical\n")

    found = detect_stale_demo_sibling(demo_dir=demo_dir, canonical_demo=canonical)

    assert found is None


def test_canonical_at_root_does_not_self_detect(tmp_path: Path) -> None:
    """If the canonical demo IS at demo_dir/demo.py (no subfolder),
    the brain doesn't mistake it for stale."""
    demo_dir = tmp_path / "model_x"
    demo_dir.mkdir()
    canonical = demo_dir / "demo.py"
    canonical.write_text("# canonical at root\n")

    found = detect_stale_demo_sibling(demo_dir=demo_dir, canonical_demo=canonical)

    assert found is None


# ---------------------------------------------------------------------------
# Recovery decision flow
# ---------------------------------------------------------------------------


def test_brain_decides_archive_and_retry_on_stale_sibling(tmp_path: Path) -> None:
    """When a stale sibling is detected on first failure, brain says
    archive_and_retry."""
    demo_dir = tmp_path / "model_x"
    (demo_dir / "demo").mkdir(parents=True)
    canonical = demo_dir / "demo" / "demo.py"
    canonical.write_text("# canonical\n")
    (demo_dir / "demo.py").write_text("# stale\n")

    v = decide_demo_recovery(
        demo_dir=demo_dir,
        canonical_demo=canonical,
        retries_attempted=0,
    )

    assert v.action == "archive_and_retry"
    assert len(v.archive_paths) == 1
    assert v.archive_paths[0].name == "demo.py"
    assert "stale sibling" in v.reason


def test_brain_falls_back_to_disable_last_wired_when_no_match(tmp_path: Path) -> None:
    """When the parser can't pinpoint a broken component from the
    pytest tail but wired_components is non-empty, the brain falls
    back to brute-force: disable the LAST wired component (most
    suspect — recently added) and retry."""
    demo_dir = tmp_path / "model_x"
    (demo_dir / "demo").mkdir(parents=True)
    canonical = demo_dir / "demo" / "demo.py"
    canonical.write_text("# canonical\n")

    v = decide_demo_recovery(
        demo_dir=demo_dir,
        canonical_demo=canonical,
        retries_attempted=0,
        max_retries=2,
        pytest_tail="RuntimeError: TT_FATAL @ layernorm.cpp:80 ...",  # opaque
        wired_components=["safe_a", "safe_b", "suspect_c"],
    )

    assert v.action == "disable_component_and_retry"
    assert getattr(v, "broken_component", None) == "suspect_c"
    assert "brute-force" in v.reason


def test_brain_retries_on_flake_when_no_wired(tmp_path: Path) -> None:
    """No wired components available → brain falls back to plain retry
    for flake protection."""
    demo_dir = tmp_path / "model_x"
    (demo_dir / "demo").mkdir(parents=True)
    canonical = demo_dir / "demo" / "demo.py"
    canonical.write_text("# canonical\n")

    v = decide_demo_recovery(
        demo_dir=demo_dir,
        canonical_demo=canonical,
        retries_attempted=0,
        max_retries=2,
        wired_components=[],
    )

    assert v.action == "retry"
    assert v.archive_paths == []
    assert "flake protection" in v.reason


def test_brain_gives_up_after_max_retries(tmp_path: Path) -> None:
    """After max_retries, brain surfaces the failure instead of looping
    forever. The PCC suite already proved the model is on device."""
    demo_dir = tmp_path / "model_x"
    (demo_dir / "demo").mkdir(parents=True)
    canonical = demo_dir / "demo" / "demo.py"
    canonical.write_text("# canonical\n")

    v = decide_demo_recovery(
        demo_dir=demo_dir,
        canonical_demo=canonical,
        retries_attempted=1,
        max_retries=1,
    )

    assert v.action == "give_up"
    assert "PCC suite already proved" in v.reason


# ---------------------------------------------------------------------------
# Archive action
# ---------------------------------------------------------------------------


def test_archive_demo_files_renames_to_stale(tmp_path: Path) -> None:
    f = tmp_path / "demo.py"
    f.write_text("# stale\n")

    archived = archive_demo_files([f])

    assert len(archived) == 1
    assert archived[0].name == "demo.py.stale_demo_sibling"
    assert archived[0].is_file()
    assert not f.exists()


def test_archive_demo_files_idempotent(tmp_path: Path) -> None:
    """If the archive name already exists, leave the original alone
    (no clobber)."""
    f = tmp_path / "demo.py"
    f.write_text("# orig\n")
    existing_archive = tmp_path / "demo.py.stale_demo_sibling"
    existing_archive.write_text("# prior archive\n")

    archived = archive_demo_files([f])

    assert archived == []
    assert f.read_text() == "# orig\n"  # untouched
    assert existing_archive.read_text() == "# prior archive\n"  # untouched


def test_archive_demo_files_handles_missing(tmp_path: Path) -> None:
    """If a path doesn't exist, archive_demo_files silently skips it."""
    archived = archive_demo_files([tmp_path / "nonexistent.py"])
    assert archived == []


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


# Removed 2026-06-04: test_cli_wires_brain_demo_recovery_on_failure
# tested wiring inside _emit_and_verify_runnable_demo, which has been
# deleted along with the rest of the smoke-test demo machinery.


def test_verdict_shape() -> None:
    v = DemoRecoveryVerdict(action="retry", archive_paths=[], reason="test")
    assert v.action == "retry"
    assert v.archive_paths == []
    assert v.reason == "test"


# ---------------------------------------------------------------------------
# Diagnostic extraction (the bug we saw on 2026-05-30: the demo error was
# hidden behind nanobind leak messages because the cli only kept the last
# 12 lines of pytest output)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Identify which wired component is broken from pytest failure tail
# ---------------------------------------------------------------------------


def test_parse_failing_wired_component_from_sam2_tail() -> None:
    """The exact SAM2 case: TT_FATAL on layer_norm during demo runtime;
    the failure trace mentions `video_layer_norm` (the wired component
    whose `_apply_*` raised). Brain identifies it."""
    tail = """
        return ttnn.layer_norm(x_tiled, weight=self.w_weight, bias=self.w_bias, ...)
    File "models/demos/vision/segmentation/sam2_hiera_tiny/_stubs/video_layer_norm.py", line 42
RuntimeError: TT_FATAL @ layernorm_device_operation.cpp:80: a.logical_shape()[-1] == gamma.value().logical_shape()[-1]
FAILED models/demos/vision/segmentation/sam2_hiera_tiny/demo/demo.py::test_demo
""".strip()
    wired = [
        "decoder_head",
        "prompt_encoder_config",
        "encoder_stack",
        "video_layer_norm",
        "video_memory_fuser",
    ]
    found = parse_failing_wired_component(pytest_tail=tail, wired_components=wired)
    assert found == "video_layer_norm"


def test_parse_prefers_stub_path_over_substring_match() -> None:
    """If the tail mentions multiple wired components but one appears
    in a `_stubs/<comp>.py` traceback frame, prefer that one. Otherwise
    a substring match elsewhere could pick the wrong component."""
    tail = (
        '  File "models/demos/X/_stubs/component_b.py", line 88\n'
        "RuntimeError: TT_FATAL\n"
        "  ... also mentions component_a somewhere\n"
        "FAILED test_demo\n"
    )
    wired = ["component_a", "component_b"]
    found = parse_failing_wired_component(pytest_tail=tail, wired_components=wired)
    assert found == "component_b"


def test_parse_matches_import_path() -> None:
    """Traceback may include the import path `_stubs.<comp>` rather
    than the file path."""
    tail = (
        "RuntimeError: TT_FATAL\n"
        "    raised in models.demos.X._stubs.video_layer_norm.LayerNormStub.__call__\n"
        "FAILED test_demo\n"
    )
    wired = ["video_layer_norm", "encoder_stack"]
    found = parse_failing_wired_component(pytest_tail=tail, wired_components=wired)
    assert found == "video_layer_norm"


def test_parse_returns_none_when_no_failure_markers() -> None:
    """If tail doesn't have TT_FATAL / RuntimeError / CPU_FALLBACK,
    the brain shouldn't randomly pick a component."""
    tail = "PASSED\n  some_component looks fine"
    found = parse_failing_wired_component(
        pytest_tail=tail,
        wired_components=["some_component", "other"],
    )
    assert found is None


def test_parse_returns_none_when_no_match() -> None:
    """Failure exists but none of the wired component names appear → None."""
    tail = "RuntimeError: TT_FATAL @ allocator.cpp:23"
    found = parse_failing_wired_component(
        pytest_tail=tail,
        wired_components=["comp_a", "comp_b"],
    )
    assert found is None


def test_parse_handles_empty_inputs() -> None:
    assert parse_failing_wired_component(pytest_tail="", wired_components=["x"]) is None
    assert parse_failing_wired_component(pytest_tail="error", wired_components=[]) is None


# ---------------------------------------------------------------------------
# Remove component from demo wiring
# ---------------------------------------------------------------------------


def test_remove_component_from_wiring(tmp_path: Path) -> None:
    demo = tmp_path / "demo.py"
    demo.write_text(
        "WIRED_COMPONENTS = [\n"
        "    ('a', 'pkg._stubs.foo', 'foo'),\n"
        "    ('b', 'pkg._stubs.bar', 'bar'),\n"
        "    ('c', 'pkg._stubs.baz', 'baz'),\n"
        "]\n"
    )
    changed = remove_component_from_wiring(demo_path=demo, component="bar")
    assert changed is True
    new_src = demo.read_text()
    assert "'foo'" in new_src
    assert "'bar'" not in new_src
    assert "'baz'" in new_src


def test_remove_component_no_op_when_missing(tmp_path: Path) -> None:
    demo = tmp_path / "demo.py"
    demo.write_text("WIRED_COMPONENTS = [\n" "    ('a', 'pkg._stubs.foo', 'foo'),\n" "]\n")
    changed = remove_component_from_wiring(demo_path=demo, component="notpresent")
    assert changed is False


# ---------------------------------------------------------------------------
# Full recovery flow: disable broken component, retry, then give up
# ---------------------------------------------------------------------------


def test_brain_decides_disable_when_broken_component_identifiable(tmp_path: Path) -> None:
    """End-to-end: pytest tail identifies broken component → brain
    decides to disable it."""
    demo_dir = tmp_path / "model_x"
    (demo_dir / "demo").mkdir(parents=True)
    canonical = demo_dir / "demo" / "demo.py"
    canonical.write_text("# canonical\n")

    tail = "RuntimeError: TT_FATAL on layer_norm in `video_layer_norm` ..."
    v = decide_demo_recovery(
        demo_dir=demo_dir,
        canonical_demo=canonical,
        retries_attempted=0,
        max_retries=2,
        pytest_tail=tail,
        wired_components=["video_layer_norm", "encoder_stack"],
    )
    assert v.action == "disable_component_and_retry"
    assert getattr(v, "broken_component", None) == "video_layer_norm"
    assert "video_layer_norm" in v.reason
    assert "mixed mode" in v.reason
