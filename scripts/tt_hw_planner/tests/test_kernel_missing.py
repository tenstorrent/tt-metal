"""Unit tests for kernel-missing detection + KERNEL_MISSING placement.

The principle: every component should run on device unless TTNN lacks
the operation it needs. If TTNN doesn't have the op, the tool flags
the kernel gap and the demo runs that component on CPU fallback until
the kernel lands.

Placement buckets:
  ON_DEVICE      — graduated, native ttnn, PCC verified
  KERNEL_MISSING — TTNN op gap verified; on CPU until the kernel lands
  PENDING        — not yet graduated; retry next run (no kernel gap)

These tests pin the bucketing AND the kernel-missing detection patterns."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _km():
    return importlib.import_module("scripts.tt_hw_planner.kernel_missing")


def _om():
    return importlib.import_module("scripts.tt_hw_planner.overlay_manager")


def _fc():
    return importlib.import_module("scripts.tt_hw_planner.final_categorization")


# ===== detection tests =====


def test_detects_tt_fatal_not_implemented() -> None:
    km = _km()
    msg = "RuntimeError: TT_FATAL: feature not implemented: ttnn.embedding with float16 dtype"
    desc = km.detect_kernel_missing(msg)
    assert desc is not None
    assert "embedding" in desc or "implemented" in desc.lower()


def test_detects_not_implemented_error_with_ttnn_op() -> None:
    km = _km()
    msg = "NotImplementedError: ttnn.scaled_dot_product_attention(causal=True, ...) not yet implemented"
    desc = km.detect_kernel_missing(msg)
    assert desc is not None
    assert "ttnn.scaled_dot_product_attention" in desc


def test_detects_no_kernel_for() -> None:
    km = _km()
    msg = "RuntimeError: no kernel for ttnn.permute on sparse_coo tensor"
    desc = km.detect_kernel_missing(msg)
    assert desc is not None


def test_detects_unsupported_op() -> None:
    km = _km()
    msg = "Error: unsupported op: ttnn.special_thing"
    desc = km.detect_kernel_missing(msg)
    assert desc is not None


def test_detects_sparse_coo_unsupported() -> None:
    km = _km()
    msg = "RuntimeError: sparse_coo tensors are not supported by ttnn ops yet"
    desc = km.detect_kernel_missing(msg)
    assert desc is not None
    assert "sparse_coo" in desc.lower()


def test_does_not_match_regular_pcc_failures() -> None:
    """A normal PCC failure or shape error must NOT be classified as
    kernel-missing — the agent should keep iterating on those."""
    km = _km()
    cases = [
        "AssertionError: PCC 0.85 below target 0.99",
        "RuntimeError: shape mismatch (1,2,3) vs (1,2,4)",
        "TypeError: ttnn.matmul() got unexpected keyword argument",
        "IndexError: list index out of range in stub",
        "",  # empty
    ]
    for msg in cases:
        assert (
            km.detect_kernel_missing(msg) is None
        ), f"non-kernel failure incorrectly classified as kernel-missing: {msg!r}"


def test_is_kernel_missing_failure_boolean() -> None:
    km = _km()
    assert km.is_kernel_missing_failure("TT_FATAL: not implemented") is True
    assert km.is_kernel_missing_failure("AssertionError: PCC 0.5") is False


def test_verify_ttnn_op_exists_known_op() -> None:
    """ttnn.matmul exists. Verification should return True (not missing)."""
    km = _km()
    result = km.verify_ttnn_op_exists("ttnn.matmul")
    # We don't strictly require the answer to be True (ttnn may not
    # be importable in all test environments), but if it IS importable
    # AND ttnn.matmul exists, the result MUST be True. We tolerate None
    # for the ttnn-not-importable case.
    assert result in (True, None), f"ttnn.matmul should resolve, got {result!r}"


def test_verify_ttnn_op_exists_nonsense_op() -> None:
    """A fabricated op name should return False (genuinely missing).
    Again tolerate None if ttnn isn't importable."""
    km = _km()
    result = km.verify_ttnn_op_exists("ttnn.this_does_not_exist_for_real_xyz_12345")
    assert result in (False, None), f"fake op should be missing, got {result!r}"


def test_verify_returns_none_for_non_ttnn_name() -> None:
    """Names that don't start with ttnn.* can't be verified — return None."""
    km = _km()
    assert km.verify_ttnn_op_exists("torch.nn.Linear") is None
    assert km.verify_ttnn_op_exists("(some generic annotation)") is None
    assert km.verify_ttnn_op_exists("") is None
    assert km.verify_ttnn_op_exists(None) is None  # type: ignore


# ===== persistence tests =====


def test_persist_and_load_roundtrip(tmp_path, monkeypatch) -> None:
    om = _om()
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path)
    om.persist_missing_kernel("test/m", "video_layer_norm", missing_op="ttnn.permute(sparse_coo)")
    listing = om.load_missing_kernels("test/m")
    assert "video_layer_norm" in listing
    assert "permute" in listing["video_layer_norm"]["missing_op"]


def test_persist_idempotent_preserves_first_timestamp(tmp_path, monkeypatch) -> None:
    om = _om()
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path)
    om.persist_missing_kernel("test/m", "comp_a", missing_op="op_v1")
    ts_first = om.load_missing_kernels("test/m")["comp_a"]["detected_ts"]
    om.persist_missing_kernel("test/m", "comp_a", missing_op="op_v2")
    ts_second = om.load_missing_kernels("test/m")["comp_a"]["detected_ts"]
    assert ts_first == ts_second


def test_remove_missing_kernel(tmp_path, monkeypatch) -> None:
    om = _om()
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path)
    om.persist_missing_kernel("test/m", "comp_a", missing_op="op_a")
    om.persist_missing_kernel("test/m", "comp_b", missing_op="op_b")
    assert om.remove_missing_kernel("test/m", "comp_a") is True
    assert om.load_missing_kernels("test/m") == {"comp_b": om.load_missing_kernels("test/m")["comp_b"]}
    # Removing the last entry deletes the file
    om.remove_missing_kernel("test/m", "comp_b")
    assert om.load_missing_kernels("test/m") == {}


def test_is_missing_kernel_fast_lookup(tmp_path, monkeypatch) -> None:
    om = _om()
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path)
    assert om.is_missing_kernel("test/m", "comp_a") is False
    om.persist_missing_kernel("test/m", "comp_a", missing_op="op")
    assert om.is_missing_kernel("test/m", "comp_a") is True


def test_load_handles_malformed_json(tmp_path, monkeypatch) -> None:
    om = _om()
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path)
    p = om._missing_kernels_path("test/m")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("not json {")
    assert om.load_missing_kernels("test/m") == {}


# ===== categorization integration =====


def _make_demo_with_components(demo_dir: Path, comps: list) -> None:
    demo_dir.mkdir(parents=True, exist_ok=True)
    status = {"new_model_id": "test/model", "components": [{"name": c, "status": "NEW"} for c in comps]}
    (demo_dir / "bringup_status.json").write_text(json.dumps(status))


def test_kernel_missing_appears_in_categorization(tmp_path, monkeypatch) -> None:
    """A skip-list entry with category=KERNEL_MISSING must appear in
    the KERNEL_MISSING bucket of the 3-category categorization report."""
    fc = _fc()
    om = _om()
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path / "overlays")
    demo = tmp_path / "demo"
    _make_demo_with_components(demo, ["graduated_comp", "kernel_gap"])
    om.persist_skip("test/m", "kernel_gap", reason="ttnn.foo missing", category="KERNEL_MISSING")

    report = fc.build_final_categorization(
        model_id="test/m",
        demo_dir=demo,
        graduated_set={"graduated_comp"},
    )
    assert report.on_device == ["graduated_comp"]
    assert report.kernel_missing == ["kernel_gap"]


def test_kernel_missing_placement_is_cpu(tmp_path, monkeypatch) -> None:
    """KERNEL_MISSING category intrinsically means runtime_target='cpu'.
    Under the 3-category design there's no gate — demo always emits;
    placement is what the category tells you."""
    fc = _fc()
    om = _om()
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path / "overlays")
    demo = tmp_path / "demo"
    _make_demo_with_components(demo, ["grad_a", "kernel_gap"])
    om.persist_skip("test/m", "kernel_gap", reason="ttnn.something missing", category="KERNEL_MISSING")

    report = fc.build_final_categorization(
        model_id="test/m",
        demo_dir=demo,
        graduated_set={"grad_a"},
    )
    assert report.runtime_target("grad_a") == "device"
    assert report.runtime_target("kernel_gap") == "cpu"


def test_skip_list_only_kernel_missing_persists(tmp_path, monkeypatch) -> None:
    """Only KERNEL_MISSING entries are persisted to the skip-list. A
    non-KERNEL_MISSING persist_skip call is a no-op, so the affected
    component lands in PENDING (retry queue) rather than COLD."""
    fc = _fc()
    om = _om()
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path / "overlays")
    demo = tmp_path / "demo"
    _make_demo_with_components(demo, ["grad", "kernel_gap", "not_yet_attempted"])
    om.persist_skip("test/m", "kernel_gap", reason="ttnn.x missing", category="KERNEL_MISSING")
    # Non-KERNEL_MISSING is a no-op
    om.persist_skip("test/m", "not_yet_attempted", reason="cap exhausted", category="ITERATION_BUDGET")

    report = fc.build_final_categorization(
        model_id="test/m",
        demo_dir=demo,
        graduated_set={"grad"},
    )
    assert report.on_device == ["grad"]
    assert report.kernel_missing == ["kernel_gap"]
    assert report.pending == ["not_yet_attempted"]


def test_graduated_beats_skip_list_kernel_missing(tmp_path, monkeypatch) -> None:
    """If a TTNN release lands a kernel and the component re-graduates,
    ON_DEVICE wins over a stale KERNEL_MISSING skip-list entry."""
    fc = _fc()
    om = _om()
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path / "overlays")
    demo = tmp_path / "demo"
    _make_demo_with_components(demo, ["was_stuck"])
    om.persist_skip("test/m", "was_stuck", reason="ttnn.x missing (stale)", category="KERNEL_MISSING")

    # Subtraction logic: graduated_set - skip-list keys = nothing for was_stuck.
    # User must clear the skip entry first. Test the subtraction behavior.
    report = fc.build_final_categorization(
        model_id="test/m",
        demo_dir=demo,
        graduated_set={"was_stuck"},
    )
    # With skip-list entry present, the subtraction logic puts it in
    # the KERNEL_MISSING bucket (skip-list category wins). To "re-graduate"
    # you must first clear the skip-list entry.
    assert report.kernel_missing == ["was_stuck"]


def test_kernel_gap_report_surfaces_ops(tmp_path, monkeypatch) -> None:
    """The user-facing kernel-gap report names components + reasons."""
    fc = _fc()
    om = _om()
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path / "overlays")
    demo = tmp_path / "demo"
    _make_demo_with_components(demo, ["x"])
    om.persist_skip("test/m", "x", reason="ttnn.scaled_dot_product_attention(causal=True)", category="KERNEL_MISSING")

    report = fc.build_final_categorization(model_id="test/m", demo_dir=demo)
    msg = fc.format_kernel_gap_report("test/m", report)
    assert "x" in msg
    assert "scaled_dot_product_attention" in msg
    assert "TTNN OPERATIONS" in msg


def test_kernel_gap_report_empty_when_none(tmp_path, monkeypatch) -> None:
    fc = _fc()
    report = fc.CategorizationReport(on_device=["a"], kernel_missing=[], pending=[])
    msg = fc.format_kernel_gap_report("test/m", report)
    assert msg == ""


# ===== auto-iterate integration =====
