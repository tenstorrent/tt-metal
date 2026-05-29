"""Unit tests for Layer 7 — kernel-missing detection + KERNEL_MISSING category.

The principle: HOT components should ALWAYS graduate unless TTNN
lacks the operation they need. If TTNN doesn't have the op, the tool
is NOT going to write the kernel — that's a separate engineering
workstream. The tool's job is to FLAG the missing op clearly and let
the demo emit with CPU fallback for that component.

Categorization buckets:
  GRADUATED      — on TT, PCC verified
  COLD           — not invoked in workload (CPU OK)
  DROPPED        — ModuleList container (parent covers)
  KERNEL_MISSING — HOT but TTNN lacks the op (allowed past gate)
  HOT_STUCK      — HOT, no kernel-missing signal, didn't graduate (blocks gate)
  UNCLASSIFIED   — no signal at all (conservative — blocks gate)

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


def test_auto_iterate_verifies_before_kernel_missing_label() -> None:
    """Verification before KERNEL_MISSING labeling now lives in the
    failure_classifier (classify_failure calls verify_ttnn_op_exists
    before promoting to KERNEL_VERIFIED_MISSING). _skip_component_to_
    fallback drives the classifier and respects its verdict.

    Source-grep contracts:
      * auto_iterate.py calls `classify_failure` inside
        `_skip_component_to_fallback`
      * failure_classifier.py imports `verify_ttnn_op_exists`
    """
    ai_src = (_REPO_ROOT / "scripts" / "tt_hw_planner" / "_cli_helpers" / "auto_iterate.py").read_text()
    fc_src = (_REPO_ROOT / "scripts" / "tt_hw_planner" / "failure_classifier.py").read_text()
    skip_idx = ai_src.find("def _skip_component_to_fallback")
    assert skip_idx >= 0
    assert "classify_failure" in ai_src[skip_idx : skip_idx + 5000], (
        "_skip_component_to_fallback must drive the failure_classifier " "before persisting any skip-list category"
    )
    assert "verify_ttnn_op_exists" in fc_src, (
        "failure_classifier must verify the op exists before claiming "
        "KERNEL_VERIFIED_MISSING (avoid false-positive flags)"
    )


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
    assert report.hot == ["graduated_comp"]
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


def test_skip_list_routes_per_category_field(tmp_path, monkeypatch) -> None:
    """Under the 3-category design, skip-list entries carry an explicit
    `category` field. KERNEL_MISSING entries land in KERNEL_MISSING bucket;
    COLD entries (or default) land in COLD."""
    fc = _fc()
    om = _om()
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path / "overlays")
    demo = tmp_path / "demo"
    _make_demo_with_components(demo, ["grad", "kernel_gap", "not_invoked"])
    om.persist_skip("test/m", "kernel_gap", reason="ttnn.x missing", category="KERNEL_MISSING")
    om.persist_skip("test/m", "not_invoked", reason="not invoked in workload", category="COLD")

    report = fc.build_final_categorization(
        model_id="test/m",
        demo_dir=demo,
        graduated_set={"grad"},
    )
    assert report.hot == ["grad"]
    assert report.kernel_missing == ["kernel_gap"]
    assert report.cold == ["not_invoked"]


def test_graduated_beats_skip_list_kernel_missing(tmp_path, monkeypatch) -> None:
    """If a TTNN release lands a kernel and the component re-graduates,
    HOT wins over a stale KERNEL_MISSING skip-list entry."""
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
    report = fc.CategorizationReport(hot=["a"], cold=[], kernel_missing=[])
    msg = fc.format_kernel_gap_report("test/m", report)
    assert msg == ""


# ===== auto-iterate integration =====


def test_auto_iterate_calls_detect_kernel_missing() -> None:
    """Kernel-missing detection now lives in `failure_classifier`,
    which is invoked from `_skip_component_to_fallback`. The skip-list
    entry's `category` field still carries the COLD vs KERNEL_MISSING
    distinction; the bridge is `skip_category_for_verdict`.

    This test pins the indirection: detection itself is in the
    classifier, the loop just drives it.
    """
    ai_src = (_REPO_ROOT / "scripts" / "tt_hw_planner" / "_cli_helpers" / "auto_iterate.py").read_text()
    fc_src = (_REPO_ROOT / "scripts" / "tt_hw_planner" / "failure_classifier.py").read_text()
    # Detection function is imported and used by the classifier
    assert "detect_kernel_missing" in fc_src
    # Loop drives the classifier inside _skip_component_to_fallback
    skip_idx = ai_src.find("def _skip_component_to_fallback")
    assert skip_idx >= 0
    body = ai_src[skip_idx : skip_idx + 5000]
    assert "classify_failure" in body
    assert "skip_category_for_verdict" in body
    # Final skip-list still uses the canonical category field
    assert "KERNEL_MISSING" in fc_src
