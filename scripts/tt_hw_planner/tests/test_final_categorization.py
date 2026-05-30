"""Unit tests for the placement model.

Every NEW non-structural component lands in exactly one of:
  ON_DEVICE       — graduated to native ttnn, PCC verified
  KERNEL_MISSING  — skip-list entry with verified missing TTNN op
  PENDING         — not yet graduated, retry next run

ModuleList containers (no_emit list) are STRUCTURAL EXCLUSIONS — not a
placement category.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _fc():
    return importlib.import_module("scripts.tt_hw_planner.final_categorization")


def _om():
    return importlib.import_module("scripts.tt_hw_planner.overlay_manager")


def _make_demo_with_components(demo_dir: Path, comps: list) -> None:
    demo_dir.mkdir(parents=True, exist_ok=True)
    status = {
        "new_model_id": "test/model",
        "components": [{"name": c, "status": "NEW"} for c in comps],
    }
    (demo_dir / "bringup_status.json").write_text(json.dumps(status))


def test_report_has_three_placement_fields() -> None:
    fc = _fc()
    fields = set(fc.CategorizationReport.__dataclass_fields__)
    assert fields == {"on_device", "kernel_missing", "pending", "structural_excluded"}, fields


def test_runtime_target_for_on_device_is_device() -> None:
    fc = _fc()
    report = fc.CategorizationReport(on_device=["a"], kernel_missing=["b"], pending=["c"])
    assert report.runtime_target("a") == "device"
    assert report.runtime_target("b") == "cpu"
    assert report.runtime_target("c") is None
    assert report.runtime_target("not_in_report") is None


def test_graduated_routes_to_on_device(tmp_path, monkeypatch) -> None:
    fc = _fc()
    om = _om()
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path / "overlays")
    demo = tmp_path / "demo"
    _make_demo_with_components(demo, ["grad_a", "grad_b"])

    report = fc.build_final_categorization(
        model_id="test/m",
        demo_dir=demo,
        graduated_set={"grad_a", "grad_b"},
    )
    assert sorted(report.on_device) == ["grad_a", "grad_b"]
    assert report.kernel_missing == []
    assert report.pending == []


def test_kernel_missing_skip_routes_to_kernel_missing(tmp_path, monkeypatch) -> None:
    fc = _fc()
    om = _om()
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path / "overlays")
    demo = tmp_path / "demo"
    _make_demo_with_components(demo, ["missing_op_comp"])
    om.persist_skip(
        "test/m",
        "missing_op_comp",
        reason="TT_FATAL: ttnn.permute(sparse_coo) not implemented",
        category="KERNEL_MISSING",
    )

    report = fc.build_final_categorization(
        model_id="test/m",
        demo_dir=demo,
        graduated_set=set(),
    )
    assert report.kernel_missing == ["missing_op_comp"]
    assert report.on_device == []
    assert report.pending == []


def test_non_kernel_missing_skip_is_dropped_and_component_pending(tmp_path, monkeypatch) -> None:
    """Only KERNEL_MISSING persists. Other categories are silently dropped
    by persist_skip, so the component is treated as never-skipped and
    lands in PENDING for the next run."""
    fc = _fc()
    om = _om()
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path / "overlays")
    demo = tmp_path / "demo"
    _make_demo_with_components(demo, ["tool_bug_comp"])
    om.persist_skip("test/m", "tool_bug_comp", reason="harness issue", category="TOOL_BUG")

    # persist_skip should have been a no-op for TOOL_BUG — verify nothing
    # got written.
    assert "tool_bug_comp" not in om.load_persistent_skips("test/m")

    report = fc.build_final_categorization(
        model_id="test/m",
        demo_dir=demo,
        graduated_set=set(),
    )
    assert report.pending == ["tool_bug_comp"]
    assert report.kernel_missing == []
    assert report.on_device == []


def test_no_emit_components_are_structural_not_categorized(tmp_path, monkeypatch) -> None:
    fc = _fc()
    om = _om()
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path / "overlays")
    demo = tmp_path / "demo"
    _make_demo_with_components(demo, ["modulelist_container"])
    om.persist_no_emit_test("test/m", "modulelist_container", reason="ModuleList drop")

    report = fc.build_final_categorization(
        model_id="test/m",
        demo_dir=demo,
        graduated_set=set(),
    )
    assert report.structural_excluded == ["modulelist_container"]
    assert report.on_device == []
    assert report.kernel_missing == []
    assert report.pending == []
    assert report.runtime_target("modulelist_container") is None


def test_summary_lists_three_placement_buckets() -> None:
    fc = _fc()
    report = fc.CategorizationReport(on_device=["a"], kernel_missing=["b"], pending=["c"])
    summary = fc.format_categorization_summary(report)
    assert "ON_DEVICE" in summary
    assert "KERNEL_MISSING" in summary
    assert "PENDING" in summary
    # HOT/COLD vocabulary is gone
    assert "HOT" not in summary
    assert "COLD" not in summary


def test_summary_shows_structural_when_present() -> None:
    fc = _fc()
    report = fc.CategorizationReport(on_device=["a"], structural_excluded=["modulelist_a"])
    summary = fc.format_categorization_summary(report)
    assert "modulelist_a" in summary
    assert "structural" in summary.lower()


def test_kernel_gap_report_lists_components_with_reasons(tmp_path, monkeypatch) -> None:
    fc = _fc()
    om = _om()
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path / "overlays")
    demo = tmp_path / "demo"
    _make_demo_with_components(demo, ["needs_op"])
    om.persist_skip(
        "test/m",
        "needs_op",
        reason="ttnn.scaled_dot_product_attention(causal=True) missing",
        category="KERNEL_MISSING",
    )

    report = fc.build_final_categorization(model_id="test/m", demo_dir=demo)
    msg = fc.format_kernel_gap_report("test/m", report)
    assert "needs_op" in msg
    assert "scaled_dot_product_attention" in msg
    assert "TTNN OPERATIONS NEEDED" in msg


def test_kernel_gap_report_empty_when_none() -> None:
    fc = _fc()
    report = fc.CategorizationReport(on_device=["a"])
    msg = fc.format_kernel_gap_report("test/m", report)
    assert msg == ""


def test_persist_skip_persists_only_kernel_missing(tmp_path, monkeypatch) -> None:
    """Verify the new persist_skip rule: only KERNEL_MISSING writes a
    skip-list entry; every other category is a no-op (component stays
    in the retry queue)."""
    om = _om()
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path)

    # KERNEL_MISSING persists
    om.persist_skip("test/m", "kernel_comp", reason="ttnn.foo missing", category="KERNEL_MISSING")
    assert "kernel_comp" in om.load_persistent_skips("test/m")

    # Every other category is dropped
    for cat in ("COLD", "TOOL_BUG", "HF_ERROR", "CONSTRAINT_MISMATCH", "ITERATION_BUDGET", "AGENT_STUCK"):
        comp = f"comp_{cat.lower()}"
        om.persist_skip("test/m", comp, reason="r", category=cat)
        assert comp not in om.load_persistent_skips("test/m"), f"category={cat} should not persist"


def test_persist_skip_kernel_missing_can_update_reason(tmp_path, monkeypatch) -> None:
    """A second KERNEL_MISSING persist updates the reason; captured_ts
    survives so the audit trail is intact."""
    om = _om()
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path)
    om.persist_skip("test/m", "x", reason="r1", category="KERNEL_MISSING")
    ts_first = om.load_persistent_skips("test/m")["x"]["captured_ts"]
    om.persist_skip("test/m", "x", reason="r2 with more detail", category="KERNEL_MISSING")
    listing = om.load_persistent_skips("test/m")
    assert listing["x"]["reason"] == "r2 with more detail"
    assert listing["x"]["captured_ts"] == ts_first


def test_cmd_up_uses_placement_summary() -> None:
    """Source-grep: cmd_up still imports the categorization helpers."""
    cli_mod = importlib.import_module("scripts.tt_hw_planner.cli")
    src = Path(cli_mod.__file__).read_text()
    assert "build_final_categorization(" in src
    assert "format_categorization_summary" in src
    assert "format_kernel_gap_report" in src
