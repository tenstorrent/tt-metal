"""Unit tests for the 3-category placement model.

Every NEW non-structural component lands in exactly one of:
  HOT             — invoked in workload, on TT device
  COLD            — not invoked in workload, on CPU
  KERNEL_MISSING  — invoked, TTNN op verified missing, on CPU

Placement is INTRINSIC to category:
  HOT             → "device"
  COLD            → "cpu"
  KERNEL_MISSING  → "cpu"

ModuleList containers (no_emit list) are EXCLUDED — not a category."""

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


def test_report_has_exactly_three_categories() -> None:
    """The dataclass has hot, cold, kernel_missing (plus structural_excluded
    which is NOT a category — just a side list)."""
    fc = _fc()
    fields = set(fc.CategorizationReport.__dataclass_fields__)
    assert fields == {"hot", "cold", "kernel_missing", "structural_excluded"}, fields


def test_runtime_target_intrinsic_to_category() -> None:
    """Placement is determined by the category — no separate flag."""
    fc = _fc()
    report = fc.CategorizationReport(
        hot=["a"],
        cold=["b"],
        kernel_missing=["c"],
    )
    assert report.runtime_target("a") == "device"
    assert report.runtime_target("b") == "cpu"
    assert report.runtime_target("c") == "cpu"
    assert report.runtime_target("not_in_report") is None


def test_graduated_routes_to_hot(tmp_path, monkeypatch) -> None:
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
    assert sorted(report.hot) == ["grad_a", "grad_b"]
    assert report.cold == []
    assert report.kernel_missing == []


def test_skip_list_with_cold_category(tmp_path, monkeypatch) -> None:
    fc = _fc()
    om = _om()
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path / "overlays")
    demo = tmp_path / "demo"
    _make_demo_with_components(demo, ["not_invoked_comp"])
    om.persist_skip("test/m", "not_invoked_comp", reason="not invoked in workload", category="COLD")

    report = fc.build_final_categorization(
        model_id="test/m",
        demo_dir=demo,
        graduated_set=set(),
    )
    assert report.cold == ["not_invoked_comp"]
    assert report.hot == []
    assert report.kernel_missing == []


def test_skip_list_with_kernel_missing_category(tmp_path, monkeypatch) -> None:
    fc = _fc()
    om = _om()
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path / "overlays")
    demo = tmp_path / "demo"
    _make_demo_with_components(demo, ["missing_op_comp"])
    om.persist_skip(
        "test/m",
        "missing_op_comp",
        reason="harness: TT_FATAL: ttnn.permute(sparse_coo) not implemented",
        category="KERNEL_MISSING",
    )

    report = fc.build_final_categorization(
        model_id="test/m",
        demo_dir=demo,
        graduated_set=set(),
    )
    assert report.kernel_missing == ["missing_op_comp"]
    assert report.cold == []
    assert report.hot == []


def test_no_emit_components_are_structural_not_categorized(tmp_path, monkeypatch) -> None:
    """ModuleList containers (on no_emit list) are STRUCTURAL EXCLUSIONS,
    not a category. They appear in structural_excluded, not in any of
    hot/cold/kernel_missing."""
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
    assert report.hot == []
    assert report.cold == []
    assert report.kernel_missing == []
    # And no runtime_target — not a graduation unit
    assert report.runtime_target("modulelist_container") is None


def test_skip_list_without_category_defaults_to_cold(tmp_path, monkeypatch) -> None:
    """Backwards compat: a skip-list entry without an explicit category
    defaults to COLD. The user can run classify-hot-cold or detect
    kernel-missing to upgrade the category if needed."""
    fc = _fc()
    om = _om()
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path / "overlays")
    demo = tmp_path / "demo"
    _make_demo_with_components(demo, ["legacy_comp"])
    # Don't pass category — defaults to COLD via persist_skip's default arg
    om.persist_skip("test/m", "legacy_comp", reason="harness incompatible")

    report = fc.build_final_categorization(
        model_id="test/m",
        demo_dir=demo,
        graduated_set=set(),
    )
    assert report.cold == ["legacy_comp"]


def test_no_hot_stuck_or_unclassified_buckets() -> None:
    """Pin: the dataclass MUST NOT have hot_stuck or unclassified fields."""
    fc = _fc()
    fields = set(fc.CategorizationReport.__dataclass_fields__)
    assert "hot_stuck" not in fields
    assert "unclassified" not in fields
    assert "dropped" not in fields


def test_skip_list_graduates_to_hot_takes_precedence(tmp_path, monkeypatch) -> None:
    """If a component is somehow on BOTH the graduated set AND the
    skip-list (stale data), the graduated set wins. (But the categorizer
    subtracts skip-list from graduated_set first, so the comp lands in
    its skip-list category — defensive against stale graduated_set.)"""
    fc = _fc()
    om = _om()
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path / "overlays")
    demo = tmp_path / "demo"
    _make_demo_with_components(demo, ["stale"])
    om.persist_skip("test/m", "stale", reason="stale entry", category="COLD")

    report = fc.build_final_categorization(
        model_id="test/m",
        demo_dir=demo,
        graduated_set={"stale"},  # stale graduated_set
    )
    # Subtracted: stale lands in COLD per its skip-list entry
    assert report.cold == ["stale"]
    assert report.hot == []


def test_summary_lists_three_categories() -> None:
    fc = _fc()
    report = fc.CategorizationReport(
        hot=["a"],
        cold=["b"],
        kernel_missing=["c"],
    )
    summary = fc.format_categorization_summary(report)
    assert "HOT" in summary
    assert "COLD" in summary
    assert "KERNEL_MISSING" in summary
    # No HOT_STUCK / UNCLASSIFIED / DROPPED
    assert "HOT_STUCK" not in summary
    assert "UNCLASSIFIED" not in summary
    assert "DROPPED" not in summary


def test_summary_shows_structural_when_present() -> None:
    fc = _fc()
    report = fc.CategorizationReport(
        hot=["a"],
        structural_excluded=["modulelist_a"],
    )
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
        "test/m", "needs_op", reason="ttnn.scaled_dot_product_attention(causal=True) missing", category="KERNEL_MISSING"
    )

    report = fc.build_final_categorization(model_id="test/m", demo_dir=demo)
    msg = fc.format_kernel_gap_report("test/m", report)
    assert "needs_op" in msg
    assert "scaled_dot_product_attention" in msg
    assert "TTNN OPERATIONS NEEDED" in msg


def test_kernel_gap_report_empty_when_none() -> None:
    fc = _fc()
    report = fc.CategorizationReport(hot=["a"])
    msg = fc.format_kernel_gap_report("test/m", report)
    assert msg == ""


def test_cmd_up_uses_3_category_categorization() -> None:
    """Source-grep: cmd_up imports the new categorization function and
    formats with the 3-category framing."""
    cli_mod = importlib.import_module("scripts.tt_hw_planner.cli")
    src = (Path(cli_mod.__file__)).read_text()
    assert "build_final_categorization(" in src
    assert "format_categorization_summary" in src
    assert "format_kernel_gap_report" in src
    # The old gate-failure machinery is gone
    assert "format_gate_failure" not in src
    assert "can_emit_demo" not in src


def test_persist_skip_accepts_category(tmp_path, monkeypatch) -> None:
    """The persist_skip function accepts a category argument and stores
    it in the JSON entry. This is how the 3 categories are encoded into
    skip-list data."""
    om = _om()
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path)
    om.persist_skip("test/m", "x", reason="r", category="KERNEL_MISSING")
    listing = om.load_persistent_skips("test/m")
    assert listing["x"]["category"] == "KERNEL_MISSING"


def test_persist_skip_defaults_category_to_cold(tmp_path, monkeypatch) -> None:
    """A persist_skip call without an explicit category defaults to COLD
    (safest interpretation when we lack workload-invocation signal)."""
    om = _om()
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path)
    om.persist_skip("test/m", "y", reason="r")
    listing = om.load_persistent_skips("test/m")
    assert listing["y"]["category"] == "COLD"


def test_hot_cold_signal_promotes_skip_cold_to_kernel_missing(tmp_path, monkeypatch) -> None:
    """Bug-1 fix: if hot_cold.json says HOT and skip-list says COLD,
    the categorizer must promote to KERNEL_MISSING. A HOT component on
    CPU fallback is either TTNN dev work or tool gap — should NOT be
    silently classified as COLD."""
    fc = _fc()
    om = _om()
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path / "overlays")
    demo = tmp_path / "demo"
    _make_demo_with_components(demo, ["hot_but_skipped_cold"])
    # Skip-list says COLD (default category for a tool that gave up)
    om.persist_skip("test/m", "hot_but_skipped_cold", reason="tool gave up", category="COLD")
    # But workload probe says HOT
    om.persist_hot_cold("test/m", {"hot_but_skipped_cold": "HOT"})

    report = fc.build_final_categorization(
        model_id="test/m",
        demo_dir=demo,
        graduated_set=set(),
    )
    # Workload signal wins: HOT component should be flagged as KERNEL_MISSING
    assert report.kernel_missing == ["hot_but_skipped_cold"]
    assert report.cold == []


def test_hot_cold_hot_without_skip_routes_to_kernel_missing(tmp_path, monkeypatch) -> None:
    """If hot_cold says HOT and the component isn't graduated AND isn't
    on skip-list, that means the tool didn't finish. Treat as KERNEL_MISSING
    (the tool surfacing a need-for-investigation case)."""
    fc = _fc()
    om = _om()
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path / "overlays")
    demo = tmp_path / "demo"
    _make_demo_with_components(demo, ["hot_pending"])
    om.persist_hot_cold("test/m", {"hot_pending": "HOT"})

    report = fc.build_final_categorization(
        model_id="test/m",
        demo_dir=demo,
        graduated_set=set(),
    )
    assert report.kernel_missing == ["hot_pending"]
    assert report.cold == []


def test_persist_skip_upgrades_cold_to_kernel_missing(tmp_path, monkeypatch) -> None:
    """Bug-3 fix: an early persist_skip with default COLD must not block
    a later persist_skip with KERNEL_MISSING for the same component.
    Auto-iterate's seed-phase persists default COLD; _skip_component_to_fallback
    later may detect kernel-missing and need to upgrade the entry."""
    om = _om()
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path)
    # First persist (seed-phase style — default COLD)
    om.persist_skip("test/m", "x", reason="harness incompatible")
    assert om.load_persistent_skips("test/m")["x"]["category"] == "COLD"

    # Second persist (kernel-missing detection upgrades category)
    om.persist_skip("test/m", "x", reason="ttnn.foo verified missing", category="KERNEL_MISSING")
    listing = om.load_persistent_skips("test/m")
    assert listing["x"]["category"] == "KERNEL_MISSING", (
        "Bug-3: COLD → KERNEL_MISSING upgrade must work; previously the " "early-return on duplicate blocked it."
    )
    # Reason also updated
    assert "verified missing" in listing["x"]["reason"]


def test_persist_skip_does_not_downgrade_kernel_missing(tmp_path, monkeypatch) -> None:
    """The reverse direction: a KERNEL_MISSING entry must NOT be
    downgraded to COLD by a later persist. (TTNN op verification is
    intentional; we don't want a default-COLD call to undo it.)"""
    om = _om()
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path)
    om.persist_skip("test/m", "x", reason="ttnn.foo missing", category="KERNEL_MISSING")
    # Try to "downgrade" to COLD
    om.persist_skip("test/m", "x", reason="harness incompat", category="COLD")
    assert om.load_persistent_skips("test/m")["x"]["category"] == "KERNEL_MISSING"


def test_persist_skip_preserves_captured_ts_on_update(tmp_path, monkeypatch) -> None:
    """The captured_ts is the audit-trail timestamp — preserved across
    updates so we know when the component was ORIGINALLY flagged."""
    om = _om()
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path)
    om.persist_skip("test/m", "x", reason="r1")
    ts_first = om.load_persistent_skips("test/m")["x"]["captured_ts"]
    # Tick forward
    import time as _t

    _t.sleep(0.001)
    om.persist_skip("test/m", "x", reason="r2", category="KERNEL_MISSING")
    ts_second = om.load_persistent_skips("test/m")["x"]["captured_ts"]
    assert ts_first == ts_second, "captured_ts must survive category/reason updates (audit-trail)"


def test_hot_cold_cold_without_skip_routes_to_cold(tmp_path, monkeypatch) -> None:
    """If hot_cold says COLD and the component isn't graduated, it
    correctly lands in COLD (no need to be on skip-list — workload
    probe is sufficient signal)."""
    fc = _fc()
    om = _om()
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path / "overlays")
    demo = tmp_path / "demo"
    _make_demo_with_components(demo, ["cold_pending"])
    om.persist_hot_cold("test/m", {"cold_pending": "COLD"})

    report = fc.build_final_categorization(
        model_id="test/m",
        demo_dir=demo,
        graduated_set=set(),
    )
    assert report.cold == ["cold_pending"]
    assert report.kernel_missing == []
