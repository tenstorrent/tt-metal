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


def test_kernel_gap_report_splits_missing_op_from_constraint(tmp_path, monkeypatch) -> None:
    """Bug-J regression: after CONSTRAINT_MISMATCH joined the
    kernel_missing placement bucket, the gap report must split the
    two cases so the TTNN dev planner sees them distinctly. Missing-op
    components need a new op; constraint-mismatch components need
    extended dtype/layout/shape coverage on an existing op."""
    fc = _fc()
    om = _om()
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path / "overlays")
    demo = tmp_path / "demo"
    _make_demo_with_components(demo, ["missing_op_comp", "constraint_comp"])
    om.persist_skip("test/m", "missing_op_comp", reason="ttnn.foo missing", category="KERNEL_MISSING")
    om.persist_skip(
        "test/m", "constraint_comp", reason="ttnn.conv2d float16 unsupported", category="CONSTRAINT_MISMATCH"
    )

    report = fc.build_final_categorization(model_id="test/m", demo_dir=demo)
    msg = fc.format_kernel_gap_report("test/m", report)

    # Both sections must appear and be labeled distinctly.
    assert "TTNN OPERATIONS NEEDED" in msg
    assert "TTNN CONSTRAINT EXTENSIONS NEEDED" in msg

    # Each component must show up under its OWN section.
    missing_idx = msg.find("TTNN OPERATIONS NEEDED")
    constraint_idx = msg.find("TTNN CONSTRAINT EXTENSIONS NEEDED")
    assert missing_idx >= 0 and constraint_idx >= 0
    # missing_op_comp belongs under "OPERATIONS NEEDED"
    # constraint_comp belongs under "CONSTRAINT EXTENSIONS NEEDED"
    # Either order is fine; just verify the two are in DIFFERENT sections.
    sec_a_start = min(missing_idx, constraint_idx)
    sec_b_start = max(missing_idx, constraint_idx)
    section_a = msg[sec_a_start:sec_b_start]
    section_b = msg[sec_b_start:]
    if "TTNN OPERATIONS NEEDED" in section_a:
        assert "missing_op_comp" in section_a
        assert "constraint_comp" in section_b
    else:
        assert "missing_op_comp" in section_b
        assert "constraint_comp" in section_a


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


def test_persist_skip_kernel_missing_is_sticky(tmp_path, monkeypatch) -> None:
    """Once a component has category=KERNEL_MISSING (a verified TTNN
    gap), later persist calls with a different category MUST NOT
    downgrade it. The gap doesn't disappear just because a later run's
    failure trace points elsewhere."""
    om = _om()
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path)
    om.persist_skip("test/m", "x", reason="initial", category="KERNEL_MISSING")
    om.persist_skip("test/m", "x", reason="later", category="COLD")
    listing = om.load_persistent_skips("test/m")
    assert listing["x"]["category"] == "KERNEL_MISSING"
    om.persist_skip("test/m", "x", reason="later2", category="TOOL_BUG")
    assert om.load_persistent_skips("test/m")["x"]["category"] == "KERNEL_MISSING"


def test_persist_skip_accepts_expanded_categories(tmp_path, monkeypatch) -> None:
    """The expanded category schema (CONSTRAINT_MISMATCH, TOOL_BUG,
    HF_ERROR, ITERATION_BUDGET, AGENT_STUCK) must round-trip through
    persist_skip / load_persistent_skips intact."""
    om = _om()
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path)
    for cat in ("CONSTRAINT_MISMATCH", "TOOL_BUG", "HF_ERROR", "ITERATION_BUDGET", "AGENT_STUCK"):
        om.persist_skip("test/m", f"comp_{cat.lower()}", reason="r", category=cat)
    listing = om.load_persistent_skips("test/m")
    for cat in ("CONSTRAINT_MISMATCH", "TOOL_BUG", "HF_ERROR", "ITERATION_BUDGET", "AGENT_STUCK"):
        assert listing[f"comp_{cat.lower()}"]["category"] == cat


def test_persist_skip_non_kernel_categories_can_be_updated(tmp_path, monkeypatch) -> None:
    """A non-KERNEL_MISSING entry can be overwritten with a different
    non-KERNEL_MISSING category at the same specificity level (e.g.
    ITERATION_BUDGET -> AGENT_STUCK when the next run hits a different
    signal). Both are specificity=2 so newer wins."""
    om = _om()
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path)
    om.persist_skip("test/m", "x", reason="r1", category="ITERATION_BUDGET")
    om.persist_skip("test/m", "x", reason="r2", category="AGENT_STUCK")
    listing = om.load_persistent_skips("test/m")
    assert listing["x"]["category"] == "AGENT_STUCK"


def test_persist_skip_specific_category_not_downgraded_to_cold(tmp_path, monkeypatch) -> None:
    """Bug-F regression: A specific category (TOOL_BUG, HF_ERROR, ...)
    must NOT be overwritten by a less-specific COLD on a later persist.
    Without this guard, a generic default-COLD persist (e.g. from the
    seed-phase fallback) would silently erase the diagnostic label."""
    om = _om()
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path)
    for specific in ("TOOL_BUG", "HF_ERROR", "CONSTRAINT_MISMATCH", "ITERATION_BUDGET", "AGENT_STUCK"):
        comp = f"c_{specific.lower()}"
        om.persist_skip("test/m", comp, reason="initial", category=specific)
        # Later generic default-COLD persist (e.g. seed-phase fallback)
        om.persist_skip("test/m", comp, reason="later", category="COLD")
        listing = om.load_persistent_skips("test/m")
        assert listing[comp]["category"] == specific, f"{specific} category should NOT have been downgraded to COLD"


def test_persist_skip_cold_upgraded_to_specific(tmp_path, monkeypatch) -> None:
    """Forward direction of the specificity ladder: a generic COLD CAN
    be upgraded to a specific category (TOOL_BUG, KERNEL_MISSING, ...)
    by a later run with better signal."""
    om = _om()
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path)
    om.persist_skip("test/m", "c1", reason="initial", category="COLD")
    om.persist_skip("test/m", "c1", reason="later", category="TOOL_BUG")
    assert om.load_persistent_skips("test/m")["c1"]["category"] == "TOOL_BUG"


def test_persist_skip_specific_upgradeable_to_kernel_missing(tmp_path, monkeypatch) -> None:
    """Specificity ladder top: any category can be upgraded to
    KERNEL_MISSING when verification confirms the TTNN gap."""
    om = _om()
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path)
    om.persist_skip("test/m", "c1", reason="initial", category="CONSTRAINT_MISMATCH")
    om.persist_skip("test/m", "c1", reason="later", category="KERNEL_MISSING")
    assert om.load_persistent_skips("test/m")["c1"]["category"] == "KERNEL_MISSING"


def test_cold_bucket_means_workload_uninvoked_only(tmp_path, monkeypatch) -> None:
    """COLD bucket is reserved for workload-uninvoked components ONLY.
    A HOT-by-workload component with category=TOOL_BUG/HF_ERROR/
    ITERATION_BUDGET/AGENT_STUCK MUST NOT land in COLD — it belongs to
    kernel_missing so the operator sees it's stuck-but-needed."""
    fc = _fc()
    om = _om()
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path / "overlays")
    demo = tmp_path / "demo"
    _make_demo_with_components(demo, ["hot_tool_bug", "hot_hf_error", "hot_iter_budget", "hot_agent_stuck"])
    for c, cat in [
        ("hot_tool_bug", "TOOL_BUG"),
        ("hot_hf_error", "HF_ERROR"),
        ("hot_iter_budget", "ITERATION_BUDGET"),
        ("hot_agent_stuck", "AGENT_STUCK"),
    ]:
        om.persist_skip("test/m", c, reason=f"stuck via {cat}", category=cat)
    # Workload probe: all four are HOT
    om.persist_hot_cold(
        "test/m", {c: "HOT" for c in ("hot_tool_bug", "hot_hf_error", "hot_iter_budget", "hot_agent_stuck")}
    )

    report = fc.build_final_categorization(model_id="test/m", demo_dir=demo)
    for c in ("hot_tool_bug", "hot_hf_error", "hot_iter_budget", "hot_agent_stuck"):
        assert c not in report.cold, f"{c} is HOT-by-workload — must NOT be in COLD bucket"
        assert c in report.kernel_missing, f"{c} is HOT-by-workload + stuck — must be in kernel_missing"


def test_cold_workload_routes_to_cold_regardless_of_skip_category(tmp_path, monkeypatch) -> None:
    """If the workload probe says COLD, the component is genuinely
    not invoked — it belongs in COLD bucket no matter what the
    skip-list category says (TOOL_BUG would have been irrelevant
    anyway since the workload didn't try to invoke it)."""
    fc = _fc()
    om = _om()
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path / "overlays")
    demo = tmp_path / "demo"
    _make_demo_with_components(demo, ["uninvoked"])
    om.persist_skip("test/m", "uninvoked", reason="harness issue", category="TOOL_BUG")
    om.persist_hot_cold("test/m", {"uninvoked": "COLD"})

    report = fc.build_final_categorization(model_id="test/m", demo_dir=demo)
    assert "uninvoked" in report.cold
    assert "uninvoked" not in report.kernel_missing


def test_no_workload_signal_falls_back_to_skip_category(tmp_path, monkeypatch) -> None:
    """When hot_cold.json doesn't classify a component (probe didn't
    run or didn't reach it), fall back to the skip-list category.
    TTNN-gap categories go to kernel_missing; everything else COLD."""
    fc = _fc()
    om = _om()
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path / "overlays")
    demo = tmp_path / "demo"
    _make_demo_with_components(demo, ["no_signal_kmiss", "no_signal_tool"])
    om.persist_skip("test/m", "no_signal_kmiss", reason="ttnn.foo missing", category="KERNEL_MISSING")
    om.persist_skip("test/m", "no_signal_tool", reason="harness issue", category="TOOL_BUG")
    # NO hot_cold persist — leave the probe unspoken

    report = fc.build_final_categorization(model_id="test/m", demo_dir=demo)
    assert "no_signal_kmiss" in report.kernel_missing
    assert "no_signal_tool" in report.cold  # safe-default for unknown


def test_categorizer_routes_constraint_mismatch_to_kernel_missing_bucket(tmp_path, monkeypatch) -> None:
    """CONSTRAINT_MISMATCH (op exists but dtype/layout/shape failed) is
    a TTNN gap from the user's perspective — must surface in the
    KERNEL_MISSING placement bucket."""
    fc = _fc()
    om = _om()
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path / "overlays")
    demo = tmp_path / "demo"
    _make_demo_with_components(demo, ["constraint_comp"])
    om.persist_skip("test/m", "constraint_comp", reason="dtype issue", category="CONSTRAINT_MISMATCH")
    report = fc.build_final_categorization(model_id="test/m", demo_dir=demo)
    assert "constraint_comp" in report.kernel_missing
    assert "constraint_comp" not in report.cold


def test_categorizer_routes_tool_bug_to_cold_bucket(tmp_path, monkeypatch) -> None:
    """TOOL_BUG is a scaffolder-side issue, NOT a TTNN gap. Component
    still lands on CPU but in the COLD bucket (not KERNEL_MISSING) so
    it doesn't pollute the TTNN dev team's flag list."""
    fc = _fc()
    om = _om()
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path / "overlays")
    demo = tmp_path / "demo"
    _make_demo_with_components(demo, ["tool_bug_comp"])
    om.persist_skip("test/m", "tool_bug_comp", reason="harness issue", category="TOOL_BUG")
    report = fc.build_final_categorization(model_id="test/m", demo_dir=demo)
    assert "tool_bug_comp" in report.cold
    assert "tool_bug_comp" not in report.kernel_missing


def test_categorizer_routes_iteration_budget_to_cold_bucket(tmp_path, monkeypatch) -> None:
    """ITERATION_BUDGET — the loop ran out of attempts. CPU placement
    is correct for THIS run; the detailed category in the skip-list
    enables retry next run (auto-iterate auto-loads non-permanent
    skip categories)."""
    fc = _fc()
    om = _om()
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path / "overlays")
    demo = tmp_path / "demo"
    _make_demo_with_components(demo, ["budget_comp"])
    om.persist_skip("test/m", "budget_comp", reason="cap", category="ITERATION_BUDGET")
    report = fc.build_final_categorization(model_id="test/m", demo_dir=demo)
    assert "budget_comp" in report.cold


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
