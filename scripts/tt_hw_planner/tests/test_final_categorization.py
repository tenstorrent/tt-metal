"""Unit tests for the placement model.

Every NEW component lands in exactly one of:
  ON_DEVICE       — graduated to native ttnn, PCC verified
  KERNEL_MISSING  — skip-list entry with verified missing TTNN op
  PENDING         — not yet graduated, retry next run

A decomposed parent (no_emit list) earns ON_DEVICE only by passing its
OWN test. While split it is PENDING — children graduating does not credit
it; the recompose path restores it as a whole-module target once its
children are on device. A parent blocked by a kernel-missing child rolls
up as KERNEL_MISSING. There is no separate "structural" bucket.
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


def test_report_has_placement_fields() -> None:
    fc = _fc()
    fields = set(fc.CategorizationReport.__dataclass_fields__)
    assert fields == {"on_device", "kernel_missing", "pending", "cpu_reuse"}, fields


def test_runtime_target_for_on_device_is_device() -> None:
    fc = _fc()
    report = fc.CategorizationReport(on_device=["a"], kernel_missing=["b"], pending=["c"], cpu_reuse=["d"])
    assert report.runtime_target("a") == "device"
    assert report.runtime_target("b") == "cpu"
    assert report.runtime_target("d") == "cpu"
    assert report.runtime_target("c") is None
    assert report.runtime_target("not_in_report") is None


def test_bare_reuse_not_wired_routes_to_cpu_reuse_not_on_device(tmp_path, monkeypatch) -> None:
    fc = _fc()
    om = _om()
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path / "overlays")
    demo = tmp_path / "demo"
    demo.mkdir(parents=True)
    (demo / "_stubs").mkdir()
    status = {
        "new_model_id": "test/m",
        "components": [
            {"name": "grad", "status": "NEW", "submodule_path": "s0"},
            {"name": "reuse_bare", "status": "REUSE", "tt_reuse_target": "models/common/rmsnorm.py"},
            {"name": "adapt_bare", "status": "ADAPT", "tt_reuse_target": "models/tt_transformers/tt/attn.py"},
        ],
    }
    (demo / "bringup_status.json").write_text(json.dumps(status))

    report = fc.build_final_categorization(model_id="test/m", demo_dir=demo, graduated_set={"grad"})
    assert report.on_device == ["grad"], report.on_device
    assert sorted(report.cpu_reuse) == ["adapt_bare", "reuse_bare"], report.cpu_reuse


def test_wired_reuse_target_routes_to_on_device(tmp_path, monkeypatch) -> None:
    fc = _fc()
    om = _om()
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path / "overlays")
    demo = tmp_path / "demo"
    (demo / "_stubs").mkdir(parents=True)
    (demo / "tt").mkdir()
    (demo / "tt" / "rmsnorm.py").write_text("# sibling ttnn module copied into this demo\n")
    status = {
        "new_model_id": "test/m",
        "components": [
            {"name": "reuse_wired", "status": "REUSE", "tt_reuse_target": "models/common/rmsnorm.py"},
        ],
    }
    (demo / "bringup_status.json").write_text(json.dumps(status))

    report = fc.build_final_categorization(model_id="test/m", demo_dir=demo, graduated_set=set())
    assert report.on_device == ["reuse_wired"], report.on_device
    assert report.cpu_reuse == [], report.cpu_reuse


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


def _make_demo_with_raw_components(demo_dir: Path, comps: list, plan: list = None) -> None:
    demo_dir.mkdir(parents=True, exist_ok=True)
    status = {"new_model_id": "test/model", "components": comps}
    (demo_dir / "bringup_status.json").write_text(json.dumps(status))
    if plan is not None:
        (demo_dir / "decomposition_plan.json").write_text(json.dumps(plan))


def test_no_emit_parent_without_plan_is_pending(tmp_path, monkeypatch) -> None:
    fc = _fc()
    om = _om()
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path / "overlays")
    demo = tmp_path / "demo"
    _make_demo_with_raw_components(demo, [{"name": "parent", "status": "NEW", "submodule_path": "parent"}])
    om.persist_no_emit_test("test/m", "parent", reason="decomposition consumer split parent into children")

    report = fc.build_final_categorization(model_id="test/m", demo_dir=demo, graduated_set=set())
    assert report.pending == ["parent"]
    assert report.on_device == []


def test_no_emit_parent_pending_even_when_all_children_graduated(tmp_path, monkeypatch) -> None:
    """Children graduating does NOT credit the parent — it stays PENDING
    until it passes its own recomposed test. But it IS reported ready to
    recompose so the loop restores it as a whole-module target."""
    fc = _fc()
    om = _om()
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path / "overlays")
    demo = tmp_path / "demo"
    _make_demo_with_raw_components(
        demo,
        [
            {"name": "parent", "status": "NEW", "submodule_path": "blk"},
            {"name": "child_a", "status": "NEW", "submodule_path": "blk.a"},
            {"name": "child_b", "status": "NEW", "submodule_path": "blk.b"},
        ],
        plan=[{"parent_name": "parent", "children": [{"submodule_path": "blk.a"}, {"submodule_path": "blk.b"}]}],
    )
    om.persist_no_emit_test("test/m", "parent", reason="decomposition consumer split parent into children")

    report = fc.build_final_categorization(model_id="test/m", demo_dir=demo, graduated_set={"child_a", "child_b"})
    assert sorted(report.on_device) == ["child_a", "child_b"]
    assert report.pending == ["parent"]

    ready = fc.parents_ready_to_recompose(model_id="test/m", demo_dir=demo, graduated_set={"child_a", "child_b"})
    assert ready == ["parent"]


def test_parent_ready_to_recompose_with_reuse_child(tmp_path, monkeypatch) -> None:
    fc = _fc()
    om = _om()
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path / "overlays")
    demo = tmp_path / "demo"
    _make_demo_with_raw_components(
        demo,
        [
            {"name": "parent", "status": "NEW", "submodule_path": "blk"},
            {"name": "child_a", "status": "NEW", "submodule_path": "blk.a"},
            {"name": "child_b", "status": "REUSE", "submodule_path": "blk.b"},
        ],
        plan=[{"parent_name": "parent", "children": [{"submodule_path": "blk.a"}, {"submodule_path": "blk.b"}]}],
    )
    om.persist_no_emit_test("test/m", "parent", reason="decomposition consumer split parent into children")

    ready = fc.parents_ready_to_recompose(model_id="test/m", demo_dir=demo, graduated_set={"child_a"})
    assert ready == ["parent"], f"expected REUSE child to count as on-device; got {ready!r}"


def test_parent_not_ready_to_recompose_when_a_child_pending(tmp_path, monkeypatch) -> None:
    fc = _fc()
    om = _om()
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path / "overlays")
    demo = tmp_path / "demo"
    _make_demo_with_raw_components(
        demo,
        [
            {"name": "parent", "status": "NEW", "submodule_path": "blk"},
            {"name": "child_a", "status": "NEW", "submodule_path": "blk.a"},
            {"name": "child_b", "status": "NEW", "submodule_path": "blk.b"},
        ],
        plan=[{"parent_name": "parent", "children": [{"submodule_path": "blk.a"}, {"submodule_path": "blk.b"}]}],
    )
    om.persist_no_emit_test("test/m", "parent", reason="decomposition consumer split parent into children")

    ready = fc.parents_ready_to_recompose(model_id="test/m", demo_dir=demo, graduated_set={"child_a"})
    assert ready == []


def test_no_emit_parent_kernel_missing_when_child_blocked(tmp_path, monkeypatch) -> None:
    """A decomposed parent whose child is blocked by a missing TTNN op rolls
    up as KERNEL_MISSING (not PENDING) — it cannot complete until the op lands."""
    fc = _fc()
    om = _om()
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path / "overlays")
    demo = tmp_path / "demo"
    _make_demo_with_raw_components(
        demo,
        [
            {"name": "parent", "status": "NEW", "submodule_path": "blk"},
            {"name": "child_a", "status": "NEW", "submodule_path": "blk.a"},
            {"name": "child_b", "status": "NEW", "submodule_path": "blk.b"},
        ],
        plan=[{"parent_name": "parent", "children": [{"submodule_path": "blk.a"}, {"submodule_path": "blk.b"}]}],
    )
    om.persist_no_emit_test("test/m", "parent", reason="decomposition consumer split parent into children")
    om.persist_skip("test/m", "child_b", reason="ttnn.foo missing", category="KERNEL_MISSING")

    report = fc.build_final_categorization(model_id="test/m", demo_dir=demo, graduated_set={"child_a"})
    assert "parent" in report.kernel_missing
    assert "child_b" in report.kernel_missing
    assert "child_a" in report.on_device


def test_no_emit_parent_pending_when_a_child_not_graduated(tmp_path, monkeypatch) -> None:
    fc = _fc()
    om = _om()
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path / "overlays")
    demo = tmp_path / "demo"
    _make_demo_with_raw_components(
        demo,
        [
            {"name": "parent", "status": "NEW", "submodule_path": "blk"},
            {"name": "child_a", "status": "NEW", "submodule_path": "blk.a"},
            {"name": "child_b", "status": "NEW", "submodule_path": "blk.b"},
        ],
        plan=[{"parent_name": "parent", "children": [{"submodule_path": "blk.a"}, {"submodule_path": "blk.b"}]}],
    )
    om.persist_no_emit_test("test/m", "parent", reason="decomposition consumer split parent into children")

    report = fc.build_final_categorization(model_id="test/m", demo_dir=demo, graduated_set={"child_a"})
    assert "parent" in report.pending
    assert "child_a" in report.on_device
    assert "child_b" in report.pending


def test_no_emit_parent_pending_when_a_child_branch_was_never_ported(tmp_path, monkeypatch) -> None:
    fc = _fc()
    om = _om()
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path / "overlays")
    demo = tmp_path / "demo"
    _make_demo_with_raw_components(
        demo,
        [
            {"name": "parent", "status": "NEW", "submodule_path": "blk"},
            {"name": "child_a", "status": "NEW", "submodule_path": "blk.a"},
        ],
        plan=[{"parent_name": "parent", "children": [{"submodule_path": "blk.a"}, {"submodule_path": "blk.b"}]}],
    )
    om.persist_no_emit_test("test/m", "parent", reason="decomposition consumer split parent into children")

    report = fc.build_final_categorization(model_id="test/m", demo_dir=demo, graduated_set={"child_a"})
    assert "parent" in report.pending
    assert "child_a" in report.on_device


def test_no_emit_parent_on_device_via_own_graduated_stub(tmp_path, monkeypatch) -> None:
    fc = _fc()
    om = _om()
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path / "overlays")
    demo = tmp_path / "demo"
    _make_demo_with_raw_components(
        demo,
        [{"name": "parent", "status": "NEW", "submodule_path": "blk"}],
        plan=[{"parent_name": "parent", "children": [{"submodule_path": "blk.a"}, {"submodule_path": "blk.b"}]}],
    )
    om.persist_no_emit_test("test/m", "parent", reason="decomposition consumer split parent into children")

    report = fc.build_final_categorization(model_id="test/m", demo_dir=demo, graduated_set={"parent"})
    assert report.on_device == ["parent"]
    assert report.pending == []


def test_nested_no_emit_parents_roll_up_recursively(tmp_path, monkeypatch) -> None:
    fc = _fc()
    om = _om()
    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path / "overlays")
    demo = tmp_path / "demo"
    comps = [
        {"name": "outer", "status": "NEW", "submodule_path": "enc"},
        {"name": "inner", "status": "NEW", "submodule_path": "enc.layers.0"},
        {"name": "leaf_ffn", "status": "NEW", "submodule_path": "enc.layers.0.ffn"},
        {"name": "leaf_attn", "status": "NEW", "submodule_path": "enc.layers.0.attn"},
    ]
    plan = [
        {"parent_name": "outer", "children": [{"submodule_path": "enc.layers.0"}]},
        {
            "parent_name": "inner",
            "children": [{"submodule_path": "enc.layers.0.ffn"}, {"submodule_path": "enc.layers.0.attn"}],
        },
    ]
    _make_demo_with_raw_components(demo, comps, plan=plan)
    om.persist_no_emit_test("test/m", "outer", reason="decomposition")
    om.persist_no_emit_test("test/m", "inner", reason="decomposition")

    report = fc.build_final_categorization(model_id="test/m", demo_dir=demo, graduated_set={"leaf_ffn"})
    assert "outer" in report.pending
    assert "inner" in report.pending

    report2 = fc.build_final_categorization(model_id="test/m", demo_dir=demo, graduated_set={"leaf_ffn", "leaf_attn"})
    assert sorted(report2.on_device) == ["leaf_attn", "leaf_ffn"]
    assert sorted(report2.pending) == ["inner", "outer"]

    ready = fc.parents_ready_to_recompose(model_id="test/m", demo_dir=demo, graduated_set={"leaf_ffn", "leaf_attn"})
    assert sorted(ready) == ["inner", "outer"]


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
