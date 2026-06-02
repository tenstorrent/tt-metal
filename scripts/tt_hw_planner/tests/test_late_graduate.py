"""Unit tests for late-graduate handling (Item 4).

When the e2e synthesizer (Item 3) discovers it needs a TTNN module
the decomposer didn't identify, this module adds it to the manifest
with LATE_GRADUATE status and runs it through the per-component
iterate path. On failure: routes to CPU fallback so e2e synthesis
isn't blocked.

Tests cover the manifest mutation primitives (add / list /
mark-as-CPU-fallback) and the orchestrator with the component-iterate
callable mocked.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from scripts.tt_hw_planner._cli_helpers.late_graduate import (
    LATE_GRADUATE_STATUS,
    LateGraduateComponentSpec,
    LateGraduateResult,
    _gradable_statuses,
    add_late_graduate_to_manifest,
    list_late_graduate_components,
    mark_late_graduate_as_cpu_fallback,
    run_late_graduate,
)


def _write_status(tmp_path: Path, components: list) -> Path:
    status = tmp_path / "bringup_status.json"
    status.write_text(json.dumps({"components": components}))
    return status


def _read_status(tmp_path: Path) -> Dict[str, Any]:
    return json.loads((tmp_path / "bringup_status.json").read_text())


def _spec(name: str = "fpn") -> LateGraduateComponentSpec:
    return LateGraduateComponentSpec(
        name=name,
        hf_reference=f"vision_encoder.{name}",
        class_name="FPNAdapter",
    )


# ─── LateGraduateComponentSpec ──────────────────────────────────────


def test_spec_renders_manifest_entry() -> None:
    s = _spec("fpn")
    e = s.to_manifest_entry()
    assert e["name"] == "fpn"
    assert e["status"] == LATE_GRADUATE_STATUS
    assert e["hf_reference"] == "vision_encoder.fpn"
    assert e["class_name"] == "FPNAdapter"


def test_spec_extras_propagate_to_manifest() -> None:
    s = LateGraduateComponentSpec(
        name="fpn",
        hf_reference="x",
        class_name="X",
        extras={"complexity": 3, "decomp_source": "synthesis-discovery"},
    )
    e = s.to_manifest_entry()
    assert e["complexity"] == 3
    assert e["decomp_source"] == "synthesis-discovery"


# ─── add_late_graduate_to_manifest ──────────────────────────────────


def test_add_returns_false_when_no_manifest(tmp_path: Path) -> None:
    """Missing bringup_status.json → False, no crash."""
    assert add_late_graduate_to_manifest(tmp_path, _spec()) is False


def test_add_returns_false_for_malformed_manifest(tmp_path: Path) -> None:
    (tmp_path / "bringup_status.json").write_text("{not json")
    assert add_late_graduate_to_manifest(tmp_path, _spec()) is False


def test_add_appends_new_component(tmp_path: Path) -> None:
    _write_status(tmp_path, [{"name": "attention", "status": "NEW"}])
    assert add_late_graduate_to_manifest(tmp_path, _spec("fpn")) is True
    data = _read_status(tmp_path)
    names = [c["name"] for c in data["components"]]
    assert "attention" in names  # existing preserved
    assert "fpn" in names  # new added
    fpn = next(c for c in data["components"] if c["name"] == "fpn")
    assert fpn["status"] == LATE_GRADUATE_STATUS


def test_add_updates_existing_component_in_place(tmp_path: Path) -> None:
    """Idempotent: re-adding the same component name updates the
    existing entry, doesn't duplicate."""
    _write_status(
        tmp_path,
        [{"name": "fpn", "status": "NEW", "hf_reference": "old", "complexity": 5}],
    )
    add_late_graduate_to_manifest(tmp_path, _spec("fpn"))
    data = _read_status(tmp_path)
    fpns = [c for c in data["components"] if c["name"] == "fpn"]
    assert len(fpns) == 1  # not duplicated
    assert fpns[0]["status"] == LATE_GRADUATE_STATUS  # flipped
    assert fpns[0]["hf_reference"] == "vision_encoder.fpn"  # updated
    assert fpns[0]["complexity"] == 5  # decomposer extras preserved


# ─── list_late_graduate_components ──────────────────────────────────


def test_list_returns_empty_when_none(tmp_path: Path) -> None:
    _write_status(tmp_path, [{"name": "a", "status": "NEW"}])
    assert list_late_graduate_components(tmp_path) == []


def test_list_filters_by_late_graduate_status(tmp_path: Path) -> None:
    _write_status(
        tmp_path,
        [
            {"name": "a", "status": "NEW"},
            {"name": "b", "status": LATE_GRADUATE_STATUS},
            {"name": "c", "status": "ADAPT"},
            {"name": "d", "status": LATE_GRADUATE_STATUS},
        ],
    )
    late = list_late_graduate_components(tmp_path)
    names = {c["name"] for c in late}
    assert names == {"b", "d"}


def test_list_returns_empty_for_missing_manifest(tmp_path: Path) -> None:
    assert list_late_graduate_components(tmp_path) == []


# ─── mark_late_graduate_as_cpu_fallback ─────────────────────────────


def test_mark_cpu_fallback_flips_status(tmp_path: Path) -> None:
    _write_status(tmp_path, [{"name": "fpn", "status": LATE_GRADUATE_STATUS}])
    assert mark_late_graduate_as_cpu_fallback(tmp_path, "fpn") is True
    data = _read_status(tmp_path)
    fpn = data["components"][0]
    assert fpn["status"] == "LATE_GRADUATE_CPU_FALLBACK"
    assert "cpu_fallback_reason" in fpn


def test_mark_cpu_fallback_returns_false_for_unknown_component(tmp_path: Path) -> None:
    _write_status(tmp_path, [{"name": "fpn", "status": LATE_GRADUATE_STATUS}])
    assert mark_late_graduate_as_cpu_fallback(tmp_path, "does-not-exist") is False


def test_mark_cpu_fallback_returns_false_for_missing_manifest(tmp_path: Path) -> None:
    assert mark_late_graduate_as_cpu_fallback(tmp_path, "any") is False


# ─── run_late_graduate orchestrator ─────────────────────────────────


def _converging_iterate(*, demo_dir, component_name, max_iters, pcc_target):
    """Mock component-iterate that always converges immediately."""

    class _Res:
        converged = True
        pcc = 0.995
        iters_used = 1

    return _Res()


def _failing_iterate(*, demo_dir, component_name, max_iters, pcc_target):
    """Mock component-iterate that exhausts the budget."""

    class _Res:
        converged = False
        pcc = 0.42
        iters_used = max_iters

    return _Res()


def _raising_iterate(*, demo_dir, component_name, max_iters, pcc_target):
    raise RuntimeError("iterate boom")


def test_run_converges_and_keeps_late_graduate_status(tmp_path: Path) -> None:
    _write_status(tmp_path, [])
    result = run_late_graduate(
        demo_dir=tmp_path,
        spec=_spec("fpn"),
        component_iterate=_converging_iterate,
    )
    assert result.converged is True
    assert result.pcc == 0.995
    assert result.fallback_to_cpu is False
    # Manifest entry kept its LATE_GRADUATE status — not CPU fallback
    data = _read_status(tmp_path)
    fpn = next(c for c in data["components"] if c["name"] == "fpn")
    assert fpn["status"] == LATE_GRADUATE_STATUS


def test_run_routes_to_cpu_on_iterate_failure(tmp_path: Path) -> None:
    _write_status(tmp_path, [])
    result = run_late_graduate(
        demo_dir=tmp_path,
        spec=_spec("fpn"),
        component_iterate=_failing_iterate,
    )
    assert result.converged is False
    assert result.fallback_to_cpu is True
    assert "budget exhausted" in result.diagnostic.lower() or "exhausted" in result.diagnostic.lower()
    data = _read_status(tmp_path)
    fpn = next(c for c in data["components"] if c["name"] == "fpn")
    assert fpn["status"] == "LATE_GRADUATE_CPU_FALLBACK"


def test_run_routes_to_cpu_on_iterate_exception(tmp_path: Path) -> None:
    _write_status(tmp_path, [])
    result = run_late_graduate(
        demo_dir=tmp_path,
        spec=_spec("fpn"),
        component_iterate=_raising_iterate,
    )
    assert result.converged is False
    assert result.fallback_to_cpu is True
    assert "iterate boom" in result.diagnostic
    data = _read_status(tmp_path)
    fpn = next(c for c in data["components"] if c["name"] == "fpn")
    assert fpn["status"] == "LATE_GRADUATE_CPU_FALLBACK"


def test_run_returns_failure_when_manifest_missing(tmp_path: Path) -> None:
    """No bringup_status.json → can't add to manifest → return early
    without invoking iterate."""
    iterate_called = [False]

    def _track(**kwargs):
        iterate_called[0] = True
        return _converging_iterate(**kwargs)

    result = run_late_graduate(
        demo_dir=tmp_path,
        spec=_spec("fpn"),
        component_iterate=_track,
    )
    assert result.converged is False
    assert "failed to add to bringup_status" in result.diagnostic
    assert iterate_called[0] is False


def test_run_returns_failure_when_no_iterate_callable(tmp_path: Path) -> None:
    """No component_iterate provided → fail-fast diagnostic."""
    _write_status(tmp_path, [])
    result = run_late_graduate(demo_dir=tmp_path, spec=_spec("fpn"))
    assert result.converged is False
    assert "no component_iterate" in result.diagnostic


def test_run_accepts_dict_return_from_iterate(tmp_path: Path) -> None:
    """Iterate callable may return a dict instead of dataclass — loose
    parsing supports both."""
    _write_status(tmp_path, [])

    def _dict_iterate(**kwargs):
        return {"converged": True, "pcc": 0.99, "iters_used": 3}

    result = run_late_graduate(
        demo_dir=tmp_path,
        spec=_spec("fpn"),
        component_iterate=_dict_iterate,
    )
    assert result.converged is True
    assert result.pcc == 0.99
    assert result.iters_used == 3


# ─── _gradable_statuses (filter helper) ─────────────────────────────


def test_gradable_statuses_includes_late_graduate() -> None:
    """The shared filter must include LATE_GRADUATE alongside NEW/ADAPT
    so iterate-loop callers can centralize the status check rather
    than each having to remember to update its hardcoded tuple."""
    statuses = _gradable_statuses()
    assert "NEW" in statuses
    assert "ADAPT" in statuses
    assert LATE_GRADUATE_STATUS in statuses
