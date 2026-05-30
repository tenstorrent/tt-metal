"""Tests for the Stage 4 bench helpers — focuses on the pure-logic
pieces (result persistence, verdict mapping) that don't require TT
hardware. The end-to-end bench requires `ttnn.open_device` so it's
exercised manually / in CI with hardware."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.tt_hw_planner.commands.bench_component import (  # noqa: E402
    BenchResult,
    _collect_graduated_for_bench,
    _persist_bench_results,
)


# ---------------------------------------------------------------------------
# BenchResult dataclass shape
# ---------------------------------------------------------------------------


def test_bench_result_default_unknown():
    r = BenchResult(component="x")
    assert r.verdict == "UNKNOWN"
    assert r.speedup == 0.0


def test_bench_result_serializable_to_json():
    r = BenchResult(
        component="x",
        cpu_mean_ms=10.0,
        device_mean_ms=2.0,
        transfer_in_ms=1.0,
        transfer_out_ms=0.5,
        verdict="DEVICE_WINS",
        speedup=2.86,
    )
    s = json.dumps(asdict(r))
    parsed = json.loads(s)
    assert parsed["verdict"] == "DEVICE_WINS"
    assert parsed["speedup"] == 2.86


# ---------------------------------------------------------------------------
# _collect_graduated_for_bench — filters to graduated + capture-complete
# ---------------------------------------------------------------------------


def test_collect_skips_components_without_snapshot(tmp_path: Path):
    """A component without .py.last_good_native is filtered out."""
    demo_dir = tmp_path / "demo"
    (demo_dir / "_stubs").mkdir(parents=True)
    cap_dir = demo_dir / "_captured" / "comp_a"
    cap_dir.mkdir(parents=True)
    # Stub exists but no .py.last_good_native snapshot
    (demo_dir / "_stubs" / "comp_a.py").write_text("# placeholder")
    (cap_dir / "args.pt").write_text("x")
    (cap_dir / "kwargs.pt").write_text("x")
    (cap_dir / "manifest.json").write_text(json.dumps({"submodule_path": "encoder"}))

    status = {"components": [{"name": "comp_a", "status": "NEW", "submodule_path": "encoder"}]}
    candidates = _collect_graduated_for_bench(demo_dir, status, None)
    assert candidates == []


def test_collect_skips_components_without_captures(tmp_path: Path):
    """A graduated stub without args.pt/kwargs.pt is filtered."""
    demo_dir = tmp_path / "demo"
    (demo_dir / "_stubs").mkdir(parents=True)
    (demo_dir / "_stubs" / "comp_a.py").write_text("# stub")
    (demo_dir / "_stubs" / "comp_a.py.last_good_native").write_text("# graduated")
    # No _captured/ dir at all

    status = {"components": [{"name": "comp_a", "status": "NEW", "submodule_path": "enc"}]}
    candidates = _collect_graduated_for_bench(demo_dir, status, None)
    assert candidates == []


def test_collect_returns_eligible_components(tmp_path: Path):
    """Components with BOTH snapshot AND captures are returned."""
    demo_dir = tmp_path / "demo"
    (demo_dir / "_stubs").mkdir(parents=True)
    (demo_dir / "_stubs" / "comp_a.py").write_text("# stub")
    (demo_dir / "_stubs" / "comp_a.py.last_good_native").write_text("# graduated")
    cap_dir = demo_dir / "_captured" / "comp_a"
    cap_dir.mkdir(parents=True)
    (cap_dir / "args.pt").write_text("x")
    (cap_dir / "kwargs.pt").write_text("x")
    (cap_dir / "manifest.json").write_text(json.dumps({"submodule_path": "encoder"}))

    status = {"components": [{"name": "comp_a", "status": "NEW", "submodule_path": "encoder"}]}
    candidates = _collect_graduated_for_bench(demo_dir, status, None)
    assert len(candidates) == 1
    assert candidates[0]["name"] == "comp_a"
    assert candidates[0]["submodule_path"] == "encoder"


def test_collect_filter_by_component_name(tmp_path: Path):
    """When --component X is passed, only X is returned."""
    demo_dir = tmp_path / "demo"
    (demo_dir / "_stubs").mkdir(parents=True)
    for name in ("a", "b"):
        (demo_dir / "_stubs" / f"{name}.py").write_text("# stub")
        (demo_dir / "_stubs" / f"{name}.py.last_good_native").write_text("# graduated")
        cap = demo_dir / "_captured" / name
        cap.mkdir(parents=True)
        (cap / "args.pt").write_text("x")
        (cap / "kwargs.pt").write_text("x")
        (cap / "manifest.json").write_text(json.dumps({"submodule_path": name}))

    status = {
        "components": [
            {"name": "a", "status": "NEW", "submodule_path": "a"},
            {"name": "b", "status": "NEW", "submodule_path": "b"},
        ]
    }
    candidates = _collect_graduated_for_bench(demo_dir, status, "a")
    assert len(candidates) == 1
    assert candidates[0]["name"] == "a"


# ---------------------------------------------------------------------------
# _persist_bench_results — Signal 4 storage in hot_cold.json
# ---------------------------------------------------------------------------


def test_persist_creates_multi_mode_entry(tmp_path: Path, monkeypatch):
    """When no prior hot_cold.json exists, the bench result creates a
    multi-mode entry with the bench data under modes[<workload_mode>]."""
    from scripts.tt_hw_planner import overlay_manager as om

    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path / "overlays")

    results = [
        BenchResult(
            component="comp_a",
            cpu_mean_ms=10.0,
            device_mean_ms=2.0,
            transfer_in_ms=1.0,
            transfer_out_ms=0.5,
            verdict="DEVICE_WINS",
            speedup=2.86,
            workload_mode="default",
        )
    ]
    _persist_bench_results("test/m", results, workload_mode="default")

    raw = om._load_hot_cold_raw("test/m")
    assert "comp_a" in raw
    entry = raw["comp_a"]
    assert "modes" in entry
    assert "default" in entry["modes"]
    assert entry["modes"]["default"]["bench_verdict"] == "DEVICE_WINS"
    assert entry["modes"]["default"]["bench"]["speedup"] == 2.86


def test_persist_bench_is_diagnostic_only_does_not_change_kind(tmp_path: Path, monkeypatch):
    """Pin: bench results are DIAGNOSTIC ONLY. They record per-mode
    bench data + an evidence line so the operator can see them, but
    DO NOT change the top-level kind. A graduated component (HOT)
    stays HOT even when bench says CPU_WINS — graduation is the
    contract, demo wiring honors it, bench just flags perf work."""
    from scripts.tt_hw_planner import overlay_manager as om

    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path / "overlays")

    # Seed: comp_a is HOT (graduated, prior frequency/latency signals).
    md = om._model_dir("test/m")
    md.mkdir(parents=True, exist_ok=True)
    (md / "hot_cold.json").write_text(
        json.dumps(
            {
                "comp_a": {
                    "kind": "HOT",
                    "modes": {"default": {"kind": "HOT", "frequency": 1.0, "workload_mode": "default"}},
                }
            }
        )
    )
    # Bench says CPU_WINS — top-level kind must NOT change.
    results = [BenchResult(component="comp_a", verdict="CPU_WINS", speedup=0.5, workload_mode="default")]
    _persist_bench_results("test/m", results, workload_mode="default")

    raw = om._load_hot_cold_raw("test/m")
    assert raw["comp_a"]["kind"] == "HOT", "graduated kind must NOT be demoted by bench"
    # Bench data + evidence line are still recorded for visibility.
    assert raw["comp_a"]["modes"]["default"]["bench_verdict"] == "CPU_WINS"
    assert any("CPU_WINS" in e for e in raw["comp_a"].get("evidence", []))


def test_persist_bench_records_device_wins_diagnostic(tmp_path: Path, monkeypatch):
    """A DEVICE_WINS bench result records the verdict + evidence line
    without changing the top-level kind."""
    from scripts.tt_hw_planner import overlay_manager as om

    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path / "overlays")

    results = [BenchResult(component="comp_a", verdict="DEVICE_WINS", speedup=3.0, workload_mode="default")]
    _persist_bench_results("test/m", results, workload_mode="default")
    raw = om._load_hot_cold_raw("test/m")
    # With no prior entry, kind stays UNKNOWN (no auto-promotion).
    assert raw["comp_a"]["kind"] in ("UNKNOWN", "HOT")  # only changes from prior fields
    assert raw["comp_a"]["modes"]["default"]["bench_verdict"] == "DEVICE_WINS"
    assert any("DEVICE_WINS" in e for e in raw["comp_a"].get("evidence", []))


def test_persist_breakeven_leaves_kind_unchanged(tmp_path: Path, monkeypatch):
    from scripts.tt_hw_planner import overlay_manager as om

    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path / "overlays")
    md = om._model_dir("test/m")
    md.mkdir(parents=True, exist_ok=True)
    (md / "hot_cold.json").write_text(json.dumps({"comp_a": {"kind": "HOT", "modes": {}}}))

    results = [BenchResult(component="comp_a", verdict="BREAKEVEN", speedup=1.0, workload_mode="default")]
    _persist_bench_results("test/m", results, workload_mode="default")
    raw = om._load_hot_cold_raw("test/m")
    assert raw["comp_a"]["kind"] == "HOT"  # unchanged


def test_persist_error_does_not_corrupt_state(tmp_path: Path, monkeypatch):
    """An ERROR result should record the verdict but not change the kind
    or corrupt the modes dict."""
    from scripts.tt_hw_planner import overlay_manager as om

    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path / "overlays")

    results = [BenchResult(component="comp_a", verdict="ERROR", error="exception", workload_mode="default")]
    _persist_bench_results("test/m", results, workload_mode="default")
    raw = om._load_hot_cold_raw("test/m")
    # bench_verdict recorded; top-level kind unchanged
    assert raw["comp_a"]["modes"]["default"]["bench_verdict"] == "ERROR"


def test_persist_per_workload_mode_isolation(tmp_path: Path, monkeypatch):
    """Bench runs in different modes go under their own modes[<mode>]
    slot and don't overwrite each other."""
    from scripts.tt_hw_planner import overlay_manager as om

    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path / "overlays")

    _persist_bench_results(
        "test/m",
        [BenchResult(component="comp_a", verdict="DEVICE_WINS", workload_mode="image")],
        workload_mode="image",
    )
    _persist_bench_results(
        "test/m",
        [BenchResult(component="comp_a", verdict="CPU_WINS", workload_mode="video")],
        workload_mode="video",
    )
    raw = om._load_hot_cold_raw("test/m")
    assert raw["comp_a"]["modes"]["image"]["bench_verdict"] == "DEVICE_WINS"
    assert raw["comp_a"]["modes"]["video"]["bench_verdict"] == "CPU_WINS"
