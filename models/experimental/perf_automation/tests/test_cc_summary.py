"""cc end-of-run summary renderer: per-op × ladder-level table + old->new % speedup."""

import importlib.util
import json
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "cc_summary", str(Path(__file__).resolve().parents[1] / "cc_optimize" / "summary.py")
)
summary = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(summary)


def _log(tmp_path, rows):
    p = tmp_path / "kernlog.json"
    p.write_text(json.dumps(rows))
    return p


def test_table_marks_win_try_and_none(tmp_path):
    log = _log(
        tmp_path,
        [
            {
                "op_signature": "MatmulDeviceOperation 1024",
                "kernel_kind": "grid",
                "measured_ms": 20.1,
                "beat_baseline": True,
            },
            {
                "op_signature": "MatmulDeviceOperation 1024",
                "kernel_kind": "dtype",
                "measured_ms": 16.4,
                "beat_baseline": True,
            },
            {"op_signature": "LayerNorm", "kernel_kind": "tt-lang", "measured_ms": 16.4, "beat_baseline": False},
        ],
    )
    out = summary.render_summary(log, baseline_ms=22.94, model="bge", task="main")
    # The label keeps the SHAPE that tells two matmuls apart -- seven identical
    # "MatmulDeviceOperation" rows made the ladder matrix unreadable -- and drops the redundant
    # "DeviceOperation" suffix to make room for it.
    assert "Matmul 1024" in out and "LayerNorm" in out
    assert "MatmulDeviceOperation" not in out
    assert "✓win" in out and "·try" in out and "—" in out


def test_empty_log_is_safe(tmp_path):
    out = summary.render_summary(_log(tmp_path, []), baseline_ms=10.0, model="m")
    assert "no kernel attempts" in out


def test_live_render_shows_pending_not_delta(tmp_path):
    log = _log(tmp_path, [{"op_signature": "Op", "kernel_kind": "dtype", "measured_ms": 16.42, "beat_baseline": True}])
    out = summary.render_summary(log, baseline_ms=22.94, model="m", finalized=False)
    # The live render must mark itself in progress and must NOT state a baseline->final delta. The
    # marker used to be a sentence explaining when the delta gets finalized; the report is read to
    # validate numbers, so it is now just the state.
    assert "optimizing…" in out
    assert "ms  ->  final" not in out


def _ledger_for(tmp_path, monkeypatch, before=None, after=None, depth="16", mode="eager"):
    """Populate the ledger, the single source the headline now reads."""
    import importlib.util as ilu

    monkeypatch.setenv("PERF_MCP_LEDGER", str(tmp_path / "led.jsonl"))
    spec = ilu.spec_from_file_location(
        "meas_cc_ut", Path(__file__).resolve().parents[1] / "cc_optimize" / "measurements.py"
    )
    m = ilu.module_from_spec(spec)
    spec.loader.exec_module(m)
    if before is not None:
        m.record(m.KIND_EAGER, m.PHASE_BEFORE, before, depth=depth, mode=mode, source="test")
    if after is not None:
        m.record(m.KIND_EAGER, m.PHASE_AFTER, after, depth=depth, mode=mode, source="test")
    return m


def test_old_to_new_percent_and_speedup(tmp_path, monkeypatch):
    _ledger_for(tmp_path, monkeypatch, before=22.94, after=16.42)
    log = _log(tmp_path, [{"op_signature": "M", "kernel_kind": "grid", "measured_ms": 16.42, "beat_baseline": True}])
    out = summary.render_summary(log, model="m")
    assert "22.94" in out and "16.42" in out
    assert "+28.4%" in out and "1.40x" in out


def test_no_measurement_says_not_measured(tmp_path, monkeypatch):
    """No anchor must read as 'not measured'. It used to fall through a chain of files and could
    surface another run's number as this run's baseline."""
    _ledger_for(tmp_path, monkeypatch)
    log = _log(tmp_path, [{"op_signature": "M", "kernel_kind": "grid", "measured_ms": 16.42, "beat_baseline": True}])
    out = summary.render_summary(log, model="m")
    assert "not measured" in out


def test_the_original_anchors_the_headline_not_the_latest_before(tmp_path, monkeypatch):
    """A rerun records a new 'before' against the already-optimized model. The headline must still
    anchor on the FIRST reading ever taken, or restarting optimize silently erases the real gain."""
    m = _ledger_for(tmp_path, monkeypatch, before=42.60, after=19.83)
    m.record(m.KIND_EAGER, m.PHASE_BEFORE, 19.83, depth="16", mode="eager", source="rerun")
    m.record(m.KIND_EAGER, m.PHASE_AFTER, 18.00, depth="16", mode="eager", source="rerun")
    log = _log(tmp_path, [{"op_signature": "M", "kernel_kind": "grid", "measured_ms": 18.0, "beat_baseline": True}])
    out = summary.render_summary(log, model="m")
    assert "42.60" in out and "18.00" in out


def test_headline_reports_the_current_state_not_the_best_ever(tmp_path, monkeypatch):
    """The 'after' is the LATEST reading, not the best one ever seen: a later regression must show."""
    m = _ledger_for(tmp_path, monkeypatch, before=100.0, after=60.0)
    m.record(m.KIND_EAGER, m.PHASE_AFTER, 75.0, depth="16", mode="eager", source="later")
    log = _log(tmp_path, [{"op_signature": "M", "kernel_kind": "grid", "measured_ms": 75.0, "beat_baseline": True}])
    out = summary.render_summary(log, model="m")
    assert "75.00" in out and "60.00" not in out
