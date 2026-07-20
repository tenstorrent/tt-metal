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
    assert "MatmulDeviceOperation" in out and "LayerNorm" in out
    assert "✓win" in out and "·try" in out and "—" in out


def test_old_to_new_percent_and_speedup(tmp_path):
    log = _log(tmp_path, [{"op_signature": "Op", "kernel_kind": "dtype", "measured_ms": 16.42, "beat_baseline": True}])
    out = summary.render_summary(log, baseline_ms=22.94, model="m")
    assert "22.94" in out and "16.42" in out
    assert "28.4%" in out  # (22.94-16.42)/22.94*100
    assert "1.40x" in out  # 22.94/16.42


def test_no_baseline_degrades(tmp_path):
    log = _log(tmp_path, [{"op_signature": "Op", "kernel_kind": "grid", "measured_ms": 5.0, "beat_baseline": True}])
    out = summary.render_summary(log, baseline_ms=None, model="m")
    assert "unavailable" in out


def test_empty_log_is_safe(tmp_path):
    out = summary.render_summary(_log(tmp_path, []), baseline_ms=10.0, model="m")
    assert "no kernel attempts" in out


def test_live_render_shows_pending_not_delta(tmp_path):
    log = _log(tmp_path, [{"op_signature": "Op", "kernel_kind": "dtype", "measured_ms": 16.42, "beat_baseline": True}])
    out = summary.render_summary(log, baseline_ms=22.94, model="m", finalized=False)
    assert "finalized when the module converges" in out
    assert "ms  ->  final" not in out


def test_original_baseline_anchors_headline(tmp_path):
    log = _log(tmp_path, [{"op_signature": "Op", "kernel_kind": "dtype", "measured_ms": 19.83, "beat_baseline": True}])
    out = summary.render_summary(
        log,
        baseline_ms=19.83,
        model="m",
        original_baseline_ms=42.60,
        final_override_ms=19.83,
    )
    assert "42.60 ms  ->  final 19.83 ms" in out
    assert "53.5%" in out
    assert "2.15x" in out


def test_final_override_pins_current_not_best_win(tmp_path):
    log = _log(tmp_path, [{"op_signature": "Op", "kernel_kind": "dtype", "measured_ms": 50.0, "beat_baseline": True}])
    out = summary.render_summary(
        log,
        baseline_ms=100.0,
        model="m",
        original_baseline_ms=100.0,
        final_override_ms=80.0,
    )
    assert "100.00 ms  ->  final 80.00 ms" in out
    assert "20.0%" in out
