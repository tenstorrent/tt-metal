"""Stress test: RUN_REPORT Roofline & utilization and Block-level timing render DETERMINISTICALLY —
never gated on the throughput temp file existing or the agent passing stages_json. Self-contained
(synthetic profiles), so it holds in a fresh checkout / CI with no runtime artifacts.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_PA = Path(__file__).resolve().parents[1]
if str(_PA) not in sys.path:
    sys.path.insert(0, str(_PA))
_CC = _PA / "cc_optimize"

_PROF = {"buckets": [{"id": "datamove", "device_ms": 70.0}, {"id": "matmul", "device_ms": 50.0},
                     {"id": "reduction", "device_ms": 14.0}]}


def _summary():
    spec = importlib.util.spec_from_file_location("cc_summary_test", str(_CC / "summary.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _tp(floor_ms):
    theo = 1000.0 / floor_ms
    return {"scope": "model", "is_llm_decode": False, "theoretical_tok_s": theo,
            "band": [0.6 * theo, 0.8 * theo], "active_bytes": 0, "peak_bw_gbps": 0.0,
            "tp_degree": 1, "modeled_floor_ms": floor_ms}


def _render(sm, tmp_path, *, baseline_profile=None, throughput=None, final_ms=100.0, attempts=None):
    kl = tmp_path / "kl.json"
    kl.write_text(json.dumps(attempts or []))
    return sm.render_summary(str(kl), final_ms, model="s", task="main", metric="device_ms",
                             baseline_profile=baseline_profile, throughput=throughput,
                             final_override_ms=final_ms, finalized=True)


def test_stages_from_profile_direct():
    sm = _summary()
    rows = sm._stages_from_profile(_PROF)
    assert rows and rows[0]["name"] == "datamove" and rows[0].get("dominant")


def test_block_level_renders_from_profile_without_stages(tmp_path):
    sm = _summary()
    text = _render(sm, tmp_path, baseline_profile=_PROF, throughput=_tp(100.0), attempts=[])
    assert "Block-level timing (per-stage trace)" in text
    assert "datamove" in text


def test_roofline_renders_when_throughput_none(tmp_path, monkeypatch):
    sm = _summary()
    monkeypatch.setattr(sm, "_throughput_from_profile", lambda bp: _tp(100.0))
    text = _render(sm, tmp_path, baseline_profile=_PROF, throughput=None, final_ms=200.0, attempts=[])
    assert "Roofline & utilization" in text


def test_prefers_agent_stages_when_present(tmp_path):
    sm = _summary()
    attempts = [{"op_signature": "MatmulDeviceOperation", "kernel_kind": "dtype", "measured_ms": 10.0,
                 "beat_baseline": True, "stages": [{"name": "matmul", "ms": 9.0, "dominant": True}]}]
    text = _render(sm, tmp_path, baseline_profile=_PROF, attempts=attempts)
    assert "Block-level timing (per-stage trace) — latest lever on Matmul" in text


def test_roofline_has_achievable_and_status(tmp_path):
    sm = _summary()
    text = _render(sm, tmp_path, baseline_profile=_PROF, throughput=_tp(100.0), final_ms=200.0)
    assert "achievable (60-80%)" in text
    assert "BELOW_BAND" in text


def test_status_in_band(tmp_path):
    sm = _summary()
    text = _render(sm, tmp_path, baseline_profile=_PROF, throughput=_tp(100.0), final_ms=150.0)
    assert "IN_BAND" in text


def test_status_above_band(tmp_path):
    sm = _summary()
    text = _render(sm, tmp_path, baseline_profile=_PROF, throughput=_tp(100.0), final_ms=90.0)
    assert "ABOVE_BAND" in text


def test_stress_many_profiles(tmp_path):
    sm = _summary()
    for i in range(60):
        prof = {"buckets": [{"id": f"op{j}", "device_ms": (j + 1) * (i % 5 + 1) * 1.0} for j in range(1 + i % 6)]}
        text = _render(sm, tmp_path, baseline_profile=prof, throughput=_tp(50.0 + i), final_ms=100.0 + i, attempts=[])
        assert "Roofline & utilization" in text, f"iter {i}: roofline missing"
        assert "Block-level timing (per-stage trace)" in text, f"iter {i}: block-level missing"
