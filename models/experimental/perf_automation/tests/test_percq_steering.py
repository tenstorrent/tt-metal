import os
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cc_optimize import perf_mcp as m


def _stub_profile_env(monkeypatch):
    seen = {}

    def fake_measure_runs(ctx):
        seen["cq"] = os.environ.get("TT_PERF_NUM_CQ")
        return [{"device_ms": 1.0}]

    monkeypatch.setattr(m, "measure_runs", fake_measure_runs)
    monkeypatch.setattr(
        m, "_Ctx", lambda: types.SimpleNamespace(run=types.SimpleNamespace(dir="/tmp", profiles_dir="/tmp"))
    )
    monkeypatch.setattr(m.roofline, "annotate_profile", lambda p, e: p)
    monkeypatch.setattr(m, "_detect_partial_capture", lambda d: None)
    monkeypatch.setattr(m, "_persist_artifacts", lambda p: p)
    monkeypatch.setattr(m, "_reap_measurement_dir", lambda d: None)
    return seen


def test_profile_once_leaves_cq_env_untouched(monkeypatch):
    # The tool is trace+1cq end to end: _profile_once never sets TT_PERF_NUM_CQ (the device opens
    # with a single command queue via the perf-test fixture), so the profiling env is left as-is.
    seen = _stub_profile_env(monkeypatch)
    saved = os.environ.get("TT_PERF_NUM_CQ")
    try:
        os.environ.pop("TT_PERF_NUM_CQ", None)
        m._profile_once()
        assert seen["cq"] is None
        assert "TT_PERF_NUM_CQ" not in os.environ
    finally:
        if saved is None:
            os.environ.pop("TT_PERF_NUM_CQ", None)
        else:
            os.environ["TT_PERF_NUM_CQ"] = saved
