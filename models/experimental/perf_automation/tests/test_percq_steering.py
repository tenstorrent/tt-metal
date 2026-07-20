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


def test_profile_once_steering_uses_1cq_and_restores(monkeypatch):
    seen = _stub_profile_env(monkeypatch)
    saved = os.environ.get("TT_PERF_NUM_CQ")
    try:
        os.environ.pop("TT_PERF_NUM_CQ", None)
        m._profile_once(cq=1)
        assert seen["cq"] == "1"
        assert "TT_PERF_NUM_CQ" not in os.environ
    finally:
        if saved is None:
            os.environ.pop("TT_PERF_NUM_CQ", None)
        else:
            os.environ["TT_PERF_NUM_CQ"] = saved


def test_profile_once_bookend_uses_2cq(monkeypatch):
    seen = _stub_profile_env(monkeypatch)
    saved = os.environ.get("TT_PERF_NUM_CQ")
    try:
        os.environ.pop("TT_PERF_NUM_CQ", None)
        m._profile_once(cq=2)
        assert seen["cq"] == "2"
    finally:
        if saved is None:
            os.environ.pop("TT_PERF_NUM_CQ", None)
        else:
            os.environ["TT_PERF_NUM_CQ"] = saved


def test_profile_once_restores_prior_value(monkeypatch):
    seen = _stub_profile_env(monkeypatch)
    saved = os.environ.get("TT_PERF_NUM_CQ")
    try:
        os.environ["TT_PERF_NUM_CQ"] = "2"
        m._profile_once(cq=1)
        assert seen["cq"] == "1"
        assert os.environ["TT_PERF_NUM_CQ"] == "2"
    finally:
        if saved is None:
            os.environ.pop("TT_PERF_NUM_CQ", None)
        else:
            os.environ["TT_PERF_NUM_CQ"] = saved


def test_profile_once_none_leaves_env_untouched(monkeypatch):
    seen = _stub_profile_env(monkeypatch)
    saved = os.environ.get("TT_PERF_NUM_CQ")
    try:
        os.environ.pop("TT_PERF_NUM_CQ", None)
        m._profile_once()
        assert seen["cq"] is None
    finally:
        if saved is None:
            os.environ.pop("TT_PERF_NUM_CQ", None)
        else:
            os.environ["TT_PERF_NUM_CQ"] = saved
