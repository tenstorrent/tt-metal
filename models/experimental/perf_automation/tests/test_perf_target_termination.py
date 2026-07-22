# SPDX-License-Identifier: Apache-2.0
"""termination_check band hook: opt-in, per-module scores its own floor, full-model falls back to
the floor without a sidecar, fail-open. Uses trace+1cq via _reliable_forward_ms (mocked)."""
import importlib.util
from pathlib import Path

_S = importlib.util.spec_from_file_location(
    "perf_mcp_ptterm", str(Path(__file__).resolve().parents[1] / "cc_optimize" / "perf_mcp.py"))
pm = importlib.util.module_from_spec(_S)
_S.loader.exec_module(pm)


def test_band_off_returns_none(monkeypatch):
    monkeypatch.delenv("PERF_MCP_TARGET_BAND", raising=False)
    assert pm._perf_target_status({"modeled_floor_ms": 2.0}, 2.2) is None


def test_per_module_in_band(monkeypatch):
    monkeypatch.setenv("PERF_MCP_TARGET_BAND", "1")
    monkeypatch.setenv("TT_PERF_MODULE_LEVEL", "1")
    monkeypatch.setattr(pm, "_reliable_forward_ms", lambda dev: 2.2)  # near the 2.0ms floor
    s = pm._perf_target_status({"modeled_floor_ms": 2.0}, 99.0)
    assert s["scope"] == "module" and s["status"] == "IN_BAND"


def test_per_module_below_band(monkeypatch):
    monkeypatch.setenv("PERF_MCP_TARGET_BAND", "1")
    monkeypatch.setenv("TT_PERF_MODULE_LEVEL", "1")
    monkeypatch.setattr(pm, "_reliable_forward_ms", lambda dev: 3.5)
    assert pm._perf_target_status({"modeled_floor_ms": 2.0}, 99.0)["status"] == "BELOW_BAND"


def test_full_model_falls_back_to_floor(monkeypatch):
    monkeypatch.setenv("PERF_MCP_TARGET_BAND", "1")
    monkeypatch.delenv("TT_PERF_MODULE_LEVEL", raising=False)
    monkeypatch.setattr(pm, "_load_perf_target_inputs", lambda: None)
    monkeypatch.setattr(pm, "_reliable_forward_ms", lambda dev: 2.2)
    s = pm._perf_target_status({"modeled_floor_ms": 2.0}, 99.0)
    assert s["scope"] == "model" and s["status"] == "IN_BAND"


def test_fail_open_on_error(monkeypatch):
    monkeypatch.setenv("PERF_MCP_TARGET_BAND", "1")
    monkeypatch.setenv("TT_PERF_MODULE_LEVEL", "1")
    def boom(_):
        raise RuntimeError("x")
    monkeypatch.setattr(pm, "_reliable_forward_ms", boom)
    assert pm._perf_target_status({"modeled_floor_ms": 2.0}, 2.0) is None
