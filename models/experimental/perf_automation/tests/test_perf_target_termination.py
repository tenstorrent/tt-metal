# SPDX-License-Identifier: Apache-2.0
"""termination_check band hook: opt-in, unit-matched scoring (per-profile floor scored against
per-profile device_ms, NOT the per-token trace), per-module uses its own floor, fail-open."""
import importlib.util
from pathlib import Path

_S = importlib.util.spec_from_file_location(
    "perf_mcp_ptterm", str(Path(__file__).resolve().parents[1] / "cc_optimize" / "perf_mcp.py"))
pm = importlib.util.module_from_spec(_S)
_S.loader.exec_module(pm)


def test_band_off_returns_none(monkeypatch):
    monkeypatch.delenv("PERF_MCP_TARGET_BAND", raising=False)
    assert pm._perf_target_status({"modeled_floor_ms": 2.0}, 2.2) is None


def test_per_module_scores_dev_against_floor_in_band(monkeypatch):
    # floor path scores dev (per-profile), NOT the per-token trace. dev 2.2 near the 2.0 floor -> IN_BAND
    monkeypatch.setenv("PERF_MCP_TARGET_BAND", "1")
    monkeypatch.setenv("TT_PERF_MODULE_LEVEL", "1")
    s = pm._perf_target_status({"modeled_floor_ms": 2.0}, 2.2)
    assert s["scope"] == "module" and s["status"] == "IN_BAND"


def test_per_module_below_band(monkeypatch):
    monkeypatch.setenv("PERF_MCP_TARGET_BAND", "1")
    monkeypatch.setenv("TT_PERF_MODULE_LEVEL", "1")
    assert pm._perf_target_status({"modeled_floor_ms": 2.0}, 3.5)["status"] == "BELOW_BAND"


def test_unit_match_no_false_above_band(monkeypatch):
    # the real regression: per-profile floor 1.913 vs per-profile dev 3.533 -> BELOW_BAND, never ABOVE.
    # (the bug scored the per-token 0.197ms trace against this floor -> ABOVE_BAND every time.)
    monkeypatch.setenv("PERF_MCP_TARGET_BAND", "1")
    monkeypatch.setenv("TT_PERF_MODULE_LEVEL", "1")
    s = pm._perf_target_status({"modeled_floor_ms": 1000 / 522.821}, 3.533)
    assert s["status"] == "BELOW_BAND", s


def test_full_model_falls_back_to_floor_scores_dev(monkeypatch):
    monkeypatch.setenv("PERF_MCP_TARGET_BAND", "1")
    monkeypatch.delenv("TT_PERF_MODULE_LEVEL", raising=False)
    monkeypatch.setattr(pm, "_load_perf_target_inputs", lambda: None)
    s = pm._perf_target_status({"modeled_floor_ms": 2.0}, 2.2)
    assert s["scope"] == "model" and s["status"] == "IN_BAND"


def test_fail_open_on_error(monkeypatch):
    monkeypatch.setenv("PERF_MCP_TARGET_BAND", "1")
    monkeypatch.setenv("TT_PERF_MODULE_LEVEL", "1")
    monkeypatch.setattr(pm.perf_target, "score", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    assert pm._perf_target_status({"modeled_floor_ms": 2.0}, 2.0) is None
