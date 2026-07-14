# SPDX-License-Identifier: Apache-2.0
"""Full-pipeline bookend gate: trace+2CQ vs trace+1CQ fidelity (#3) + reserved-budget guidance (#4).

The trace+2CQ number is a BOOKEND (baseline at start, final at end). A final bookend that silently
degrades to 1CQ — even if the raw ms LOOKS faster — makes before/after incomparable, so it must be
flagged 'degraded' and NOT banked. A legitimate fidelity UPGRADE (eager->trace, 1cq->2cq) re-baselines.
"""
import importlib.util
import json
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "perf_mcp_fullpipe",
    str(Path(__file__).resolve().parents[1] / "cc_optimize" / "perf_mcp.py"),
)
perf_mcp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(perf_mcp)

_cfpl = getattr(perf_mcp.check_full_pipeline_latency, "fn", perf_mcp.check_full_pipeline_latency)


def _drive(monkeypatch, ms, method, path):
    monkeypatch.setattr(perf_mcp, "_run_full_pipeline_ms", lambda: (ms, method, None, path))
    return _cfpl()


def test_mode_helpers():
    assert perf_mcp._fullpipe_mode("trace", "trace+2cq") == "trace+2cq"
    assert perf_mcp._fullpipe_mode("trace", "trace+1cq") == "trace+1cq"
    assert perf_mcp._fullpipe_mode("trace", None) == "trace"
    assert perf_mcp._fullpipe_mode("eager", None) == "eager"
    assert perf_mcp._mode_rank("trace+2cq") > perf_mcp._mode_rank("trace+1cq") > perf_mcp._mode_rank("eager")


def test_degrade_2cq_to_1cq_is_flagged_not_banked(tmp_path, monkeypatch):
    monkeypatch.setattr(perf_mcp, "_FULLPIPE_BASELINE_PATH", tmp_path / "base.json")
    monkeypatch.setattr(perf_mcp, "_FULLPIPE_TARGET_MS", 0.0)

    r1 = _drive(monkeypatch, 100.0, "trace", "trace+2cq")
    assert r1["status"] == "ok" and r1["mode"] == "trace+2cq"

    r2 = _drive(monkeypatch, 80.0, "trace", "trace+2cq")
    assert r2["status"] == "ok" and r2["delta_pct"] == -20.0

    # 1CQ number is numerically smaller (50 < 80) but a fidelity DOWNGRADE -> must be rejected.
    r3 = _drive(monkeypatch, 50.0, "trace", "trace+1cq")
    assert r3["status"] == "degraded"
    assert r3["baseline_mode"] == "trace+2cq"
    base = json.loads((tmp_path / "base.json").read_text())
    assert base["full_pipeline_ms"] == 80.0 and base["mode"] == "trace+2cq"  # NOT re-baselined to the weaker mode


def test_fidelity_upgrade_rebaselines(tmp_path, monkeypatch):
    p = tmp_path / "base.json"
    p.write_text(json.dumps({"full_pipeline_ms": 500.0, "method": "eager", "mode": "eager"}))
    monkeypatch.setattr(perf_mcp, "_FULLPIPE_BASELINE_PATH", p)
    monkeypatch.setattr(perf_mcp, "_FULLPIPE_TARGET_MS", 0.0)
    r = _drive(monkeypatch, 90.0, "trace", "trace+2cq")
    assert r["status"] == "ok" and r["mode"] == "trace+2cq"


def test_budget_guidance_present_only_when_2cq(monkeypatch):
    monkeypatch.setenv("TT_PERF_TRACE", "1")
    monkeypatch.setenv("TT_PERF_NUM_CQ", "2")
    b = perf_mcp._trace_budget_facts()
    assert b and b["num_command_queues"] == 2 and "trace_region_size" in b
    monkeypatch.setenv("TT_PERF_NUM_CQ", "1")
    assert perf_mcp._trace_budget_facts() is None
    monkeypatch.setenv("TT_PERF_NUM_CQ", "2")
    rk = getattr(perf_mcp.recall_knobs, "fn", perf_mcp.recall_knobs)("matmul")
    assert rk.get("budget") is not None
