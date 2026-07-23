# SPDX-License-Identifier: Apache-2.0
"""Full-pipeline gate: trace+1CQ per-iteration track vs trace+2CQ bookend track + reserved-budget guidance.

Two CQ tracks with SEPARATE baselines (PERF_MCP_FULLPIPE_CQ selects which):
  - cq=1 (mid-loop): the ROBUST per-iteration signal — always engages, banks compute-op wins; a mid-loop
    1CQ result must NOT be flagged 'degraded' against the 2CQ bookend (different track).
  - cq=2 (start/end bookend): the production ship metric. WITHIN this track a fall to 1CQ (2CQ failed to
    engage) is still flagged 'degraded' and NOT banked — before/after must stay 2CQ-comparable.
A legitimate fidelity UPGRADE (eager->trace, ->2cq) re-baselines within its track.
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
    # BOOKEND track (cq=2): a fall to 1CQ within it is a fidelity downgrade -> flagged, not banked.
    monkeypatch.setenv("PERF_MCP_FULLPIPE_CQ", "2")
    monkeypatch.setattr(perf_mcp, "_FULLPIPE_BASELINE_PATH", tmp_path / "base.json")
    monkeypatch.setattr(perf_mcp, "_FULLPIPE_BASELINE_1CQ_PATH", tmp_path / "base_1cq.json")
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


def test_1cq_midloop_is_its_own_track_not_degraded(tmp_path, monkeypatch):
    # The new split: a mid-loop 1CQ result must NOT be flagged 'degraded' against the 2CQ bookend
    # baseline — different track — and it banks a faster 1CQ candidate as a real per-iteration win.
    monkeypatch.setattr(perf_mcp, "_FULLPIPE_BASELINE_PATH", tmp_path / "base.json")
    monkeypatch.setattr(perf_mcp, "_FULLPIPE_BASELINE_1CQ_PATH", tmp_path / "base_1cq.json")
    monkeypatch.setattr(perf_mcp, "_FULLPIPE_TARGET_MS", 0.0)
    # a 2CQ bookend baseline already exists
    (tmp_path / "base.json").write_text(json.dumps({"full_pipeline_ms": 80.0, "method": "trace", "mode": "trace+2cq"}))

    monkeypatch.setenv("PERF_MCP_FULLPIPE_CQ", "1")
    r1 = _drive(monkeypatch, 90.0, "trace", "trace+1cq")
    assert r1["status"] == "ok" and r1["mode"] == "trace+1cq"  # NOT degraded despite the 2cq baseline
    r2 = _drive(monkeypatch, 84.0, "trace", "trace+1cq")
    assert r2["status"] == "ok" and r2["delta_pct"] < 0  # faster 1cq candidate banks as a win
    # the 2cq bookend baseline was left untouched by the 1cq track
    assert json.loads((tmp_path / "base.json").read_text())["mode"] == "trace+2cq"


def test_fidelity_upgrade_rebaselines(tmp_path, monkeypatch):
    monkeypatch.setenv("PERF_MCP_FULLPIPE_CQ", "2")
    p = tmp_path / "base.json"
    p.write_text(json.dumps({"full_pipeline_ms": 500.0, "method": "eager", "mode": "eager"}))
    monkeypatch.setattr(perf_mcp, "_FULLPIPE_BASELINE_PATH", p)
    monkeypatch.setattr(perf_mcp, "_FULLPIPE_BASELINE_1CQ_PATH", tmp_path / "base_1cq.json")
    monkeypatch.setattr(perf_mcp, "_FULLPIPE_TARGET_MS", 0.0)
    r = _drive(monkeypatch, 90.0, "trace", "trace+2cq")
    assert r["status"] == "ok" and r["mode"] == "trace+2cq"


def test_track_mode_collapses_2cq_in_1cq_track():
    # In the 1-CQ track a 2cq reading is not extra fidelity -> collapse to 1cq.
    assert perf_mcp._track_mode("trace+2cq", 1) == "trace+1cq"
    assert perf_mcp._track_mode("trace+1cq", 1) == "trace+1cq"
    assert perf_mcp._track_mode("trace", 1) == "trace"
    assert perf_mcp._track_mode("eager", 1) == "eager"
    # The 2-CQ bookend track keeps 2cq as the top rank.
    assert perf_mcp._track_mode("trace+2cq", 2) == "trace+2cq"


def test_stale_2cq_entry_in_1cq_file_does_not_veto(tmp_path, monkeypatch):
    # THE BUG: a leftover 1cq baseline pinned at trace+2cq (rank 2) used to veto every
    # live trace reading (rank 1) forever, because the degraded branch returns before the
    # rewrite. _track_mode collapses the stale 2cq entry to 1cq so the live reading banks.
    monkeypatch.setattr(perf_mcp, "_FULLPIPE_BASELINE_PATH", tmp_path / "base.json")
    monkeypatch.setattr(perf_mcp, "_FULLPIPE_BASELINE_1CQ_PATH", tmp_path / "base_1cq.json")
    monkeypatch.setattr(perf_mcp, "_FULLPIPE_TARGET_MS", 0.0)
    (tmp_path / "base_1cq.json").write_text(
        json.dumps({"full_pipeline_ms": 80.0, "method": "trace", "mode": "trace+2cq"})
    )
    monkeypatch.setenv("PERF_MCP_FULLPIPE_CQ", "1")
    r1 = _drive(monkeypatch, 3.5392, "trace", "trace")
    assert r1["status"] == "ok"
    r2 = _drive(monkeypatch, 3.3784, "trace", "trace")
    assert r2["status"] == "ok" and r2["delta_pct"] is not None and r2["delta_pct"] < 0


def test_reset_clears_both_baseline_files(tmp_path, monkeypatch):
    import importlib.util as _u

    spec = _u.spec_from_file_location(
        "cc_run_reset", str(Path(__file__).resolve().parents[1] / "cc_optimize" / "run.py")
    )
    run = _u.module_from_spec(spec)
    spec.loader.exec_module(run)
    monkeypatch.setattr(run.tempfile, "gettempdir", lambda: str(tmp_path))
    for name in ("perf_mcp_full_pipeline_baseline.json", "perf_mcp_full_pipeline_baseline_1cq.json"):
        (tmp_path / name).write_text("{}")
    run._reset_fullpipe_baselines()
    assert not (tmp_path / "perf_mcp_full_pipeline_baseline.json").exists()
    assert not (tmp_path / "perf_mcp_full_pipeline_baseline_1cq.json").exists()


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
