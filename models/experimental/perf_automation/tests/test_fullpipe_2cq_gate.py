# SPDX-License-Identifier: Apache-2.0
"""Full-pipeline gate: trace+1CQ end to end (single track, single baseline).

The tool is trace+1cq end to end — there is no second-command-queue bookend and no second track. One
1cq baseline file holds the best-so-far; a faster candidate banks (ratchets the best down), a slower
one is flagged 'diverged' and not banked, and a genuine fidelity UPGRADE (eager -> trace+1cq)
re-baselines.
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
    # trace+1cq is the only trace path; any other declared path collapses to bare "trace".
    assert perf_mcp._fullpipe_mode("trace", "trace+1cq") == "trace+1cq"
    assert perf_mcp._fullpipe_mode("trace", None) == "trace"
    assert perf_mcp._fullpipe_mode("eager", None) == "eager"
    # ranks: eager (0) is below trace, and trace == trace+1cq (same fidelity, one track).
    assert perf_mcp._mode_rank("trace+1cq") == perf_mcp._mode_rank("trace") > perf_mcp._mode_rank("eager")


def test_records_and_ratchets_best(tmp_path, monkeypatch):
    # First reading records the 1cq baseline; a faster candidate banks and ratchets the best down.
    monkeypatch.setattr(perf_mcp, "_FULLPIPE_BASELINE_1CQ_PATH", tmp_path / "base_1cq.json")
    monkeypatch.setattr(perf_mcp, "_FULLPIPE_TARGET_MS", 0.0)

    r1 = _drive(monkeypatch, 90.0, "trace", "trace+1cq")
    assert r1["status"] == "ok" and r1["mode"] == "trace+1cq"

    r2 = _drive(monkeypatch, 84.0, "trace", "trace+1cq")
    assert r2["status"] == "ok" and r2["delta_pct"] < 0  # faster 1cq candidate reads as a win
    # The COMMITTED best does not move on a reading alone. It used to ratchet down immediately --
    # before PCC was known and regardless of a later revert -- so a candidate that measured faster
    # and was then reverted still set the run's AFTER headline while the tree was unchanged. The
    # reading is held pending and promoted only once a commit is actually observed.
    assert json.loads((tmp_path / "base_1cq.json").read_text())["full_pipeline_ms"] == 90.0
    pend = json.loads(perf_mcp._fullpipe_pending_path().read_text())
    assert pend["full_pipeline_ms"] == 84.0, "the faster reading was not held as pending"
    assert perf_mcp._promote_fullpipe_pending() is True
    assert json.loads((tmp_path / "base_1cq.json").read_text())["full_pipeline_ms"] == 84.0


def test_slower_is_diverged_not_banked(tmp_path, monkeypatch):
    monkeypatch.setattr(perf_mcp, "_FULLPIPE_BASELINE_1CQ_PATH", tmp_path / "base_1cq.json")
    monkeypatch.setattr(perf_mcp, "_FULLPIPE_TARGET_MS", 0.0)

    r1 = _drive(monkeypatch, 90.0, "trace", "trace+1cq")
    assert r1["status"] == "ok"
    # 100 > 90 * (1 + tol) -> diverged; the best-so-far baseline stays at 90.
    r2 = _drive(monkeypatch, 100.0, "trace", "trace+1cq")
    assert r2["status"] == "diverged"
    assert json.loads((tmp_path / "base_1cq.json").read_text())["full_pipeline_ms"] == 90.0


def test_eager_to_trace_upgrade_rebaselines(tmp_path, monkeypatch):
    # A fidelity UPGRADE (eager -> trace+1cq) re-baselines rather than cross-comparing incomparable modes.
    p = tmp_path / "base_1cq.json"
    p.write_text(json.dumps({"full_pipeline_ms": 500.0, "method": "eager", "mode": "eager"}))
    monkeypatch.setattr(perf_mcp, "_FULLPIPE_BASELINE_1CQ_PATH", p)
    monkeypatch.setattr(perf_mcp, "_FULLPIPE_TARGET_MS", 0.0)
    r = _drive(monkeypatch, 90.0, "trace", "trace+1cq")
    assert r["status"] == "ok" and r["mode"] == "trace+1cq"
    assert json.loads(p.read_text())["full_pipeline_ms"] == 90.0


def test_reset_clears_1cq_baseline(tmp_path, monkeypatch):
    import importlib.util as _u

    spec = _u.spec_from_file_location(
        "cc_run_reset", str(Path(__file__).resolve().parents[1] / "cc_optimize" / "run.py")
    )
    run = _u.module_from_spec(spec)
    spec.loader.exec_module(run)
    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("PERF_MCP_LEDGER_DIR", str(tmp_path))
    monkeypatch.setattr(run.tempfile, "gettempdir", lambda: str(tmp_path))
    # the scoreboard is keyed by (model, task) now -- an unkeyed global file let any process on
    # the box overwrite a live run's AFTER number
    _f = tmp_path / run._fullpipe_1cq_name()
    _f.write_text("{}")
    run._reset_fullpipe_baselines()
    assert not _f.exists()


def test_read_fullpipe_best_1cq(tmp_path, monkeypatch):
    # The AFTER scoreboard number is the best committed 1cq verdict read back from the baseline file.
    import importlib.util as _u

    spec = _u.spec_from_file_location(
        "cc_run_read", str(Path(__file__).resolve().parents[1] / "cc_optimize" / "run.py")
    )
    run = _u.module_from_spec(spec)
    spec.loader.exec_module(run)
    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("PERF_MCP_LEDGER_DIR", str(tmp_path))
    monkeypatch.setattr(run.tempfile, "gettempdir", lambda: str(tmp_path))
    assert run._read_fullpipe_best_1cq() == (None, "")  # now returns (ms, mode): the mode decides
    # whether the AFTER number is even comparable to the BEFORE bookend
    (tmp_path / run._fullpipe_1cq_name()).write_text(
        json.dumps({"full_pipeline_ms": 42.5, "method": "trace", "mode": "trace+1cq"})
    )
    assert run._read_fullpipe_best_1cq() == (42.5, "trace+1cq")
