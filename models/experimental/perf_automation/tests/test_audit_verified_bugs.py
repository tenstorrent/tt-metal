"""RED tests for the four audit findings VERIFIED against source on 2026-07-25.

Each one is an unfixed twin of a bug already paid for in this plan, or a hole of the same
family. All four were confirmed by reading the code, not inferred:

  A  git_commit / recall_knobs are listed in run.py _ALLOWED_TOOLS but carry NO @mcp.tool()
     decorator, so the agent is instructed to call two tools that do not exist -- while the
     private _record_committed_win (which appends beat_baseline: True with no measurement,
     no PCC and no commit) IS registered and agent-callable. Present in the origin checkout
     too, so it is pre-existing, not rebase fallout.
  B  check_pcc's catch-all calls _note_device_crash for ANY host-side exception -- the exact
     bug fixed in measure_candidate (BUG 1), still live on the correctness gate, where the
     blast radius is worse: 'crash' means revert-the-edit, and two of them reset the board.
  C  kernel_kind is WRITTEN prefixed ("knob:grid", minted at perf_mcp.py:673-690) and COUNTED
     bare (== "grid"), so a wedged knob never increments its *_tries counter and the ladder
     re-issues the identical rung forever. Same mismatch as the _level_of fix (BUG 2), but on
     the ladder rather than a report column. The correct normaliser already exists at :1810.
  D  A zero-op capture yields device_ms == 0.0, and the below-floor physics guard is gated
     `if i_ms > 0` -- so it is SKIPPED for exactly that case, and 0.0 ms is banked as
     verdict=valid, pct_faster=100.0, is_real_gain=True, then written as the new baseline.

Hermetic: no device, no agent, no claude subprocess.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def _fresh_perf_mcp(tmp_path):
    """Import perf_mcp against a throwaway manifest (it parses one at import time)."""
    run = tmp_path / "models/experimental/perf_automation/runs/2026-01-01T00-00-00"
    (run / "profiles").mkdir(parents=True, exist_ok=True)
    (run / "manifest.json").write_text(
        json.dumps({"config": {"timeout": 10800, "metric": "device_ms"}, "perf_test_resolved": {"path": "t.py"}})
    )
    (run / "events.jsonl").write_text(
        json.dumps({"stage": "tracy_baseline", "event": "done", "seconds": 146.72}) + "\n"
    )
    saved = {k: os.environ.get(k) for k in ("PERF_MCP_MANIFEST", "PERF_MCP_KERNEL_LOG")}
    os.environ["PERF_MCP_MANIFEST"] = str(run / "manifest.json")
    os.environ["PERF_MCP_KERNEL_LOG"] = str(tmp_path / "kernlog.json")
    try:
        spec = importlib.util.spec_from_file_location("perf_mcp_audit_ut", _ROOT / "cc_optimize" / "perf_mcp.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["perf_mcp_audit_ut"] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


def _registered_tools() -> set:
    """Which functions actually carry @mcp.tool(). Parsed, not imported, so it needs no device."""
    tree = ast.parse((_ROOT / "cc_optimize" / "perf_mcp.py").read_text())
    out = set()
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(
            (isinstance(d, ast.Call) and getattr(d.func, "attr", "") == "tool") or getattr(d, "attr", "") == "tool"
            for d in n.decorator_list
        ):
            out.add(n.name)
    return out


def _allowed_tools() -> set:
    """The perf-mcp tools run.py tells the agent it may call."""
    txt = (_ROOT / "cc_optimize" / "run.py").read_text()
    start = txt.index("_ALLOWED_TOOLS")
    block = txt[start : txt.index("]", start)]
    return {ln.split("mcp__perf-mcp__")[1].strip().strip('",') for ln in block.splitlines() if "mcp__perf-mcp__" in ln}


# --------------------------------------------------------------------------- A: registration


def test_every_allowed_tool_is_actually_registered():
    """The agent is told to call these by name. An unregistered one is not a soft failure:
    git_commit IS the bank-a-verified-win action and recall_knobs is the mandatory
    prior-knowledge lookup the prompt requires before editing."""
    missing = _allowed_tools() - _registered_tools()
    assert not missing, (
        f"run.py instructs the agent to call {sorted(missing)}, but they carry no @mcp.tool() "
        "decorator, so they are not exposed -- the agent cannot bank a win or recall a knob"
    )


def test_no_private_helper_is_exposed_to_the_agent():
    """_record_committed_win appends beat_baseline: True with no measurement, no PCC and no
    commit. It must not be an agent-callable tool."""
    private = {t for t in _registered_tools() if t.startswith("_")}
    assert not private, (
        f"private helpers exposed as agent tools: {sorted(private)} -- "
        "_record_committed_win can fabricate a verified win in one call"
    )


# --------------------------------------------------------------------------- B: check_pcc twin


def test_check_pcc_host_failure_is_not_a_device_crash(tmp_path):
    """A pytest collection error / missing node / launcher timeout is a MEASUREMENT failure.
    Reporting it as 'crash' reverts a correct edit and resets the board after two."""
    m = _fresh_perf_mcp(tmp_path)
    calls = []
    m._note_device_crash = lambda *a, **k: calls.append(a)
    m.run_pcc = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("unexpected CSV header in /tmp/x.csv: '\\n'"))
    out = m.check_pcc()
    assert out.get("status") != "crash", f"host-side parse failure reported as a device crash: {out}"
    assert not calls, "a host-side failure incremented the device-crash counter (two of these reset the board)"


def test_check_pcc_real_crash_is_still_a_crash(tmp_path):
    """The fix must not blind the gate to genuine device faults."""
    m = _fresh_perf_mcp(tmp_path)
    calls = []
    m._note_device_crash = lambda *a, **k: calls.append(a)
    m.run_pcc = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("TT_FATAL @ tt_cluster.cpp:281"))
    assert m.check_pcc().get("status") == "crash"
    assert calls, "a real TT_FATAL must still count as a device crash"


# --------------------------------------------------------------------------- C: ladder prefix


def test_wedged_knob_rung_counts_toward_its_own_retry_limit(tmp_path):
    """`knob:grid` != `grid`, so grid_tries stayed 0 and termination_check re-issued the same
    rung forever. The ladder deadlocks; this is worse than the report-column instance."""
    m = _fresh_perf_mcp(tmp_path)
    assert hasattr(m, "_normalise_rung"), "no shared rung normaliser: the prefixed/bare mismatch is still live"
    for prefixed, bare in (("knob:grid", "grid"), ("knob:dtype", "dtype"), ("rung:shard", "shard")):
        assert m._normalise_rung(prefixed) == bare, f"{prefixed} does not normalise to {bare}"
    assert m._normalise_rung("tt-lang") == "tt-lang", "a bare rung must survive normalisation unchanged"


def test_recorded_attempt_stores_a_bare_rung(tmp_path):
    """Normalise at the WRITE site too, so the counters work on logs written from now on."""
    m = _fresh_perf_mcp(tmp_path)
    m._persist_target({"op": "MatmulDeviceOperation", "rung": "knob:grid"})
    m._autorecord_wedge("device wedge")
    rows = json.loads((tmp_path / "kernlog.json").read_text())
    kinds = [r.get("kernel_kind") for r in rows if isinstance(r, dict)]
    assert "grid" in kinds, f"wedge recorded with a prefixed kind that no counter matches: {kinds}"

    # the OTHER write site: run.py records the round-killed wedge for the same target
    rt = (_ROOT / "cc_optimize" / "run.py").read_text()
    assert 'kind = target.get("rung") or "knob"' not in rt, "run.py still writes a prefixed rung"


# --------------------------------------------------------------------------- D: zero-op capture


def test_zero_device_ms_is_never_a_win():
    """A zero-op capture must not be banked. It is the same upstream zero-row condition as
    BUG 5, but it raises no exception -- so the retry never fires and 0.0 ms reads as -100%."""
    from agent.handlers import remeasure as m

    ok, reason = m._comparable({"device_ms": 100.0, "op_count": 1900}, {"device_ms": 0.0, "op_count": 0}, floor_ms=8.0)
    assert ok is False, "device_ms == 0.0 accepted as a valid measurement -> pct_faster 100.0, is_real_gain True"
    assert "zero" in (reason or "").lower() or "captur" in (reason or "").lower(), reason


def test_zero_op_baseline_does_not_whitelist_everything():
    """_comparable returned (True, None) whenever the BASELINE had zero ops, so one bad
    baseline waved through every later measurement."""
    from agent.handlers import remeasure as m

    ok, _ = m._comparable({"device_ms": 0.0, "op_count": 0}, {"device_ms": 3.0, "op_count": 12}, floor_ms=8.0)
    assert ok is False, "a degenerate zero-op baseline whitelisted a later measurement"
