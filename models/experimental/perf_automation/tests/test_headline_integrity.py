"""RED tests for the headline-integrity bugs (audit round 2, items E/F + the mode-flip and
sample-asymmetry findings). All were verified against source on 2026-07-25.

E  The best-so-far full-pipeline file is ratcheted down by ANY lower reading, taken BEFORE
   PCC is known and regardless of a later revert. `after_ms` reads that file, so a candidate
   that measured faster and was then reverted for pcc_low still sets the run headline: the
   tree ends byte-identical to baseline while the report prints a speedup.

E2 Sample asymmetry: the BEFORE bookend medians 3 samples (run.py sets
   PERF_MCP_FULLPIPE_SAMPLES=3 for its own subprocess) while the per-lever gate in the MCP
   server defaults to 1. AFTER = min over many 1-sample readings vs BEFORE = median of 3
   manufactures the full noise range as a gain on every run.

E3 Mode flip: `base_mode != mode or best <= 0` re-baselines and returns status "ok" -- the
   agent's bank-a-win signal -- however much SLOWER the reading is, and across a unit change
   (per-token decode ms vs summed pipeline ms both arrive in TRACE_PER_TOKEN_MS).

F  When the run achieved nothing (real baseline not better than final), summary.py replaces
   the baseline with the SLOWEST measurement ever recorded, so 100 -> 105 with a 180 ms
   failed experiment prints `baseline 180.00 -> final 105.00 (+41.7%)`.

Hermetic: no device, no agent.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def _fresh_perf_mcp(tmp_path, **env):
    run = tmp_path / "models/experimental/perf_automation/runs/2026-01-01T00-00-00"
    (run / "profiles").mkdir(parents=True, exist_ok=True)
    (run / "manifest.json").write_text(
        json.dumps({"config": {"timeout": 10800, "metric": "device_ms"}, "perf_test_resolved": {"path": "t.py"}})
    )
    (run / "events.jsonl").write_text(
        json.dumps({"stage": "tracy_baseline", "event": "done", "seconds": 146.72}) + "\n"
    )
    keys = ("PERF_MCP_MANIFEST", "PERF_MCP_KERNEL_LOG", "PERF_MCP_FULLPIPE_SAMPLES", "TMPDIR")
    saved = {k: os.environ.get(k) for k in keys}
    os.environ["PERF_MCP_MANIFEST"] = str(run / "manifest.json")
    os.environ["PERF_MCP_KERNEL_LOG"] = str(tmp_path / "kernlog.json")
    os.environ["TMPDIR"] = str(tmp_path)
    os.environ.update(env)
    try:
        spec = importlib.util.spec_from_file_location("perf_mcp_headline_ut", _ROOT / "cc_optimize" / "perf_mcp.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["perf_mcp_headline_ut"] = mod
        spec.loader.exec_module(mod)
        # Point every /tmp-rooted path at tmp_path EXPLICITLY. Setting TMPDIR above is not enough:
        # perf_mcp computes these at import time via tempfile.gettempdir(), which CACHES its result,
        # so by the time this fixture runs some earlier test has already resolved it to the real /tmp
        # and the intended isolation silently does nothing. These tests then wrote 100.0 into the REAL
        # scoreboard -- which is exactly what corrupted a live 10-hour optimize run on 2026-07-27,
        # making its AFTER number 100.0 when every measurement was ~23.9 ms.
        mod._FULLPIPE_BASELINE_1CQ_PATH = tmp_path / "fullpipe_baseline_1cq.json"
        return mod
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


# --------------------------------------------------------------------------- E


def test_committed_best_is_separate_from_a_mere_reading(tmp_path):
    """There must be a COMMITTED best distinct from the latest candidate reading, and the
    promotion must be an explicit act tied to a commit."""
    m = _fresh_perf_mcp(tmp_path)
    assert hasattr(m, "_fullpipe_pending_path"), (
        "no pending/committed split: any lower reading overwrites the committed best, so a "
        "reverted candidate sets the run headline"
    )
    assert hasattr(m, "_promote_fullpipe_pending"), "no explicit promote-on-commit step exists"


def test_a_reverted_candidate_does_not_move_the_committed_best(tmp_path):
    """The exact fabrication: read 70 against a committed 100, then revert. The committed
    best must still be 100, because nothing was committed."""
    m = _fresh_perf_mcp(tmp_path)
    m._FULLPIPE_BASELINE_1CQ_PATH.write_text(json.dumps({"full_pipeline_ms": 100.0, "mode": "trace+1cq"}))
    m._record_fullpipe_candidate(70.0, "trace", "trace+1cq")
    m._discard_fullpipe_pending()
    best = json.loads(m._FULLPIPE_BASELINE_1CQ_PATH.read_text())["full_pipeline_ms"]
    assert best == 100.0, f"a reverted candidate moved the committed best to {best} -> fake headline speedup"


def test_a_committed_candidate_does_move_the_committed_best(tmp_path):
    """The fix must not block real wins."""
    m = _fresh_perf_mcp(tmp_path)
    m._FULLPIPE_BASELINE_1CQ_PATH.write_text(json.dumps({"full_pipeline_ms": 100.0, "mode": "trace+1cq"}))
    m._record_fullpipe_candidate(70.0, "trace", "trace+1cq")
    m._promote_fullpipe_pending()
    assert json.loads(m._FULLPIPE_BASELINE_1CQ_PATH.read_text())["full_pipeline_ms"] == 70.0


# --------------------------------------------------------------------------- E2


def test_candidate_and_bookend_use_the_same_sample_count(tmp_path):
    """AFTER = min over 1-sample readings vs BEFORE = median of 3 is a gain generator."""
    m = _fresh_perf_mcp(tmp_path)
    run_txt = (_ROOT / "cc_optimize" / "run.py").read_text()
    bookend = run_txt.split('PERF_MCP_FULLPIPE_SAMPLES", "')[1].split('"')[0]
    assert m._FULLPIPE_SAMPLES == int(bookend), (
        f"per-lever gate takes {m._FULLPIPE_SAMPLES} sample(s) but the BEFORE bookend medians "
        f"{bookend} -- the asymmetry alone manufactures a gain"
    )


# --------------------------------------------------------------------------- E3


def test_mode_flip_is_not_reported_as_a_bankable_win(tmp_path):
    """A unit/mode change cannot be differenced against the old baseline. It must re-establish
    the baseline WITHOUT returning the agent's bank-a-win status."""
    m = _fresh_perf_mcp(tmp_path)
    out = m._fullpipe_verdict_for(ms=5000.0, method="trace", mode="trace", best=60.0, base_mode="trace+1cq")
    assert out.get("status") != "ok", (
        f"a mode flip to a SLOWER, differently-united reading returned {out.get('status')!r} -- "
        "the agent's IRON RULE banks that as a verified win"
    )


def test_no_baseline_yet_is_still_recorded_cleanly(tmp_path):
    """Genuinely having no baseline is not a failure; it just is not a win either."""
    m = _fresh_perf_mcp(tmp_path)
    out = m._fullpipe_verdict_for(ms=100.0, method="trace", mode="trace+1cq", best=0.0, base_mode="")
    assert out.get("status") in ("baseline", "ok"), out
    # a first reading establishes the baseline; there is nothing to compare it to, so no delta
    assert out.get("delta_pct") is None, out


# --------------------------------------------------------------------------- F


def test_baseline_is_never_replaced_by_the_worst_measurement(tmp_path, monkeypatch):
    """A run that regressed must not print a speedup against an invented baseline."""
    from cc_optimize import summary as S

    attempts = [
        {"op_signature": "Matmul", "kernel_kind": "grid", "beat_baseline": True, "measured_ms": 105.0},
        {"op_signature": "Matmul", "kernel_kind": "dtype", "beat_baseline": False, "measured_ms": 180.0},
    ]
    p = tmp_path / "kernlog.json"
    p.write_text(json.dumps(attempts))
    import importlib.util as _ilu

    monkeypatch.setenv("PERF_MCP_LEDGER", str(tmp_path / "led.jsonl"))
    _sp = _ilu.spec_from_file_location("meas_int_ut", Path(S.__file__).with_name("measurements.py"))
    _m = _ilu.module_from_spec(_sp)
    _sp.loader.exec_module(_m)
    _m.record(_m.KIND_EAGER, _m.PHASE_BEFORE, 100.0, depth="16", mode="eager", source="test")
    _m.record(_m.KIND_EAGER, _m.PHASE_AFTER, 105.0, depth="16", mode="eager", source="test")
    out = S.render_summary(p, 100.0, final_override_ms=105.0, model="m")
    # 180.00 legitimately appears in the per-attempt table; the HEADLINE is what must be honest.
    hdr = next(ln for ln in out.splitlines() if "device time" in ln)  # headline was relabelled: it now
    # names WHAT was measured and over how many layers, because a bare "baseline -> final" hid a
    # 2-layer number being compared against a 16-layer one.
    assert "180.00" not in hdr, f"the slowest failed experiment was substituted as the baseline: {hdr.strip()}"
    assert "100.00" in hdr, f"headline lost the real baseline: {hdr.strip()}"
    assert "+" not in hdr.split("->")[1], f"printed a gain for a run that regressed: {hdr.strip()}"
    # split on "->" rather than "final": the headline now reads
    # "eager per-op device time (N layers):  A ms  ->  B ms", naming the measurement and depth
