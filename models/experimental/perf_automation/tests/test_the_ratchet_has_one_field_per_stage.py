# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The ratchet remembered ONE stage, so work on any other stage read as no-gain and was reverted.

The bar is `TRACE_PER_TOKEN_MS`, which trace_replay sets from the DECODE stage -- that is what
tok/s/u means, and it is the right headline. But it is one stage of two. A prefill lever leaves it
flat by construction, so:

    gate_set_new_best:  ms < prev            <- prev is the decode best; prefill cannot enter
    _record_fullpipe_candidate(ms, ...)      <- the doc had no field for a prefill reading at all

On gemma-3-12b-it the traced-prefill fix took TTFT from 95.19 to 54.81 ms -- a 42% cut in the metric
a user waits on before the first token appears -- and there was nowhere for it to be recorded. The
lever scores no-gain, the loop reverts it, and the next round is free to try it again.

So the bar carries every stage the pipeline declares, and each field ratchets on its own: the
headline never rises and neither does any stage. A commit that beats prefill while decode holds
flat now promotes the prefill best, which a whole-document promote/refuse could not express.

Two things this must NOT do:

  * credit a lever that moves time from one stage into another -- stage_win requires that some stage
    improved AND that none regressed;
  * let a stage override a decode REGRESSION -- the status still follows the headline, because
    tok/s/u is the number the run is being optimized for.

  r1  the gemma3 prefill case
  r2  the stage rule: improvement is not enough on its own
  r3  each field ratchets separately through a commit
  r4  the delta is stated in the metric that actually moved
"""

import importlib.util
import sys
from pathlib import Path

_PA = Path(__file__).resolve().parent.parent
_CC = _PA / "cc_optimize"
sys.path.insert(0, str(_PA))


def _pm(monkeypatch, tmp_path, model="gemma3"):
    monkeypatch.setenv("PERF_MCP_MODEL_ROOT", str(tmp_path / model))
    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("PERF_MCP_TASK", "main")
    (tmp_path / model).mkdir(parents=True, exist_ok=True)
    (tmp_path / "state").mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location("pmcp_stage_ratchet", str(_CC / "perf_mcp.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _verdict(m, monkeypatch, ms, best, stages=None, stage_win=False):
    fp = {"status": "ok", "full_pipeline_ms": ms, "best_ms": best}
    if stages is not None:
        fp["stages"] = stages
        fp["stage_win"] = stage_win
    monkeypatch.setattr(m, "gate_verdicts", lambda: {"full_pipeline": fp})


def _row(ms, best, improved=False, regressed=False):
    return {"ms": ms, "best": best, "improved": improved, "regressed": regressed}


# --------------------------------------------------------------------------- r1 THE CASE
def test_r1_a_prefill_win_with_a_flat_decode_is_a_win(monkeypatch, tmp_path):
    """TTFT 95.19 -> 54.81 while the decode headline holds at 32.22. The old rule scored no-gain."""
    m = _pm(monkeypatch, tmp_path)
    _verdict(
        m,
        monkeypatch,
        32.22,
        32.22,
        {"prefill": _row(54.81, 95.19, improved=True), "decode": _row(32.22, 32.22)},
        stage_win=True,
    )
    assert m.gate_set_new_best() is True


def test_r1_a_flat_everything_is_still_not_a_win(monkeypatch, tmp_path):
    m = _pm(monkeypatch, tmp_path)
    _verdict(m, monkeypatch, 32.22, 32.22, {"prefill": _row(95.19, 95.19), "decode": _row(32.22, 32.22)})
    assert m.gate_set_new_best() is False


# --------------------------------------------------------------------------- r2 THE STAGE RULE
def test_r2_moving_time_between_stages_is_not_a_win(monkeypatch, tmp_path):
    """A lever that buys prefill at decode's expense improves one stage and regresses the other.
    stage_win is False, so the whole question falls back to the headline."""
    m = _pm(monkeypatch, tmp_path)
    now = {"prefill": 40.0, "decode": 40.0}
    bar = {"prefill": 95.19, "decode": 32.22}
    d = m._stage_deltas(now, bar)
    assert d["prefill"]["improved"] and d["decode"]["regressed"]
    assert m._win_from_verdict({"stage_win": False, "stages": d}, 40.0, 32.22)[0] is False


def test_r2_an_improvement_inside_the_tolerance_is_not_one(monkeypatch, tmp_path):
    """The same tolerance the headline uses. Below the board's spread it is not a result."""
    m = _pm(monkeypatch, tmp_path)
    d = m._stage_deltas({"prefill": 95.0}, {"prefill": 95.19})
    assert d["prefill"]["improved"] is False


def test_r2_a_stage_with_no_bar_yet_is_neither(monkeypatch, tmp_path):
    m = _pm(monkeypatch, tmp_path)
    d = m._stage_deltas({"prefill": 54.81}, {})
    assert d["prefill"]["improved"] is False and d["prefill"]["regressed"] is False
    assert d["prefill"]["best"] is None and d["prefill"]["delta_pct"] is None


def test_r2_a_decode_regression_is_a_regression_whatever_prefill_did(monkeypatch, tmp_path):
    """The status follows the headline: tok/s/u is what the run is being optimized for."""
    m = _pm(monkeypatch, tmp_path)
    _verdict(
        m,
        monkeypatch,
        40.00,
        32.22,
        {"prefill": _row(54.81, 95.19, improved=True), "decode": _row(40.0, 32.22, regressed=True)},
        stage_win=False,
    )
    assert m.gate_set_new_best() is False


# --------------------------------------------------------------------------- r3 THE RATCHET
def test_r3_each_field_keeps_its_own_minimum(monkeypatch, tmp_path):
    """A prefill best must survive a commit whose headline did not beat the committed one -- the
    whole-document promote/refuse could not express that."""
    m = _pm(monkeypatch, tmp_path)
    m._write_fullpipe_bar(
        {"full_pipeline_ms": 32.22, "mode": "trace+1cq", "stages": {"prefill": 95.19, "decode": 32.22}}
    )
    m._fullpipe_pending_path().write_text(
        __import__("json").dumps(
            {
                "full_pipeline_ms": 32.60,  # slower headline
                "mode": "trace+1cq",
                "sha": "deadbeef",
                "stages": {"prefill": 54.81, "decode": 32.60},
            }
        )
    )
    monkeypatch.setattr(m, "_head_sha_quiet", lambda: "cafebabe")
    assert m._promote_fullpipe_if_committed() is True
    bar = __import__("json").loads(m._FULLPIPE_BASELINE_1CQ_PATH.read_text())
    assert bar["full_pipeline_ms"] == 32.22, "the headline moved backwards"
    assert bar["stages"]["prefill"] == 54.81, "the prefill best was not ratcheted"
    assert bar["stages"]["decode"] == 32.22, "a stage moved backwards"


def test_r3_a_mode_change_rebaselines_the_stages_too(monkeypatch, tmp_path):
    """eager and trace numbers are not comparable, so the old stage bests are meaningless rather
    than better -- the same rule the headline has always had."""
    m = _pm(monkeypatch, tmp_path)
    m._write_fullpipe_bar({"full_pipeline_ms": 20.0, "mode": "eager", "stages": {"prefill": 10.0}})
    m._fullpipe_pending_path().write_text(
        __import__("json").dumps(
            {"full_pipeline_ms": 32.22, "mode": "trace+1cq", "sha": "a", "stages": {"prefill": 54.81}}
        )
    )
    monkeypatch.setattr(m, "_head_sha_quiet", lambda: "b")
    m._promote_fullpipe_if_committed()
    bar = __import__("json").loads(m._FULLPIPE_BASELINE_1CQ_PATH.read_text())
    assert bar["stages"]["prefill"] == 54.81 and bar["full_pipeline_ms"] == 32.22


def test_r3_min_stages_keeps_a_stage_present_in_only_one_side(monkeypatch, tmp_path):
    m = _pm(monkeypatch, tmp_path)
    assert m._min_stages({"prefill": 95.0}, {"decode": 32.0}) == {"prefill": 95.0, "decode": 32.0}
    assert m._min_stages({"prefill": 95.0}, {"prefill": 54.0}) == {"prefill": 54.0}
    assert m._min_stages({"prefill": 54.0}, {"prefill": 95.0}) == {"prefill": 54.0}
    assert m._min_stages(None, {"prefill": 0}) == {}  # a zero reading is not a measurement


# --------------------------------------------------------------------------- r4 THE DELTA
def test_r4_the_delta_is_stated_in_the_metric_that_moved(monkeypatch, tmp_path):
    """Otherwise the report prints a ✓ beside `+0.00 ms`, which is how the Δ column and the win
    flag came to disagree the first time."""
    m = _pm(monkeypatch, tmp_path)
    fp = {"stage_win": True, "stages": {"prefill": _row(54.81, 95.19, improved=True)}}
    win, delta, metric = m._win_from_verdict(fp, 32.22, 32.22)
    assert win is True and metric == "prefill"
    assert abs(delta - (54.81 - 95.19)) < 1e-6


def test_r4_a_headline_win_is_still_stated_end_to_end(monkeypatch, tmp_path):
    m = _pm(monkeypatch, tmp_path)
    fp = {"stage_win": True, "stages": {"prefill": _row(54.81, 95.19, improved=True)}}
    win, delta, metric = m._win_from_verdict(fp, 30.00, 32.22)
    assert win is True and metric == "end_to_end" and abs(delta - (30.00 - 32.22)) < 1e-6


def test_r4_stage_win_with_nothing_to_attribute_it_to_falls_back(monkeypatch, tmp_path):
    """An internally inconsistent verdict must not invent a metric."""
    m = _pm(monkeypatch, tmp_path)
    win, delta, metric = m._win_from_verdict({"stage_win": True, "stages": {}}, 32.22, 32.22)
    assert win is False and metric == "end_to_end"


def test_r4_one_rule_not_three(monkeypatch, tmp_path):
    """gate_set_new_best and the per-attempt verdict must answer 'is this a win' identically --
    three disagreeing implementations is a bug this module has already had."""
    src = (_CC / "perf_mcp.py").read_text()
    for fn in ("def gate_set_new_best", "def _attempt_fullpipe_verdict"):
        body = src[src.index(fn) : src.index("\ndef ", src.index(fn) + 1)]
        assert "_win_from_verdict" in body, fn
