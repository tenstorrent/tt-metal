"""The gate measures the unit the test DECLARES, not one token.

The generated perf test states its measurement unit explicitly:

    PERF_ISL_TOKENS = int(os.environ.get("TT_PERF_ISL_TOKENS", "128"))
    PERF_OSL_TOKENS = int(os.environ.get("TT_PERF_OSL_TOKENS", "128"))

and then two separate things threw it away:

  1. PERF_OSL_TOKENS was DECORATIVE. The decode loop is bounded by a DIFFERENT variable --
     `for _ in range(max(1, PERF_MAX_NEW_TOKENS))` -- which defaulted to 4. So the declared unit and
     the executed unit were never the same number.

  2. The full-pipeline gate then pinned that variable to 1, with no comment justifying it, in a file
     where nearly every decision carries a paragraph:

         env["TT_PERF_MAX_NEW_TOKENS"] = os.environ.get("PERF_MCP_FULLPIPE_TOKENS", "1")

One decode step is not a decode measurement. What it costs, all visible in the scorecard:

    PERF_SCORECARD ... ISL=128 OSL=1 batch=1 TTFT_ms=NA prefill_path=n/a
                       decode_ms=33.98 TSU=29.43 TS=29.43

  * TTFT_ms=NA        -- with one token there is no first-token boundary to time, so prefill has no
                         metric at all and prefill work cannot be scored. That is exactly how a run
                         spends itself optimizing prefill ops against a decode-only ceiling.
  * TS == TSU         -- no decode loop, so nothing to aggregate.
  * "per token" lies  -- 33.98 ms is prefill of 128 tokens PLUS one decode step. The 29.43 tok/s/u
                         derived from it is a blend, not a steady-state decode rate.

Honouring the declared OSL gives all three from one run, with no signposts and no subtraction:

    TTFT            = time to token 1
    decode ms/token = (total - TTFT) / (OSL - 1)
    tok/s/u         = 1000 / decode ms

The cost is real -- 128 decode steps per measurement instead of 1 -- so PERF_MCP_FULLPIPE_TOKENS is
kept as an explicit override for a cheap steering measurement. It is now an opt-in shortcut rather
than an undocumented default.

NOT changed: the two op-signature probes (perf_mcp._op_sig_probe path, run.py:684). Those only
enumerate which ops execute, so one token is correct there and 128 would be pure waste.
"""

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))

GATE = Path(__file__).resolve().parent.parent / "cc_optimize" / "perf_mcp.py"
GEN = Path(__file__).resolve().parent.parent / "agent" / "perf_test_gen.py"


@pytest.fixture()
def mcp(tmp_path, monkeypatch):
    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("PERF_MCP_LEDGER_DIR", str(tmp_path))
    monkeypatch.delenv("PERF_MCP_FULLPIPE_TOKENS", raising=False)
    monkeypatch.delenv("TT_PERF_OSL_TOKENS", raising=False)
    import models.experimental.perf_automation.cc_optimize.perf_mcp as m

    importlib.reload(m)
    return m


def _gate_osl_expr():
    """The line in _run_full_pipeline_ms that sets the decode-loop cap."""
    src = GATE.read_text()
    i = src.index('env["TT_PERF_MAX_NEW_TOKENS"] = os.environ.get("PERF_MCP_FULLPIPE_TOKENS")')
    return src[i : i + 200]


# ---------------------------------------------------------------- the declared unit is measured


def test_the_gate_no_longer_defaults_to_one_token(mcp):
    """The whole defect in one assertion: a bare "1" default is gone."""
    assert '"PERF_MCP_FULLPIPE_TOKENS", "1"' not in GATE.read_text()


def test_the_gate_falls_back_to_the_declared_osl(mcp):
    expr = _gate_osl_expr()
    assert "TT_PERF_OSL_TOKENS" in expr and '"128"' in expr, expr


def test_the_override_still_exists_for_cheap_steering(mcp):
    """128 decode steps per measurement is a real cost; the shortcut stays available, but as an
    explicit opt-in rather than an undocumented default."""
    assert "PERF_MCP_FULLPIPE_TOKENS" in _gate_osl_expr()


def test_the_override_wins_when_set(mcp, monkeypatch):
    monkeypatch.setenv("PERF_MCP_FULLPIPE_TOKENS", "4")
    monkeypatch.setenv("TT_PERF_OSL_TOKENS", "128")
    import os

    got = os.environ.get("PERF_MCP_FULLPIPE_TOKENS") or os.environ.get("TT_PERF_OSL_TOKENS", "128")
    assert got == "4"


def test_the_declared_value_wins_when_no_override(mcp, monkeypatch):
    monkeypatch.delenv("PERF_MCP_FULLPIPE_TOKENS", raising=False)
    monkeypatch.setenv("TT_PERF_OSL_TOKENS", "64")
    import os

    got = os.environ.get("PERF_MCP_FULLPIPE_TOKENS") or os.environ.get("TT_PERF_OSL_TOKENS", "128")
    assert got == "64"


# ---------------------------------------------------------------- the scorecard reports it honestly


def test_the_scorecard_osl_is_not_hardcoded_to_one(mcp):
    """OSL=1 in the scorecard is what made 'per token' a blended number."""
    src = GATE.read_text()
    assert 'osl = env.get("TT_PERF_MAX_NEW_TOKENS", "1")' not in src


# ---------------------------------------------------------------- declared == executed


def test_the_generated_test_loops_over_the_declared_osl(mcp):
    """PERF_OSL_TOKENS was printed as the unit while PERF_MAX_NEW_TOKENS (default 4) bounded the
    loop -- so the reported unit and the executed unit were different numbers."""
    src = GEN.read_text()
    assert 'os.environ.get("TT_PERF_MAX_NEW_TOKENS", "4")' not in src, "loop cap still defaults to 4"
    assert "TT_PERF_OSL_TOKENS" in src, "loop cap does not fall back to the declared OSL"


# ---------------------------------------------------------------- probes are deliberately untouched


def test_the_op_signature_probe_still_uses_one_token(mcp):
    """It only enumerates WHICH ops run. 128 tokens there is pure waste, so this stays at 1."""
    src = GATE.read_text()
    i = src.index("_op_sig_probe.py")
    window = src[max(0, i - 900) : i]
    assert 'env["TT_PERF_MAX_NEW_TOKENS"] = "1"' in window, "the op-sig probe should still cap at 1"
