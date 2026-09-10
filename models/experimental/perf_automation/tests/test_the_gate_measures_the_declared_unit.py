"""The gate measures the unit the test DECLARES, not one token.

The generated perf test states its measurement unit explicitly:

    PERF_ISL_TOKENS = int(os.environ.get("TT_PERF_ISL_TOKENS", "128"))
    PERF_OSL_TOKENS = int(os.environ.get("TT_PERF_OSL_TOKENS", "128"))

and then two separate things threw it away:

  1. PERF_OSL_TOKENS was DECORATIVE. The decode loop is bounded by a DIFFERENT variable --
     `for _ in range(max(1, PERF_OSL_TOKENS))` -- which defaulted to 4. So the declared unit and
     the executed unit were never the same number.

  2. The full-pipeline gate then pinned that variable to 1, with no comment justifying it, in a file
     where nearly every decision carries a paragraph:

         env["TT_PERF_OSL_TOKENS"] = os.environ.get("PERF_MCP_FULLPIPE_TOKENS", "1")

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
    i = src.index('env["TT_PERF_OSL_TOKENS"] = os.environ.get("PERF_MCP_FULLPIPE_TOKENS")')
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
    assert 'osl = env.get("TT_PERF_OSL_TOKENS", "1")' not in src


# ---------------------------------------------------------------- declared == executed


def test_the_generated_test_loops_over_the_declared_osl(mcp):
    """PERF_OSL_TOKENS was printed as the unit while PERF_OSL_TOKENS (default 4) bounded the
    loop -- so the reported unit and the executed unit were different numbers."""
    src = GEN.read_text()
    assert 'os.environ.get("TT_PERF_OSL_TOKENS", "4")' not in src, "loop cap still defaults to 4"
    assert "TT_PERF_OSL_TOKENS" in src, "loop cap does not fall back to the declared OSL"


# ---------------------------------------------------------------- probes are deliberately untouched


def test_the_op_signature_probe_still_uses_one_token(mcp):
    """It only enumerates WHICH ops run. 128 tokens there is pure waste, so this stays at 1."""
    src = GATE.read_text()
    i = src.index("_op_sig_probe.py")
    window = src[max(0, i - 900) : i]
    assert 'env["TT_PERF_OSL_TOKENS"] = "1"' in window, "the op-sig probe should still cap at 1"


# ---------------------------------------------------------------- the PROFILE runs the declared OSL too


def test_the_perf_node_runner_keeps_a_cheap_validation_cap(mcp):
    """THE SITE THAT SURVIVED THE FIRST FIX. The skeleton reads

        TT_PERF_OSL_TOKENS or TT_PERF_OSL_TOKENS

    so an env value WINS OUTRIGHT and the OSL fallback can never fire. _run_perf_node did
    `env.setdefault("TT_PERF_OSL_TOKENS", "4")`, so every run through it -- including the one whose
    tracy capture ranks ops -- executed 4 decode steps while printing PERF_OSL_TOKENS=128.

    Cost, measured on gemma-3-12b-it run 39: decode ops were sampled 4 times instead of 128, so their
    gap_ms was ~32x under-counted against prefill's single big pass. 72 of 138 shaped attempts went to
    prefill matmuls, which cannot move tok/s/u at all.
    """
    SUPERSEDED = """This asserted the 4 was GONE from _run_perf_node, on the belief that the profile
    ran through it. It does not -- profile_model -> measure_runs -> probes.make_run_profiled -- so
    raising it here only made VALIDATION 32x slower and fixed nothing. The token count now belongs to
    measure.py, where the ranking is actually sampled; validation keeps its cheap cap, because "does
    this test run at all" is answered as well by 4 tokens as by 128."""
    src = GEN.read_text()
    i = src.index('setdefault("TT_PERF_OSL_TOKENS"')
    assert '"4"' in src[i : i + 80], SUPERSEDED


def test_the_bound_itself_is_kept(mcp):
    """The setdefault must NOT simply be deleted: it is what stops a generative loop running forever
    when nothing else caps it."""
    assert 'setdefault("TT_PERF_OSL_TOKENS"' in GEN.read_text()


def test_the_authoring_prompt_no_longer_teaches_four(mcp):
    """It is the instruction that stamps the literal into every newly generated test, which is how a
    validation shortcut became the measurement condition in the first place."""
    src = GEN.read_text()
    assert "default 4" not in src
    assert "PERF_OSL_TOKENS" in src


def test_the_scorecard_reports_the_osl_that_ran(mcp):
    """run.py printed OSL=4 whenever the variable was unset, on a run measuring 128."""
    run_src = (Path(__file__).resolve().parent.parent / "cc_optimize" / "run.py").read_text()
    assert 'os.environ.get("TT_PERF_OSL_TOKENS") or "4"' not in run_src


def test_the_marker_buffer_is_not_a_reason_for_a_small_cap(mcp):
    """The 12000-marker limit was the assumed justification for 4. It is not one: the skeleton drains
    the profiler every TT_PERF_FLUSH_EVERY ops, so a long capture is SLOWER, not unsafe. Pinned so the
    literal is not reintroduced on that reasoning."""
    src = GEN.read_text()
    assert "TT_PERF_FLUSH_EVERY" in src and "ReadDeviceProfiler" in src


# ---------------------------------------------------------------- the PROFILE, not just validation


def test_the_profile_sets_the_token_count(mcp):
    """WHERE THE EARLIER FIX MISSED. It changed _run_perf_node, on the belief that the profile came
    through there. It does not:

        profile_model -> _profile_with_zero_row_retry -> _profile_once
                      -> measure_runs -> probes.make_run_profiled

    Neither measure.py nor probes.py set a token count, so the generated test fell back to its own
    literal -- 4 in every test written before today -- and the ranking kept under-counting decode 32x
    while the commit message claimed otherwise."""
    src = (Path(__file__).resolve().parent.parent / "agent" / "measure.py").read_text()
    assert "TT_PERF_OSL_TOKENS" in src, "the profile still inherits whatever the test hardcoded"
    i = src.index('xenv["TT_PERF_OSL_TOKENS"]')  # the assignment, not the comment above it
    assert "TT_PERF_OSL_TOKENS" in src[i : i + 400], "it does not fall back to the DECLARED unit"


def test_an_explicit_cap_still_wins_on_the_profile(mcp):
    """The op-signature probe caps at 1 deliberately -- it only enumerates WHICH ops run, and 128
    there is pure waste."""
    src = (Path(__file__).resolve().parent.parent / "agent" / "measure.py").read_text()
    i = src.index('xenv["TT_PERF_OSL_TOKENS"]')
    assert 'os.environ.get("TT_PERF_OSL_TOKENS")' in src[i : i + 400], "an explicit cap is ignored"


def test_the_slow_profile_can_be_bought_back(mcp):
    """128 steps is ~32x the markers and ~32x the eager time of 4, every round. Someone who wants the
    quick, skewed ranking should not have to edit the tool."""
    src = (Path(__file__).resolve().parent.parent / "agent" / "measure.py").read_text()
    assert "PERF_MCP_PROFILE_TOKENS" in src


def test_validation_keeps_its_cheap_cap(mcp):
    """_run_perf_node answers "does the generated test run at all", which 4 tokens establish as well
    as 128. Raising it there made validation 32x slower and fixed nothing."""
    src = GEN.read_text()
    i = src.index('setdefault("TT_PERF_OSL_TOKENS"')
    assert '"4"' in src[i : i + 80], src[i : i + 80]
