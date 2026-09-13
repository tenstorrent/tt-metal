"""The committed-best bar is a ratchet. A ratchet does not turn backwards.

`_promote_fullpipe_pending` copies the pending reading into the baseline file unconditionally:

    _FULLPIPE_BASELINE_1CQ_PATH.write_text(src.read_text())      # perf_mcp.py:2558

so the bar tracks the LAST committed measurement rather than the BEST one. One bad reading that
happens to be committed replaces a good reference, and every later verdict is graded against it.

That is not hypothetical. On gemma-3-12b-it the ledger reads:

    35.3772   committed-best
    34.9066   committed-best
    55.0264   committed-best        <- a measured REGRESSION became the bar

55.0264 ms is 18.2 tok/s/u against a model that runs at 28.6. The next run then graded every attempt
against 18.2, so anything at all would have banked as a ~20 ms win and been written into the ladder
as conclusive. The run was stopped before that happened, but nothing in the code prevented it.

Where the 55 came from is its own story -- the concat-heads L1 shard is a real 1.6-1.9x regression the
agent kept re-applying -- but the bar must survive a bad commit regardless of why one occurs.

So: a promotion that would move the bar UP is refused, and the previous best stands. A mode change
(eager -> trace) still re-baselines, because those numbers are not comparable and the old value is
meaningless rather than better.
"""

import importlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))


@pytest.fixture()
def mcp(tmp_path, monkeypatch):
    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("PERF_MCP_LEDGER_DIR", str(tmp_path))
    import models.experimental.perf_automation.cc_optimize.perf_mcp as m

    importlib.reload(m)
    return m


def _bar(mcp):
    p = mcp._FULLPIPE_BASELINE_1CQ_PATH
    return json.loads(p.read_text())["full_pipeline_ms"] if p.exists() else None


def _set_bar(mcp, ms, mode="trace+1cq"):
    mcp._FULLPIPE_BASELINE_1CQ_PATH.write_text(json.dumps({"full_pipeline_ms": ms, "mode": mode, "method": "trace"}))


def _pending(mcp, ms, mode="trace+1cq"):
    mcp._fullpipe_pending_path().write_text(json.dumps({"full_pipeline_ms": ms, "mode": mode, "method": "trace"}))


# ---------------------------------------------------------------- the reported case


def test_a_regression_does_not_become_the_bar(mcp):
    """The gemma3 case: 34.9066 was the best, a 55.0264 commit replaced it."""
    _set_bar(mcp, 34.9066)
    _pending(mcp, 55.0264)
    mcp._promote_fullpipe_pending()
    assert _bar(mcp) == 34.9066, _bar(mcp)


def test_an_improvement_still_moves_the_bar(mcp):
    """The ratchet must still turn the way it is meant to."""
    _set_bar(mcp, 35.0)
    _pending(mcp, 34.5)
    mcp._promote_fullpipe_pending()
    assert _bar(mcp) == 34.5


def test_an_equal_reading_is_accepted(mcp):
    """Not worse, so nothing to protect against; refusing it would leave the pending file dangling."""
    _set_bar(mcp, 35.0)
    _pending(mcp, 35.0)
    mcp._promote_fullpipe_pending()
    assert _bar(mcp) == 35.0


def test_the_first_ever_reading_sets_the_bar(mcp):
    """No bar yet -- there is nothing to compare against and the reading IS the reference."""
    _pending(mcp, 40.0)
    mcp._promote_fullpipe_pending()
    assert _bar(mcp) == 40.0


# ---------------------------------------------------------------- a mode change still re-baselines


def test_switching_measurement_mode_rebaselines_even_if_slower(mcp):
    """eager and trace numbers are not comparable, so the old value is meaningless rather than
    better. Refusing here would pin the bar to a number from a different measurement entirely."""
    _set_bar(mcp, 20.0, mode="eager")
    _pending(mcp, 35.0, mode="trace+1cq")
    mcp._promote_fullpipe_pending()
    assert _bar(mcp) == 35.0


# ---------------------------------------------------------------- housekeeping


def test_the_pending_file_is_consumed_either_way(mcp):
    """Left behind, it would be promoted again on the next call and the refusal would not stick."""
    _set_bar(mcp, 34.0)
    _pending(mcp, 55.0)
    mcp._promote_fullpipe_pending()
    assert not mcp._fullpipe_pending_path().exists()


def test_no_pending_reading_is_a_no_op(mcp):
    _set_bar(mcp, 34.0)
    mcp._promote_fullpipe_pending()
    assert _bar(mcp) == 34.0


def test_a_corrupt_pending_file_does_not_destroy_the_bar(mcp):
    """This runs on every gate call; a bad write must not take the reference with it."""
    _set_bar(mcp, 34.0)
    mcp._fullpipe_pending_path().write_text("{not json")
    mcp._promote_fullpipe_pending()
    assert _bar(mcp) == 34.0
