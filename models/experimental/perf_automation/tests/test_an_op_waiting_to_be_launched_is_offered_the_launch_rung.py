"""An op that is waiting to be launched must be offered the rung that addresses waiting.

A profile tags an op stalled on the launch loop `dispatch`. The whole-run bucket for the same wait
is tagged `host`. _op_ladder_status read only the second, so an op carrying the first fell through
to the knob rungs -- and _RUNG_PRIORITY, four hundred lines above, already says of that exact case:
"Nothing on the knob rungs addresses dispatch: the op is waiting on the host loop that launches it."

On voxtral_mini_3b_2507 (2026-09-08) 236 of the run's 238 dispatch-tagged ops were prefill, the only
stage still short of its band. Every one was answered with an arithmetic knob while the stage sat at
92.6 of 512 GB/s -- 18% of the roof it binds on -- and eight of nine rungs returned +0.00 ms. Decode
carried no dispatch-tagged op at all, which is why only the stuck stage was affected.

The two spellings are one state and are named once, so a third reader cannot pick a different half.
"""

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))


@pytest.fixture()
def mcp(tmp_path, monkeypatch):
    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("PERF_MCP_LEDGER_DIR", str(tmp_path))
    monkeypatch.setenv("PERF_MCP_KERNEL_LOG", str(tmp_path / "kernel_attempts.json"))
    import models.experimental.perf_automation.cc_optimize.perf_mcp as m

    importlib.reload(m)
    return m


def _rung(m, bound):
    """The rung the ladder offers an untouched op carrying this bound tag."""
    op = {"op": "MatmulDeviceOperation 3328 x 3072 x 8192", "bound_by": bound, "bucket": "matmul"}
    return m._op_ladder_status(op, "matmul", [])[1]


def test_every_spelling_of_waiting_reaches_the_same_rung(mcp):
    """The bug: one spelling got the launch rung, the other got arithmetic."""
    offered = {bound: _rung(mcp, bound) for bound in sorted(mcp._DISPATCH_BOUND)}

    assert len(mcp._DISPATCH_BOUND) > 1, "there is nothing to disagree about"
    assert len(set(offered.values())) == 1, offered


def test_a_waiting_op_is_not_sent_to_tune_arithmetic(mcp):
    """The point of the rung, stated the way the ladder states it.

    _RUNG_PRIORITY's dispatch entry says of the knob rungs: "Nothing on the knob rungs addresses
    dispatch: the op is waiting on the host loop that launches it." So the thing to assert is not
    which lever is named -- that is the ladder's business and may be re-ordered -- but that a
    waiting op is NOT handed one of the knobs. _KNOBS owns that list; none is typed here.
    """
    for bound in sorted(mcp._DISPATCH_BOUND):
        assert mcp._normalise_rung(_rung(mcp, bound)) not in mcp._KNOBS, (bound, _rung(mcp, bound))


def test_an_op_that_is_not_waiting_still_gets_a_knob(mcp):
    """The fix must widen one branch, not capture every op into it."""
    others = [b for b in mcp._RUNG_PRIORITY if b and b not in mcp._DISPATCH_BOUND]
    assert others, "no non-waiting bound to compare against"

    for bound in others + [""]:
        assert mcp._normalise_rung(_rung(mcp, bound)) in mcp._KNOBS, (bound, _rung(mcp, bound))


def test_the_pair_is_named_once(mcp):
    """Three readers of this state drifted apart; a restated pair is how that happened."""
    src = Path(mcp.__file__).read_text()

    assert src.count('("host", "dispatch")') == 0, "a reader is restating the pair instead of reading it"
    assert src.count('bound == "host"') == 0, "a reader still hears only one spelling"


def _spent_launch_attempts(mcp, n):
    """n attempts on the launch rung, recorded the way the lever is actually applied."""
    return [{"kernel_kind": "trace-capture", "op_signature": "anything", "beat_baseline": False} for _ in range(n)]


def _cap(mcp):
    import os

    return int(os.environ.get("PERF_MCP_MAX_HOST_ATTEMPTS", "3") or "3")


def test_settling_the_launch_rung_does_not_retire_the_op(mcp):
    """The regression this fix could have introduced, and the reason the ladder lists knobs after.

    The launch rung is capped globally -- the attempts are counted wherever they were recorded,
    because the lever transforms the loop rather than an op. Widening the branch to hear `dispatch`
    therefore puts every dispatch-tagged op behind that one shared cap, and if settling it retired
    the op, the run's third launch attempt -- on ANY op -- would retire all of them at once, each
    without a single knob tried. _RUNG_PRIORITY lists the knobs after `host` precisely so they still
    get their sweep.
    """
    op = {"op": "MatmulDeviceOperation 3328 x 3072 x 8192", "bound_by": "dispatch", "bucket": "matmul"}

    done, rung, _ = mcp._op_ladder_status(op, "matmul", _spent_launch_attempts(mcp, _cap(mcp)))

    assert done is False, "a dispatch-tagged op retired without its completeness sweep"
    assert mcp._normalise_rung(rung) in mcp._KNOBS, rung


def test_the_launch_gap_itself_still_retires(mcp):
    """The synthetic whole-run op stands for the gap, not for work -- it has no knobs to sweep."""
    gap = {
        "op_code": "host_overhead",
        "bucket": "host_fallback",
        "bound_by": "host",
        "gap_ms": 33.5,
        "grid": "",
        "weight_dtype": "",
    }

    assert mcp._op_ladder_status(gap, "host_overhead", _spent_launch_attempts(mcp, 0))[0] is False
    assert mcp._op_ladder_status(gap, "host_overhead", _spent_launch_attempts(mcp, _cap(mcp)))[0] is True


def test_an_attempt_on_the_launch_rung_counts_under_either_name(mcp):
    """The same defect as the bound tag, one layer down, and the one that makes the rung unbounded.

    The agent is told to record this lever as `trace-capture`, but it also records it under the rung
    it was handed, `host`. Both are the same attempt on the same rung. Counting only the lever
    spelling left the cap permanently at zero -- on voxtral_mini_3b_2507 (2026-09-08) the rung was
    offered 322 times against 2 recorded attempts, none counted. Harmless while the rung reached
    almost nothing; unbounded the moment it reaches every dispatch-tagged op.
    """
    op = {"op": "MatmulDeviceOperation 3328 x 3072 x 8192", "bound_by": "dispatch", "bucket": "matmul"}
    rung_name = mcp._RUNG_PRIORITY["dispatch"][0]
    lever_name = mcp._op_ladder_status(op, "matmul", [])[1]

    assert rung_name != lever_name, "there is nothing to disagree about"

    for name in (rung_name, lever_name):
        spent = [{"kernel_kind": name, "op_signature": "anything", "beat_baseline": False} for _ in range(_cap(mcp))]
        done, rung, _ = mcp._op_ladder_status(op, "matmul", spent)
        assert mcp._normalise_rung(rung) in mcp._KNOBS, (name, rung, "the cap never advanced")
