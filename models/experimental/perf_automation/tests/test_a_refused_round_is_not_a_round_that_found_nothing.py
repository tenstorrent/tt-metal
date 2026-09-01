"""A round that was never let in must not be counted as a round that found no win.

An expired or rejected credential produces a round that runs, writes a transcript and exits
cleanly having done nothing. To the loop that is indistinguishable from an agent that looked and
found nothing, so it spent all ten rounds of a 7h37m run on it and the report said "no kernel
attempts recorded" -- which reads as "the model is already optimal" rather than "nobody was
allowed in". The credential had lapsed overnight; every round carried the client's own words:

    Failed to authenticate. API Error: 403 Access Denied: Your IP address is not in the
    allowed range for this organization
"""

from __future__ import annotations

import sys
from pathlib import Path

PERF = Path(__file__).resolve().parents[1]
if str(PERF) not in sys.path:
    sys.path.insert(0, str(PERF))

from agent.probes import detect_auth_failure  # noqa: E402
from cc_optimize.run import (  # noqa: E402
    _MAX_AUTH_RECOVERIES,
    _MAX_AUTH_STRIKES,
    _agent_auth_failure,
)

_REAL_REFUSAL = (
    "Failed to authenticate. API Error: 403 Access Denied: "
    "Your IP address is not in the allowed range for this organization"
)


def test_the_refusal_this_actually_cost_us_is_recognised():
    assert detect_auth_failure(_REAL_REFUSAL) == "Failed to authenticate"


def test_a_working_round_is_left_alone():
    """A false positive would halt a healthy run, which is worse than the bug being fixed."""
    for benign in (
        "",
        "applied a full-grid program_config; check_pcc ok",
        "the model failed to converge",
        "authentication succeeded",
    ):
        assert detect_auth_failure(benign) is None, benign


def test_the_phrase_is_returned_so_the_operator_is_told_which_failure():
    """Re-login, a bad key and an org restriction need three different actions -- quote, don't guess."""
    assert detect_auth_failure("Invalid API key; please check") == "Invalid API key"
    assert detect_auth_failure("OAuth token has expired") == "OAuth token has expired"


def test_it_reads_the_rounds_own_transcript(tmp_path):
    kernel_log = tmp_path / "kernel.json"
    (tmp_path / "kernel.json.agent.log").write_text(_REAL_REFUSAL + "\n")
    assert _agent_auth_failure(kernel_log) == "Failed to authenticate"

    (tmp_path / "kernel.json.agent.log").write_text("recorded a kernel attempt\n")
    assert _agent_auth_failure(kernel_log) is None


def test_an_old_failure_earlier_in_the_log_does_not_halt_a_recovered_run(tmp_path):
    """Only this round's tail counts: a refusal that was retried past must not stop the run later."""
    kernel_log = tmp_path / "kernel.json"
    body = "\n".join("progress line %d" % i for i in range(200))
    (tmp_path / "kernel.json.agent.log").write_text(_REAL_REFUSAL + "\n" + body)
    assert _agent_auth_failure(kernel_log) is None


def test_a_missing_transcript_is_not_treated_as_a_refusal(tmp_path):
    assert _agent_auth_failure(tmp_path / "never-written.json") is None


def test_one_blip_is_retried_before_stopping():
    assert _MAX_AUTH_STRIKES >= 2, "halting on a single refusal would abort a run over one flaky call"


def _loop(attempts, recovers, max_rounds=10):
    """The round loop's auth branch, in isolation: (rounds_run, halted, trace).

    `attempts` is per ATTEMPT, not per round -- a renewed round is re-run, so it consumes another
    attempt without advancing the round counter, which is exactly the shape that can spin.
    """
    rounds = recoveries = strikes = attempt = 0
    halted = False
    trace = []
    guard = 0
    while rounds < max_rounds:
        guard += 1
        if guard > 200:  # the property under test: this must never trip
            return rounds, halted, trace + ["SPUN"]
        refused = attempts[attempt] if attempt < len(attempts) else False
        attempt += 1
        if refused:
            if recovers and recoveries < _MAX_AUTH_RECOVERIES:
                recoveries += 1
                strikes = 0
                trace.append("renew@%d" % rounds)
                continue
            strikes += 1
            if strikes >= _MAX_AUTH_STRIKES:
                halted = True
                trace.append("halt@%d" % rounds)
                break
            trace.append("strike@%d" % rounds)
        else:
            strikes = 0
        rounds += 1
    return rounds, halted, trace


def test_a_renewed_credential_keeps_the_run_going():
    """The whole point: a blip must cost a retry, not the run and its measured baseline."""
    rounds, halted, trace = _loop([True], recovers=True)
    assert not halted, "the run stopped even though renewing worked"
    assert rounds == 10, "every round should still have been run"
    assert trace == ["renew@0"], "one blip should cost exactly one renewal"


def test_a_refusal_that_renewing_cannot_fix_still_stops_the_run():
    rounds, halted, trace = _loop([True] * 20, recovers=False)
    assert halted, "a permanently refused run must not spend its remaining rounds being told no"
    assert trace == ["strike@0", "halt@1"], trace
    assert rounds < 10, "it stopped early instead of burning every round"


def test_the_loop_always_finishes_even_when_renewing_never_helps():
    """Retrying without consuming a round is unbounded unless it is budgeted -- so budget it."""
    rounds, halted, trace = _loop([True] * 50, recovers=True)
    assert "SPUN" not in trace, "renew-and-retry looped forever without finishing the run"
    assert halted, "past the recovery budget a permanent refusal must stop the run"
    assert trace.count("renew@0") == _MAX_AUTH_RECOVERIES, "the recovery budget was not enforced"


def test_a_healthy_run_is_untouched():
    assert _loop([], recovers=True) == (10, False, [])


def test_the_recovery_budget_leaves_room_for_a_real_blip():
    assert _MAX_AUTH_RECOVERIES >= 1
