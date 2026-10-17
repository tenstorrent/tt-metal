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


# --- Out of budget is not out of credentials -------------------------------------------------
# The failure auth handling gets wrong in the most expensive way: a spent quota looks exactly like
# a refused credential -- a round that runs, writes a transcript and does nothing -- but renewing
# is not the remedy. The credential is already valid, so the renewal "succeeds", the round is
# retried, the retry is refused again, and the run drains its recovery budget before telling the
# operator to re-login over a problem re-logging in cannot touch.

from agent.probes import detect_quota_exhausted  # noqa: E402


def test_the_shapes_a_spent_budget_arrives_in_are_recognised():
    for text in (
        "Claude usage limit reached. Your limit will reset at 9pm.",
        "API Error: 429 rate_limit_error",
        "You have exceeded your usage quota for this organization",
        "Credit balance is too low",
    ):
        assert detect_quota_exhausted(text), text


def test_budget_and_credentials_never_claim_the_same_failure():
    """They have opposite remedies, so a text matching both would send the operator the wrong way."""
    spent = "Claude usage limit reached. Your limit will reset at 9pm."
    refused = "Failed to authenticate. API Error: 403 Access Denied"
    assert detect_quota_exhausted(spent) and detect_auth_failure(spent) is None
    assert detect_auth_failure(refused) and detect_quota_exhausted(refused) is None


def test_a_healthy_round_trips_neither():
    for benign in ("", "recorded a kernel attempt; check_pcc ok", "authentication succeeded"):
        assert detect_quota_exhausted(benign) is None and detect_auth_failure(benign) is None


def test_the_two_are_read_from_the_rounds_own_transcript(tmp_path):
    from cc_optimize.run import _agent_auth_failure as auth, _agent_quota_exhausted as spent

    log = tmp_path / "k.json"
    (tmp_path / "k.json.agent.log").write_text("Claude usage limit reached. Resets at 9pm.")
    assert spent(log) and auth(log) is None
    (tmp_path / "k.json.agent.log").write_text("Failed to authenticate. 403 Access Denied")
    assert auth(log) and spent(log) is None


def _loop_with_budget(spent_at, attempts, recovers, max_rounds=10):
    """The loop's refusal branch with BOTH checks, budget first: (rounds, halted, trace)."""
    rounds = recoveries = strikes = attempt = 0
    halted = False
    trace = []
    guard = 0
    while rounds < max_rounds:
        guard += 1
        if guard > 200:
            return rounds, halted, trace + ["SPUN"]
        if rounds >= spent_at:  # out of budget from this round on
            trace.append("out-of-budget@%d" % rounds)
            halted = True
            break
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
        else:
            strikes = 0
        rounds += 1
    return rounds, halted, trace, recoveries


def test_a_spent_budget_stops_the_run_without_spending_a_recovery():
    """Renewing an account that is merely broke succeeds, so the loop must not try: it would
    retry, be refused again, and drain the recovery budget before reporting the wrong cause."""
    rounds, halted, trace, recoveries = _loop_with_budget(spent_at=0, attempts=[], recovers=True)
    assert halted, "a spent budget must stop the run"
    assert recoveries == 0, "a spent budget must cost zero renewal attempts"
    assert trace == ["out-of-budget@0"]


def test_a_budget_that_runs_out_midway_keeps_the_rounds_already_done():
    rounds, halted, trace, _ = _loop_with_budget(spent_at=4, attempts=[], recovers=True)
    assert halted and rounds == 4, "the rounds already completed must still count"


def test_the_recovery_probe_is_not_fooled_by_a_valid_but_broke_account():
    """The probe asks the agent to work. An account with a valid credential and no budget left
    ANSWERS -- so checking only for an auth failure reports a successful recovery that is not one."""
    from agent.probes import detect_auth_failure, detect_quota_exhausted

    answer = "Claude usage limit reached. Your limit will reset at 9pm."
    assert detect_auth_failure(answer) is None, "the credential really is fine -- that is the trap"
    recovered = detect_auth_failure(answer) is None and detect_quota_exhausted(answer) is None
    assert not recovered, "the probe reported a recovery for an account that cannot do any work"
