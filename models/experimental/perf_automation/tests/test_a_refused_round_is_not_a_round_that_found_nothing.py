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
from cc_optimize.run import _MAX_AUTH_STRIKES, _agent_auth_failure  # noqa: E402

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
