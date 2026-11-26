# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Teardown sanity, level 1, C++: the DistributedContext alone, no device.

Two C++ ranks (``rank_probe.cpp`` built as ``teardown_context_probe``) under tt-run on the CPU, one injected fault
per row. What the rank side must do, with no Python above it: a signal is seen (the C handler sets the flag, the
wait in flight throws, the uncaught exception reaches ``std::terminate`` and ``teardown``); a failure (an uncaught
exception, ``exit(n)``, ``abort(n)``) writes the failed word to the wrapper and reaches ``teardown`` with the right
code; a peer's death or exit is delivered as an exception; a wait past the operation timeout fails the rank.
The same rows run from Python in tests/ttnn/distributed/teardown_sanity/test_context_py.py.

Run: pytest tests/tt_metal/distributed/multiprocess/teardown_sanity/test_context_cpp.py -v (a tree built with
BUILD_TESTS, the python environment; the rows skip when the probe is not built).
"""

from __future__ import annotations

import re
import signal
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rows_common import *  # noqa: E402,F401,F403 - the harness, the checks and the helpers
from rows_common import probe_fixture  # noqa: E402

PROBE_BIN = ROOT / "build/test/tt_metal/distributed/teardown_context_probe"
_probe = probe_fixture(PROBE_BIN)
UNCAUGHT = "uncaught C++ exception: injected failure at after_open"


# ---------------------------------------------------------------------------------------------------------------
# the stop: SIGTERM is seen wherever the rank is
# ---------------------------------------------------------------------------------------------------------------
def test_clean():
    """Two ranks, a barrier, exit 0: no teardown line, nothing stopped, nothing left."""
    r = run_case("cpp_clean", {}, DEADLINE_END)
    ended(r)
    assert r.exit == 0 and "outcome=clean exit=0" in r.log, str(r)
    assert "teardown initiated" not in r.log and "stopping the job" not in r.log, r.log[-3000:]
    clean(r)


@pytest.mark.parametrize(
    "coll, victim",
    [("barrier_slow", 0), ("recv_no_send", 1), ("split_slow", 0)],
    ids=["in_barrier", "in_recv", "in_split"],
)
def test_sigterm_in_a_wait(coll, victim):
    """SIGTERM lands on rank 1 while it waits inside a collective (a barrier its peer joins late, a receive nobody
    sends, a communicator split its peer joins late): the C handler sets the flag, the polled wait throws, the
    uncaught exception reaches std::terminate, which leaves through teardown(SIGTERM) with 143. Its wrapper saw no
    stop, so it reports "exited with code 143" and stops the job; rank 0 tears down on the stop."""
    env = {"PROBE_COLL": coll, "PROBE_VICTIM": str(victim), "PROBE_SLOW_S": "8"}
    r = run_case(
        f"cpp_sigterm_{coll}", env, DEADLINE_END, rank_signal=(1, signal.SIGTERM, 1.5), signal_on_marker="coll_ready"
    )
    ended(r)
    stopped_by(r, 1, "exited with code 143", 143)
    tore_down(r, ranks=(1,), reason="SIGTERM")
    tore_down(r, ranks=(0,), reason="SIGTERM")
    clean(r)


def test_sigterm_outside_any_wait_is_forced():
    """SIGTERM lands on rank 1 while it sleeps in its own code, never calling the context again: the C handler only
    sets the flag and nothing reads it (a C++ rank has no Python handler to run at the next bytecode). Documented:
    the job's --job-timeout stops the job and the wrapper forces rank 1 after its stop grace, with the tt-smi line."""
    r = run_case(
        "cpp_sigterm_idle",
        {"PROBE_SCENARIO": "hang", "PROBE_PHASE": "after_open", "TT_RUN_STOP_GRACE_S": "4"},
        DEADLINE_END,
        cmd=legacy_cmd(ttrun_extra=("--job-timeout", "6")),
    )
    ended(r)
    assert re.search(r"\[rank 1\] did not exit within its stop grace of \d+ s: forced", r.log), r.log[-3000:]
    assert "tt-smi -r" in r.log, r.log[-3000:]
    tore_down(r, ranks=(0,))
    clean(r)


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM], ids=["ctrl_c", "sigterm"])
def test_stop_at_ttrun(sig):
    """Ctrl-C or SIGTERM at tt-run while both ranks sit in a slow barrier: every rank's wait throws on the SIGTERM
    and leaves through teardown; 130 / 143."""
    r = run_case(
        f"cpp_stop_{sig.name}",
        {"PROBE_COLL": "barrier_slow", "PROBE_SLOW_S": "8"},
        DEADLINE_END,
        launcher_signal=sig,
        launcher_ready=(0, "coll_ready"),
        signal_after=1.0,
    )
    fault_delivered(r)
    stopped_at_ttrun(r, sig)
    clean(r)


def test_sigterm_to_one_rank():
    """SIGTERM to rank 1's pid only (kill, scancel of one task) while it waits: it leaves with 143 without a stop
    having reached its wrapper, which reports "exited with code 143" and stops the job."""
    r = run_case(
        "cpp_sigterm_one_rank",
        {"PROBE_COLL": "barrier_slow", "PROBE_VICTIM": "0", "PROBE_SLOW_S": "8"},
        DEADLINE_END,
        rank_signal=(1, signal.SIGTERM, 1.0),
        signal_on_marker="coll_ready",
    )
    ended(r)
    stopped_by(r, 1, "exited with code 143", 143)
    tore_down(r)
    clean(r)


# ---------------------------------------------------------------------------------------------------------------
# a failure: the failed word reaches the wrapper, the rank leaves through teardown with its code
# ---------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "scenario, reason, code",
    [
        ("throw", UNCAUGHT, 1),
        ("exit1", "exited with code 1", 1),
        ("abort_n", "abort(2)", 2),
    ],
    ids=["uncaught_exception", "exit_1", "abort_2"],
)
def test_failure_stops_the_job(scenario, reason, code):
    """Rank 1 fails in its own code while rank 0 waits in a barrier: an uncaught exception (the TT_FATAL shape,
    std::terminate), exit(1) (on_exit sees the status), DistributedContext::abort(2). The rank writes the failed
    word and leaves through teardown with its code; the wrapper stops the job with that reason; rank 0 tears down."""
    r = run_case(f"cpp_{scenario}", {"PROBE_SCENARIO": scenario, "PROBE_PHASE": "after_open"}, DEADLINE_END)
    ended(r)
    stopped_by(r, 1, reason, code)
    tore_down(r, ranks=(1,), reason=reason)
    tore_down(r, ranks=(0,))
    clean(r)


def test_exit_without_hooks():
    """Rank 1 _exit(1)s: no hook runs and no teardown line; its wrapper sees the exit and stops the job."""
    r = run_case(
        "cpp_exit_nofinalize", {"PROBE_SCENARIO": "exit_nofinalize", "PROBE_PHASE": "after_open"}, DEADLINE_END
    )
    ended(r)
    stopped_by(r, 1, "exited with code 1", 1)
    assert "[rank 1] teardown initiated" not in r.log, r.log[-3000:]
    tore_down(r, ranks=(0,))
    clean(r)


@pytest.mark.parametrize(
    "scenario, signum",
    [("sigkill", 9), ("segv", 11), ("abort", 6), ("sigint", 2)],
    ids=["sigkill", "segv", "abort", "sigint"],
)
def test_death_stops_the_job(scenario, signum):
    """Rank 1 dies on a signal (SIGKILL, SIGSEGV, abort(), a SIGINT to its pid: the default disposition for a C++
    rank): its wrapper stops the job in its place, naming the signal; rank 0 tears down; 128 + the signal."""
    r = run_case(f"cpp_{scenario}", {"PROBE_SCENARIO": scenario, "PROBE_PHASE": "after_open"}, DEADLINE_END)
    ended(r)
    stopped_by(r, 1, f"killed by signal {signum}", 128 + signum)
    tore_down(r, ranks=(0,))
    clean(r)


def test_throw_while_peer_in_split():
    """Rank 1 throws while rank 0 is inside a communicator split (no non-blocking form in MPI; it runs on a helper
    thread and is polled): rank 0 leaves it on the SIGTERM and tears down; 1 naming rank 1."""
    r = run_case("cpp_split_peer_throws", {"PROBE_COLL": "split_peer_throws"}, DEADLINE_END)
    ended(r)
    stopped_by(r, 1, "uncaught C++ exception: injected failure while the peers are in split", 1)
    reached(r, 0, "coll_split_enter")
    tore_down(r, ranks=(0,))
    clean(r)


# ---------------------------------------------------------------------------------------------------------------
# a peer's death or exit, delivered to the survivor
# ---------------------------------------------------------------------------------------------------------------
def test_peer_dead_ulfm_leaves_as_stopped():
    """Rank 1 SIGKILLed with ULFM on, so rank 0's barrier sees MPI_ERR_PROC_FAILED: the wait gives the dead rank's
    wrapper's SIGTERM up to 2 s and rank 0 leaves as a stopped rank (the wait throws "abandoned", std::terminate,
    teardown(SIGTERM)), not as a second failure."""
    r = run_case(
        "cpp_peer_dead_ulfm",
        {"PROBE_COLL": "barrier_dead_peer", "TT_RUN_STOP_GRACE_S": "6"},
        DEADLINE_HANG,
        cmd=legacy_cmd("--with-ft ulfm"),
    )
    ended(r)
    assert re.search(r"\[rank 1\] .*killed by signal 9.*stopping the job \(exit 137\)", r.log), r.log[-3000:]
    tore_down(r, ranks=(0,), reason="SIGTERM")
    assert "[rank 0] teardown initiated: uncaught" not in r.log, r.log[-3000:]
    clean(r)


def test_peer_exit_notify_is_an_exception():
    """tt-run --on-peer-exit notify: rank 1 exits 1 while rank 0 waits in a receive from it, and the job goes on.
    Rank 0's wait throws PeerExitError (the SIGUSR1 receiver); the probe catches it (PROBE_ON_ERROR=continue),
    closes and exits 0."""
    r = run_case(
        "cpp_peer_exit_notify",
        {
            "PROBE_SCENARIO": "exit1",
            "PROBE_PHASE": "after_open",
            "PROBE_COLL": "recv_no_send",
            "PROBE_ON_ERROR": "continue",
        },
        DEADLINE_END,
        cmd=legacy_cmd(ttrun_extra=("--on-peer-exit", "notify")),
    )
    ended(r)
    assert "the job goes on" in r.log, r.log[-3000:]
    rc_contains(r, 0, "PeerExitError")
    reached(r, 0, "main_exit")
    assert "stopping the job" not in r.log, r.log[-3000:]
    clean(r)


# ---------------------------------------------------------------------------------------------------------------
# the clocks
# ---------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "coll, what", [("barrier_skip", "barrier"), ("recv_no_send", "recv"), ("allgather_skip", "all_gather")]
)
def test_operation_timeout_fails_the_rank(coll, what):
    """Rank 1 never joins; TT_METAL_OPERATION_TIMEOUT_SECONDS=4: rank 0's wait gives up, the rank fails on its own
    ("did not complete within ... 4 s") and leaves through teardown; the job stops with 1 naming rank 0."""
    r = run_case(f"cpp_timeout_{coll}", {"PROBE_COLL": coll, "TT_METAL_OPERATION_TIMEOUT_SECONDS": "4"}, DEADLINE_END)
    ended(r)
    stopped_by(r, 0, f"{what} did not complete within TT_METAL_OPERATION_TIMEOUT_SECONDS=4 s", 1)
    tore_down(r, ranks=(0,), reason="did not complete within")
    clean(r)


def test_slow_collective_is_not_a_timeout():
    """Rank 1 joins the barrier 2 s late under a 5 s timeout: nothing fails, exit 0."""
    r = run_case(
        "cpp_slow_collective_ok",
        {"PROBE_COLL": "barrier_slow", "PROBE_SLOW_S": "2", "TT_METAL_OPERATION_TIMEOUT_SECONDS": "5"},
        DEADLINE_END,
    )
    ended(r)
    assert r.exit == 0 and "stopping the job" not in r.log, str(r)
    clean(r)


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="P30: with no timeout set anywhere a hung collective never ends; policy: a default for "
    "TT_METAL_OPERATION_TIMEOUT_SECONDS in the context, or tt-run setting it per rank",
)
def test_hang_without_any_timeout():
    """Rank 1 skips the barrier and no clock is set: rank 0 waits until the row's deadline (the policy row)."""
    r = run_case("cpp_hang_no_timeout", {"PROBE_COLL": "barrier_skip"}, DEADLINE_HANG)
    ended(r)


def test_init_timeout_ends_a_late_rank():
    """Rank 1 starts 8 s late under TT_METAL_DISTRIBUTED_INIT_TIMEOUT_SECONDS=3: rank 0's MPI_Init watchdog exits 1
    and the launcher ends the job."""
    r = run_case(
        "cpp_init_timeout",
        {"PROBE_START_DELAY_S": "8", "TT_METAL_DISTRIBUTED_INIT_TIMEOUT_SECONDS": "3"},
        DEADLINE_END,
    )
    ended(r)
    nonzero(r)
    assert "MPI_Init did not complete within" in r.log, r.log[-3000:]
    clean(r)


def test_early_exit0_peer_times_out():
    """Rank 1 exits 0 while rank 0 enters a barrier (P27: no error anywhere): with the operation timeout set, rank
    0's barrier gives up and fails the rank; the job ends 1 naming rank 0."""
    r = run_case(
        "cpp_exit0_early",
        {"PROBE_SCENARIO": "exit0", "PROBE_PHASE": "after_open", "TT_METAL_OPERATION_TIMEOUT_SECONDS": "4"},
        DEADLINE_END,
    )
    ended(r)
    stopped_by(r, 0, "barrier did not complete within", 1)
    clean(r)


def test_ctrl_z_pauses_and_resumes():
    """SIGTSTP then SIGCONT at tt-run while the ranks are in a slow barrier: the job pauses, resumes, and ends 0."""
    r = run_case(
        "cpp_ctrl_z",
        {"PROBE_COLL": "barrier_slow", "PROBE_SLOW_S": "6"},
        DEADLINE_END,
        launcher_signal=signal.SIGTSTP,
        launcher_ready=(0, "coll_ready"),
        signal_after=1.0,
        then_signals=[(signal.SIGCONT, 3.0)],
    )
    fault_delivered(r)
    ended(r)
    assert r.exit == 0 and "outcome=clean exit=0" in r.log, str(r)
    reached(r, 0, "main_exit")
    clean(r)
