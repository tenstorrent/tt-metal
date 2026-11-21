# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Teardown sanity, tt-metal ranks: does a job of tt-metal ranks end gracefully when one rank raises, or on a signal
at tt-run?

CPU only, on the mock dual galaxy (two ranks of 8x4, no device), launched by tt-run like every real job. The rank
program is ``sanity_probe.py`` next to this file; the harness is in
tests/tt_metal/distributed/multiprocess/teardown_sanity.

The rows: an exception on rank 1 with the mesh open (a Python exception, ``sys.exit(1)``, an exception in a worker
thread, a TT_FATAL raised in Python, an uncaught C++ exception), and signals at tt-run (Ctrl-Z, tt-run killed). The
expected end is the graceful one: the job is stopped (SIGTERM to every rank), every rank still alive closes its mesh,
and tt-run names the rank that failed and returns its code. tt-run alone cannot give that end: a failing rank has to
stop the job itself, and the others have to close on the SIGTERM, which is the DistributedContext's job stop
(#64546). Until it lands each such row is a strict xfail naming what happens instead, so the lane stays green and the
row turns red (XPASS) when the fix makes it pass. Ctrl-Z needs nothing from the ranks and is a gate.

Run: pytest tests/ttnn/distributed/teardown_sanity -v   (from the repository root, with a built tree and the python
environment; the fabric CPU lane runs it as group ``teardown-sanity-ttmetal`` of
tests/scripts/multihost/run_fabric_cpu_only_unit_tests.sh).
"""

from __future__ import annotations

import os
import re
import signal
import sys
from pathlib import Path

import pytest

# The harness (sanity_common) is shared with the raw-MPI rows in tests/tt_metal/distributed/multiprocess/teardown_sanity.
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "tests/tt_metal/distributed/multiprocess/teardown_sanity"))
import sanity_common  # noqa: E402

sanity_common.PROBE = Path(__file__).resolve().parent / "sanity_probe.py"
from sanity_common import clean, ended, fault_delivered, run_case  # noqa: E402

# Each row's deadline. The graceful end takes a few seconds after the mesh opens; the rows that hang today (the
# exception unwinding into MPI_Finalize) are killed at the deadline and fail, so it is kept short.
DEADLINE = float(os.environ.get("TT_SANITY_DEADLINE_EXCEPTION", "45"))

# What the job stop of #64546 fixes; every row below fails without it.
NEEDS_RANK_STOP = "#64546 (the DistributedContext job stop)"


def closed(r, ranks=(0, 1)) -> None:
    """Every listed rank closed its mesh on the way out (the close_done marker): a graceful end, not a kill."""
    for rank in ranks:
        assert any(
            f"[probe r={rank} " in line and "close_done" in line for line in r.log.splitlines()
        ), f"rank {rank} did not close its mesh; it ended at {r.reached.get(rank)!r}\n{r.log[-3000:]}"


def stopped_by(r, rank: int, reason: str, code: int) -> None:
    """Rank ``rank`` stopped the job for ``reason``: it printed the [tt-stop] line, every other rank was sent SIGTERM
    and exited within the grace (nothing forced), and tt-run names the rank and returns its code."""
    m = re.search(rf"\[tt-stop\] rank {rank} on \S+ \(pid \d+\): (.*?); stopping the job \(exit {code}\)", r.log)
    assert m, f"no [tt-stop] line from rank {rank} with exit {code}\n{r.log[-3000:]}"
    assert reason in m.group(1), f"rank {rank} stopped the job for {m.group(1)!r}, expected {reason!r}"
    assert r.exit == code, f"tt-run returned {r.exit}, expected the stopping rank's {code}: {r}"
    assert "the other ranks were sent SIGTERM and exited" in r.log, r.log[-3000:]
    assert f"origin=rank {rank} on" in r.log and "outcome=failed" in r.log, r.log[-3000:]
    assert "job forced down" not in r.log, f"the stop was forced, not graceful\n{r.log[-3000:]}"


@pytest.mark.xfail(
    strict=True,
    reason=f"{NEEDS_RANK_STOP}: the exception unwinds into MPI_Finalize, which waits on rank 0 still in its barrier; "
    "the job hangs",
)
def test_raise_after_open():
    """Rank 1 raises after open: it stops the job, rank 0 leaves its barrier on the SIGTERM, both close their mesh;
    tt-run returns 1 naming rank 1 and the exception."""
    r = run_case("raise_after_open", {"PROBE_SCENARIO": "raise", "PROBE_PHASE": "after_open"}, DEADLINE)
    ended(r)
    stopped_by(r, 1, "Python exception RuntimeError: injected failure at after_open", 1)
    closed(r)
    clean(r)


@pytest.mark.xfail(
    strict=True,
    reason=f"{NEEDS_RANK_STOP}: sys.exit(1) unwinds into MPI_Finalize, which waits on rank 0 still in its barrier; "
    "the job hangs",
)
def test_exit1_stops_the_job():
    """Rank 1 calls sys.exit(1) after open, the way click or a script's error return does: the job is stopped, every
    rank closes; tt-run returns 1 naming rank 1."""
    r = run_case("exit1_after_open", {"PROBE_SCENARIO": "exit1", "PROBE_PHASE": "after_open"}, DEADLINE)
    ended(r)
    stopped_by(r, 1, "sys.exit(1)", 1)
    closed(r)
    clean(r)


@pytest.mark.xfail(
    strict=True,
    reason=f"{NEEDS_RANK_STOP}: Python only prints a worker thread's exception; nothing stops the job",
)
def test_thread_exception_stops_the_job():
    """An uncaught exception in a worker thread of rank 1 (threading.excepthook): the job is stopped with code 1."""
    r = run_case("thread_raise", {"PROBE_SCENARIO": "thread_raise", "PROBE_PHASE": "after_open"}, DEADLINE)
    ended(r)
    stopped_by(r, 1, "uncaught Python exception in thread worker RuntimeError: worker failed", 1)
    closed(r)
    clean(r)


@pytest.mark.xfail(
    strict=True,
    reason=f"{NEEDS_RANK_STOP}: the TT_FATAL reaches Python as RuntimeError and unwinds into MPI_Finalize like any "
    "Python exception, while the other rank waits for it; the job hangs",
)
def test_bad_shape():
    """Rank 1 asks for a mesh the descriptor cannot hold: the shape check in open is a TT_FATAL (a C++ exception,
    raised in Python as RuntimeError, its message several lines long). The job is stopped with code 1, and tt-run
    names the rank from its one-line [tt-stop] line, not prterun's banner."""
    r = run_case("bad_shape_rank1", {"PROBE_MESH_SHAPE_VICTIM": "9x9"}, DEADLINE)
    ended(r)
    m = re.search(
        r"\[tt-stop\] rank (\d+) on \S+ \(pid \d+\): (?:uncaught )?Python exception RuntimeError: TT_FATAL.*"
        r"; stopping the job \(exit 1\)",
        r.log,
    )
    assert m, f"no one-line [tt-stop] line for the TT_FATAL\n{r.log[-3000:]}"
    assert r.exit == 1 and f"origin=rank {m.group(1)} on" in r.log and "outcome=failed exit=1" in r.log, str(r)
    assert "the launcher then killed the other ranks" not in r.log and "job forced down" not in r.log, r.log[-3000:]
    clean(r)


@pytest.mark.xfail(
    strict=True,
    reason=f"{NEEDS_RANK_STOP}: std::terminate aborts rank 1 (SIGABRT) and its wrapper stops the job with 134, but "
    "rank 0 leaves on the failed barrier (MPI_ERR_PROC_FAILED) without closing its mesh",
)
def test_uncaught_cxx_exception_stops_the_job():
    """An uncaught C++ exception on rank 1 (a std::runtime_error with no handler, the way a TT_FATAL escapes C++
    code): the terminate handler stops the job instead of aborting, rank 0 leaves its barrier on the SIGTERM and closes
    its mesh; tt-run returns 1 naming rank 1 and the exception."""
    r = run_case("cxx_throw", {"PROBE_SCENARIO": "cxx_throw", "PROBE_PHASE": "after_open"}, DEADLINE)
    ended(r)
    stopped_by(r, 1, "uncaught C++ exception: injected C++ failure", 1)
    closed(r, ranks=(0,))
    clean(r)


# ---------------------------------------------------------------------------------------------------------------
# signals at tt-run, with tt-metal ranks (how the launcher passes them on is in teardown_sanity/test_teardown_sanity_mpi)
# ---------------------------------------------------------------------------------------------------------------
def test_ctrl_z_pauses_and_resumes_the_job():
    """Ctrl-Z (SIGTSTP) at tt-run while the ranks are in a collective pauses every rank; SIGCONT (fg / bg) resumes
    them and the job finishes as if nothing happened: every rank closes its mesh, tt-run returns 0."""
    r = run_case(
        "ctrl_z_in_collective",
        {"PROBE_COLL": "barrier_slow", "PROBE_SLOW_S": "8"},
        DEADLINE,
        launcher_signal=signal.SIGTSTP,
        launcher_ready=(0, "coll_ready"),
        signal_after=1.0,
        then_signals=[(signal.SIGCONT, 3.0)],
    )
    fault_delivered(r)
    ended(r)
    paused = re.findall(r"ranks: ([^;]*)", r.signal_sent)[0]
    assert paused == "rank_0=T rank_1=T", f"SIGTSTP did not pause every rank: {paused!r}\n{r}"
    assert r.exit == 0, f"the paused and resumed job did not finish cleanly: {r}\n{r.log[-3000:]}"
    closed(r)
    clean(r)


@pytest.mark.xfail(
    strict=True,
    reason=f"{NEEDS_RANK_STOP}: tt-run's guard stops the job (SIGTERM to every rank) but the ranks have no SIGTERM "
    "handler yet and die without closing their mesh",
)
def test_ttrun_killed_ranks_close_their_mesh():
    """tt-run is SIGKILLed while the ranks sit in a deadlocked collective: its guard (tt-run-rank --guard) stops the
    job, SIGTERM to every rank, and every rank leaves its wait and closes its mesh, as for a Ctrl-C at tt-run."""
    r = run_case(
        "ttrun_killed_in_deadlock",
        {"PROBE_COLL": "send_no_recv"},
        DEADLINE,
        launcher_signal=signal.SIGKILL,
        launcher_ready=(0, "coll_ready"),
        signal_after=3.0,
    )
    fault_delivered(r)
    ended(r)
    assert "[tt-run guard] sent SIGTERM to all ranks" in r.log, r.log[-3000:]
    assert "job forced down" not in r.log, r.log[-3000:]
    closed(r)
    clean(r)
