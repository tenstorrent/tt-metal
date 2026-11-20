# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Teardown sanity: does a distributed tt-metal job end, end cleanly, and report it, after one failure?

CPU only, on the mock dual galaxy (two ranks of 8x4, no device), launched by tt-run like every real job. Every
test is one fault with one assertion, taken from the teardown plan of tt-metal #56902 (tt-blaze
tools/teardown_probe, lane L4): the rank program is ``sanity_probe.py`` next to this file; the harness is in
tests/tt_metal/distributed/multiprocess/teardown_sanity. The faults are the
two kinds a real job meets: the code failing on one rank (an exception, a shape the descriptor cannot hold, a
descriptor that disagrees, a collective with a wrong or dead peer) and the user (a signal at tt-run).

Requirements, as in the plan: a failure never becomes a hang (R1), a hang ends by itself and leaves nothing
(R2), nobody kills processes by hand (R3), the exit code reports the failure (R4).

Modes. A row that passes today is a gate. A row that fails today is marked ``@known("P..: ...")`` with the
plan's error id and fix: it fails, so the job is red until the fix lands, and its failure message names the
error; when the fix lands the row passes and the marker is dropped. Each row has a deadline; a job that is still running at the
deadline is killed by its process group and counted as a hang.

Run: pytest tests/ttnn/distributed/teardown_sanity -v   (from the repository root, with a
built tree and the python environment; the fabric CPU lane runs it as group ``teardown-sanity`` of
tests/scripts/multihost/run_fabric_cpu_only_unit_tests.sh).
"""

from __future__ import annotations

import os
import re
import signal
import sys
import tempfile
from pathlib import Path

import pytest

# The harness (sanity_common) is shared with the raw-MPI rows in tests/tt_metal/distributed/multiprocess/teardown_sanity.
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "tests/tt_metal/distributed/multiprocess/teardown_sanity"))
import sanity_common  # noqa: E402

sanity_common.PROBE = Path(__file__).resolve().parent / "sanity_probe.py"
from sanity_common import (
    _session_dirs,
    CONFLICT_DESC,
    DEADLINE_END,
    DEADLINE_HANG,
    clean,
    conflict_mapping,
    ended,
    fault_delivered,
    legacy_cmd,
    mpirun_path,
    nonzero,
    rc_contains,
    run_case,
    solve_cmd,
    MGD_QUAD,
    PYTHON,
    ROOT,
)


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


def stopped_at_ttrun(r, sig: int) -> None:
    """A stop at tt-run: every rank was sent SIGTERM, closed its mesh and exited within the grace; 128 + the signal."""
    assert r.exit == 128 + sig, f"expected {128 + sig}: {r}"
    assert "sent SIGTERM to all ranks" in r.log and f"outcome=stopped exit={128 + sig}" in r.log, r.log[-3000:]
    assert "job forced down" not in r.log, r.log[-3000:]
    closed(r)


# ---------------------------------------------------------------------------------------------------------------
# rows that pass today (gates)
# ---------------------------------------------------------------------------------------------------------------
def test_clean():
    """Control: both ranks open the mock galaxy, one barrier, close, exit 0, nothing left."""
    r = run_case("clean", {}, DEADLINE_END)
    ended(r)
    assert r.exit == 0, str(r)
    clean(r)


def test_dead_peer_barrier():
    """Rank 1 is SIGKILLed before a barrier. Its wrapper (tt-run-rank) stops the job in its place: rank 0 leaves the
    barrier on the SIGTERM, closes its mesh, and tt-run returns 137 naming rank 1."""
    r = run_case("dead_peer_barrier", {"PROBE_COLL": "barrier_dead_peer"}, DEADLINE_END)
    ended(r)
    stopped_by(r, 1, "was killed by signal 9 (SIGKILL)", 137)
    closed(r, ranks=(0,))
    clean(r)


def test_dead_peer_recv():
    """Rank 1 is SIGKILLed before rank 0's recv: rank 0 leaves the recv on the SIGTERM and closes; 137, rank 1."""
    r = run_case("dead_peer_recv", {"PROBE_COLL": "recv_dead_peer"}, DEADLINE_END)
    ended(r)
    stopped_by(r, 1, "was killed by signal 9 (SIGKILL)", 137)
    closed(r, ranks=(0,))
    clean(r)


def test_fabric_mismatch():
    """One rank sets FABRIC_2D before open and the other does not: TT_FATAL on one rank, the job ends non-zero."""
    r = run_case("fabric_mismatch", {"PROBE_FABRIC_RANK": "1"}, DEADLINE_END)
    ended(r)
    nonzero(r)
    clean(r)


def test_prte_timeout():
    """A send nobody receives under --timeout 60: at 60 s tt-run sends SIGTERM to every rank (the timeout is graceful;
    PRRTE's own kill stays only as a backstop after the stop grace), both ranks close, and tt-run returns 241.

    The timeout is longer than the ttnn import on a loaded runner, and the row checks that both ranks were past
    open when the clock fired, so it is the deadlock that was ended and not the start-up."""
    r = run_case(
        "send_no_recv+prte_timeout60", {"PROBE_COLL": "send_no_recv"}, DEADLINE_END + 60, cmd=legacy_cmd("--timeout 60")
    )
    ended(r)
    assert r.exit == 241, str(r)
    # Require positive evidence that the wall clock ended the intended deadlock, not a slow start-up: both ranks
    # must have entered the collective section (a ``coll_*`` marker in their output), past open — not merely "not a
    # startup marker", which would also accept ``open_enter`` (entered open, never completed). The last marker is no
    # longer the collective itself: PRRTE's timeout ends the ranks with SIGTERM, which since P28 makes the collective
    # return and the rank close its device (``close_done``) on the way out.
    for rank in (0, 1):
        assert f"[probe r={rank} " in r.log and any(
            f"[probe r={rank} " in line and "] coll_" in line for line in r.log.splitlines()
        ), f"rank {rank} had not reached the collective when the wall clock fired: {r}"
    closed(r)  # the --timeout is graceful: both ranks leave the collective on SIGTERM and close their mesh
    assert "the job reached its --timeout of 60 s: sent SIGTERM to all ranks" in r.log, r.log[-3000:]
    assert "outcome=stopped exit=241" in r.log and "job forced down" not in r.log, r.log[-3000:]
    clean(r)


def test_exit_nofinalize():
    """Rank 1 calls os._exit(1) with the mesh open (no atexit, no hook): its wrapper stops the job, rank 0 closes."""
    r = run_case(
        "exit_nofinalize_after_open", {"PROBE_SCENARIO": "exit_nofinalize", "PROBE_PHASE": "after_open"}, DEADLINE_END
    )
    ended(r)
    stopped_by(r, 1, "exited with code 1 without stopping the job", 1)
    closed(r, ranks=(0,))
    clean(r)


def test_invalid_rank():
    """A receive from a rank that does not exist raises at once; the job goes on and ends."""
    r = run_case("recv_bad_rank", {"PROBE_COLL": "recv_bad_rank", "PROBE_ON_ERROR": "continue"}, DEADLINE_END)
    ended(r)
    rc_contains(r, 0, "RuntimeError")
    clean(r)


def test_use_before_init():
    """A collective before init_distributed_context raises 'not initialized' instead of crashing or hanging."""
    r = run_case("use_before_init", {"PROBE_SCENARIO": "use_before_init", "PROBE_ON_ERROR": "continue"}, DEADLINE_END)
    ended(r)
    rc_contains(r, 1, "not initialized")
    assert r.exit == 0, str(r)
    clean(r)


# ---------------------------------------------------------------------------------------------------------------
# rows that fail today. Each marker names the error of the plan and its fix; drop the marker with the fix.
# ---------------------------------------------------------------------------------------------------------------
def test_raise_after_open():
    """Rank 1 raises after open: on the way out its finally closes the mesh, and that close stops the job first (the
    exception is in flight), so rank 0 leaves its barrier on the SIGTERM and closes too, instead of rank 1's close
    waiting on rank 0. tt-run returns 1 naming rank 1 and the exception."""
    r = run_case("raise_after_open", {"PROBE_SCENARIO": "raise", "PROBE_PHASE": "after_open"}, DEADLINE_END)
    ended(r)
    stopped_by(r, 1, "Python exception RuntimeError: injected failure at after_open (closing the mesh)", 1)
    closed(r)
    clean(r)


def test_bad_shape():
    """Rank 1 asks for a mesh the descriptor cannot hold: the shape check in open is a TT_FATAL (a C++ exception,
    raised in Python as RuntimeError, its message several lines long) on every rank. The job is stopped with code 1,
    and tt-run reads the one-line [tt-stop] line and names the rank, not prterun's banner."""
    r = run_case("bad_shape_rank1", {"PROBE_MESH_SHAPE_VICTIM": "9x9"}, DEADLINE_END)
    ended(r)
    m = re.search(
        r"\[tt-stop\] rank (\d+) on \S+ \(pid \d+\): (?:uncaught )?Python exception RuntimeError: TT_FATAL.*"
        r"ranks disagree on the requested mesh shape.*; stopping the job \(exit 1\)",
        r.log,
    )
    assert m, f"no one-line [tt-stop] line for the TT_FATAL\n{r.log[-3000:]}"
    assert r.exit == 1 and f"origin=rank {m.group(1)} on" in r.log and "outcome=failed exit=1" in r.log, str(r)
    assert "the launcher then killed the other ranks" not in r.log and "job forced down" not in r.log, r.log[-3000:]
    clean(r)


def test_bad_descriptor(tmp_path):
    r = run_case("conflicting_descriptor", {}, DEADLINE_HANG, cmd=legacy_cmd(mapping=conflict_mapping(tmp_path)))
    ended(r)
    nonzero(r)
    clean(r)


def test_barrier_timeout():
    """P16, the per-collective timeout: rank 1 never joins the barrier, so rank 0 would wait forever. With
    TT_METAL_OPERATION_TIMEOUT_SECONDS set, the barrier gives up at the deadline and tears the whole job down (both
    ranks end, nothing left), rather than one rank raising while the peer keeps waiting. Ends well before the hang
    deadline, which proves the operation timeout fired, not the harness kill."""
    r = run_case(
        "barrier_skip", {"PROBE_COLL": "barrier_skip", "TT_METAL_OPERATION_TIMEOUT_SECONDS": "8"}, DEADLINE_END
    )
    ended(r)
    nonzero(r)
    clean(r)


def test_send_timeout():
    """P16 on send: a send nobody receives gives up at TT_METAL_OPERATION_TIMEOUT_SECONDS and the job is torn down."""
    r = run_case(
        "send_no_recv", {"PROBE_COLL": "send_no_recv", "TT_METAL_OPERATION_TIMEOUT_SECONDS": "8"}, DEADLINE_END
    )
    ended(r)
    nonzero(r)
    clean(r)


def test_sigkill_before_init():
    """Rank 1 is SIGKILLed before the context exists: its wrapper stops the job, rank 0 leaves MPI_Init's wait."""
    r = run_case("sigkill_before_init", {"PROBE_SCENARIO": "sigkill", "PROBE_PHASE": "before_init"}, DEADLINE_HANG)
    ended(r)
    assert r.exit == 137 and "[tt-stop] rank 1 on" in r.log, str(r)
    clean(r)


def test_dead_peer_in_open():
    # A peer dies *inside* open_mesh_device (the control plane's distributed init). The kill is sent by the parent
    # harness, gated on the victim's ``open_enter`` marker, because open holds the GIL — a Python timer set in the
    # rank could not fire inside open. The job must end non-zero with nothing left.
    import signal

    r = run_case(
        "sigkill_in_open",
        {"PROBE_SCENARIO": "clean", "PROBE_VICTIM": "1"},
        DEADLINE_END,
        rank_signal=(1, signal.SIGKILL, 0.2),
        signal_on_marker="open_enter",
    )
    ended(r)
    clean(r)
    nonzero(r)
    assert "FAULT NOT DELIVERED" not in r.signal_sent, r  # the victim really was killed inside open


def test_ctrl_c_in_deadlock():
    """SIGINT at tt-run while the ranks sit in the send/barrier deadlock: every rank gets SIGTERM, leaves its polled
    wait, closes its mesh and exits; tt-run returns 130."""
    r = run_case(
        "ctrl_c_in_deadlock",
        {"PROBE_COLL": "send_no_recv"},
        DEADLINE_END,
        launcher_signal=signal.SIGINT,
        launcher_ready=(0, "coll_ready"),  # wait until the ranks reach the collective, not a 45 s wall-clock guess
        signal_after=3.0,
    )
    fault_delivered(r)
    ended(r)
    stopped_at_ttrun(r, signal.SIGINT)
    clean(r)


def test_ctrl_c_in_init():
    """SIGINT at tt-run while rank 0 waits in MPI_Init for a rank 1 that starts 75 s late: nothing may be left."""
    r = run_case(
        "ctrl_c_in_init",
        {"PROBE_START_DELAY_S": "75", "PROBE_START_DELAY_RANK": "1"},
        DEADLINE_END + 30,
        launcher_signal=signal.SIGINT,
        launcher_ready=(0, "import_done"),  # rank 0 is up and about to block in init waiting for the late rank 1
        signal_after=5.0,
    )
    fault_delivered(r)
    ended(r)
    clean(r)


def test_sigterm_at_ttrun():
    """SIGTERM at tt-run (scancel of the job step, a CI timeout): as Ctrl-C, exit 143."""
    r = run_case(
        "sigterm_at_ttrun",
        {"PROBE_COLL": "send_no_recv"},
        DEADLINE_END,
        launcher_signal=signal.SIGTERM,
        launcher_ready=(0, "coll_ready"),
        signal_after=3.0,
    )
    fault_delivered(r)
    ended(r)
    stopped_at_ttrun(r, signal.SIGTERM)
    clean(r)


def test_solve_rank_killed():
    r = run_case("solve_rank_killed", {}, DEADLINE_HANG, cmd=solve_cmd(), phase1_kill=True)
    fault_delivered(r)  # the phase-1 (generate_rank_bindings) process really was found and killed
    ended(r)
    nonzero(r)
    clean(r)


# ---------------------------------------------------------------------------------------------------------------
# the rest of the plan's open errors that one host can show
# ---------------------------------------------------------------------------------------------------------------
def rank_saw(r, rank: int, marker: str) -> None:
    """The rank printed this marker at some point (not necessarily its last one)."""
    assert any(
        f"[probe r={rank} " in line and marker in line for line in r.log.splitlines()
    ), f"rank {rank} never printed {marker!r}; it ended at {r.reached.get(rank)!r}\n{r.log[-3000:]}"


def test_survivor_closes_ulfm():
    """Opt-in ULFM so the death is delivered as an error (the only mode in which the survivor runs on); rank 1 killed
    with the mesh open, rank 0 records the error and goes on to close: its close must complete."""
    r = run_case(
        "survivor_close_ulfm",
        # Under ULFM recovery the launcher does not end the job when ranks exit, so the dead rank's wrapper stands in
        # for the whole stop grace + 10 s before it leaves; a short grace keeps the row inside its deadline.
        {
            "PROBE_SCENARIO": "sigkill",
            "PROBE_PHASE": "after_open",
            "PROBE_ON_ERROR": "continue",
            "TT_RUN_STOP_GRACE_S": "10",
        },
        DEADLINE_HANG,
        cmd=legacy_cmd("--with-ft ulfm"),
    )
    rank_saw(r, 0, "close_done")
    ended(r)
    clean(r)


def test_abort_core_dump():
    """P8's shape: rank 1 aborts (SIGABRT) with the mesh open and the core limit raised, so the kernel writes a core
    that can take tens of seconds. The job must still end within 20 s with a non-zero code — it must not wait for the
    dying rank's core to finish. The fix is two-fold: the DistributedContext's own error path ends with ``_Exit``
    rather than ``std::abort`` (no core on that path), and the survivor's per-collective operation timeout tears the
    whole job down (``MPI_Abort`` → the launcher SIGKILLs the still-dumping rank), so teardown no longer waits on the
    core write however long it takes."""
    r = run_case(
        "abort_with_core",
        {
            "PROBE_SCENARIO": "abort",
            "PROBE_PHASE": "after_open",
            "PROBE_CORE_UNLIMITED": "1",
            "TT_METAL_OPERATION_TIMEOUT_SECONDS": "8",
        },
        DEADLINE_END,
    )
    ended(r)
    nonzero(r)
    # tt-run's own job time (launch to the end of the job), not the harness wall clock, which also counts tt-run's and
    # the ranks' start-up on a loaded runner.
    m = re.search(r"job end: .* elapsed=([0-9.]+)s", r.log)
    assert m, r.log[-2000:]
    assert float(m.group(1)) < 20, f"P8: the aborting rank's core dump held the job for {m.group(1)} s\n{r.log[-2000:]}"
    clean(r)


def test_unsat_mgd():
    """A quad-galaxy descriptor asked of the dual-galaxy mock: phase 1 cannot place it and must end non-zero."""
    r = run_case("unsat_mgd", {}, DEADLINE_HANG, cmd=solve_cmd(mgd=MGD_QUAD))
    ended(r)
    nonzero(r)
    clean(r)


def test_late_rank():
    """P10: a peer that never joins the MPI wire-up hangs rank 0 inside MPI_Init, which the per-collective operation
    timeout cannot reach (MPI_Init is not a DistributedContext collective). Here rank 1 starts 20 s late. With
    TT_METAL_DISTRIBUTED_INIT_TIMEOUT_SECONDS=5 rank 0's MPI_Init gives up at 5 s and exits, and the launcher ends the
    job — well before the 20 s late start, instead of rank 0 waiting the whole time."""
    r = run_case(
        "late_rank_init_timeout",
        {"PROBE_START_DELAY_S": "20", "PROBE_START_DELAY_RANK": "1", "TT_METAL_DISTRIBUTED_INIT_TIMEOUT_SECONDS": "3"},
        DEADLINE_END,
    )
    ended(r)
    nonzero(r)
    # Bounded: rank 0 gives up on MPI_Init at 3 s and exits; the rest is the launcher tearing the late rank down.
    # Well under the ~35 s it takes when rank 0 waits out the 20 s late start (the unbounded behaviour this fixes).
    m = re.search(r"job end: .* elapsed=([0-9.]+)s", r.log)  # tt-run's job time, not start-up on a loaded runner
    assert m, r.log[-2000:]
    assert (
        float(m.group(1)) < 18
    ), f"rank 0 waited {m.group(1)} s for the late rank with the init timeout at 3 s\n{r.log[-2000:]}"
    clean(r)


def test_shape_mismatch():
    r = run_case("shape_mismatch", {"PROBE_MESH_SHAPE_VICTIM": "4x4"}, DEADLINE_END)
    ended(r)
    nonzero(r)
    clean(r)


def test_allgather_timeout():
    """The user's example on all_gather: rank 1 never joins, so the others would wait in distributed_context_allgather_int
    forever. With the operation timeout set, the call gives up at the deadline and aborts the job (every rank ends),
    not a local raise that leaves the peers waiting."""
    r = run_case(
        "allgather_missing_peer",
        {"PROBE_COLL": "allgather_skip", "TT_METAL_OPERATION_TIMEOUT_SECONDS": "8"},
        DEADLINE_END,
    )
    ended(r)
    nonzero(r)
    clean(r)


def test_recv_timeout():
    """P16 on recv: rank 0 waits in recv for a peer that never sends; the per-call timeout cancels it at the deadline
    and tears the job down."""
    r = run_case(
        "recv_without_send", {"PROBE_COLL": "recv_no_send", "TT_METAL_OPERATION_TIMEOUT_SECONDS": "8"}, DEADLINE_END
    )
    ended(r)
    nonzero(r)
    clean(r)


def test_slow_collective_ok():
    """The timeout must fire only on a hang, not on a legitimately slow collective. Rank 1 joins the barrier 4 s late
    with the operation timeout at 30 s: the barrier completes and the job exits 0. A per-call timeout that is too eager
    would tear this down, so this gate guards against that."""
    r = run_case(
        "slow_collective_ok",
        {"PROBE_COLL": "barrier_slow", "PROBE_SLOW_S": "2", "TT_METAL_OPERATION_TIMEOUT_SECONDS": "10"},
        DEADLINE_END,
    )
    ended(r)
    assert r.exit == 0, str(r)
    clean(r)


def test_singleton_shm(tmp_path):
    """A tt-metal process nobody launched inits the distributed context as an MPI singleton. Open MPI would leave
    both a 16 MB shared-memory segment and its session directory (``/tmp/ompi.<host>.<uid>/...``) behind, because the
    launcher that reaps them never ran (#63870). The DistributedContext init asks for the ``self`` transport only (no
    shm segment) and steers the session directory into a private ``TMPDIR`` it removes at exit (nothing left). This is
    the less-invasive counterpart of tt-metal #63954, which fixes the same leak by not initialising MPI at all for a
    singleton; both are carried in this PR. Slow, because it imports ttnn; kept in Python like the rest of the suite.

    ``TMPDIR`` is pointed at a fresh directory so the session-dir check is scoped to this job (the shared
    ``/tmp/ompi.*`` tree can hold a concurrent job's subdirs)."""
    sess = tmp_path / "mpi"
    sess.mkdir()
    r = run_case(
        "singleton",
        {"TMPDIR": str(sess)},
        DEADLINE_END,
        cmd=[PYTHON, "-c", "import ttnn; ttnn.init_distributed_context()"],
    )
    ended(r)
    assert r.exit == 0, str(r)
    assert not r.shm_leaked, f"a tt-metal singleton left a shm segment {r.shm_leaked} (#63870)"
    session_dirs = sorted(p.name for p in sess.iterdir() if p.name.startswith(("tt_mpi_singleton", "ompi.")))
    assert not session_dirs, f"a tt-metal singleton left an Open MPI session directory {session_dirs} (#63870)"


def _singleton(name: str, code: str, tmp_path, deadline=DEADLINE_END, env=None):
    """Run one unlaunched tt-metal process with TMPDIR scoped to the row; return the result and what it left there."""
    sess = tmp_path / "mpi"
    sess.mkdir()
    r = run_case(name, {"TMPDIR": str(sess), **(env or {})}, deadline, cmd=[PYTHON, "-c", code])
    left = sorted(p.name for p in sess.iterdir() if p.name.startswith(("tt_mpi_singleton", "ompi.")))
    return r, left


def test_singleton_open_close_leaves_nothing(tmp_path):
    """The full path a one-host pytest takes without a launcher: init, open a 1x1 mesh on the mock, close, exit.
    Nothing may be left: no segment, no session directory (the #63870 shape, the P11 fix on the clean path)."""
    r, left = _singleton(
        "singleton_open_close",
        "import ttnn; ttnn.init_distributed_context(); m = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1)); "
        "ttnn.close_mesh_device(m)",
        tmp_path,
        env={"TT_METAL_MOCK_CLUSTER_DESC_PATH": str(ROOT / CONFLICT_DESC)},  # a one-galaxy mock: no device here
    )
    ended(r)
    assert r.exit == 0, str(r)
    assert not r.shm_leaked, f"a tt-metal singleton left a shm segment {r.shm_leaked} (#63870)"
    assert not left, f"a tt-metal singleton left an Open MPI session directory {left} (#63870)"


def test_singleton_raise_leaves_nothing(tmp_path):
    """A plain pytest on one host hits an exception after init. P31: a world of one has nobody to abort for, so the
    exception exits normally and the atexit that removes the private session directory runs."""
    r, left = _singleton(
        "singleton_raise", "import ttnn; ttnn.init_distributed_context(); raise RuntimeError('boom')", tmp_path
    )
    ended(r)
    nonzero(r)
    assert not r.shm_leaked, f"a tt-metal singleton left a shm segment {r.shm_leaked} (#63870)"
    assert not left, f"a tt-metal singleton that raised left an Open MPI session directory {left} (P31)"


def test_singleton_killed_leaves_nothing(tmp_path):
    """A plain one-host process killed after init (OOM, kill -9) cannot remove its private session directory. P32:
    the next singleton init on the host sweeps the directories whose owner pid is gone, so the one left here is
    removed by a second singleton run against the same TMPDIR. (#63954 makes this moot by starting no MPI at all.)"""
    r, left = _singleton(
        "singleton_killed",
        "import ttnn, os, signal; ttnn.init_distributed_context(); os.kill(os.getpid(), signal.SIGKILL)",
        tmp_path,
    )
    ended(r)
    nonzero(r)
    assert not r.shm_leaked, f"a tt-metal singleton left a shm segment {r.shm_leaked} (#63870)"
    assert left, "the killed singleton was expected to leave its directory for the next init to sweep"
    sess = tmp_path / "mpi"
    r2 = run_case(
        "singleton_sweep",
        {"TMPDIR": str(sess)},
        DEADLINE_END,
        cmd=[PYTHON, "-c", "import ttnn; ttnn.init_distributed_context()"],
    )
    ended(r2)
    assert r2.exit == 0, str(r2)
    left_after = sorted(p.name for p in sess.iterdir() if p.name.startswith(("tt_mpi_singleton", "ompi.")))
    assert not left_after, f"the next singleton init did not sweep the killed singleton's directory: {left_after} (P32)"


# ---------------------------------------------------------------------------------------------------------------
# gaps the launcher covers today that the DistributedContext should own (plan P23-P25). Each row fails today and is a
# strict xfail naming the error it exposes, so the suite stays green and the row turns red (XPASS) the day the fix
# lands, which is the signal to drop the marker. The TODO at the line that should own the fix names the row.
# ---------------------------------------------------------------------------------------------------------------
def test_sigint_one_rank():
    """SIGINT reaches rank 1 only (a user interrupts one pid): KeyboardInterrupt. P23: it stops the job with 130 (from
    the mesh close it unwinds through), so the peer leaves its collective and closes."""
    r = run_case("sigint_one_rank", {"PROBE_SCENARIO": "sigint", "PROBE_PHASE": "after_open"}, DEADLINE_END)
    ended(r)
    stopped_by(r, 1, "Python exception KeyboardInterrupt (closing the mesh)", 130)
    closed(r)
    clean(r)


def test_helper_outlives_rank():
    """Rank 1 forks a helper that ignores SIGTERM, then finishes its work and exits cleanly. The launcher kills a rank's
    whole process group on an abort but only reaps the rank on a clean exit, so the helper used to live on until
    tt-run's session sweep found it (launch host only) and turned the clean job into exit 1. P24: the rank's own exit
    path ends the process group it leads (SIGTERM, a grace, SIGKILL) before MPI_Finalize, so the job is clean and exit 0
    without the launcher or tt-run having to sweep."""
    r = run_case("helper_outlives_rank", {"PROBE_SCENARIO": "fork_helper", "PROBE_PHASE": "after_open"}, DEADLINE_END)
    ended(r)
    clean(r)
    assert "process(es) of its job alive" not in r.log, (
        "the helper was ended by tt-run's sweep, not by the rank's own exit path\n" + r.log[-2000:]
    )
    assert r.exit == 0, str(r)


def test_helper_ended_by_rank_without_ttrun(tmp_path):
    """P24 at the MPI-context level alone: launched with mpirun directly, so neither tt-run's session sweep nor
    its exit-code handling is in the picture. A one-rank job inits the context, forks a helper that ignores SIGTERM,
    prints the helper's pid and exits cleanly. The MPIContext's exit path must have ended the helper by the time the
    launcher returns; the row checks the pid itself, not a log line."""
    import subprocess

    code = (
        "import ttnn, subprocess, signal, sys; ttnn.init_distributed_context(); "
        "h = subprocess.Popen([sys.executable, '-c', "
        "'import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(600)'], "
        "stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); "
        "print('HELPER_PID', h.pid, flush=True)"
    )
    cmd = [
        mpirun_path(),
        "--mca",
        "btl",
        "self,tcp",
        "--allow-run-as-root",
        "--oversubscribe",
        "-np",
        "1",
        PYTHON,
        "-c",
        code,
    ]
    r = run_case("helper_without_ttrun", {"TT_METAL_MOCK_CLUSTER_DESC_PATH": str(CONFLICT_DESC)}, DEADLINE_END, cmd=cmd)
    ended(r)
    assert r.exit == 0, str(r)
    m = re.search(r"HELPER_PID (\d+)", r.log)
    assert m, "the rank did not report its helper's pid\n" + r.log[-2000:]
    helper = int(m.group(1))
    alive = os.path.exists(f"/proc/{helper}")
    if alive:
        os.kill(helper, signal.SIGKILL)  # do not leave it for the next row
    assert not alive, (
        f"the helper (pid {helper}) outlived the rank: the MPIContext exit path did not end it\n" + r.log[-2000:]
    )
    assert "ending 1 process(es) this rank left behind" in r.log, (
        "the context did not report ending the helper\n" + r.log[-2000:]
    )
    clean(r)


def test_launcher_killed_no_session_dir():
    """kill -9 of prterun while the ranks sit in a collective: tt-run ends the orphaned ranks and, since prterun's exit
    path never ran, removes its session directory under /tmp itself (P25, #65121). Nothing may be left, directories
    included."""
    r = run_case(
        "launcher_killed",
        {"PROBE_COLL": "send_no_recv"},
        DEADLINE_END,
        launcher_ready=(0, "coll_ready"),
        signal_after=3.0,
        kill_launcher=True,
    )
    fault_delivered(r)
    ended(r)
    nonzero(r)
    clean(r)


# ---------------------------------------------------------------------------------------------------------------
# the remaining gaps only the DistributedContext can close (plan P26-P30): one strict xfail each, a TODO at the owner
# ---------------------------------------------------------------------------------------------------------------
def test_context_sweeps_stale_launcher_dir(tmp_path):
    """P25, the fail-safe inside the rank: tt-run's sweep (#65121) reaches the launch host only. A launcher killed on a
    remote host leaves its session directory where only the next job's ranks can see it, so a launched rank's context
    removes the stale launcher directories of its own host at init: the ones whose owner pid (in the name,
    prterun.<host>.<pid>.<uid>) is gone, found through PMIx's PMIX_SYSTEM_TMPDIR, never this job's own
    (PMIX_SERVER_TMPDIR). Launched with mpirun directly so tt-run's own sweep cannot be the one that removes it."""
    import socket
    import subprocess

    helper = subprocess.Popen(["true"])  # a pid that has exited: nobody owns it any more
    helper.wait()
    stale = Path(tempfile.gettempdir()) / f"prterun.{socket.gethostname()}.{helper.pid}.{os.getuid()}"
    (stale / "hwloc.sm").mkdir(parents=True, exist_ok=True)
    assert str(stale) in _session_dirs()
    code = "import ttnn; ttnn.init_distributed_context(); print('CTX_OK', flush=True)"
    cmd = [
        mpirun_path(),
        "--mca",
        "btl",
        "self,tcp",
        "--allow-run-as-root",
        "--oversubscribe",
        "-np",
        "1",
        PYTHON,
        "-c",
        code,
    ]
    r = run_case("context_sweeps_stale", {"TT_METAL_MOCK_CLUSTER_DESC_PATH": str(CONFLICT_DESC)}, DEADLINE_END, cmd=cmd)
    ended(r)
    assert r.exit == 0, str(r)
    assert "CTX_OK" in r.log, str(r)
    assert not stale.exists(), f"the stale launcher directory survived the next job's init: {stale}"
    clean(r)


def test_exit1_stops_the_job():
    """Rank 1 calls sys.exit(1) after open, the way click or a script's error return does. P26: sys.exit stops the job
    at the call (SIGTERM to every rank), then the SystemExit unwinds through the finally that closes the mesh. Every
    rank closes and leaves together; tt-run returns 1 naming rank 1."""
    r = run_case("exit1_after_open", {"PROBE_SCENARIO": "exit1", "PROBE_PHASE": "after_open"}, DEADLINE_END)
    ended(r)
    stopped_by(r, 1, "sys.exit(1)", 1)
    closed(r)
    clean(r)


def test_early_exit0_hangs_peer():
    """Rank 1 returns early with exit 0 (a logic bug) while rank 0 enters the barrier. Nothing is an error, so
    nothing aborts: rank 1 sits in its finalize fence, rank 0 in the barrier. P27: with the operation timeout set,
    rank 0's barrier expires and aborts the job, which releases rank 1's fence too. The fence itself is not bounded:
    a rank cannot tell a peer that is still working (compiling kernels, dumping a profile) from one that is stuck,
    and bounding it aborted real jobs whose ranks finish at different times (the streaming profiler torus test).
    Without a timeout both wait forever (P30, test_hang_without_any_timeout)."""
    r = run_case(
        "exit0_after_open",
        {"PROBE_SCENARIO": "exit0", "PROBE_PHASE": "after_open", "TT_METAL_OPERATION_TIMEOUT_SECONDS": "8"},
        DEADLINE_HANG,
    )
    ended(r)
    nonzero(r)
    clean(r)


def test_sigterm_runs_close():
    """SIGTERM reaches rank 1 alone from outside the launcher (kill <pid>, scancel of one task). P28: the rank sends
    it on to the whole job, every rank leaves its wait, closes its mesh and exits; tt-run returns 143 naming rank 1."""
    r = run_case(
        "sigterm_one_rank",
        {"PROBE_COLL": "barrier_slow", "PROBE_SLOW_S": "8"},
        DEADLINE_END,
        rank_signal=(1, signal.SIGTERM, 0.5),
        signal_on_marker="coll_ready",
    )
    fault_delivered(r)
    ended(r)
    stopped_by(r, 1, "received SIGTERM from pid", 143)
    closed(r)
    clean(r)


def test_bare_launch_dead_rank_leaves_no_shm():
    """P29 guard: tt-run --bare drops the TCP-only BTL line, so the shared-memory transport is back. A rank killed
    with the mesh open must still leave no sm_segment behind. If this row fails, the context should set the BTL
    default itself before MPI_Init instead of relying on tt-run's launch line."""
    cmd = legacy_cmd()
    cmd.insert(cmd.index(PYTHON), "--bare")
    r = run_case("bare_dead_rank", {"PROBE_SCENARIO": "sigkill", "PROBE_PHASE": "after_open"}, DEADLINE_END, cmd=cmd)
    ended(r)
    nonzero(r)
    clean(r)


# ---------------------------------------------------------------------------------------------------------------
# the job-stop table: every way a job ends, each a graceful close of every rank that is still alive
# ---------------------------------------------------------------------------------------------------------------
def test_segfault_stops_the_job():
    """Rank 1 segfaults with the mesh open: its wrapper stops the job, rank 0 leaves its barrier and closes; 139."""
    r = run_case("segv_after_open", {"PROBE_SCENARIO": "segv", "PROBE_PHASE": "after_open"}, DEADLINE_END)
    ended(r)
    stopped_by(r, 1, "was killed by signal 11 (SIGSEGV)", 139)
    closed(r, ranks=(0,))
    clean(r)


def test_sigkill_with_mesh_open_stops_the_job():
    """Rank 1 is SIGKILLed with the mesh open (OOM, kill -9): its wrapper stops the job, rank 0 closes; 137."""
    r = run_case("sigkill_after_open", {"PROBE_SCENARIO": "sigkill", "PROBE_PHASE": "after_open"}, DEADLINE_END)
    ended(r)
    stopped_by(r, 1, "was killed by signal 9 (SIGKILL)", 137)
    closed(r, ranks=(0,))
    clean(r)


def test_thread_exception_stops_the_job():
    """An uncaught exception in a worker thread of rank 1 (threading.excepthook): the job is stopped with code 1."""
    r = run_case("thread_raise", {"PROBE_SCENARIO": "thread_raise", "PROBE_PHASE": "after_open"}, DEADLINE_END)
    ended(r)
    stopped_by(r, 1, "uncaught Python exception in thread worker RuntimeError: worker failed", 1)
    closed(r)
    clean(r)


def test_clean_exit_waits_for_a_slow_rank():
    """Rank 1 has finished and exits 0 while rank 0 is still working 15 s longer (longer than tt-run's old 10 s
    settle). A rank that exits 0 starts nothing: no stop, no warning, nothing killed; the job waits for rank 0 and
    exits 0."""
    r = run_case("slow_exit", {"PROBE_SCENARIO": "slow_exit", "PROBE_VICTIM": "0", "PROBE_SLOW_S": "15"}, DEADLINE_END)
    ended(r)
    assert r.exit == 0 and "outcome=clean exit=0" in r.log, str(r)
    assert r.wall_s >= 15, f"the job ended before the slow rank finished: {r}"
    for bad in ("[tt-stop]", "sent SIGTERM", "job forced down", "a leak", "still running"):
        assert bad not in r.log, f"{bad!r} on a clean job\n{r.log[-3000:]}"
    assert any("[probe r=0 " in line and "main_exit" in line for line in r.log.splitlines()), r.log[-3000:]
    clean(r)


@pytest.mark.parametrize("how", ["second_ctrl_c", "grace_expired"])
def test_stuck_rank_is_forced(how):
    """Ctrl-C while rank 1 is stuck and does not react to SIGTERM (a device wait): rank 0 closes, then tt-run forces
    the job, on a second Ctrl-C or when the stop grace expires, says the devices may need a reset, and returns 130."""
    env = {"PROBE_SCENARIO": "ignore_sigterm", "PROBE_PHASE": "after_open", "PROBE_COLL": "barrier"}
    if how == "grace_expired":
        env["TT_RUN_STOP_GRACE_S"] = "6"
    r = run_case(
        f"stuck_rank_{how}",
        env,
        DEADLINE_END,
        launcher_signal=signal.SIGINT,
        launcher_ready=(0, "coll_ready"),
        signal_after=3.0,
        second_launcher_signal_after=3.0 if how == "second_ctrl_c" else None,
    )
    fault_delivered(r)
    ended(r)
    assert r.exit == 130, str(r)
    why = "second SIGINT" if how == "second_ctrl_c" else "ranks did not exit within 6 s"
    assert f"job forced down" in r.log and why in r.log and "tt-smi -r" in r.log, r.log[-3000:]
    assert "outcome=forced exit=130" in r.log, r.log[-3000:]
    closed(r, ranks=(0,))
    clean(r)


@pytest.mark.xfail(
    strict=True,
    reason="P30: with no timeout set anywhere a hung collective never ends; policy: a default for TT_METAL_OPERATION_TIMEOUT_SECONDS, or tt-run setting it per rank",
)
def test_hang_without_any_timeout():
    """Rank 1 skips the barrier and nobody set a timeout: not the operation timeout, not PRRTE's --timeout. R2 says
    the job ends by itself; today both defaults are off, so it does not."""
    assert not os.environ.get("TT_METAL_OPERATION_TIMEOUT_SECONDS"), "this row needs the operation timeout unset"
    r = run_case("hang_no_timeout", {"PROBE_COLL": "barrier_skip"}, DEADLINE_HANG)
    ended(r)
    clean(r)
