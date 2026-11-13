# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Teardown sanity: does a distributed tt-metal job end, end cleanly, and report it, after one failure?

CPU only, on the mock dual galaxy (two ranks of 8x4, no device), launched by tt-run like every real job. Every
test is one fault with one assertion, taken from the teardown plan of tt-metal #56902 (tt-blaze
tools/teardown_probe, lane L4): the rank program is ``sanity_probe.py`` next to this file. The faults are the
two kinds a real job meets: the code failing on one rank (an exception, a shape the descriptor cannot hold, a
descriptor that disagrees, a collective with a wrong or dead peer) and the user (a signal at tt-run).

Requirements, as in the plan: a failure never becomes a hang (R1), a hang ends by itself and leaves nothing
(R2), nobody kills processes by hand (R3), the exit code reports the failure (R4).

Modes. A row that passes today is a gate. A row that fails today is marked ``@known("P..: ...")`` with the
plan's error id and fix: it fails, so the job is red until the fix lands, and its failure message names the
error; when the fix lands the row passes and the marker is dropped. Each row has a deadline; a job that is still running at the
deadline is killed by its process group and counted as a hang.

Run: pytest tests/tt_metal/distributed/multiprocess/teardown_sanity -v   (from the repository root, with a
built tree and the python environment; the fabric CPU lane runs it as group ``teardown-sanity`` of
tests/scripts/multihost/run_fabric_cpu_only_unit_tests.sh).
"""

from __future__ import annotations

import signal
import sys
from pathlib import Path

import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parent)
)  # sanity_common lives next to this file, whatever the import mode
from sanity_common import (
    known,
    DEADLINE_END,
    DEADLINE_HANG,
    clean,
    conflict_mapping,
    ended,
    fault_delivered,
    legacy_cmd,
    nonzero,
    rc_contains,
    run_case,
    solve_cmd,
    MGD_QUAD,
    PYTHON,
)


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
    """Default launcher (no ULFM): rank 1 dies before a barrier, so the whole job ends with a non-zero code and nothing
    is left. A hung (not dead) peer is the per-collective-timeout case instead (test_barrier_skipped_ends, P16)."""
    r = run_case("dead_peer_barrier", {"PROBE_COLL": "barrier_dead_peer"}, DEADLINE_END)
    ended(r)
    nonzero(r)
    clean(r)


def test_dead_peer_recv():
    """Default launcher: rank 1 dies before rank 0's recv, so the job ends non-zero instead of rank 0 waiting forever."""
    r = run_case("dead_peer_recv", {"PROBE_COLL": "recv_dead_peer"}, DEADLINE_END)
    ended(r)
    nonzero(r)
    clean(r)


def test_fabric_mismatch():
    """One rank sets FABRIC_2D before open and the other does not: TT_FATAL on one rank, the job ends non-zero."""
    r = run_case("fabric_mismatch", {"PROBE_FABRIC_RANK": "1"}, DEADLINE_END)
    ended(r)
    nonzero(r)
    clean(r)


def test_prte_timeout():
    """A send nobody receives under PRRTE --timeout 60: the wall clock ends the job with 241 and nothing left.

    The timeout is longer than the ttnn import on a loaded runner, and the row checks that both ranks were past
    open when the clock fired, so it is the deadlock that was ended and not the start-up."""
    r = run_case(
        "send_no_recv+prte_timeout60", {"PROBE_COLL": "send_no_recv"}, DEADLINE_END + 60, cmd=legacy_cmd("--timeout 60")
    )
    ended(r)
    assert r.exit == 241, str(r)
    # Require positive evidence that the wall clock ended the intended deadlock, not a slow start-up: both ranks
    # must have reached the collective section (a ``coll_*`` marker), past open — not merely "not a startup marker",
    # which would also accept ``open_enter`` (entered open, never completed).
    for rank in (0, 1):
        assert r.reached.get(rank, "").startswith(
            "coll"
        ), f"rank {rank} had not reached the collective when the wall clock fired: {r}"
    clean(r)


def test_exit_nofinalize():
    """Rank 1 _exits with the mesh open (the crash path, no finalize): the job ends non-zero on the production line."""
    r = run_case(
        "exit_nofinalize_after_open", {"PROBE_SCENARIO": "exit_nofinalize", "PROBE_PHASE": "after_open"}, DEADLINE_END
    )
    ended(r)
    nonzero(r)
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
    r = run_case("raise_after_open", {"PROBE_SCENARIO": "raise", "PROBE_PHASE": "after_open"}, DEADLINE_HANG)
    ended(r)
    nonzero(r)
    clean(r)


def test_bad_shape():
    r = run_case("bad_shape_rank1", {"PROBE_MESH_SHAPE_VICTIM": "9x9"}, DEADLINE_HANG)
    ended(r)
    nonzero(r)
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
    r = run_case("sigkill_before_init", {"PROBE_SCENARIO": "sigkill", "PROBE_PHASE": "before_init"}, DEADLINE_HANG)
    ended(r)
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
    """SIGINT at tt-run while the ranks sit in the send/barrier deadlock: tt-run ends and nothing may be left."""
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
    r = run_case(
        "sigterm_at_ttrun",
        {"PROBE_COLL": "send_no_recv"},
        DEADLINE_HANG,
        launcher_signal=signal.SIGTERM,
        launcher_ready=(0, "coll_ready"),
        signal_after=3.0,
    )
    fault_delivered(r)
    ended(r)
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
        {"PROBE_SCENARIO": "sigkill", "PROBE_PHASE": "after_open", "PROBE_ON_ERROR": "continue"},
        DEADLINE_HANG,
        cmd=legacy_cmd("--with-ft ulfm"),
    )
    rank_saw(r, 0, "close_done")
    ended(r)
    clean(r)


@known(
    "P8: an aborting rank holds the whole job for as long as its core dump takes (38 s on the sandbox with the core limit raised); fix: P1, no SIGABRT on the error path"
)
def test_abort_core_dump():
    """P8's shape: rank 1 aborts with the mesh open and the core limit raised. The job must still end within 20 s
    with a non-zero code: a slow core dump (apport) holds every other rank for its whole length."""
    r = run_case(
        "abort_with_core",
        {"PROBE_SCENARIO": "abort", "PROBE_PHASE": "after_open", "PROBE_CORE_UNLIMITED": "1"},
        DEADLINE_END,
    )
    ended(r)
    nonzero(r)
    assert r.wall_s < 20, f"P8: the aborting rank's core dump held the job for {r.wall_s:.0f} s\n{r.log[-2000:]}"
    clean(r)


@known(
    "P9: an unsatisfiable mesh graph descriptor makes generate_rank_bindings TT_FATAL on rank 0 while rank 1 waits forever in phase 1"
)
def test_unsat_mgd():
    """A quad-galaxy descriptor asked of the dual-galaxy mock: phase 1 cannot place it and must end non-zero."""
    r = run_case("unsat_mgd", {}, DEADLINE_HANG, cmd=solve_cmd(mgd=MGD_QUAD))
    ended(r)
    nonzero(r)
    clean(r)


@known(
    "P10: the operation timeout does not cover startup; rank 0 waits the whole 20 s for a late rank with TT_METAL_OPERATION_TIMEOUT_SECONDS=1; fix: bounded discovery (P16) or the launcher's clock (P5)"
)
def test_late_rank():
    r = run_case(
        "late_rank_op_timeout",
        {"PROBE_START_DELAY_S": "20", "PROBE_START_DELAY_RANK": "1", "TT_METAL_OPERATION_TIMEOUT_SECONDS": "1"},
        DEADLINE_END,
    )
    ended(r)
    assert r.wall_s < 15, f"rank 0 waited {r.wall_s:.0f} s for the late rank with the timeout at 1 s\n{r.log[-2000:]}"


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
