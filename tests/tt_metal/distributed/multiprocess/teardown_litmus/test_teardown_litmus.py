# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Teardown litmus: does a distributed tt-metal job end, end cleanly, and report it, after one failure?

CPU only, on the mock dual galaxy (two ranks of 8x4, no device), launched by tt-run like every real job. Every
test is one fault with one assertion, taken from the teardown plan of tt-metal #56902 (tt-blaze
tools/teardown_probe, lane L4): the rank program is ``litmus_probe.py`` next to this file. The faults are the
two kinds a real job meets: the code failing on one rank (an exception, a shape the descriptor cannot hold, a
descriptor that disagrees, a collective with a wrong or dead peer) and the user (a signal at tt-run).

Requirements, as in the plan: a failure never becomes a hang (R1), a hang ends by itself and leaves nothing
(R2), nobody kills processes by hand (R3), the exit code reports the failure (R4).

Modes. A row that passes today is a gate. A row that fails today is marked ``@known("P..: ...")`` with the
plan's error id and fix: it fails, so the job is red until the fix lands, and its failure message names the
error; when the fix lands the row passes and the marker is dropped. Each row has a deadline; a job that is still running at the
deadline is killed by its process group and counted as a hang.

Run: pytest tests/tt_metal/distributed/multiprocess/teardown_litmus -v   (from the repository root, with a
built tree and the python environment; the fabric CPU lane runs it as group ``teardown-litmus`` of
tests/scripts/multihost/run_fabric_cpu_only_unit_tests.sh).
"""

from __future__ import annotations

import signal
import sys
from pathlib import Path

import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parent)
)  # litmus_common lives next to this file, whatever the import mode
from litmus_common import (
    known,
    DEADLINE_END,
    DEADLINE_HANG,
    clean,
    conflict_mapping,
    ended,
    legacy_cmd,
    nonzero,
    rc_contains,
    run_case,
    solve_cmd,
    MGD_QUAD,
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


def test_dead_peer_barrier_ends():
    """Default launcher (no ULFM): rank 1 dies before a barrier, so the whole job ends with a non-zero code and nothing
    is left. A hung (not dead) peer is the per-collective-timeout case instead (test_barrier_skipped_ends, P16)."""
    r = run_case("dead_peer_barrier", {"PROBE_COLL": "barrier_dead_peer"}, DEADLINE_END)
    ended(r)
    nonzero(r)
    clean(r)


def test_dead_peer_recv_ends():
    """Default launcher: rank 1 dies before rank 0's recv, so the job ends non-zero instead of rank 0 waiting forever."""
    r = run_case("dead_peer_recv", {"PROBE_COLL": "recv_dead_peer"}, DEADLINE_END)
    ended(r)
    nonzero(r)
    clean(r)


def test_fabric_config_mismatch_ends_nonzero():
    """One rank sets FABRIC_2D before open and the other does not: TT_FATAL on one rank, the job ends non-zero."""
    r = run_case("fabric_mismatch", {"PROBE_FABRIC_RANK": "1"}, DEADLINE_END)
    ended(r)
    nonzero(r)
    clean(r)


def test_prte_timeout_ends_deadlock():
    """A send nobody receives under PRRTE --timeout 60: the wall clock ends the job with 241 and nothing left.

    The timeout is longer than the ttnn import on a loaded runner, and the row checks that both ranks were past
    open when the clock fired, so it is the deadlock that was ended and not the start-up."""
    r = run_case(
        "send_no_recv+prte_timeout60", {"PROBE_COLL": "send_no_recv"}, DEADLINE_END + 60, cmd=legacy_cmd("--timeout 60")
    )
    ended(r)
    assert r.exit == 241, str(r)
    for rank in (0, 1):
        assert r.reached.get(rank, "") not in (
            "",
            "start",
            "start_delay",
            "import_done",
            "ctx_init",
        ), f"rank {rank} had not reached open when the wall clock fired: {r}"
    clean(r)


def test_exit_nofinalize_ends_nonzero():
    """Rank 1 _exits with the mesh open (the crash path, no finalize): the job ends non-zero on the production line."""
    r = run_case(
        "exit_nofinalize_after_open", {"PROBE_SCENARIO": "exit_nofinalize", "PROBE_PHASE": "after_open"}, DEADLINE_END
    )
    ended(r)
    nonzero(r)
    clean(r)


def test_invalid_rank_raises():
    """A receive from a rank that does not exist raises at once; the job goes on and ends."""
    r = run_case("recv_bad_rank", {"PROBE_COLL": "recv_bad_rank", "PROBE_ON_ERROR": "continue"}, DEADLINE_END)
    ended(r)
    rc_contains(r, 0, "RuntimeError")
    clean(r)


def test_use_before_init_raises():
    """A collective before init_distributed_context raises 'not initialized' instead of crashing or hanging."""
    r = run_case("use_before_init", {"PROBE_SCENARIO": "use_before_init", "PROBE_ON_ERROR": "continue"}, DEADLINE_END)
    ended(r)
    rc_contains(r, 1, "not initialized")
    assert r.exit == 0, str(r)
    clean(r)


# ---------------------------------------------------------------------------------------------------------------
# rows that fail today. Each marker names the error of the plan and its fix; drop the marker with the fix.
# ---------------------------------------------------------------------------------------------------------------
@known(
    "P1: a rank that raises leaves the others in their barrier forever; fix: abort on an uncaught exception in the DistributedContext"
)
def test_raise_after_open_ends():
    r = run_case("raise_after_open", {"PROBE_SCENARIO": "raise", "PROBE_PHASE": "after_open"}, DEADLINE_HANG)
    ended(r)
    nonzero(r)
    clean(r)


@known(
    "P21: a TT_FATAL inside open on one rank (a shape the descriptor cannot hold) leaves its peer in the discovery barrier forever"
)
def test_bad_shape_on_one_rank_ends():
    r = run_case("bad_shape_rank1", {"PROBE_MESH_SHAPE_VICTIM": "9x9"}, DEADLINE_HANG)
    ended(r)
    nonzero(r)
    clean(r)


@known(
    "P21: a rank whose descriptor describes another machine makes the mapper TT_FATAL on one rank while the other waits forever"
)
def test_conflicting_descriptor_ends(tmp_path):
    r = run_case("conflicting_descriptor", {}, DEADLINE_HANG, cmd=legacy_cmd(mapping=conflict_mapping(tmp_path)))
    ended(r)
    nonzero(r)
    clean(r)


@known("P16: a collective on the distributed context has no timeout; a skipped barrier hangs forever")
def test_barrier_skipped_ends():
    r = run_case("barrier_skip", {"PROBE_COLL": "barrier_skip"}, DEADLINE_HANG)
    ended(r)
    clean(r)


@known(
    "P16: a send nobody receives (the ssend/irecv deadlock of the topology mapper) hangs forever without PRRTE --timeout"
)
def test_send_without_recv_ends():
    r = run_case("send_no_recv", {"PROBE_COLL": "send_no_recv"}, DEADLINE_HANG)
    ended(r)
    clean(r)


def test_sigkill_before_init_ends():
    r = run_case("sigkill_before_init", {"PROBE_SCENARIO": "sigkill", "PROBE_PHASE": "before_init"}, DEADLINE_HANG)
    ended(r)
    clean(r)


def test_dead_peer_in_open_exit_nonzero():
    r = run_case(
        "sigkill_in_open",
        {"PROBE_SCENARIO": "sigkill", "PROBE_PHASE": "in_open", "PROBE_DELAY_MS": "100"},
        DEADLINE_END,
    )
    ended(r)
    clean(r)
    nonzero(r)


@known(
    "P6: Ctrl-C at tt-run in the deadlock leaves prterun running (seen in CI with the wheel's tt-run; a sandbox tt-run cleaned up)"
)
def test_ctrl_c_in_deadlock_leaves_nothing():
    """SIGINT at tt-run while the ranks sit in the send/barrier deadlock: tt-run ends and nothing may be left."""
    r = run_case(
        "ctrl_c_in_deadlock",
        {"PROBE_COLL": "send_no_recv"},
        DEADLINE_END,
        launcher_signal=signal.SIGINT,
        signal_after=45.0,  # both ranks are past open and in the deadlock by then
    )
    ended(r)
    clean(r)


@known("P6: Ctrl-C at tt-run while a rank waits in the context init SIGKILLs prterun and leaves that rank behind")
def test_ctrl_c_in_init_leaves_nothing():
    """SIGINT at tt-run while rank 0 waits in MPI_Init for a rank 1 that starts 75 s late: nothing may be left."""
    r = run_case(
        "ctrl_c_in_init",
        {"PROBE_START_DELAY_S": "75", "PROBE_START_DELAY_RANK": "1"},
        DEADLINE_END + 30,
        launcher_signal=signal.SIGINT,
        signal_after=50.0,
    )
    ended(r)
    clean(r)


@known("P7: SIGTERM at tt-run is not forwarded; prterun and the ranks stay in the deadlock")
def test_sigterm_at_ttrun_ends():
    r = run_case(
        "sigterm_at_ttrun",
        {"PROBE_COLL": "send_no_recv"},
        DEADLINE_HANG,
        launcher_signal=signal.SIGTERM,
        signal_after=20.0,
    )
    ended(r)
    clean(r)


def test_solve_rank_killed_ends():
    r = run_case("solve_rank_killed", {}, DEADLINE_HANG, cmd=solve_cmd(), phase1_kill=True)
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


@known(
    "P3: the survivor of a dead peer crashes inside its own close (the context barrier throws out of close_impl), so the device is never closed; fix: no collective in close after a detected failure"
)
def test_survivor_closes_after_peer_death_ulfm():
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
def test_abort_with_core_dump_ends_quickly():
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
def test_unsat_mgd_solve_ends():
    """A quad-galaxy descriptor asked of the dual-galaxy mock: phase 1 cannot place it and must end non-zero."""
    r = run_case("unsat_mgd", {}, DEADLINE_HANG, cmd=solve_cmd(mgd=MGD_QUAD))
    ended(r)
    nonzero(r)
    clean(r)


@known(
    "P10: the operation timeout does not cover startup; rank 0 waits the whole 20 s for a late rank with TT_METAL_OPERATION_TIMEOUT_SECONDS=1; fix: bounded discovery (P16) or the launcher's clock (P5)"
)
def test_late_rank_in_open_bounded():
    r = run_case(
        "late_rank_op_timeout",
        {"PROBE_START_DELAY_S": "20", "PROBE_START_DELAY_RANK": "1", "TT_METAL_OPERATION_TIMEOUT_SECONDS": "1"},
        DEADLINE_END,
    )
    ended(r)
    assert r.wall_s < 15, f"rank 0 waited {r.wall_s:.0f} s for the late rank with the timeout at 1 s\n{r.log[-2000:]}"


@known("P17: a mesh-shape disagreement between ranks (4x4 against 8x4) is accepted silently and the job exits 0")
def test_shape_mismatch_refused():
    r = run_case("shape_mismatch", {"PROBE_MESH_SHAPE_VICTIM": "4x4"}, DEADLINE_END)
    ended(r)
    nonzero(r)
    clean(r)


@known(
    "P16: an all_gather where one rank never calls it hangs the others forever; fix: a per-call timeout on the collective that cancels and tears the job down"
)
def test_allgather_missing_peer_ends():
    """The user's example: rank 1 does not join the all_gather, so ranks 0 and 2 wait in distributed_context_allgather_int
    forever. The fix is step 3: the call gives up at its deadline and aborts the job, not a local raise that leaves peers.
    """
    r = run_case("allgather_missing_peer", {"PROBE_COLL": "allgather_skip"}, DEADLINE_HANG)
    ended(r)
    nonzero(r)
    clean(r)


@known(
    "P16: a recv whose peer never sends hangs rank 0 forever; fix: the per-call timeout on recv cancels and tears the job down"
)
def test_recv_without_send_ends():
    """The recv side of the per-collective timeout: rank 0 waits in recv for a rank that never sends."""
    r = run_case("recv_without_send", {"PROBE_COLL": "recv_no_send"}, DEADLINE_HANG)
    ended(r)
    nonzero(r)
    clean(r)


def test_slow_collective_within_deadline_completes():
    """The timeout must fire only on a hang, not on a legitimately slow collective. Rank 1 joins the barrier 4 s late
    with the operation timeout at 30 s: the barrier completes and the job exits 0. A per-call timeout that is too eager
    would tear this down, so this gate guards against that."""
    r = run_case(
        "slow_collective_ok",
        {"PROBE_COLL": "barrier_slow", "PROBE_SLOW_S": "4", "TT_METAL_OPERATION_TIMEOUT_SECONDS": "30"},
        DEADLINE_END,
    )
    ended(r)
    assert r.exit == 0, str(r)
    clean(r)
