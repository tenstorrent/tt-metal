# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Teardown sanity, level 2, Python: a mock MeshDevice open on the dual Blackhole galaxy.

Two ttnn ranks under tt-run on the CPU, each opening a mock mesh (the control plane's distributed init:
discovery, the fabric-config check, the topology solve, the exchange; no kernels), one injected fault per row:
a death inside open, a shape, fabric or descriptor conflict, a Python failure unwinding into the close, a rank
dying with the mesh open. The rank program is ``mesh_probe.py``; the context-only rows are ``test_context_py.py``.

Run: pytest tests/ttnn/distributed/teardown_sanity/test_mesh_py.py -v (a built tree, the python environment).
"""

from __future__ import annotations

import os
import re
import signal
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "tests/tt_metal/distributed/multiprocess/teardown_sanity"))
from rows_common import *  # noqa: E402,F401,F403 - the harness, the checks and the helpers
from rows_common import _both_ranks_were_in_the_collective, _session_dirs, _singleton, probe_fixture  # noqa: E402

_probe = probe_fixture(Path(__file__).resolve().parent / "mesh_probe.py")


def test_fabric_mismatch():
    """One rank sets FABRIC_2D before open and the other does not: TT_FATAL on one rank, the job ends non-zero."""
    r = run_case("fabric_mismatch", {"PROBE_FABRIC_RANK": "1"}, DEADLINE_END)
    ended(r)
    nonzero(r)
    clean(r)


def test_invalid_rank_uncaught_stops_the_job():
    """The same receive from a rank that does not exist, but nothing catches it: the RuntimeError unwinds into the
    mesh close, the close stops the job, rank 1 leaves the barrier it was in on the SIGTERM and closes, and tt-run
    returns 1 naming rank 0 and the MPI error."""
    r = run_case("recv_bad_rank_uncaught", {"PROBE_COLL": "recv_bad_rank_peer_barrier"}, DEADLINE_END)
    ended(r)
    stopped_by(r, 0, "Python exception RuntimeError: MPI_Irecv", 1)
    assert "MPI_ERR_RANK" in r.log, r.log[-3000:]
    tore_down(r, ranks=(1,))
    tore_down(r, ranks=(0,), reason="Python exception RuntimeError")
    clean(r)


def test_raise_while_peer_in_split():
    """Rank 1 raises while rank 0 is inside a communicator split (the one context call with no non-blocking form,
    used by every mesh open and by the control plane): rank 0 must still leave it on the SIGTERM, close its mesh and
    exit, so the job ends with 1 naming rank 1 instead of rank 0 sitting in MPI_Comm_split until the force."""
    r = run_case("raise_while_peer_in_split", {"PROBE_COLL": "split_peer_raises"}, DEADLINE_END)
    ended(r)
    stopped_by(r, 1, "Python exception RuntimeError: injected failure while the peers are in split", 1)
    reached(r, 0, "coll_split_enter")
    tore_down(r, ranks=(0,))
    tore_down(r, ranks=(1,), reason="Python exception")
    clean(r)


def test_raise_after_open():
    """Rank 1 raises after open: on the way out its finally closes the mesh, and that close stops the job first (the
    exception is in flight), so rank 0 leaves its barrier on the SIGTERM and closes too, instead of rank 1's close
    waiting on rank 0. tt-run returns 1 naming rank 1 and the exception."""
    r = run_case("raise_after_open", {"PROBE_SCENARIO": "raise", "PROBE_PHASE": "after_open"}, DEADLINE_END)
    ended(r)
    stopped_by(r, 1, "Python exception RuntimeError: injected failure at after_open", 1)
    tore_down(r, ranks=(0,))
    tore_down(r, ranks=(1,), reason="Python exception")
    clean(r)


def test_bad_shape():
    """Rank 1 asks for a mesh the descriptor cannot hold: the shape check in open is a TT_FATAL (a C++ exception,
    raised in Python as RuntimeError, its message several lines long) on every rank. The job is stopped with code 1,
    and tt-run reads the wrapper's one-line stop line and names the rank, not prterun's banner."""
    r = run_case("bad_shape_rank1", {"PROBE_MESH_SHAPE_VICTIM": "9x9"}, DEADLINE_END)
    ended(r)
    m = re.search(
        r"\[rank (\d+)\] Python exception RuntimeError: TT_FATAL.*"
        r"ranks disagree on the requested mesh shape.* \(on \S+, pid \d+\); stopping the job \(exit 1\)",
        r.log,
    )
    assert m, f"no one-line stop line for the TT_FATAL\n{r.log[-3000:]}"
    assert r.exit == 1 and f"origin=rank {m.group(1)} on" in r.log and "outcome=failed exit=1" in r.log, str(r)
    assert "the launcher then killed the other ranks" not in r.log and "job forced down" not in r.log, r.log[-3000:]
    clean(r)


def test_bad_descriptor(tmp_path):
    r = run_case(
        "conflicting_descriptor",
        {"PROBE_SCENARIO": "mesh_open"},
        DEADLINE_HANG,
        cmd=legacy_cmd(mapping=conflict_mapping(tmp_path)),
    )
    ended(r)
    nonzero(r)
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


def test_survivor_tears_down_ulfm():
    """Opt-in ULFM so the death is delivered as an error; rank 1 killed with the mesh open. Rank 0's wait sees the
    error, gives the dead rank's wrapper's SIGTERM its few ms to arrive and leaves as a stopped rank: through
    teardown, not as a second failure. (The survivor's own close is the later teardown step.)"""
    r = run_case(
        "survivor_close_ulfm",
        # Under ULFM recovery the launcher does not end the job when ranks exit, so the dead rank's wrapper stands in
        # for the whole stop grace + 10 s before it leaves; a short grace keeps the row inside its deadline.
        {
            "PROBE_SCENARIO": "sigkill",
            "PROBE_PHASE": "after_open",
            "PROBE_ON_ERROR": "continue",
            "TT_RUN_STOP_GRACE_S": "6",
        },
        DEADLINE_HANG,
        cmd=legacy_cmd("--with-ft ulfm"),
    )
    tore_down(r, ranks=(0,))
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
            "TT_METAL_OPERATION_TIMEOUT_SECONDS": "4",
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


def test_shape_mismatch():
    r = run_case("shape_mismatch", {"PROBE_MESH_SHAPE_VICTIM": "4x4"}, DEADLINE_END)
    ended(r)
    nonzero(r)
    clean(r)


def test_sigint_one_rank():
    """SIGINT reaches rank 1 only (a user interrupts one pid): KeyboardInterrupt. P23: it stops the job with 130 (from
    the mesh close it unwinds through), so the peer leaves its collective and closes."""
    r = run_case("sigint_one_rank", {"PROBE_SCENARIO": "sigint", "PROBE_PHASE": "after_open"}, DEADLINE_END)
    ended(r)
    stopped_by(r, 1, "Python exception KeyboardInterrupt", 130)
    tore_down(r, ranks=(0,))
    tore_down(r, ranks=(1,), reason="Python exception")
    clean(r)


def test_exit1_stops_the_job():
    """Rank 1 calls sys.exit(1) after open, the way click or a script's error return does. P26: sys.exit stops the job
    at the call (SIGTERM to every rank), then the SystemExit unwinds through the finally that closes the mesh. Every
    rank closes and leaves together; tt-run returns 1 naming rank 1."""
    r = run_case("exit1_after_open", {"PROBE_SCENARIO": "exit1", "PROBE_PHASE": "after_open"}, DEADLINE_END)
    ended(r)
    stopped_by(r, 1, "Python exception SystemExit: 1", 1)
    tore_down(r, ranks=(0,))
    tore_down(r, ranks=(1,), reason="Python exception")
    clean(r)


def test_segfault_stops_the_job():
    """Rank 1 segfaults with the mesh open: its wrapper stops the job, rank 0 leaves its barrier and closes; 139."""
    r = run_case("segv_after_open", {"PROBE_SCENARIO": "segv", "PROBE_PHASE": "after_open"}, DEADLINE_END)
    ended(r)
    stopped_by(r, 1, "was killed by signal 11 (SIGSEGV)", 139)
    tore_down(r, ranks=(0,))
    clean(r)


def test_sigkill_with_mesh_open_stops_the_job():
    """Rank 1 is SIGKILLed with the mesh open (OOM, kill -9): its wrapper stops the job, rank 0 closes; 137."""
    r = run_case(
        "sigkill_after_open",
        {"PROBE_SCENARIO": "sigkill", "PROBE_PHASE": "after_open"},
        DEADLINE_END,
    )
    ended(r)
    stopped_by(r, 1, "was killed by signal 9 (SIGKILL)", 137)
    tore_down(r, ranks=(0,))
    clean(r)
