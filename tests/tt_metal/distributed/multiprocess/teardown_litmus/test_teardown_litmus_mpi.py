# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Teardown litmus, MPI only: the same questions asked of OpenMPI and PRRTE alone, with no tt-metal in the ranks.

``litmus_mpi_probe.c`` is a C rank program that uses MPI the way tt-metal's DistributedContext does
(MPI_Init_thread MULTIPLE, atexit MPI_Finalize, a dup of WORLD with MPI_ERRORS_RETURN, blocking collectives) and
misbehaves on one rank. These rows say which of the failures belong to the launcher and the transport, so they are
the rows to run again when the MPI install changes: ``TT_LITMUS_MPI_PREFIX=/opt/openmpi-<version>`` picks the
mpirun and the mpicc, and the probe is recompiled against it. Rows that fail today are ``xfail(strict=True)`` with
the error id of the teardown plan (tt-blaze tools/teardown_probe).
"""

from __future__ import annotations

import signal
import sys
from pathlib import Path

import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parent)
)  # litmus_common lives next to this file, whatever the import mode
from litmus_common import DEADLINE_END, DEADLINE_HANG, clean, ended, mpi_cmd, nonzero, rank_reached, run_case

NP = {"PROBE_NP": "3"}


# ---- gates -----------------------------------------------------------------------------------------------------
def test_mpi_clean():
    """Three ranks, one barrier, finalize: exit 0, nothing left."""
    r = run_case("mpi_clean", {**NP, "PROBE_SCENARIO": "clean"}, DEADLINE_END, cmd=mpi_cmd())
    ended(r)
    assert r.exit == 0, str(r)
    clean(r)


def test_mpi_dead_rank_ends_ulfm():
    """A rank SIGKILLs itself under --with-ft ulfm: the survivors' barrier returns an error, they finalize, the job ends."""
    r = run_case("mpi_dead_rank_ulfm", {**NP, "PROBE_SCENARIO": "victim_sigkill"}, DEADLINE_END, cmd=mpi_cmd())
    ended(r)
    clean(r)


def test_mpi_dead_rank_exit_nonzero_plain():
    """The same kill under plain mpirun (no ULFM): the job ends with the victim's code, the truthful exit code."""
    r = run_case(
        "mpi_dead_rank_plain", {**NP, "PROBE_SCENARIO": "victim_sigkill"}, DEADLINE_END, cmd=mpi_cmd(ulfm=False)
    )
    ended(r)
    nonzero(r)
    clean(r)


def test_mpi_survivor_abort_exit_nonzero():
    """The fix shape: the survivors call MPI_Abort after the error, so the launcher reports it under ULFM too."""
    r = run_case(
        "mpi_survivor_abort",
        {**NP, "PROBE_SCENARIO": "victim_sigkill", "PROBE_SURVIVOR_ACTION": "abort"},
        DEADLINE_END,
        cmd=mpi_cmd(),
    )
    ended(r)
    nonzero(r)
    clean(r)


def test_mpi_exit_nofinalize_ends():
    """A rank that _exits without finalize, on the production transport line (btl self,tcp): the job ends non-zero."""
    r = run_case(
        "mpi_exit_nofinalize", {**NP, "PROBE_SCENARIO": "victim_exit1_nofinalize"}, DEADLINE_END, cmd=mpi_cmd()
    )
    ended(r)
    nonzero(r)
    clean(r)


def test_mpi_hang_timeout_ends():
    """A rank that spins forever under PRRTE --timeout 20: exit 241, nothing left."""
    r = run_case(
        "mpi_hang+prte_timeout20",
        {**NP, "PROBE_SCENARIO": "victim_hang"},
        DEADLINE_END,
        cmd=mpi_cmd(extra="--timeout 20"),
    )
    ended(r)
    assert r.exit == 241, str(r)
    clean(r)


def test_mpi_ctrl_c_at_mpirun_leaves_nothing():
    """SIGINT at mpirun itself while every rank hangs: PRRTE forwards it and the whole tree is gone."""
    r = run_case(
        "mpi_ctrl_c_at_mpirun",
        {**NP, "PROBE_SCENARIO": "all_hang"},
        DEADLINE_END,
        cmd=mpi_cmd(),
        launcher_signal=signal.SIGINT,
        signal_after=3.0,
    )
    ended(r)
    clean(r)


# ---- report rows: what the launcher and the transport do today ----------------------------------------------
@pytest.mark.xfail(
    strict=True, reason="P4: under --with-ft ulfm a dead rank is not a job failure; the launcher returns 0"
)
def test_mpi_dead_rank_exit_nonzero_ulfm():
    r = run_case("mpi_dead_rank_ulfm_code", {**NP, "PROBE_SCENARIO": "victim_sigkill"}, DEADLINE_END, cmd=mpi_cmd())
    ended(r)
    nonzero(r)


@pytest.mark.xfail(
    strict=True,
    reason="P1: a rank that exits after an error sits in its atexit finalize and the others wait in their barrier forever",
)
def test_mpi_exit_after_error_ends():
    r = run_case("mpi_exit_after_error", {**NP, "PROBE_SCENARIO": "victim_exit1"}, DEADLINE_HANG, cmd=mpi_cmd())
    ended(r)
    clean(r)


@pytest.mark.xfail(strict=True, reason="P5: a genuine hang never ends without a wall clock")
def test_mpi_hang_never_ends():
    r = run_case("mpi_hang", {**NP, "PROBE_SCENARIO": "victim_hang"}, DEADLINE_HANG, cmd=mpi_cmd())
    ended(r)


@pytest.mark.xfail(
    strict=True, reason="P14: with a wrong TCP interface every rank dies at MPI_Init and prterun never ends the job"
)
def test_mpi_wrong_interface_ends():
    r = run_case(
        "mpi_wrong_interface",
        {**NP, "PROBE_SCENARIO": "clean"},
        DEADLINE_HANG,
        cmd=mpi_cmd(extra="--mca btl_tcp_if_include nosuchif"),
    )
    ended(r)
    clean(r)


@pytest.mark.xfail(
    strict=True,
    reason="P2: with the shared-memory layer on, a rank that _exits without finalize parks the survivors in MPI_Finalize",
)
def test_mpi_exit_nofinalize_sm_layer_ends():
    r = run_case(
        "mpi_exit_nofinalize+sm",
        {**NP, "PROBE_SCENARIO": "victim_exit1_nofinalize"},
        DEADLINE_HANG,
        cmd=mpi_cmd(btl=""),
    )
    ended(r)
    clean(r)
