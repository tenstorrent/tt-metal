# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Teardown litmus, MPI only: the same questions asked of OpenMPI and PRRTE alone, with no tt-metal in the ranks.

``litmus_mpi_probe.c`` is a C rank program that uses MPI the way tt-metal's DistributedContext does
(MPI_Init_thread MULTIPLE, atexit MPI_Finalize, a dup of WORLD with MPI_ERRORS_RETURN, blocking collectives) and
misbehaves on one rank. These rows say which of the failures belong to the launcher and the transport, so they are
the rows to run again when the MPI install changes: ``TT_LITMUS_MPI_PREFIX=/opt/openmpi-<version>`` picks the
mpirun and the mpicc, and the probe is recompiled against it. Rows that fail today are marked ``@known`` with
the error id of the teardown plan (tt-blaze tools/teardown_probe) and fail until the fix lands.
"""

from __future__ import annotations

import re
import signal
import sys
from pathlib import Path

import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parent)
)  # litmus_common lives next to this file, whatever the import mode
from litmus_common import (
    ROOT,
    known,
    DEADLINE_END,
    DEADLINE_HANG,
    clean,
    ended,
    mpi_cmd,
    mpi_probe_path,
    nonzero,
    rank_reached,
    run_case,
)

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


# ---- rows that fail today: what the launcher and the transport do ----------------------------------------------
@known("P4: under --with-ft ulfm a dead rank is not a job failure; the launcher returns 0")
def test_mpi_dead_rank_exit_nonzero_ulfm():
    r = run_case("mpi_dead_rank_ulfm_code", {**NP, "PROBE_SCENARIO": "victim_sigkill"}, DEADLINE_END, cmd=mpi_cmd())
    ended(r)
    nonzero(r)


@known("P1: a rank that exits after an error sits in its atexit finalize and the others wait in their barrier forever")
def test_mpi_exit_after_error_ends():
    r = run_case("mpi_exit_after_error", {**NP, "PROBE_SCENARIO": "victim_exit1"}, DEADLINE_HANG, cmd=mpi_cmd())
    ended(r)
    clean(r)


@known("P5: a genuine hang never ends without a wall clock")
def test_mpi_hang_never_ends():
    r = run_case("mpi_hang", {**NP, "PROBE_SCENARIO": "victim_hang"}, DEADLINE_HANG, cmd=mpi_cmd())
    ended(r)


def test_mpi_wrong_interface_ends():
    """Default launcher (no ULFM) with a bogus TCP interface: every rank fails MPI_Init and the job ends fast with a
    non-zero code. Under --with-ft ulfm this instead hangs (P14), which is why tt-run launches the plain mpirun."""
    r = run_case(
        "mpi_wrong_interface",
        {**NP, "PROBE_SCENARIO": "clean"},
        DEADLINE_END,
        cmd=mpi_cmd(ulfm=False, extra="--mca btl_tcp_if_include nosuchif"),
    )
    ended(r)
    nonzero(r)
    clean(r)


@known("P2: with the shared-memory layer on, a rank that _exits without finalize parks the survivors in MPI_Finalize")
def test_mpi_exit_nofinalize_sm_layer_ends():
    r = run_case(
        "mpi_exit_nofinalize+sm",
        {**NP, "PROBE_SCENARIO": "victim_exit1_nofinalize"},
        DEADLINE_HANG,
        cmd=mpi_cmd(btl=""),
    )
    ended(r)
    clean(r)


# ---- the OpenMPI housekeeping rows and the launch-line rule -----------------------------------------------------
@known("P11: a singleton (no launcher) never finalizes its shared-memory segment; tt-metal #63870")
def test_mpi_singleton_leaves_no_shm():
    """The C probe started on its own, no mpirun: one rank, exit 0, and /dev/shm must hold nothing new."""
    r = run_case("mpi_singleton", {"PROBE_SCENARIO": "clean", "PROBE_NP": "1"}, DEADLINE_END, cmd=[mpi_probe_path()])
    ended(r)
    assert r.exit == 0, str(r)
    assert not r.shm_leaked, f"R3: the singleton left {r.shm_leaked}"


@known("P11: a job killed with the shared-memory layer on leaves its 16 MB sm_segment files in /dev/shm")
def test_mpi_killed_job_leaves_no_shm():
    """Default transports (shared memory on), a rank spins, SIGKILL at mpirun: whatever else is left (P7), the
    segments must not be."""
    r = run_case(
        "mpi_sigkill_sm",
        {**NP, "PROBE_SCENARIO": "victim_hang"},
        DEADLINE_END,
        cmd=mpi_cmd(btl=""),
        launcher_signal=signal.SIGKILL,
        signal_after=3,
    )
    assert not r.shm_leaked, f"R3: /dev/shm segments leaked: {r.shm_leaked}"


@known(
    "P15: every bare mpirun (one not started by tt-run) must carry --mca btl self,tcp, the line tt-run uses; run_dual_t3k_tests.sh launches with the default transports"
)
def test_multihost_scripts_use_tcp_only_transport():
    """Static: each mpirun launch in tests/scripts/multihost/*.sh carries the tcp-only transport, on the line or
    through the argument variable the script builds with it."""
    bad = []
    for sh in sorted((ROOT / "tests/scripts/multihost").glob("*.sh")):
        text = sh.read_text()
        has_rule = "--mca btl self,tcp" in text
        for n, line in enumerate(text.splitlines(), 1):
            code = line.split("#", 1)[0]
            if not re.search(r"(^|\s)mpirun(-ulfm)?\s", code):
                continue
            if "--mca btl self,tcp" in code or (has_rule and "mpirun_args" in code):
                continue
            bad.append(f"{sh.name}:{n}: {code.strip()[:100]}")
    assert not bad, "mpirun launches without --mca btl self,tcp:\n" + "\n".join(bad)
