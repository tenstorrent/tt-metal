# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Teardown sanity, the launcher: does tt-run address the MPI failure modes?

Every row launches the raw C++ probe (``sanity_mpi_probe.cpp``, which uses MPI the way tt-metal's DistributedContext
does) THROUGH tt-run, then asks whether tt-run ends the job correctly: the right exit code, the ranks on every host
gone, nothing left in ``/dev/shm``. tt-run picks the plain launcher and builds the production mpirun line
(``--mca btl self,tcp``); a group-E configuration variation (a transport, a component, an interface) goes through
``--mpi-args``, which reaches PRRTE and OpenMPI. These rows are the ones to run again when the MPI install changes:
``TT_SANITY_MPI_PREFIX=/opt/openmpi-<version>`` picks the mpirun and the mpicxx and the probe is recompiled.

A row that passes is a gate: tt-run already handles that failure. A row marked ``@known("P..")`` fails until the
tt-run fix in this PR is in the tt-run on PATH (the launcher in the CI image is the one this PR changes).
"""

from __future__ import annotations

import re
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parent)
)  # sanity_common lives next to this file, whatever the import mode
from sanity_common import (
    ROOT,
    DEADLINE_END,
    DEADLINE_HANG,
    clean,
    ended,
    fault_delivered,
    mpirun_path,
    nonzero,
    run_case,
    ttrun_mpi_cmd,
)

NP = {"PROBE_NP": "3"}


# ---------------------------------------------------------------------------------------------------------------
# does tt-run end the job correctly on each failure? (the fault classes, through tt-run)
# ---------------------------------------------------------------------------------------------------------------
def test_ttrun_clean():
    """Baseline: three ranks through tt-run, one barrier, finalize. tt-run exits 0 and leaves nothing."""
    r = run_case("ttrun_clean", {**NP, "PROBE_SCENARIO": "clean"}, DEADLINE_END, cmd=ttrun_mpi_cmd())
    ended(r)
    assert r.exit == 0, str(r)
    clean(r)


def test_ttrun_dead_rank():
    """A rank is killed. Because tt-run launches the plain mpirun, the job ends with a non-zero code (not the ULFM
    false 0, P4) and nothing is left."""
    r = run_case("ttrun_dead_rank", {**NP, "PROBE_SCENARIO": "victim_sigkill"}, DEADLINE_END, cmd=ttrun_mpi_cmd())
    ended(r)
    nonzero(r)
    clean(r)
    assert "died by signal 9 (SIGKILL)" in r.log, r.log[-2000:]  # tt-run says why, not the stale-cache hint


def test_ttrun_hang_timeout():
    """A rank spins forever. With PRRTE's --timeout passed through --mpi-args, tt-run ends the job (exit 241) and
    leaves nothing — the launcher wall clock reaches PRRTE through tt-run (P5)."""
    r = run_case(
        "ttrun_hang_timeout",
        {**NP, "PROBE_SCENARIO": "victim_hang"},
        DEADLINE_END,
        cmd=ttrun_mpi_cmd("--timeout 20 --report-state-on-timeout"),
    )
    ended(r)
    nonzero(r)
    assert r.exit == 241, f"expected PRRTE's timeout code 241, got {r.exit} — the wall clock was not what ended it: {r}"
    assert 20 <= r.wall_s < 40, f"the job did not end at the ~20 s --timeout but at {r.wall_s:.0f} s: {r}"
    clean(r)


def test_ttrun_no_shm_leak():
    """A rank dies under tt-run. Because tt-run builds the mpirun line with --mca btl self,tcp, no shared-memory
    segment is ever mapped, so even a failed job leaves nothing in /dev/shm (P11)."""
    r = run_case("ttrun_no_shm_leak", {**NP, "PROBE_SCENARIO": "victim_sigkill"}, DEADLINE_END, cmd=ttrun_mpi_cmd())
    ended(r)
    assert not r.shm_leaked, f"a job launched by tt-run left shm segments: {r.shm_leaked}"


# ---------------------------------------------------------------------------------------------------------------
# a group-E configuration variation through tt-run: tt-run still ends the job with a known shape
# ---------------------------------------------------------------------------------------------------------------
def test_ttrun_bad_interface():
    """A bogus TCP interface through --mpi-args: every rank fails MPI_Init and tt-run ends the job fast with a
    non-zero code, not a hang (P14). The production line uses the real interface; this is the one-variable departure."""
    r = run_case(
        "ttrun_bad_interface",
        {**NP, "PROBE_SCENARIO": "clean"},
        DEADLINE_END,
        cmd=ttrun_mpi_cmd("--mca btl_tcp_if_include nosuchif"),
    )
    ended(r)
    nonzero(r)
    clean(r)
    # tt-run names the interface and the ones this host has, instead of the generic stale-cache hint
    assert "TCP interface 'nosuchif'" in r.log, r.log[-2000:]
    assert "golden" not in r.log.lower(), r.log[-2000:]


def test_scripts_tcp_only():
    """Static: every bare mpirun in tests/scripts/multihost/*.sh carries --mca btl self,tcp, the line tt-run builds,
    so none of them maps the shared-memory layer (P15). No probe launched."""
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


# ---------------------------------------------------------------------------------------------------------------
# the signal cases: a signal at tt-run is forwarded to the launcher and leaves nothing (this PR's fix)
# ---------------------------------------------------------------------------------------------------------------
def test_ttrun_ctrl_c():
    """A rank hangs, then SIGINT is sent to tt-run. tt-run forwards it to the launcher, which kills every rank;
    nothing is left (this PR forwards the signal to the launcher)."""
    r = run_case(
        "ttrun_ctrl_c",
        {**NP, "PROBE_SCENARIO": "victim_hang"},
        DEADLINE_HANG,
        cmd=ttrun_mpi_cmd(),
        launcher_signal=signal.SIGINT,
        launcher_ready=(0, "ready"),  # the ranks are up and hanging before we interrupt, not still starting
        signal_after=1,
    )
    fault_delivered(r)
    ended(r)
    clean(r)


def test_ttrun_sigterm():
    """A rank hangs, then SIGTERM is sent to tt-run. tt-run forwards it and waits; nothing is left."""
    r = run_case(
        "ttrun_sigterm",
        {**NP, "PROBE_SCENARIO": "victim_hang"},
        DEADLINE_HANG,
        cmd=ttrun_mpi_cmd(),
        launcher_signal=signal.SIGTERM,
        launcher_ready=(0, "ready"),
        signal_after=1,
    )
    fault_delivered(r)
    ended(r)
    clean(r)


# ---------------------------------------------------------------------------------------------------------------
# group E: one-variable departures from the production mpirun line, through tt-run; each needs a known teardown shape
# production line: pml ob1, --mca btl self,tcp, --mca btl_tcp_if_include <nic>. Departures go through --mpi-args.
# ---------------------------------------------------------------------------------------------------------------
def test_ttrun_abort_ends():
    """A rank aborts (SIGABRT, the TT_FATAL shape). With the plain launcher tt-run ends the job quickly once the
    rank is reaped, and the OTHER ranks are torn down with it: the job exits non-zero and nothing is left alive on
    any host. It must not stall for the length of a core dump (P8)."""
    r = run_case("ttrun_abort", {**NP, "PROBE_SCENARIO": "victim_abort"}, DEADLINE_END, cmd=ttrun_mpi_cmd())
    ended(r)
    nonzero(r)
    clean(r)  # every surviving rank is gone, nothing left on any host
    assert r.wall_s < 30, f"tt-run stalled on the aborting rank: {r}"


def test_ttrun_shm_layer_still_ends():
    """Group-E departure: the shared-memory layer put back (btl self,sm,tcp). A clean job through tt-run must still
    end cleanly; the layer changes the transport, not tt-run's teardown (P15). tt-run uses self,tcp by default, so
    this is the guard that the departure does not hang."""
    r = run_case(
        "ttrun_shm_layer",
        {**NP, "PROBE_SCENARIO": "clean"},
        DEADLINE_END,
        cmd=ttrun_mpi_cmd("--mca btl self,sm,tcp"),
    )
    ended(r)
    assert r.exit == 0, str(r)


def test_ttrun_lo_interface():
    """Group-E departure: the loopback interface. On a single host a clean job through tt-run still completes; the
    row is the guard that an interface change has a known shape (compare test_ttrun_bad_interface, which must fail
    fast)."""
    r = run_case(
        "ttrun_lo_interface",
        {**NP, "PROBE_SCENARIO": "clean"},
        DEADLINE_END,
        cmd=ttrun_mpi_cmd("--mca btl_tcp_if_include lo"),
    )
    ended(r)
    assert r.exit == 0, str(r)


def test_ttrun_pml_ucx_known_shape():
    """Group-E departure: the UCX point-to-point layer instead of ob1. Through tt-run a clean job must end with a
    known shape — either it runs (exit 0) or UCX refuses and tt-run ends the job with a non-zero code, never a
    hang."""
    r = run_case(
        "ttrun_pml_ucx",
        {**NP, "PROBE_SCENARIO": "clean"},
        DEADLINE_END,
        cmd=ttrun_mpi_cmd("--mca pml ucx"),
    )
    ended(r)
    clean(r)


# ---------------------------------------------------------------------------------------------------------------
# MPI install & configuration requirements: does the OpenMPI / PRRTE build on this machine meet what tt-run needs?
# Gates only — each queries or exercises the install, no fault injected. A host with the wrong MPI on PATH, or a
# build missing a component tt-run's launch line relies on, is caught here before any teardown row runs. Honours
# TT_SANITY_MPI_PREFIX (sanity_common.mpirun_path), so a second install can be checked with the same rows.
# ---------------------------------------------------------------------------------------------------------------
def _mpi_bin(name: str) -> str:
    """The sibling tool of the mpirun tt-run would use (ompi_info, mpirun-ulfm), honouring TT_SANITY_MPI_PREFIX."""
    cand = Path(mpirun_path()).resolve().parent / name
    return str(cand) if cand.exists() else (shutil.which(name) or name)


def _run(cmd: list, timeout: int = 30) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def test_mpi_build_is_open_mpi_5_plus():
    """tt-run's launch line (PRRTE --timeout, --map-by, the self,tcp btl line) is Open MPI 5.x / PRRTE. Confirm the
    mpirun tt-run would use is that build, so an older or foreign MPI on PATH is caught early, not mid-job."""
    out = (_run([mpirun_path(), "--version"]).stdout or "") + (_run([mpirun_path(), "--version"]).stderr or "")
    m = re.search(r"Open MPI[)\s]+v?(\d+)\.(\d+)", out)
    assert m, f"not an Open MPI build: {out.strip()[:120]!r}"
    assert int(m.group(1)) >= 5, f"Open MPI >= 5.0 required (PRRTE launcher); got {m.group(1)}.{m.group(2)}"


def test_launcher_supports_timeout():
    """tt-run's opt-in wall clock (P5) is PRRTE's own --timeout, reachable only through --mpi-args. Confirm this
    launcher accepts it — a trivial job with --timeout must run, not be rejected as an unknown option."""
    r = _run([mpirun_path(), "--timeout", "10", "--allow-run-as-root", "-np", "1", "true"])
    assert r.returncode == 0, f"the launcher does not accept --timeout (P5 backstop unreachable): {r.stderr[:200]}"


def test_required_mca_components_available():
    """The production line tt-run builds is pml ob1 over --mca btl self,tcp (P15). Confirm the install actually
    carries those components, so the line is honoured rather than silently falling back to another transport."""
    info = _run([_mpi_bin("ompi_info")]).stdout
    assert "MCA btl: self" in info, "btl self missing from this MPI build"
    assert "MCA btl: tcp" in info, "btl tcp missing from this MPI build"
    assert "MCA pml: ob1" in info, "pml ob1 missing from this MPI build"


def test_production_transport_line_runs_clean():
    """The exact transport line tt-run adds (pml ob1, --mca btl self,tcp) must run on this install: a trivial
    two-rank job under that line completes, so the line tt-run builds is not just present but working."""
    r = _run(
        [
            mpirun_path(),
            "--allow-run-as-root",
            "--oversubscribe",
            "-np",
            "2",
            "--mca",
            "pml",
            "ob1",
            "--mca",
            "btl",
            "self,tcp",
            "true",
        ]
    )
    assert r.returncode == 0, f"the self,tcp / ob1 line failed on this install: {r.stderr[:200]}"


def test_ulfm_launcher_available():
    """tt-metal's DistributedContext recovery can opt into ULFM via --mpi-args; the install must ship it. Confirm
    mpirun-ulfm is present and a trivial --with-ft ulfm job runs, so the capability exists when it is asked for."""
    ulfm = _mpi_bin("mpirun-ulfm")
    if not Path(ulfm).exists() and shutil.which("mpirun-ulfm") is None:
        pytest.skip("mpirun-ulfm not installed in this prefix")
    r = _run([ulfm, "--with-ft", "ulfm", "--allow-run-as-root", "-np", "1", "true"])
    assert r.returncode == 0, f"--with-ft ulfm failed on this install: {r.stderr[:200]}"
