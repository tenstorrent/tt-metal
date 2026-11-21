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

import os
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
    mpi_cmd,
    mpi_install_id,
    mpi_installs,
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
    """A rank is killed. Its wrapper (tt-run-rank) stops the job in its place: every other rank gets SIGTERM, the job
    ends with the dead rank's 137 (not the ULFM false 0, P4), tt-run names it, and nothing is left."""
    r = run_case("ttrun_dead_rank", {**NP, "PROBE_SCENARIO": "victim_sigkill"}, DEADLINE_END, cmd=ttrun_mpi_cmd())
    ended(r)
    nonzero(r)
    clean(r)
    # tt-run names the rank, where it ran, how it ended and what happened to the others, not the stale-cache hint
    assert re.search(r"rank 1 on \S+ \(pid \d+\): was killed by signal 9 \(SIGKILL\)", r.log), r.log[-2000:]
    assert "the other ranks were sent SIGTERM and exited" in r.log, r.log[-2000:]
    assert r.exit == 137 and "outcome=failed exit=137" in r.log, r.log[-2000:]


def test_ttrun_hang_timeout():
    """A rank spins forever. With --timeout 20 in --mpi-args, tt-run sends SIGTERM to every rank at 20 s (graceful; tt-run
    runs the clock itself and takes PRRTE's --timeout off the launch line), the job ends with 241 and leaves nothing (P5).
    """
    r = run_case(
        "ttrun_hang_timeout",
        {**NP, "PROBE_SCENARIO": "victim_hang"},
        DEADLINE_END,
        cmd=ttrun_mpi_cmd("--timeout 20 --report-state-on-timeout"),
    )
    ended(r)
    nonzero(r)
    assert r.exit == 241, f"expected PRRTE's timeout code 241, got {r.exit} — the wall clock was not what ended it: {r}"
    # tt-run's own job time (launch to the end of the job), not the harness wall clock that also counts start-up
    m = re.search(r"job end: .* elapsed=([0-9.]+)s", r.log)
    assert m, r.log[-2000:]
    assert 20 <= float(m.group(1)) < 40, f"the job did not end at the ~20 s --timeout but at {m.group(1)} s: {r}"
    assert "the job reached its --timeout of 20 s: sent SIGTERM to all ranks" in r.log, r.log[-2000:]
    clean(r)


@pytest.mark.xfail(
    reason="P1 for a raw MPI program: a rank that calls exit(1) runs its own atexit MPI_Finalize, which is collective "
    "and waits on the peers in their barrier, so the job rides the --timeout wall clock (241). Neither tt-run nor its "
    "rank wrapper can end a rank that is still alive; a tt-metal rank stops the job from its exit instead (#64546).",
    strict=True,
)
def test_ttrun_error_exit_fails_fast_not_hang():
    """A raw MPI rank (no tt-metal context) exits on an error with MPI_Finalize in its atexit (victim_exit1). A PRRTE
    --timeout bounds the run so the row can never hang the suite. The row stays a strict xfail: it documents the one
    error exit the launcher layer cannot see, and turns red if that ever changes."""
    r = run_case(
        "ttrun_error_exit_hang",
        {**NP, "PROBE_SCENARIO": "victim_exit1"},
        DEADLINE_END,
        cmd=ttrun_mpi_cmd("--timeout 20 --report-state-on-timeout"),
    )
    ended(r)
    nonzero(r)
    clean(r)
    # The contract: a fast end, NOT riding the --timeout wall clock to 241.
    assert r.exit != 241, f"P1: the error-exit deadlocked and the job rode the --timeout wall clock (241): {r}"


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
            # Skip references that are not launches: resolver lookups that only locate the launcher binary
            # (command -v / which / type mpirun) and mpirun named inside an echoed message. These carry no mca
            # line by nature; the actual launches run through the resolved "$launcher", which is checked above.
            if re.search(r"\b(command\s+-v|which|type)\b", code) or "echo" in code:
                continue
            if "--mca btl self,tcp" in code or (has_rule and "mpirun_args" in code):
                continue
            bad.append(f"{sh.name}:{n}: {code.strip()[:100]}")
    assert not bad, "mpirun launches without --mca btl self,tcp:\n" + "\n".join(bad)


# ---------------------------------------------------------------------------------------------------------------
# the signal cases: a signal at tt-run is forwarded to the launcher and leaves nothing (this PR's fix)
# ---------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "sig,case",
    [(signal.SIGINT, "delivered"), (signal.SIGTERM, "delivered"), (signal.SIGINT, "no_helper")],
    ids=["ctrl_c", "sigterm", "ctrl_c_no_helper"],
)
def test_ttrun_stop(sig, case):
    """The ranks hang, then tt-run gets a stop signal. tt-run sends SIGTERM to every rank through PRRTE job
    control: in all_hang every rank records the signal it caught, so each must report signal 15; the job reports
    128 + the signal and nothing is left. Without the helper tt-run forces at once through the launcher and says
    the devices may be unhealthy. (What a rank does with the SIGTERM is the rank's business, not tt-run's.)"""
    _ttrun_stop(sig, case)


@pytest.mark.parametrize("prefix", mpi_installs(), ids=lambda p: p.strip("/").replace("/", "_") or "path")
def test_ttrun_stop_on_each_mpi_install(prefix):
    """The stop relies on PRRTE internals (prterun's tool rendezvous file, the job-control signal path), and the
    MPI build differs by machine: Open MPI 5.0.7 built from source in the CI images, the openmpi-ulfm .deb on bare
    metal, the distribution's own on every host. Run the delivery check against every install on this host and say
    which build it was; an install tt-run does not support (Open MPI < 5, no PRRTE) is skipped with that reason."""
    ident, major = mpi_install_id(prefix)
    print(f"\n[sanity] MPI install {prefix or '(PATH)'}: {ident}")
    if major < 5:
        pytest.skip(f"{prefix or '(PATH)'} is {ident}: tt-run needs Open MPI 5+ (PRRTE)")
    _ttrun_stop(signal.SIGINT, "delivered", mpi_prefix=prefix)


def _ttrun_stop(sig, case, mpi_prefix=None):
    env = {**NP, "PROBE_SCENARIO": "all_hang" if case == "delivered" else "victim_hang"}
    if case == "no_helper":
        env["TT_RUN_SIGNAL_HELPER"] = "/nonexistent/tt-run-rank"
    r = run_case(
        f"ttrun_stop_{signal.Signals(sig).name}_{case}",
        env,
        DEADLINE_HANG,
        cmd=ttrun_mpi_cmd(mpi_prefix=mpi_prefix),
        launcher_signal=sig,
        launcher_ready=(0, "ready"),  # the ranks are up and hanging before we stop them, not still starting
        signal_after=1,
        mpi_prefix=mpi_prefix,
    )
    fault_delivered(r)
    ended(r)
    clean(r)
    assert r.exit == 128 + sig, r
    if case == "delivered":
        got = {int(m.group(1)) for m in re.finditer(r"\[probe r=(\d+) [^\]]*\] caught signal 15\b", r.log)}
        assert got == {0, 1, 2}, f"every rank must receive SIGTERM (signal 15); got it: {sorted(got)}\n{r.log[-3000:]}"
        assert "sent SIGTERM to all ranks" in r.log and "outcome=stopped" in r.log, r.log[-3000:]
        return
    assert "could not reach prterun" in r.log and "job forced down" in r.log, r.log[-3000:]
    assert "tt-smi -r" in r.log and "outcome=forced" in r.log, r.log[-3000:]


def _ranks_got(r, signum: int) -> set:
    return {int(m.group(1)) for m in re.finditer(rf"\[probe r=(\d+) [^\]]*\] got signal {signum};", r.log)}


@pytest.mark.parametrize(
    "sig",
    [signal.SIGUSR1, signal.SIGUSR2, signal.SIGALRM, signal.SIGQUIT, signal.SIGWINCH, signal.SIGCONT],
    ids=lambda s: signal.Signals(s).name,
)
def test_ttrun_passes_signals_on(sig):
    """A signal at tt-run that is not a stop reaches every rank unchanged, through job control and each rank's
    wrapper, and the job goes on: every rank reports it and keeps running; a stop (SIGTERM) then ends the job."""
    r = run_case(
        f"ttrun_pass_{signal.Signals(sig).name}",
        {**NP, "PROBE_SCENARIO": "all_hang"},
        DEADLINE_HANG,
        cmd=ttrun_mpi_cmd(),
        launcher_signal=sig,
        launcher_ready=(0, "ready"),
        signal_after=1,
        then_signals=[(signal.SIGTERM, 0.5)],
    )
    fault_delivered(r)
    ended(r)
    clean(r)
    assert _ranks_got(r, sig) == {0, 1, 2}, f"every rank must get {signal.Signals(sig).name}\n{r.log[-3000:]}"
    assert f"{signal.Signals(sig).name} received: passed it on to all ranks" in r.log, r.log[-3000:]
    after = re.findall(r"ranks: ([^;]*)", r.signal_sent)[0]
    assert "gone" not in after and "=T" not in after, f"the {signal.Signals(sig).name} ended or paused a rank: {r}"
    got_term = {int(m.group(1)) for m in re.finditer(r"\[probe r=(\d+) [^\]]*\] caught signal 15\b", r.log)}
    assert got_term == {0, 1, 2} and r.exit == 128 + signal.SIGTERM, f"the stop after it did not end the job: {r}"


def test_ttrun_ctrl_z_pauses_and_resumes_the_ranks():
    """Ctrl-Z (SIGTSTP) at tt-run pauses every rank and tt-run itself; SIGCONT (the shell's fg / bg) resumes tt-run,
    which passes it on and resumes every rank; the job then stops on SIGTERM as usual."""
    r = run_case(
        "ttrun_ctrl_z",
        {**NP, "PROBE_SCENARIO": "all_hang"},
        DEADLINE_HANG,
        cmd=ttrun_mpi_cmd(),
        launcher_signal=signal.SIGTSTP,
        launcher_ready=(0, "ready"),
        signal_after=1,
        then_signals=[(signal.SIGCONT, 1.0), (signal.SIGTERM, 0.5)],
    )
    fault_delivered(r)
    ended(r)
    clean(r)
    paused, resumed = re.findall(r"ranks: ([^;]*)", r.signal_sent)[:2]
    assert paused == "rank_0=T rank_1=T rank_2=T", f"SIGTSTP did not pause every rank: {paused!r}\n{r}"
    assert "=T" not in resumed and "gone" not in resumed, f"SIGCONT did not resume every rank: {resumed!r}\n{r}"
    assert _ranks_got(r, signal.SIGCONT) == {0, 1, 2}, r.log[-3000:]
    assert "the job is paused" in r.log and r.exit == 128 + signal.SIGTERM, r.log[-3000:]


def test_ttrun_sigterm_to_every_process_stops_gracefully():
    """SIGTERM to tt-run and prterun at the same instant: what scancel of an srun step or a Slurm time limit does. tt-run
    starts prterun with SIGTERM blocked, so prterun's own kill path (SIGKILL 250 ms after its SIGTERM) does not run: the
    ranks get tt-run's graceful stop and each takes its 2 s to exit (finalize_done from every rank, none killed)."""
    r = run_case(
        "ttrun_sigterm_everyone",
        {**NP, "PROBE_SCENARIO": "all_hang", "PROBE_STOP_DELAY_MS": "2000"},
        DEADLINE_HANG,
        cmd=ttrun_mpi_cmd(),
        launcher_signal=signal.SIGTERM,
        launcher_ready=(0, "ready"),
        signal_after=1,
        launcher_too=True,
    )
    fault_delivered(r)
    ended(r)
    clean(r)
    done = {int(m.group(1)) for m in re.finditer(r"\[probe r=(\d+) [^\]]*\] finalize_done", r.log)}
    assert done == {
        0,
        1,
        2,
    }, f"ranks killed during their exit (prterun's kill path): only {sorted(done)} finished\n{r.log[-3000:]}"
    assert r.exit == 128 + signal.SIGTERM and "outcome=stopped" in r.log, r.log[-3000:]


def test_ttrun_hangup_keeps_the_job():
    """SIGHUP at tt-run (the terminal closed, an ssh session dropped): the job keeps running with tt-run still
    supervising it; nothing reaches the ranks, and a later stop (SIGTERM) ends the job gracefully as usual."""
    r = run_case(
        "ttrun_sighup",
        {**NP, "PROBE_SCENARIO": "all_hang"},
        DEADLINE_HANG,
        cmd=ttrun_mpi_cmd(),
        launcher_signal=signal.SIGHUP,
        launcher_ready=(0, "ready"),
        signal_after=1,
        then_signals=[(signal.SIGTERM, 1.0)],
    )
    fault_delivered(r)
    ended(r)
    clean(r)
    after = re.findall(r"ranks: ([^;]*)", r.signal_sent)[0]
    assert "gone" not in after, f"the hangup ended a rank: {r}"
    assert "SIGHUP received" in r.log and "the job keeps running" in r.log, r.log[-3000:]
    got = {int(m.group(1)) for m in re.finditer(r"\[probe r=(\d+) [^\]]*\] caught signal 15\b", r.log)}
    assert got == {0, 1, 2} and r.exit == 128 + signal.SIGTERM, f"the stop after the hangup did not end the job: {r}"


def test_ttrun_killed_stops_the_ranks():
    """tt-run itself is SIGKILLed (or aborts, or crashes) while the job runs: it can do nothing, but its guard
    (tt-run-rank --guard) sees it gone and stops the job as tt-run would, SIGTERM to every rank through job control.
    Every rank receives it and the job ends; nothing is left, the guard included."""
    r = run_case(
        "ttrun_sigkill",
        {**NP, "PROBE_SCENARIO": "all_hang"},
        DEADLINE_HANG,
        cmd=ttrun_mpi_cmd(),
        launcher_signal=signal.SIGKILL,
        launcher_ready=(0, "ready"),
        signal_after=1,
    )
    fault_delivered(r)
    ended(r)
    clean(r)
    got = {int(m.group(1)) for m in re.finditer(r"\[probe r=(\d+) [^\]]*\] caught signal 15\b", r.log)}
    assert got == {0, 1, 2}, f"every rank must receive SIGTERM (signal 15); got it: {sorted(got)}\n{r.log[-3000:]}"
    assert "[tt-run guard] sent SIGTERM to all ranks" in r.log, r.log[-3000:]
    assert "[tt-run guard] the job ended after the stop" in r.log, r.log[-3000:]
    assert "job forced down" not in r.log, r.log[-3000:]


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
    assert re.search(r"rank 1 on \S+ \(pid \d+\): was killed by signal 6 \(SIGABRT\)", r.log), r.log[-2000:]
    assert r.exit == 134, str(r)


def test_ttrun_dead_peer_survivors_abort():
    """A rank is SIGKILLed, the survivors get MPI_ERR_PROC_FAILED from the barrier and call MPI_Abort — the way
    tt-metal's DistributedContext reacts to a dead peer (see the comment at the MPI_Abort in sanity_mpi_probe.cpp).
    tt-run should end the whole job non-zero with nothing left.

    On the PRRTE this image ships that is not guaranteed — the prrte#2815 race can leave prterun up after the
    ranks are gone, and this harness's deadline then bounds the row. So it tolerates ONLY that one shape — the job
    hanging past the deadline after the survivors reached MPI_Abort — and fails on anything else, so a genuine
    teardown regression is never masked by a blanket xfail."""
    r = run_case(
        "ttrun_dead_peer_abort",
        {**NP, "PROBE_SCENARIO": "victim_sigkill", "PROBE_SURVIVOR_ACTION": "abort"},
        DEADLINE_END,
        cmd=ttrun_mpi_cmd(),
    )
    try:
        ended(r)
        nonzero(r)
        clean(r)
    except AssertionError:
        # The fingerprint of openpmix/prrte#2815: the survivors reached MPI_Abort and the job then hung past the
        # deadline (prterun stayed up with the ranks gone). Tolerate only that; re-raise anything else — a wrong
        # exit code, a leftover with no hang — as a real failure.
        if r.timed_out and "survivor MPI_Abort" in r.log:
            pytest.xfail(
                "openpmix/prrte#2815: prterun left up after a killed peer + the survivors' MPI_Abort "
                "(PRRTE 3.0.8 / Open MPI 5.0.7rc2; fixed in open-mpi/ompi#14590 / #14713, not in 5.0.x). "
                "https://github.com/openpmix/prrte/issues/2815"
            )
        raise


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


def _run(cmd: list, timeout: int = 30, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)


def test_mpi_build_is_open_mpi_5_plus():
    """tt-run's launch line (--map-by, the self,tcp btl line, PRRTE job control) is Open MPI 5.x / PRRTE. Confirm the
    mpirun tt-run would use is that build, so an older or foreign MPI on PATH is caught early, not mid-job."""
    out = (_run([mpirun_path(), "--version"]).stdout or "") + (_run([mpirun_path(), "--version"]).stderr or "")
    m = re.search(r"Open MPI[)\s]+v?(\d+)\.(\d+)", out)
    assert m, f"not an Open MPI build: {out.strip()[:120]!r}"
    assert int(m.group(1)) >= 5, f"Open MPI >= 5.0 required (PRRTE launcher); got {m.group(1)}.{m.group(2)}"


def test_required_mca_components_available():
    """The production line tt-run builds is pml ob1 over --mca btl self,tcp (P15). Confirm the install actually
    carries those components, so the line is honoured rather than silently falling back to another transport."""
    info = _run([_mpi_bin("ompi_info")]).stdout
    assert "MCA btl: self" in info, "btl self missing from this MPI build"
    assert "MCA btl: tcp" in info, "btl tcp missing from this MPI build"
    assert "MCA pml: ob1" in info, "pml ob1 missing from this MPI build"


def test_production_transport_line_runs_clean():
    """The exact transport line tt-run adds (pml ob1, --mca btl self,tcp) must not just be present but actually
    carry traffic. Run the probe's clean two-rank scenario (MPI_Init, a barrier, MPI_Finalize) over that line, so
    ob1/TCP is initialised and the ranks exchange a collective: the line tt-run builds is proven working, not
    merely accepted. (A bare ``true`` never calls MPI_Init and would pass even with an unusable transport.)"""
    cmd = mpi_cmd(ulfm=False, extra="--mca pml ob1", btl="self,tcp", np=2)
    env = {**os.environ, "PROBE_SCENARIO": "clean", "PROBE_NP": "2"}
    r = _run(cmd, env=env)
    assert r.returncode == 0, f"the self,tcp / ob1 line failed on this install: {r.stderr[:300]}"


def test_ulfm_launcher_available():
    """tt-metal's DistributedContext recovery can opt into ULFM via --mpi-args; the install must ship it. Confirm
    mpirun-ulfm is present and a trivial --with-ft ulfm job runs, so the capability exists when it is asked for."""
    ulfm = _mpi_bin("mpirun-ulfm")
    if not Path(ulfm).exists() and shutil.which("mpirun-ulfm") is None:
        pytest.skip("mpirun-ulfm not installed in this prefix")
    r = _run([ulfm, "--with-ft", "ulfm", "--allow-run-as-root", "-np", "1", "true"])
    assert r.returncode == 0, f"--with-ft ulfm failed on this install: {r.stderr[:200]}"
