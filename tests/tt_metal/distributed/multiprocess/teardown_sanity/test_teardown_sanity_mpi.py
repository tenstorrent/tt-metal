# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Teardown sanity, the launcher: does tt-run address the MPI failure modes?

Every row launches the raw C++ probe (``sanity_mpi_probe.cpp``, which uses MPI the way tt-metal's DistributedContext
does) THROUGH tt-run, then asks whether tt-run ends the job correctly: the right exit code, the ranks on every host
gone, nothing left in ``/dev/shm``. tt-run picks the plain launcher and builds the production mpirun line
(``--mca btl self,tcp``); a group-E configuration variation (a transport, a component, an interface) goes through
``--mpi-args``, which reaches PRRTE and OpenMPI. These rows are the ones to run again when the MPI install changes:
``TT_SANITY_MPI_PREFIX=/opt/openmpi-<version>`` picks the mpirun and the mpicxx and the probe is recompiled.

A row that passes is a gate: tt-run already handles that failure. A known gap is a strict xfail naming what fixes
it, so it turns red (XPASS) when that fix lands.
"""

from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parent)
)  # sanity_common lives next to this file, whatever the import mode
from sanity_common import (
    DEADLINE_END,
    DEADLINE_HANG,
    MAPPING,
    MGD,
    ROOT,
    TTRUN,
    ULFM_DETECTOR,
    clean,
    ended,
    fault_delivered,
    forced,
    launcher_cmd,
    marked,
    mpi_cmd,
    mpi_probe_path,
    mpirun_path,
    nonzero,
    pinned_tcp_interface,
    run_case,
    stop_lines,
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


def test_ttrun_hang_timeout():
    """A rank spins forever. With tt-run's --job-timeout 20, tt-run sends SIGTERM to every rank at 20 s (graceful), the job
    ends with 241 and leaves nothing (P5)."""
    r = run_case(
        "ttrun_hang_timeout",
        {**NP, "PROBE_SCENARIO": "victim_hang"},
        DEADLINE_END,
        cmd=ttrun_mpi_cmd(ttrun_extra=("--job-timeout", "20")),
    )
    ended(r)
    nonzero(r)
    assert r.exit == 241, f"expected the timeout code 241, got {r.exit} — the clock was not what ended it: {r}"
    # tt-run's own job time (launch to the end of the job), not the harness wall clock that also counts start-up
    m = re.search(r"job end: .* elapsed=([0-9.]+)s", r.log)
    assert m, r.log[-2000:]
    assert 20 <= float(m.group(1)) < 40, f"the job did not end at the ~20 s --timeout but at {m.group(1)} s: {r}"
    assert "the job reached its --job-timeout of 20 s: sent SIGTERM to all ranks" in r.log, r.log[-2000:]
    clean(r)


def test_ttrun_mpi_args_timeout_is_prrtes_own():
    """--mpi-args "--timeout 20" is PRRTE's own clock and reaches mpirun unchanged: tt-run warns at the start that it
    kills every rank at once, PRRTE ends the job at 20 s with 241, tt-run says it was forced, nothing is left."""
    r = run_case(
        "ttrun_mpi_args_timeout",
        {**NP, "PROBE_SCENARIO": "victim_hang"},
        DEADLINE_END,
        cmd=ttrun_mpi_cmd("--timeout 20"),
    )
    ended(r)
    clean(r)
    assert r.exit == 241, r
    assert "sets PRRTE's own --timeout 20" in r.log, r.log[-2000:]
    assert "job forced down" in r.log and "PRRTE --timeout" in r.log and "outcome=forced exit=241" in r.log, r.log[
        -2000:
    ]


@pytest.mark.xfail(
    reason="P1 for a raw MPI program: a rank that calls exit(1) runs its own atexit MPI_Finalize, which is collective "
    "and waits on the peers in their barrier, so the job rides the --timeout wall clock (241). Neither tt-run nor its "
    "rank wrapper can end a rank that is still alive; a tt-metal rank stops the job from its exit instead (https://github.com/tenstorrent/tt-metal/pull/64546).",
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
        # A launch through "$mpirun_args" counts only when the script DEFINES mpirun_args with the rule (code, not a
        # comment that happens to mention it).
        has_rule = any(
            re.match(r"\s*(local\s+|export\s+)?mpirun_args(_base)?\+?=", line.split("#", 1)[0])
            and "--mca btl self,tcp" in line.split("#", 1)[0]
            for line in text.splitlines()
        )
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
    [(signal.SIGTERM, "delivered"), (signal.SIGINT, "no_helper")],
    ids=["sigterm", "ctrl_c_no_helper"],  # Ctrl-C delivered: test_every_ending[tt-run-ctrl_c]
)
def test_ttrun_stop(sig, case):
    """The ranks hang, then tt-run gets a stop signal. tt-run sends SIGTERM to every rank through PRRTE job
    control: in all_hang every rank records the signal it caught, so each must report signal 15; the job reports
    128 + the signal and nothing is left. Without the helper tt-run forces at once through the launcher and says
    the devices may be unhealthy. (What a rank does with the SIGTERM is the rank's business, not tt-run's.)"""
    _ttrun_stop(sig, case)


def _ttrun_stop(sig, case, mpi_prefix=None):
    env = {**NP, "PROBE_SCENARIO": "all_hang" if case == "delivered" else "victim_hang"}
    cwd = None
    if case == "no_helper":  # a tree with the sources but no build/, and no installed copy: no tt-run-rank
        cwd = Path(tempfile.mkdtemp(prefix="sanity_nobuild_"))
        for d in ("tt_metal", "tests"):
            (cwd / d).symlink_to(ROOT / d)
        env["TT_METAL_HOME"] = str(cwd)
        path = os.environ.get("PATH", "").split(os.pathsep)
        env["PATH"] = os.pathsep.join(d for d in path if not os.access(os.path.join(d, "tt-run-rank"), os.X_OK))
    r = run_case(
        f"ttrun_stop_{signal.Signals(sig).name}_{case}",
        env,
        DEADLINE_HANG,
        cmd=ttrun_mpi_cmd(mpi_prefix=mpi_prefix),
        launcher_signal=sig,
        launcher_ready=(0, "ready"),  # the ranks are up and hanging before we stop them, not still starting
        signal_after=1,
        mpi_prefix=mpi_prefix,
        cwd=cwd,
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
    [signal.SIGUSR1, signal.SIGQUIT],  # one that goes on, one that would end prterun; the rest: test_ttrun.py
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
    Every rank receives it and the job ends; nothing is left, the guard included. The ranks write to stderr while they
    stop (30 lines over 3 s): with tt-run gone the guard drains the launcher's stderr, or prterun would abort the job
    on SIGPIPE (SIGKILL to every rank) and the stop would not finish."""
    r = run_case(
        "ttrun_sigkill",
        {**NP, "PROBE_SCENARIO": "all_hang", "PROBE_STOP_STDERR_LINES": "30", "PROBE_STOP_DELAY_MS": "3000"},
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
    # every rank finished its stop (a prterun abort on SIGPIPE would have killed them mid-way), and its stderr got out
    done = {int(m.group(1)) for m in re.finditer(r"\[probe r=(\d+) [^\]]*\] main_exit", r.log)}
    assert done == {0, 1, 2}, f"a rank did not finish its stop: {sorted(done)}\n{r.log[-3000:]}"
    assert all(f"[probe r={k}] stop line 29" in r.log for k in range(3)), r.log[-3000:]
    assert "SIGPIPE" not in r.log, r.log[-3000:]


# ---------------------------------------------------------------------------------------------------------------
# Phase 1 (new mode: --mesh-graph-descriptor): the same stops, signals and failures while generate_rank_bindings runs.
# Phase 1 runs through the same launcher code as Phase 2 (its ranks in tt-run-rank, the guard, --job-timeout); here
# its generate_rank_bindings is the C probe, from a stand-in tree whose build/ holds only the probe and tt-run-rank.
# ---------------------------------------------------------------------------------------------------------------
def phase1_cmd(tree: Path, ttrun_extra: tuple = ()) -> list:
    return [
        TTRUN,
        *ttrun_extra,
        "--mesh-graph-descriptor",
        str(MGD),
        "--mock-cluster-rank-binding",
        str(MAPPING),
        "--force-rediscovery",
        "--mpi-args",
        "--allow-run-as-root --oversubscribe -x PROBE_SCENARIO -x PROBE_VICTIM -x PROBE_NP -x PROBE_PIDDIR",
        "/bin/true",  # Phase 2's program: never reached, Phase 1 ends the job
    ]


def phase1_tree() -> Path:
    """A tree with the sources and a build/ whose generate_rank_bindings runs the C probe (its arguments ignored)."""
    tree = Path(tempfile.mkdtemp(prefix="sanity_phase1_"))
    for d in ("tt_metal", "tests"):
        (tree / d).symlink_to(ROOT / d)
    tools = tree / "build" / "tools" / "scaleout"
    tools.mkdir(parents=True)
    real = ROOT / "build" / "tools" / "scaleout" / "tt-run-rank"
    if not os.access(real, os.X_OK):
        pytest.skip("tt-run-rank is not built")
    (tools / "tt-run-rank").symlink_to(real)
    grb = tools / "generate_rank_bindings"
    grb.write_text(f"#!/bin/sh\nexec '{mpi_probe_path()}'\n")
    grb.chmod(0o755)
    return tree


PHASE1 = {"PROBE_NP": "2"}  # the mock dual galaxy: two Phase 1 ranks


def phase1_case(name: str, env: dict, deadline: float, ttrun_extra: tuple = (), **kw):
    tree = phase1_tree()
    try:
        return run_case(
            name,
            {**PHASE1, **env, "TT_METAL_HOME": str(tree)},
            deadline,
            cmd=phase1_cmd(tree, ttrun_extra),
            cwd=tree,
            **kw,
        )
    finally:
        shutil.rmtree(tree, ignore_errors=True)


def test_phase1_stop():
    """Ctrl-C while Phase 1 runs: SIGTERM to every Phase 1 rank through job control, every rank catches it, tt-run
    returns 130 (not a generic Phase 1 failure) and Phase 2 never starts; nothing is left."""
    r = phase1_case(
        "phase1_ctrl_c",
        {"PROBE_SCENARIO": "all_hang"},
        DEADLINE_HANG,
        launcher_signal=signal.SIGINT,
        launcher_ready=(0, "ready"),
        signal_after=1,
    )
    fault_delivered(r)
    ended(r)
    clean(r)
    got = {int(m.group(1)) for m in re.finditer(r"\[probe r=(\d+) [^\]]*\] caught signal 15\b", r.log)}
    assert got == {0, 1}, f"every Phase 1 rank must receive SIGTERM: {sorted(got)}\n{r.log[-3000:]}"
    assert r.exit == 130 and "Phase 2" not in r.log, r


def test_phase1_dead_rank():
    """A Phase 1 rank is killed: its wrapper stops Phase 1, tt-run names the rank and returns its 137."""
    r = phase1_case("phase1_dead_rank", {"PROBE_SCENARIO": "victim_sigkill", "PROBE_VICTIM": "1"}, DEADLINE_END)
    ended(r)
    clean(r)
    assert re.search(r"rank 1 on \S+ \(pid \d+\): was killed by signal 9 \(SIGKILL\)", r.log), r.log[-3000:]
    assert r.exit == 137 and "Phase 1 (generate_rank_bindings) failed" in r.log, r


def test_phase1_job_timeout():
    """A Phase 1 rank hangs: --job-timeout covers Phase 1 too, so tt-run stops it gracefully at 15 s with 241."""
    r = phase1_case(
        "phase1_hang_timeout",
        {"PROBE_SCENARIO": "victim_hang", "PROBE_VICTIM": "1"},
        DEADLINE_HANG,
        ttrun_extra=("--job-timeout", "15"),
    )
    ended(r)
    clean(r)
    assert r.exit == 241 and "the job reached its --job-timeout of 15 s" in r.log, r


# The offline Phase 1 (--offline, from a factory system descriptor): one process without MPI, through the same
# launcher (direct) and inside tt-run-rank. Its generate_rank_bindings is a script here.
def offline_case(name: str, script: str, deadline: float, **kw):
    tree = phase1_tree()
    grb = tree / "build" / "tools" / "scaleout" / "generate_rank_bindings"
    grb.write_text("#!/bin/bash\n" + script + "\n")
    fsd = next(ROOT.glob("tt_metal/third_party/tt-cluster-descriptors/**/*factory_system_descriptor.textproto"), None)
    if fsd is None:
        pytest.skip("no factory system descriptor in the tree")
    cmd = [
        TTRUN,
        "-m",
        str(MGD),
        "--factory-system-descriptor",
        str(fsd),
        "--offline",
        "--force-rediscovery",
        "/bin/true",
    ]
    try:
        return run_case(name, {"TT_METAL_HOME": str(tree)}, deadline, cmd=cmd, cwd=tree, **kw)
    finally:
        shutil.rmtree(tree, ignore_errors=True)


def test_offline_phase1_stop():
    """Ctrl-C while the offline Phase 1 runs: tt-run sends it SIGTERM itself (no job control without MPI), it exits
    gracefully, tt-run returns 130 and nothing is left."""
    r = offline_case(
        "offline_ctrl_c",
        'trap "echo offline: caught TERM; exit 0" TERM; touch "$PROBE_PIDDIR/rank_0.ready"; while :; do sleep 0.1; done',
        DEADLINE_HANG,
        launcher_signal=signal.SIGINT,
        launcher_ready=(0, "ready"),
        signal_after=0.5,
    )
    fault_delivered(r)
    ended(r)
    clean(r)
    assert "offline: caught TERM" in r.log and r.exit == 130, r
    assert "job forced down" not in r.log, r.log[-3000:]


def test_offline_phase1_dead_process():
    """The offline Phase 1 process is killed: its tt-run-rank names it (rank 0, from tt-run's TT_RUN_RANK) and tt-run
    returns its 137."""
    r = offline_case("offline_killed", "kill -9 $$", DEADLINE_END)
    ended(r)
    clean(r)
    assert re.search(r"rank 0 on \S+ \(pid \d+\): was killed by signal 9 \(SIGKILL\)", r.log), r.log[-3000:]
    assert r.exit == 137, r


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
    cmd = mpi_cmd(ulfm=False, extra=" ".join(["--mca", "pml", "ob1", *pinned_tcp_interface()]), btl="self,tcp", np=2)
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


# ---------------------------------------------------------------------------------------------------------------
# tt-run-rank under tt-run and under a plain mpirun (``mpirun --mca mpi_ft_enable 1 ... tt-run-rank -- <probe>``), on
# the probe's "job" scenario. Each row checks: every live rank closed, none was forced, the stop line names the right
# rank and code, the launcher returns that code, nothing is left.
# ---------------------------------------------------------------------------------------------------------------
LAUNCHERS = ["tt-run", "mpirun"]
JOB = {"PROBE_NP": "3", "PROBE_SCENARIO": "job", "PROBE_VICTIM": "1"}


def job(name, launcher, env, deadline=DEADLINE_END, wrapper_args=(), ttrun_extra=(), mpi_extra="", **kw):
    env = {**JOB, **env}
    cmd = launcher_cmd(launcher, env, int(env["PROBE_NP"]), wrapper_args, ttrun_extra, mpi_extra)
    return run_case(f"{name}[{launcher}]", env, deadline, cmd=cmd, **kw)


def assert_job(r, origin, code, live, follows="stopping the job", reason=""):
    ended(r)
    clean(r)
    assert live <= marked(r, "close_done"), r.log[-3000:]
    assert not forced(r) and "job forced down" not in r.log, r.log[-3000:]
    lines = stop_lines(r)
    if origin is None:
        assert not lines, lines
    else:
        assert lines and {rank for rank, *_ in lines} == {origin}, lines
        assert lines[0][2:] == (follows, code) and reason in lines[0][1], lines
    assert r.exit == code, f"{r}\n{r.log[-3000:]}"


# ending -> (the victim's PROBE_ENDING, the failed rank, the code, the ranks that close, the reason)
ENDINGS = {
    "sigkill": ("sigkill", 1, 137, {0, 2}, "killed by signal 9 (SIGKILL)"),
    "segfault": ("segfault", 1, 139, {0, 2}, "killed by signal 11 (SIGSEGV)"),
    "raise": ("raise", 1, 1, {0, 1, 2}, "RuntimeError: injected failure at after_open"),
    "exit1": ("exit1", 1, 1, {0, 2}, "exited with code 1"),
    "sigterm_one_rank": ("hang", 1, 143, {0, 1, 2}, "exited with code 143"),
    "ctrl_c": ("hang", None, 130, {0, 1, 2}, ""),
    "clean": ("none", None, 0, {0, 1, 2}, ""),
}


@pytest.mark.parametrize("ending", list(ENDINGS))
@pytest.mark.parametrize("launcher", LAUNCHERS)
def test_every_ending(launcher, ending):
    """A: a 143 with no stop is a failure, a 143 after the launcher's stop is not. A Ctrl-C at a plain mpirun is
    prterun's own kill: that row checks only that the job ends and leaves nothing."""
    action, origin, code, live, reason = ENDINGS[ending]
    kw = {}
    if ending == "sigterm_one_rank":
        kw = {"rank_signal": (1, signal.SIGTERM, 1.0)}
    elif ending == "ctrl_c":
        kw = {"launcher_signal": signal.SIGINT, "launcher_ready": (2, "ready"), "signal_after": 1}
    r = job(f"A_{ending}", launcher, {"PROBE_ENDING": action}, DEADLINE_HANG if kw else DEADLINE_END, **kw)
    if kw:
        fault_delivered(r)
    if ending == "ctrl_c" and launcher == "mpirun":
        ended(r)
        clean(r)
    else:
        assert_job(r, origin, code, live, reason=reason)


@pytest.mark.parametrize("failure", ["sigkill", "raise"])
@pytest.mark.parametrize("launcher", LAUNCHERS)
def test_exit_order(launcher, failure):
    """B: rank 0 exits at once while rank 2 takes 3 s to close: rank 2 is never killed, the code is rank 1's."""
    r = job(f"B_{failure}", launcher, {"PROBE_ENDING": failure, "PROBE_STOP_DELAY_MS_2": "3000"})
    assert_job(r, 1, {"sigkill": 137, "raise": 1}[failure], {0, 2})


@pytest.mark.parametrize("rank2,grace", [("slow", None), ("slow", "5"), ("never", "5")])
@pytest.mark.parametrize("launcher", LAUNCHERS)
def test_stop_grace(launcher, rank2, grace):
    """C: rank 2 closes in 2 s, or never. Inside the grace nothing is forced; past it its wrapper forces it (before
    tt-run would), and the code stays rank 1's."""
    env = {
        "PROBE_ENDING": "sigkill",
        **({"PROBE_STOP_DELAY_MS_2": "2000"} if rank2 == "slow" else {"PROBE_IGNORE_STOP": "2"}),
    }
    kw = {}
    if grace and launcher == "tt-run":
        env["TT_RUN_STOP_GRACE_S"] = grace
    elif grace:
        kw["wrapper_args"] = ("--stop-grace", grace)
    r = job(f"C_{rank2}_{grace}", launcher, env, DEADLINE_HANG, **kw)
    if rank2 == "slow":
        assert_job(r, 1, 137, {0, 2})
        return
    ended(r)
    clean(r)
    assert forced(r) == {2} and "tt-smi -r" in r.log and 0 in marked(r, "close_done"), r.log[-3000:]
    assert [(rank, code) for rank, _, _, code in stop_lines(r)] == [(1, 137)], stop_lines(r)
    assert r.exit == 137 and "job forced down" not in r.log, r.log[-3000:]


def test_reason_is_sent_before_the_rank_closes():
    """D: the wrapper reads the rank's "failed" word as it is written: the peers get SIGTERM before it has closed."""
    r = job("D_reason", "tt-run", {"PROBE_ENDING": "raise", "PROBE_RAISE_CLOSE_MS": "3000"})
    assert_job(r, 1, 1, {0, 1, 2}, reason="RuntimeError: injected failure at after_open")
    closed = re.search(r"\[probe r=1 [^\]]*\] close_done", r.log).start()
    for k in (0, 2):
        caught = re.search(rf"\[probe r={k} [^\]]*\] caught signal 15", r.log)
        assert caught and caught.start() < closed, r.log[-3000:]


@pytest.mark.parametrize("exit_kind", ["exit3", "sigkill"])
@pytest.mark.parametrize("mode", ["stop", "notify"])
@pytest.mark.parametrize("launcher", LAUNCHERS)
def test_peer_exit(launcher, mode, exit_kind):
    """E: the others read "exit 1 <code> <signal> <reason>". stop: then SIGTERM. notify: SIGUSR1, no stop, they finish
    their work and the job ends with rank 1's code."""
    code, sig = {"exit3": (3, 0), "sigkill": (137, 9)}[exit_kind]
    if launcher == "tt-run":
        kw = {"ttrun_extra": ("--on-peer-exit", mode)}
    else:
        kw = {"wrapper_args": ("--on-peer-exit", mode), "mpi_extra": ULFM_DETECTOR if mode == "notify" else ""}
    r = job(f"E_{mode}_{exit_kind}", launcher, {"PROBE_ENDING": exit_kind}, **kw)
    assert marked(r, f"event: exit 1 {code} {sig} ") == {0, 2}, r.log[-3000:]
    if mode == "stop":
        assert marked(r, "caught signal 15") == {0, 2}, r.log[-3000:]
        assert_job(r, 1, code, {0, 2})
    else:
        assert not marked(r, "caught signal 15") and marked(r, "work_done") == {0, 2}, r.log[-3000:]
        assert_job(r, 1, code, {0, 2}, follows="the job goes on")


def test_32_wrappers_hold_a_tool_connection_each():
    """F: 32 wrappers each connect to the PMIx server as a tool (none says it cannot); exit 0, nothing left."""
    r = job("F_scale", "mpirun", {"PROBE_NP": "32", "PROBE_VICTIM": "-1"}, DEADLINE_HANG)
    assert "cannot connect to the PMIx server" not in r.log, r.log[-3000:]
    assert_job(r, None, 0, set(range(32)))
