# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""tt-run-rank under tt-run and under a plain mpirun, on the C probe's "job" scenario. Each row checks: every live rank
closed, none was forced, the stop line names the right rank and code, the launcher returns that code, nothing is left.
"""

from __future__ import annotations

import re
import signal
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sanity_common import (
    DEADLINE_END,
    DEADLINE_HANG,
    ULFM_DETECTOR,
    clean,
    ended,
    fault_delivered,
    forced,
    launcher_cmd,
    marked,
    run_case,
    stop_lines,
)

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
    """F: every wrapper is a PMIx tool for the whole run and leaves at the end; exit 0, nothing left."""
    r = job("F_scale", "mpirun", {"PROBE_NP": "32", "PROBE_VICTIM": "-1", "TT_RUN_RANK_VERBOSE": "1"}, DEADLINE_HANG)
    on = {int(x) for x in re.findall(r"\[tt-run-rank\] rank (\d+): connected to the PMIx server as a tool", r.log)}
    off = {int(x) for x in re.findall(r"\[tt-run-rank\] rank (\d+): left the PMIx server", r.log)}
    assert on == off == set(range(32)), f"connected {sorted(on)}, left {sorted(off)}"
    assert_job(r, None, 0, set(range(32)))
