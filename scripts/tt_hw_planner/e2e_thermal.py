# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""emit-e2e's thermal protection: the same four layers optimize has, applied to emit-e2e's device work.

    hold      before a device step starts: at/above the safety ceiling, wait until the board is cool
    cooldown  the same hold after the step exits
    watch     for the whole run, record the board running over the clamp threshold
    abort     at the abort limit, end THIS run's device processes, reset the board, and re-run the
              step from a cool board

optimize gets all four from _run_device_proc. emit-e2e had none: its gate steps hold at launch only
when they go through probes._execute, and the builder agent runs `pytest` and scripts itself, which
nothing watched. Measured 2026-10-10 on a 4-chip p300c: a Devstral run held 88-94C for hours with no
[thermal-*] line in its log, then a chip dropped off the bus and the reset after it took all four.

NO POLICY LIVES HERE. Thresholds, the abort decision and the over-clamp record are cc_optimize.perf_mcp's
and run's; the hold and the run-wide names are agent.thermal_pytest_plugin's (which delegates the hold
to run's launch gate); the step retry is run.rerun_after_thermal_abort; "is this process ours" is
cc_harness._carries_tag. This module only wires them to emit-e2e's launch points.

Agent-started pytest is covered by agent.thermal_pytest_plugin, which install() puts on PYTEST_PLUGINS
for every process this run starts. A process is "this run's" when its environment carries RUN_ENV with
this run's value, so a detached (`setsid nohup`) run is still recognised, and the watcher never ends
anything else. (The reset after an abort is the shared recovery's, which reclaims every holder first.)
"""

from __future__ import annotations

import os
import signal
import sys
import tempfile
import threading
import time
from pathlib import Path

from models.experimental.perf_automation.agent.thermal_pytest_plugin import (
    ABORT_MARK,
    PLUGIN_MODULE,
    RECOVERED_MARK,
    RECOVERED_UPTO,
    RUN_ENV,
    WATCH_POLL_S,
    hold_if_hot,
    record_abort,
    thermal_owner,
)

from .cc_harness import _carries_tag

_REPO_ROOT = Path(__file__).resolve().parents[2]
_OWNER_ENV = RUN_ENV + "_OWNER"  # pid of the process whose watcher covers the run
_PLUGINS_ENV, _PATH_ENV = "PYTEST_PLUGINS", "PYTHONPATH"  # how a child's pytest finds the plugin
# How long a holder gets to end itself (the plugin says why, which tells an agent this was the board
# and not its code) before the watcher ends it. Two plugin polls.
_ABORT_GRACE_S = 2 * WATCH_POLL_S
# How long a step waits for the watcher's reset after an abort before re-running anyway.
_RECOVERY_WAIT_S = 900.0

_STATE = {"thread": None, "recover": None}
_LOCK = threading.Lock()


def record_path() -> str:
    """This run's abort record ('' when install() has not run in this process tree)."""
    return os.environ.get(RUN_ENV, "")


def _record_lines() -> list:
    try:
        return Path(record_path()).read_text().splitlines()
    except OSError:
        return []


def _counts() -> tuple:
    """(aborts recorded, aborts covered by a finished reset). A RECOVERED line names how many ABORT
    lines it covers (`upto=`), because one reset answers every abort it found."""
    aborts, covered = 0, 0
    for ln in _record_lines():
        if ln.startswith(ABORT_MARK):
            aborts += 1
        elif ln.startswith(RECOVERED_MARK):
            for tok in ln.split():
                n = tok[len(RECOVERED_UPTO) :]
                if tok.startswith(RECOVERED_UPTO) and n.isdigit():
                    covered = max(covered, int(n))
    return aborts, covered


def run_holders() -> list:
    """Pids holding the device that THIS run started -- never this process or its ancestors
    (device_holders excludes them), never anything without this run's RUN_ENV value."""
    value = record_path()
    if not value:
        return []
    try:
        from models.experimental.perf_automation.agent.device_recovery import device_holders

        return sorted(p for p in device_holders() if _carries_tag(p, value, var=RUN_ENV))
    except Exception:  # noqa: BLE001 -- a scan that cannot run must not stop the run
        return []


def _kill(pid: int) -> None:
    """End one device holder. Only the pid: its process group is usually the AGENT's (a Bash tool's
    pytest shares it), and ending the agent is not the point. A holder's own children that hold the
    device are holders too, and are ended by the same scan."""
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass


def _recover(n_aborts: int) -> str:
    """Reset the board after a thermal abort -- ALWAYS, whatever the ended run held.

    MEASURED, NOT ASSUMED. The shared rule (device_recovery.reset_is_mandatory_after_kill) makes a reset
    after a kill mandatory only for a multi-chip run, on the reasoning that a single chip has no fabric
    to wedge, and leaves a single chip to the telemetry veto. On this 4-chip p300c, 2026-10-10: a
    single-chip pytest (physical chip 2, no fabric) was ended mid-matmul at the abort limit, and the
    next open of chip 2 failed with "Device 2 init: failed to initialize FW! Try resetting the board"
    while board_needs_reset() still said False -- the ARC kept answering. Neither emit-e2e's wedge
    signatures nor device_recovery.is_dead_board recognise that error, so without a reset here every
    later run on that chip fails and is reported as the model's. _recover_board(..., fault_is_certain)
    reset it and verified it (43 s), and the chip ran again.

    The shared recover() reclaims every process holding the device before it resets; a reset cannot run
    under a holder."""
    fn = _STATE["recover"]
    if fn is None:
        return "no recovery hook installed"
    try:
        came_back, how = fn("thermal abort: a device process was ended mid-run at the abort limit", True)
        return "%s (reset after %d abort(s))" % (how, n_aborts)
    except Exception as exc:  # noqa: BLE001
        return "recovery raised: %s" % exc


def _watch_tick(run, pm, ctx: dict) -> None:
    """One poll of the run-wide watch. `ctx` carries `state` (run's watch state), `hot_since` and
    `handled` (how many ABORT lines a reset has already answered)."""
    run._thermal_watch_sample(ctx["state"], "emit-e2e run")
    hot, cur = pm.board_over_abort_limit()
    holders = run_holders() if hot else []
    if hot and holders:
        ctx["hot_since"] = ctx.get("hot_since") or time.monotonic()
        if time.monotonic() - ctx["hot_since"] >= _ABORT_GRACE_S:
            # Still holding the device after the grace: it did not end itself (a script, or a pytest
            # without the plugin), so end it here.
            for pid in holders:
                _kill(pid)
                record_abort(
                    record_path(), "%s pid=%d temp=%s by=e2e-watcher t=%d" % (ABORT_MARK, pid, cur, time.time())
                )
            print(
                "  [thermal-abort] emit-e2e: board at %.1fC, at or above the %.1fC abort limit -- ended this "
                "run's device process(es) %s; the step re-runs once the board is cool"
                % (cur, pm._ABORT_CEILING_C, holders),
                file=sys.stderr,
                flush=True,
            )
            ctx["hot_since"] = None
    else:
        ctx["hot_since"] = None
    aborts, _ = _counts()
    if aborts > ctx.get("handled", 0):
        how = _recover(aborts - ctx.get("handled", 0))
        ctx["handled"] = aborts
        record_abort(record_path(), "%s %s%d t=%d %s" % (RECOVERED_MARK, RECOVERED_UPTO, aborts, time.time(), how))
        print("  [thermal-abort] board recovery after the abort: %s" % how, file=sys.stderr, flush=True)


def _watch_loop() -> None:
    run = thermal_owner("run")
    pm = thermal_owner("perf_mcp")
    if run is None or pm is None:
        return
    ctx = {"state": run._thermal_watch_new(), "hot_since": None, "handled": _counts()[0]}
    while True:
        time.sleep(WATCH_POLL_S)
        try:
            _watch_tick(run, pm, ctx)
        except Exception:  # noqa: BLE001 -- the watcher must never take the run down
            continue


def agent_env(env: dict) -> dict:
    """`env` with this run's tag and the pytest plugin, for a child whose env is built from scratch
    (an MCP server config, the harness's agent env). Idempotent."""
    out = dict(env)
    if record_path():
        out[RUN_ENV] = record_path()
    plugins = [p for p in out.get(_PLUGINS_ENV, "").split(",") if p]
    if PLUGIN_MODULE not in plugins:
        plugins.append(PLUGIN_MODULE)
    out[_PLUGINS_ENV] = ",".join(plugins)
    paths = [p for p in out.get(_PATH_ENV, "").split(os.pathsep) if p]
    if str(_REPO_ROOT) not in paths:
        paths.insert(0, str(_REPO_ROOT))  # the plugin must ALWAYS import, or pytest refuses to start
    out[_PATH_ENV] = os.pathsep.join(paths)
    return out


def install(recover=None) -> str:
    """Turn the protection on for this process and everything it starts. Idempotent, and a no-op in a
    child of a run that already installed it (the parent's watcher already covers it). `recover` is
    (error_text, fault_is_certain) -> (came_back, how): emit-e2e's _recover_board. Returns the record path."""
    with _LOCK:
        owned_elsewhere = record_path() and os.environ.get(_OWNER_ENV) != str(os.getpid())
        if _STATE["thread"] is not None or owned_elsewhere:
            return record_path()  # this process, or an ancestor, already runs the watcher
        fd, path = tempfile.mkstemp(prefix="e2e_thermal_", suffix=".log")
        os.close(fd)
        os.environ[RUN_ENV] = path
        os.environ[_OWNER_ENV] = str(os.getpid())
        os.environ.update({k: v for k, v in agent_env(os.environ).items() if k in (_PLUGINS_ENV, _PATH_ENV)})
        _STATE["recover"] = recover
        t = threading.Thread(target=_watch_loop, name="e2e-thermal-watch", daemon=True)
        t.start()
        _STATE["thread"] = t
        print(
            "  [thermal-gate] emit-e2e thermal protection on: hold/cooldown at the safety ceiling, watch, "
            "abort at the abort limit (record %s)" % path,
            flush=True,
        )
        return path


def device_step(label: str, run_once):
    """Run one device step the way optimize's _run_device_step runs one: hold before, cool after, and
    re-run it (run.rerun_after_thermal_abort) when the board reached the abort limit while it ran.
    Returns whatever run_once returns."""
    aborted = [False]

    def _attempt():
        hold_if_hot(label)
        before = _counts()[0]
        result = run_once()
        aborted[0] = _counts()[0] > before
        if aborted[0]:
            # The watcher resets the board after an abort; do not start the re-run on top of that reset.
            deadline = time.monotonic() + _RECOVERY_WAIT_S
            while time.monotonic() < deadline:
                aborts, covered = _counts()
                if covered >= aborts:
                    break
                time.sleep(WATCH_POLL_S)
        hold_if_hot("%s (post-run cooldown)" % label)
        return result

    run = thermal_owner("run")
    if run is None:
        return _attempt()
    return run.rerun_after_thermal_abort(_attempt, lambda: aborted[0], label)
