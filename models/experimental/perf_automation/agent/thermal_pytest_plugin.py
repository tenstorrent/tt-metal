# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Thermal protection for pytest runs that no tool code launches -- the ones an AGENT starts itself.

THE GAP THIS CLOSES. optimize's device work goes through _run_device_proc / probes._execute, which hold
at the safety ceiling before a launch, watch while it runs and end it at the abort limit. emit-e2e's
builder runs its own `pytest` through Bash, so none of that ran. Measured on a 4-chip p300c,
2026-10-10: a Devstral emit-e2e run held 88-94C for hours with not one [thermal-*] line in its log,
then chip 2 dropped off the bus at ~18:41 and the reset that followed took all four chips with it.

Loaded through PYTEST_PLUGINS, which emit-e2e sets for everything it launches (see
scripts/tt_hw_planner/e2e_thermal.py). That is the only route that reaches a `pytest` the agent
started detached (`setsid nohup ... &`), and it needs nothing from the model's code.

THIS MODULE MUST ALWAYS IMPORT. pytest treats an unimportable PYTEST_PLUGINS entry as a hard error,
which would turn temperature protection into a broken test suite. So the top level imports the
standard library and pytest only, and every hook is best-effort: a gate that cannot run warns and
lets the test proceed.

THE POLICY IS NOT HERE. The hold is cc_optimize.run's launch gate and the abort decision is
cc_optimize.perf_mcp's, both reached through agent.probes' resolver. This module owns only WHEN to
ask (session start, between tests, a watch while the process holds the device) and the names every
process in an emit-e2e run agrees on; scripts/tt_hw_planner/e2e_thermal.py imports them from here.
"""

from __future__ import annotations

import os
import sys
import threading
import time

import pytest

# A run that wants this protection sets RUN_ENV to the path of its abort record; every process it
# starts inherits it, which is also how the run tells its own device holders from anyone else's.
RUN_ENV = "TT_HW_PLANNER_THERMAL_RUN"
PLUGIN_MODULE = __name__
ABORT_EXIT_CODE = os.EX_TEMPFAIL  # the run was ended to protect the board, not because it failed
WATCH_POLL_S = 5.0  # the same cadence as _run_device_proc's in-flight thermal check
# The abort record's line kinds: one ABORT line per ended process, one RECOVERED line per reset, naming
# (after RECOVERED_UPTO) how many ABORT lines it answered.
ABORT_MARK, RECOVERED_MARK, RECOVERED_UPTO = "ABORT", "RECOVERED", "upto="

_STATE = {"watch": None, "warned": False}


def thermal_owner(name: str):
    """cc_optimize.<name> through agent.probes' resolver (the one owner of that import), or None --
    saying so once, loudly, because a gate that cannot run must not be silent about it."""
    try:
        from models.experimental.perf_automation.agent import probes
    except Exception as exc:  # noqa: BLE001
        if not _STATE["warned"]:
            _STATE["warned"] = True
            print(
                "  [thermal-gate] WARNING: temperature protection is INERT (agent.probes cannot be "
                "imported: %s). Device work will continue with NO thermal gating." % exc,
                file=sys.stderr,
                flush=True,
            )
        return None
    try:
        return probes._cc_optimize(name)
    except Exception as exc:  # noqa: BLE001
        probes._warn_thermal_inert("emit-e2e thermal protection", exc)
        return None


def hold_if_hot(label: str) -> None:
    """optimize's launch gate, as is: cc_optimize.run._wait_for_thermal_headroom_before_device_work. At
    or above the safety ceiling it waits, with no deadline, until the board is back to the cool-back
    target. Only ever called at a boundary -- no test body is running and nothing is in flight."""
    run = thermal_owner("run")
    if run is not None:
        run._wait_for_thermal_headroom_before_device_work(label)


def holds_device() -> bool:
    """Whether THIS process has a device node open (device_holders() cannot answer: it excludes the
    asking process). Reads /proc; False when it cannot."""
    try:
        from models.experimental.perf_automation.agent.device_recovery import DEVICE_NODE_DIR

        base = "/proc/self/fd"
        for fd in os.listdir(base):
            try:
                if os.readlink(os.path.join(base, fd)).startswith(DEVICE_NODE_DIR):
                    return True
            except OSError:
                continue
    except Exception:  # noqa: BLE001
        return False
    return False


def record_abort(path: str, line: str) -> None:
    """Append one line to the run's abort record. Best-effort."""
    if not path:
        return
    try:
        with open(path, "a") as fh:
            fh.write(line.rstrip("\n") + "\n")
    except OSError:
        pass


def _watch_loop(record_path: str) -> None:
    """Every WATCH_POLL_S: if THIS process holds the device and the board is at the abort limit, say
    why and end the process. A process cannot be paused mid-kernel without risking a wedge; ending it
    releases the device, and saying so first is what lets whoever started it tell a protected board
    from a crash. The run's watcher (e2e_thermal) resets the board after."""
    pm = None
    while True:
        time.sleep(WATCH_POLL_S)
        if not holds_device():
            continue
        if pm is None:
            pm = thermal_owner("perf_mcp")
            if pm is None:
                return
        try:
            hot, cur = pm.board_over_abort_limit()
        except Exception:  # noqa: BLE001
            continue
        if not hot:
            continue
        msg = (
            "[thermal-abort] pytest pid %d: board at %.1fC, at or above the %.1fC abort limit -- ending this "
            "run to protect the hardware. This is NOT a code failure: re-run it; the next run waits for the "
            "board to cool first." % (os.getpid(), cur if cur is not None else -1.0, pm._ABORT_CEILING_C)
        )
        for stream in (sys.stderr, sys.stdout):
            try:
                print(msg, file=stream, flush=True)
            except Exception:  # noqa: BLE001
                pass
        record_abort(
            record_path, "%s pid=%d temp=%s by=pytest-plugin t=%d" % (ABORT_MARK, os.getpid(), cur, time.time())
        )
        os._exit(ABORT_EXIT_CODE)


@pytest.hookimpl(tryfirst=True)
def pytest_sessionstart(session):
    """Before the session's first test: hold while the board is at the ceiling, then start the watch."""
    try:
        hold_if_hot("pytest session start")
        if _STATE["watch"] is None:
            t = threading.Thread(
                target=_watch_loop, args=(os.environ.get(RUN_ENV, ""),), name="thermal-abort-watch", daemon=True
            )
            t.start()
            _STATE["watch"] = t
    except Exception:  # noqa: BLE001 -- protection that cannot start must not stop the session
        pass


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    """Between tests: offer the hold. probes.thermal_yield rate-limits itself, so this is a ~0.3 ms
    sysfs read on most tests and only waits when the board is at the ceiling. tryfirst, so it runs before
    the item's fixtures open the device."""
    try:
        from models.experimental.perf_automation.agent import probes

        probes.thermal_yield("before %s" % getattr(item, "nodeid", "test"))
    except Exception:  # noqa: BLE001
        pass
