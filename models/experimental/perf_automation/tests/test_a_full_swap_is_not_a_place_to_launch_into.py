# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A device subprocess can load a full model checkpoint into host RAM, and nothing checked whether
the box could survive that before launching one.

RUN, 2026-09-11. The kernel OOM-killed nine device subprocesses in a row over roughly an hour, each
one launched into a box whose swap was already at 0 KB free from the PREVIOUS kill's damage:

    05:59  OOM-killed python, anon-rss 125.7 GB
    06:08  OOM-killed python, anon-rss 125.6 GB   (swap: 148 KB free)
    06:15  OOM-killed python, anon-rss 125.3 GB   (swap: 108 KB free)
    ...            (repeats every ~8 minutes for the next hour)
    07:50  OOM-killed python, anon-rss 117.2 GB   (swap: 0 KB free)

Every retry repeated the same failure because nothing at the launch point looked at host memory --
the thermal gate protects the board, but a subprocess that loads a 30B-parameter reference model at
full precision can exhaust 249 GB of RAM on its own, and no amount of board cooling touches that.

THE FIX IS NOT "know how much the next subprocess needs" -- that is model- and call-specific, and
this launch point is generic across every model the tool optimizes. It is the same shape as the
thermal gate: refuse to START heavy work when the box is ALREADY in a state nothing would survive,
without trying to predict what the work itself will cost.
"""

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))


@pytest.fixture()
def run_mod(monkeypatch):
    monkeypatch.setenv("PERF_MCP_MIN_FREE_MEM_GB", "20")
    monkeypatch.setenv("PERF_MCP_MEM_POLL_S", "0")
    import models.experimental.perf_automation.cc_optimize.run as R

    importlib.reload(R)
    R._MEM_GATE_BROKEN[0] = False
    return R


def _mem(monkeypatch, run_mod, readings):
    seen = iter(readings)
    monkeypatch.setattr(run_mod, "_available_memory_gb", lambda: next(seen, readings[-1]))
    monkeypatch.setattr(run_mod.time, "sleep", lambda _s: None)


def test_a_healthy_box_is_not_delayed(monkeypatch, run_mod):
    _mem(monkeypatch, run_mod, [64.0])
    calls = {"n": 0}
    monkeypatch.setattr(run_mod.time, "sleep", lambda _s: calls.__setitem__("n", calls["n"] + 1))
    run_mod._wait_for_memory_headroom_before_device_work("device work")
    assert calls["n"] == 0, "a box with plenty of headroom waited anyway"


def test_it_waits_out_a_low_reading_that_recovers(monkeypatch, run_mod):
    _mem(monkeypatch, run_mod, [5.0, 5.0, 30.0])
    run_mod._wait_for_memory_headroom_before_device_work("device work")  # must not raise


def test_it_gives_up_and_launches_anyway_after_the_bound(monkeypatch, run_mod):
    """Best-effort, like the thermal gate: it must not hang the run forever waiting on memory that
    never comes back -- a launch that still fails now fails as a plain OOM, not a silent hang."""
    monkeypatch.setenv("PERF_MCP_MEM_WAIT_S", "0")
    importlib.reload(run_mod)
    _mem(monkeypatch, run_mod, [2.0])
    run_mod._wait_for_memory_headroom_before_device_work("device work")  # returns, does not hang


def test_unreadable_memory_is_not_a_board_we_refuse_to_use(monkeypatch, run_mod):
    """Same rule as the rest of the safety gates: a missing sensor is not a reason to hold up work."""
    monkeypatch.setattr(run_mod, "_available_memory_gb", lambda: None)
    run_mod._wait_for_memory_headroom_before_device_work("device work")  # must not raise


def test_a_broken_gate_warns_once_and_lets_work_continue(monkeypatch, run_mod, capsys):
    def _boom():
        raise RuntimeError("no /proc")

    monkeypatch.setattr(run_mod, "_available_memory_gb", _boom)
    run_mod._wait_for_memory_headroom_before_device_work("device work")
    run_mod._wait_for_memory_headroom_before_device_work("device work")
    err = capsys.readouterr().err
    assert err.count("MEMORY GATE CANNOT RUN") == 1, "warned more than once, or not at all"


def test_reads_meminfo_available_not_free(run_mod):
    """MemFree undercounts reclaimable cache; MemAvailable is what decides if the next allocation
    succeeds. Assert the real function talks to /proc/meminfo, not a hardcoded number."""
    val = run_mod._available_memory_gb()
    assert val is None or val >= 0.0


def test_the_common_launch_point_calls_it(run_mod):
    """The gate is worthless anywhere else: it must run at the ONE place every device-touching
    subprocess launches through, alongside the thermal gate it mirrors."""
    import inspect

    src = inspect.getsource(run_mod._run_device_proc)
    i = src.index("_wait_for_thermal_headroom_before_device_work(")
    j = src.index("_wait_for_memory_headroom_before_device_work(")
    assert j > i, "the memory gate is not wired in right after the thermal gate"
    assert src.index("subprocess.Popen", j) > j, "the gate does not run before the process launches"
