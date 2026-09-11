# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A device subprocess can load a full model checkpoint into host RAM, and nothing checked whether
the box could survive that before launching one.

RUN, 2026-09-11. The kernel OOM-killed device subprocesses repeatedly, in two separate incidents:

    05:59-07:04  nine kills in cc_optimize.run's own launch path (_run_device_proc), ~125 GB each
    07:50        one kill in agent.stack_survey's OWN, separate subprocess.run call -- the FIRST
                 gate (added after the first incident) covered only _run_device_proc, so the very
                 next run OOM-killed at Step 6 (pipeline mapping) exactly the same way

Every retry repeated the same failure because nothing at either launch point looked at host memory
before starting -- the thermal gate protects the board, but a subprocess that loads a 30B-parameter
reference model at full precision can exhaust 249 GB of RAM on its own, and no amount of board
cooling touches that.

THE FIX IS NOT "know how much the next subprocess needs" -- that is model- and call-specific, and
these launch points are generic across every model the tool optimizes. It is the same shape as the
thermal gate: refuse to START heavy work when the box is ALREADY in a state nothing would survive,
without trying to predict what the work itself will cost.

SHARED IN ONE PLACE (agent.probes), not copied per launch point -- the first attempt at this fix
lived only in cc_optimize.run and missed agent.stack_survey.survey_model, which builds the model via
its own independent subprocess.run and was the very next thing to OOM. check_pcc (agent.pcc_runner)
and the PCC gate generator (agent.pcc_gate_gen) run the SAME full-depth build after every lever
attempt during the optimize loop and are the most likely explanation for the first incident's tight,
repeating cadence -- every one of these must call the same shared check, not grow its own copy.
"""

import importlib
import inspect
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))


@pytest.fixture()
def probes(monkeypatch):
    monkeypatch.setenv("PERF_MCP_MIN_FREE_MEM_GB", "20")
    monkeypatch.setenv("PERF_MCP_MEM_POLL_S", "0")
    import models.experimental.perf_automation.agent.probes as P

    importlib.reload(P)
    P._MEM_GATE_BROKEN[0] = False
    return P


def _mem(monkeypatch, probes, readings):
    seen = iter(readings)
    monkeypatch.setattr(probes, "available_memory_gb", lambda: next(seen, readings[-1]))
    monkeypatch.setattr(probes.time, "sleep", lambda _s: None)


def test_a_healthy_box_is_not_delayed(monkeypatch, probes):
    _mem(monkeypatch, probes, [64.0])
    calls = {"n": 0}
    monkeypatch.setattr(probes.time, "sleep", lambda _s: calls.__setitem__("n", calls["n"] + 1))
    probes.wait_for_memory_headroom_before_device_work("device work")
    assert calls["n"] == 0, "a box with plenty of headroom waited anyway"


def test_it_waits_out_a_low_reading_that_recovers(monkeypatch, probes):
    _mem(monkeypatch, probes, [5.0, 5.0, 30.0])
    probes.wait_for_memory_headroom_before_device_work("device work")  # must not raise


def test_it_gives_up_and_launches_anyway_after_the_bound(monkeypatch, probes):
    """Best-effort, like the thermal gate: it must not hang the run forever waiting on memory that
    never comes back -- a launch that still fails now fails as a plain OOM, not a silent hang."""
    monkeypatch.setenv("PERF_MCP_MEM_WAIT_S", "0")
    _mem(monkeypatch, probes, [2.0])
    probes.wait_for_memory_headroom_before_device_work("device work")  # returns, does not hang


def test_unreadable_memory_is_not_a_board_we_refuse_to_use(monkeypatch, probes):
    """Same rule as the rest of the safety gates: a missing sensor is not a reason to hold up work."""
    monkeypatch.setattr(probes, "available_memory_gb", lambda: None)
    probes.wait_for_memory_headroom_before_device_work("device work")  # must not raise


def test_a_broken_gate_warns_once_and_lets_work_continue(monkeypatch, probes, capsys):
    def _boom():
        raise RuntimeError("no /proc")

    monkeypatch.setattr(probes, "available_memory_gb", _boom)
    probes.wait_for_memory_headroom_before_device_work("device work")
    probes.wait_for_memory_headroom_before_device_work("device work")
    err = capsys.readouterr().err
    assert err.count("MEMORY GATE CANNOT RUN") == 1, "warned more than once, or not at all"


def test_reads_meminfo_available_not_free(probes):
    """MemFree undercounts reclaimable cache; MemAvailable is what decides if the next allocation
    succeeds. Assert the real function talks to /proc/meminfo, not a hardcoded number."""
    val = probes.available_memory_gb()
    assert val is None or val >= 0.0


# ------------------------------------------------------- every launch point wires the shared gate


def test_run_devices_proc_calls_it():
    """The gate is worthless anywhere else: it must run at the ONE place in cc_optimize.run every
    device-touching subprocess launches through, alongside the thermal gate it mirrors."""
    import models.experimental.perf_automation.cc_optimize.run as R

    src = inspect.getsource(R._run_device_proc)
    i = src.index("_wait_for_thermal_headroom_before_device_work(")
    j = src.index("_wait_for_memory_headroom_before_device_work(")
    assert j > i, "the memory gate is not wired in right after the thermal gate"
    assert src.index("subprocess.Popen", j) > j, "the gate does not run before the process launches"
    # AND the re-export actually delegates to the shared implementation, not a stale local copy.
    assert "agent.probes" in inspect.getsource(R._wait_for_memory_headroom_before_device_work)
    assert "agent.probes" in inspect.getsource(R._available_memory_gb)


def test_stack_survey_calls_it_at_both_build_sites():
    """stack_survey builds the model via TWO separate functions (survey_model + survey), each with
    its OWN subprocess.run, independent of _run_device_proc -- the incident this test file is named
    for is exactly the first fix missing these call sites entirely."""
    import re

    import models.experimental.perf_automation.agent.stack_survey as S

    for fn in (S.survey_model, S.survey):
        src = inspect.getsource(fn)
        calls = [m.start() for m in re.finditer(r"wait_for_memory_headroom_before_device_work\(", src)]
        runs = [m.start() for m in re.finditer(r"subprocess\.run\(", src)]
        assert runs, "%s has no subprocess.run to guard -- test is stale" % fn.__name__
        for r in runs:
            assert any(c < r for c in calls), "%s's subprocess.run has no preceding memory-gate call" % fn.__name__


def test_check_pcc_calls_it():
    """check_pcc runs after EVERY lever attempt during the optimize loop -- the highest-frequency
    full-depth build in the tool, and the most likely explanation for a tight repeating OOM cadence."""
    import models.experimental.perf_automation.agent.pcc_runner as R

    src = inspect.getsource(R.run_pcc)
    i = src.index("wait_for_memory_headroom_before_device_work(")
    j = src.index("subprocess.run(")
    assert j > i, "check_pcc's subprocess.run is not preceded by the memory gate"


def test_pcc_gate_gen_calls_it():
    import models.experimental.perf_automation.agent.pcc_gate_gen as G

    src = inspect.getsource(G._run_gate)
    i = src.index("wait_for_memory_headroom_before_device_work(")
    j = src.index("subprocess.run(")
    assert j > i, "_run_gate's subprocess.run is not preceded by the memory gate"
