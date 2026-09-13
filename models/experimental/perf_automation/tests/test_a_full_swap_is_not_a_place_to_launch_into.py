# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A device subprocess can load a full model checkpoint into host RAM, and nothing checked whether
the box could survive that before launching one.

RUN, 2026-09-11. The kernel OOM-killed device subprocesses repeatedly, in THREE separate incidents:

    05:59-07:04  nine kills in cc_optimize.run's own launch path (_run_device_proc), ~125 GB each
    07:50        one kill in agent.stack_survey's OWN, separate subprocess.run call -- the FIRST
                 gate (added after the first incident) covered only _run_device_proc, so the very
                 next run OOM-killed at Step 6 (pipeline mapping) exactly the same way
    10:53        one kill AFTER the headroom wait was wired into every launch point, including
                 stack_survey, pcc_runner, and pcc_gate_gen. Available memory was healthy at launch
                 -- no [memory-gate] line appears anywhere in that run's log -- and the subprocess
                 still grew to 123 GB and got OOM-killed on its own.

Every retry repeated the same failure because nothing at either launch point looked at host memory
before starting -- the thermal gate protects the board, but a subprocess that loads a 30B-parameter
reference model at full precision can exhaust 249 GB of RAM on its own, and no amount of board
cooling touches that.

THE HEADROOM WAIT ANSWERS ONE QUESTION: "is the box already too full to start something new" -- a
question about the LOT, answerable before any car pulls in. It has no opinion once launched, because
it cannot know how big the car will get; that is model- and call-specific.

memory_cap_preexec_fn USED to also put a hard ceiling on the CHILD's own address space (RLIMIT_AS),
sized off whatever was available at launch. RETIRED 2026-09-11: RLIMIT_AS caps total VIRTUAL address
space, not physical use, and the device driver's own mesh-open/TLB-window setup needs large virtual
mappings that carry no real memory pressure. With the cap on, a healthy run failed hard with a
device-level error (`tt_tlb_alloc failed ... error code -12`); the identical run passed clean with
PERF_MCP_DISABLE_MEM_CAP=1 -- confirmed by a direct A/B rerun. The function is now a no-op, kept only
so every `preexec_fn=memory_cap_preexec_fn()` call site keeps working unchanged.

SHARED IN ONE PLACE (agent.probes), not copied per launch point -- the first attempt at this fix
lived only in cc_optimize.run and missed agent.stack_survey.survey_model, which builds the model via
its own independent subprocess.run and was the very next thing to OOM. check_pcc (agent.pcc_runner)
and the PCC gate generator (agent.pcc_gate_gen) run the SAME full-depth build after every lever
attempt during the optimize loop and are the most likely explanation for the first incident's tight,
repeating cadence -- every one of these must call the same shared checks, not grow its own copy.
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


# ------------------------------------------------------------ the hard cap, retired -- now a no-op


def test_the_cap_is_a_retired_no_op_regardless_of_env_or_reading(monkeypatch, probes):
    """RLIMIT_AS capped VIRTUAL address space, which collided with the device driver's own
    mesh-open/TLB-window mappings -- see the module docstring for the confirmed A/B rerun. The
    function must return None under every input now, not just the old escape-hatch/no-reading cases,
    so every existing `preexec_fn=memory_cap_preexec_fn()` call site keeps working unchanged."""
    monkeypatch.setattr(probes, "available_memory_gb", lambda: 64.0)
    assert probes.memory_cap_preexec_fn() is None
    assert probes.memory_cap_preexec_fn(margin_gb=10.0) is None
    monkeypatch.setattr(probes, "available_memory_gb", lambda: None)
    assert probes.memory_cap_preexec_fn() is None


# ------------------------------------------------------- detect a memory-cap hit, retry once, lower


class _FakeResult:
    def __init__(self, returncode, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_is_memory_cap_failure_reads_the_real_signatures(probes):
    assert probes.is_memory_cap_failure("Traceback...\nMemoryError\n")
    assert probes.is_memory_cap_failure("terminate called after throwing an instance of 'std::bad_alloc'")
    assert probes.is_memory_cap_failure("RuntimeError: [enforce fail] Cannot allocate memory")
    assert not probes.is_memory_cap_failure("AssertionError: PCC 0.71 < 0.95")
    assert not probes.is_memory_cap_failure("")


def test_a_clean_success_is_not_retried(monkeypatch, probes):
    # A HEALTHY BOX, ON PURPOSE: this isolates the REACTIVE retry from the PROACTIVE pre-launch
    # check below (should_use_low_mem_reference) -- a box actually this loaded would legitimately
    # set the signal before ever calling _run, which is a different behavior with its own tests.
    monkeypatch.setattr(probes, "available_memory_gb", lambda: 999.0)
    calls = {"n": 0}

    def _run():
        calls["n"] += 1
        return _FakeResult(0, "all good")

    env = {}
    r = probes.run_with_low_memory_fallback(_run, env)
    assert calls["n"] == 1, "a successful run must not be retried"
    assert probes.LOW_MEM_REFERENCE_ENV not in env
    assert r.returncode == 0


def test_a_non_memory_failure_is_not_retried(monkeypatch, probes):
    """Retrying a PCC-threshold failure or any other ordinary crash under a different dtype would
    not fix it and would waste a full-depth build for nothing."""
    monkeypatch.setattr(probes, "available_memory_gb", lambda: 999.0)  # see test above
    calls = {"n": 0}

    def _run():
        calls["n"] += 1
        return _FakeResult(1, "AssertionError: PCC 0.71 < 0.95")

    env = {}
    probes.run_with_low_memory_fallback(_run, env)
    assert calls["n"] == 1, "a non-memory failure must not trigger the fallback retry"
    assert probes.LOW_MEM_REFERENCE_ENV not in env


def test_a_memory_cap_failure_retries_once_with_the_signal_set(probes):
    calls = {"n": 0}

    def _run():
        calls["n"] += 1
        return _FakeResult(1, "", "MemoryError: out of memory")

    env = {}
    probes.run_with_low_memory_fallback(_run, env)
    assert calls["n"] == 2, "a memory-cap failure should be retried exactly once"
    assert env[probes.LOW_MEM_REFERENCE_ENV] == "1", "the retry did not set the standard fallback signal"


def test_a_second_memory_cap_failure_is_not_retried_again(probes):
    """Bounded, like every other retry in this tool: if the model does not honour the signal (or
    genuinely cannot fit even at low memory), give up rather than loop forever."""
    calls = {"n": 0}

    def _run():
        calls["n"] += 1
        return _FakeResult(1, "", "MemoryError: still out of memory")

    env = {}
    r = probes.run_with_low_memory_fallback(_run, env)
    assert calls["n"] == 2, "must not retry a second time"
    assert r.returncode != 0


# ------------------------------------------ decide BEFORE launch, not only after a clean failure
#
# RUN, 2026-09-12. Retrying after a clean failure cannot always fire: nvidia_nemotron_3_5_lightning_
# 30b_a3b_bf16's fp32 reference build hit ~117-120 GB RSS TWICE with ~110 GB reported available at
# launch, and the kernel OOM-killer killed the whole session's cgroup both times -- including the
# orchestrator that would have retried. should_use_low_mem_reference lets a caller decide BEFORE the
# first attempt, so a box already too loaded for the full-precision peak never has to risk it.


def test_should_use_low_mem_reference_below_the_margin(monkeypatch, probes):
    monkeypatch.setattr(probes, "available_memory_gb", lambda: 110.0)  # the exact reading measured
    assert probes.should_use_low_mem_reference() is True


def test_should_use_low_mem_reference_above_the_margin(monkeypatch, probes):
    monkeypatch.setattr(probes, "available_memory_gb", lambda: 200.0)
    assert probes.should_use_low_mem_reference() is False


def test_should_use_low_mem_reference_no_reading_is_not_a_reason_to_intervene(monkeypatch, probes):
    monkeypatch.setattr(probes, "available_memory_gb", lambda: None)
    assert probes.should_use_low_mem_reference() is False


def test_should_use_low_mem_reference_respects_the_escape_hatch(monkeypatch, probes):
    monkeypatch.setenv("PERF_MCP_DISABLE_MEM_CAP", "1")
    monkeypatch.setattr(probes, "available_memory_gb", lambda: 1.0)
    assert probes.should_use_low_mem_reference() is False


# ---------------------------------------------- sustained memory PRESSURE, a separate kill path
#
# RUN, 2026-09-12. systemd-oomd -- a userspace watchdog separate from the kernel's own OOM-killer,
# and invisible to a dmesg-only check -- killed an entire optimize run's session (10 processes,
# including the orchestrator) because /user.slice/.../user@1000.service's memory pressure held
# above 50% for over 20s. available_memory_gb() cannot see this coming: the box can show plenty
# free at LAUNCH and still build pressure minutes into a run.


def test_memory_pressure_percent_reads_a_real_value_or_none(probes):
    val = probes.memory_pressure_percent()
    assert val is None or val >= 0.0


def test_pressure_cgroup_path_falls_back_to_the_process_own_leaf(probes):
    """Not every login nests under user@<uid>.service (some sessions on this box sit in a sibling
    session-N.scope with no such ancestor, confirmed live) -- the walk must not just give up in
    that case. Exercised against the REAL /proc/self/cgroup rather than a mock: this process's own
    cgroup on this box IS one of the no-user@.service logins, so this is the real fallback path,
    not a simulated one."""
    path = probes._own_pressure_cgroup_path()
    assert path is not None, "must fall back to this process's own leaf cgroup, not give up"


def test_a_healthy_pressure_reading_does_not_warn(monkeypatch, probes, capsys):
    monkeypatch.setattr(probes, "memory_pressure_percent", lambda: 2.0)
    state = probes.memory_pressure_watch_new()
    probes.memory_pressure_watch_sample(state, "device work")
    assert state["last_report"] == 0.0
    assert "memory-pressure-watch" not in capsys.readouterr().err


def test_sustained_pressure_warns_once_per_report_window(monkeypatch, probes, capsys):
    monkeypatch.setattr(probes, "memory_pressure_percent", lambda: 45.0)
    state = probes.memory_pressure_watch_new()
    probes.memory_pressure_watch_sample(state, "device work")
    err = capsys.readouterr().err
    assert "memory-pressure-watch" in err and "45.0%" in err
    assert state["last_report"] != 0.0
    # a second sample inside the report window must not re-print
    probes.memory_pressure_watch_sample(state, "device work")
    assert capsys.readouterr().err == ""


def test_unreadable_pressure_is_not_a_reason_to_warn(monkeypatch, probes, capsys):
    monkeypatch.setattr(probes, "memory_pressure_percent", lambda: None)
    state = probes.memory_pressure_watch_new()
    probes.memory_pressure_watch_sample(state, "device work")
    assert state["last_report"] == 0.0
    assert capsys.readouterr().err == ""


def test_run_device_proc_samples_pressure_too():
    """Same funnel as the thermal/low-mem-reference gates: one call site, so this covers every
    device-touching subprocess run.py launches."""
    import inspect

    import models.experimental.perf_automation.cc_optimize.run as R

    src = inspect.getsource(R._run_device_proc)
    assert "memory_pressure_watch_new" in src and "memory_pressure_watch_sample" in src


def test_execute_samples_pressure_too(probes):
    """The OTHER streaming launcher (agent.probes._execute, used by perf_test_gen's generated-test
    validation) needs the same watch -- it is the exact path that had NEITHER gate before today."""
    import inspect

    src = inspect.getsource(probes._execute)
    assert "memory_pressure_watch_new" in src and "memory_pressure_watch_sample" in src


def test_a_loaded_box_sets_the_signal_before_the_first_attempt(monkeypatch, probes):
    """THE ACTUAL GAP: on a box already below the margin, the FIRST call must already see the
    signal -- not just a retry after it fails once for nothing."""
    monkeypatch.setattr(probes, "available_memory_gb", lambda: 110.0)
    seen_by_first_call = {}

    def _run():
        if not seen_by_first_call:
            seen_by_first_call.update(env)
        return _FakeResult(0, "all good")

    env = {}
    probes.run_with_low_memory_fallback(_run, env)
    assert (
        seen_by_first_call.get(probes.LOW_MEM_REFERENCE_ENV) == "1"
    ), "a box below the margin must set the signal before the FIRST attempt, not only on retry"


def test_a_healthy_box_never_sets_the_signal_proactively(monkeypatch, probes):
    monkeypatch.setattr(probes, "available_memory_gb", lambda: 999.0)
    env = {}
    probes.run_with_low_memory_fallback(lambda: _FakeResult(0, "all good"), env)
    assert probes.LOW_MEM_REFERENCE_ENV not in env


def test_an_explicitly_set_signal_is_left_alone(monkeypatch, probes):
    """A caller (or a previous retry) that already set the signal is not re-decided against."""
    monkeypatch.setattr(probes, "available_memory_gb", lambda: 999.0)
    env = {probes.LOW_MEM_REFERENCE_ENV: "1"}
    probes.run_with_low_memory_fallback(lambda: _FakeResult(0, "all good"), env)
    assert env[probes.LOW_MEM_REFERENCE_ENV] == "1"


# ------------------------------------------------------- every launch point wires the shared gate


def test_run_devices_proc_calls_it():
    """The gate is worthless anywhere else: it must run at the ONE place in cc_optimize.run every
    device-touching subprocess launches through, alongside the thermal gate it mirrors -- and the
    launch itself must carry the hard cap, not just the pre-launch wait."""
    import models.experimental.perf_automation.cc_optimize.run as R

    src = inspect.getsource(R._run_device_proc)
    i = src.index("_wait_for_thermal_headroom_before_device_work(")
    j = src.index("_wait_for_memory_headroom_before_device_work(")
    k = src.index("subprocess.Popen", j)
    assert j > i, "the memory gate is not wired in right after the thermal gate"
    assert k > j, "the gate does not run before the process launches"
    assert "preexec_fn=memory_cap_preexec_fn()" in src[k : k + 400], "the launch carries no hard cap"
    # AND the re-export actually delegates to the shared implementation, not a stale local copy.
    assert "agent.probes" in inspect.getsource(R._wait_for_memory_headroom_before_device_work)
    assert "agent.probes" in inspect.getsource(R._available_memory_gb)


def test_adaptive_run_calls_it_too():
    """The SIXTH launch point, found late: perf_mcp._adaptive_run drives the BEFORE/AFTER
    full-pipeline bookend, which explicitly builds the model at full uncapped depth -- the exact
    operation that OOM-killed every other full-depth launch point -- and had neither the wait nor
    the cap until this test was written. Same streaming-Popen shape as _run_device_proc, so the
    auto-retry stays out of scope here too, for the same reason."""
    import models.experimental.perf_automation.cc_optimize.perf_mcp as M

    src = inspect.getsource(M._adaptive_run)
    i = src.index("wait_for_memory_headroom_before_device_work(")
    k = src.index("_sp.Popen(", i)
    assert k > i, "the wait does not run before the process launches"
    assert "preexec_fn=memory_cap_preexec_fn()" in src[k : k + 400], "the launch carries no hard cap"


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
        assert "run_with_low_memory_fallback(" in src, "%s does not use the retry-with-fallback wrapper" % fn.__name__
        for r in runs:
            assert any(c < r for c in calls), "%s's subprocess.run has no preceding memory-gate call" % fn.__name__
            assert "preexec_fn=memory_cap_preexec_fn()" in src[r : r + 500], (
                "%s's launch carries no hard cap" % fn.__name__
            )


def test_check_pcc_calls_it():
    """check_pcc runs after EVERY lever attempt during the optimize loop -- the highest-frequency
    full-depth build in the tool, and the most likely explanation for a tight repeating OOM cadence."""
    import models.experimental.perf_automation.agent.pcc_runner as R

    src = inspect.getsource(R.run_pcc)
    i = src.index("wait_for_memory_headroom_before_device_work(")
    j = src.index("subprocess.run(")
    assert j > i, "check_pcc's subprocess.run is not preceded by the memory gate"
    assert "run_with_low_memory_fallback(" in src, "check_pcc does not use the retry-with-fallback wrapper"
    assert "preexec_fn=probes.memory_cap_preexec_fn()" in src[j : j + 500], "check_pcc's launch carries no hard cap"


def test_pcc_gate_gen_calls_it():
    import models.experimental.perf_automation.agent.pcc_gate_gen as G

    src = inspect.getsource(G._run_gate)
    i = src.index("wait_for_memory_headroom_before_device_work(")
    j = src.index("subprocess.run(")
    assert j > i, "_run_gate's subprocess.run is not preceded by the memory gate"
    assert "run_with_low_memory_fallback(" in src, "_run_gate does not use the retry-with-fallback wrapper"
    assert "preexec_fn=probes.memory_cap_preexec_fn()" in src[j : j + 500], "_run_gate's launch carries no hard cap"


def test_nemotron_pipeline_respects_the_low_memory_signal():
    """The one thing this generic mechanism cannot verify on its own: that at least the reference
    model this incident was found on actually reads the signal it is retried with."""
    import inspect

    from models.demos.nvidia_nemotron_3_5_lightning_30b_a3b_bf16.tt import pipeline as P

    src = inspect.getsource(P.build_pipeline)
    assert "PERF_MCP_LOW_MEM_REFERENCE" in src, "nemotron's build_pipeline does not honour the fallback signal"
    assert "torch.bfloat16" in src.split("load_reference")[0][-400:], "no bf16 branch precedes the reference load"
