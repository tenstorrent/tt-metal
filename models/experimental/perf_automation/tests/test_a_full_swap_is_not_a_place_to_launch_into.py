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

THE HEADROOM WAIT AND THE HARD CAP ANSWER TWO DIFFERENT QUESTIONS. The third incident is why both
exist: the wait only asks "is the box already too full to start something new" -- a question about
the LOT, answerable before any car pulls in. It has no opinion once launched, because it cannot
know how big the car will get; that is model- and call-specific, and this launch point is generic
across every model. The answer is not a better prediction -- it is not needing one. memory_cap_
preexec_fn puts a hard ceiling on the CHILD's own address space (RLIMIT_AS), sized off whatever is
available at the moment of launch, so a subprocess that outgrows that ceiling hits its own allocator
(an ordinary MemoryError/bad_alloc, the same shape of failure every caller here already treats as a
crashed measurement) instead of the kernel picking a victim from the whole machine.

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


# ------------------------------------------------------------------- the hard cap on the child itself


def test_the_cap_is_disabled_by_the_escape_hatch(monkeypatch, probes):
    monkeypatch.setenv("PERF_MCP_DISABLE_MEM_CAP", "1")
    monkeypatch.setattr(probes, "available_memory_gb", lambda: 64.0)
    assert probes.memory_cap_preexec_fn() is None


def test_no_cap_without_a_reading(monkeypatch, probes):
    """Same rule as the rest of the safety gates: a sensor that cannot be read is not a reason to
    guess a number and cap the work on a fabricated basis."""
    monkeypatch.delenv("PERF_MCP_DISABLE_MEM_CAP", raising=False)
    monkeypatch.setattr(probes, "available_memory_gb", lambda: None)
    assert probes.memory_cap_preexec_fn() is None


def test_it_returns_a_callable_sized_below_available(monkeypatch, probes):
    monkeypatch.delenv("PERF_MCP_DISABLE_MEM_CAP", raising=False)
    monkeypatch.setattr(probes, "available_memory_gb", lambda: 64.0)
    fn = probes.memory_cap_preexec_fn(margin_gb=10.0)
    assert callable(fn)


def test_a_callable_that_cannot_set_the_limit_does_not_raise(monkeypatch, probes):
    """BEST-EFFORT, LIKE EVERY OTHER GATE HERE: a platform without RLIMIT_AS, or a setrlimit call
    that fails for any reason, must not take the subprocess down before it even starts."""
    monkeypatch.delenv("PERF_MCP_DISABLE_MEM_CAP", raising=False)
    monkeypatch.setattr(probes, "available_memory_gb", lambda: 64.0)
    fn = probes.memory_cap_preexec_fn()
    import resource as _resource

    monkeypatch.setattr(_resource, "setrlimit", lambda *a: (_ for _ in ()).throw(OSError("no permission")))
    fn()  # must not raise


def test_the_cap_actually_constrains_a_real_child_process():
    """The mechanism itself, not just the sizing arithmetic: a child given a small RLIMIT_AS must
    fail to allocate past it. This is the one thing a mock cannot stand in for -- it is exactly the
    gap the third incident exposed (a healthy headroom reading said nothing about what the child
    would do once running)."""
    import resource
    import subprocess
    import sys

    cap_bytes = 200 * 1024 * 1024  # 200 MB -- enough to start Python, not enough for this allocation

    def _preexec():
        resource.setrlimit(resource.RLIMIT_AS, (cap_bytes, cap_bytes))

    script = "import sys; b = bytearray(2 * 1024 * 1024 * 1024); sys.exit(0)"  # 2 GB, over the cap
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
        preexec_fn=_preexec,
    )
    assert proc.returncode != 0, "a 2 GB allocation under a 200 MB cap should not have succeeded"


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


def test_a_clean_success_is_not_retried(probes):
    calls = {"n": 0}

    def _run():
        calls["n"] += 1
        return _FakeResult(0, "all good")

    env = {}
    r = probes.run_with_low_memory_fallback(_run, env)
    assert calls["n"] == 1, "a successful run must not be retried"
    assert probes.LOW_MEM_REFERENCE_ENV not in env
    assert r.returncode == 0


def test_a_non_memory_failure_is_not_retried(probes):
    """Retrying a PCC-threshold failure or any other ordinary crash under a different dtype would
    not fix it and would waste a full-depth build for nothing."""
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
