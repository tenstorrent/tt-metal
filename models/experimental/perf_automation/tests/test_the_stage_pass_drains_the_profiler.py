"""The per-stage pass drains the device profiler as it runs, like the test's own op wrapper.

WH Galaxy, 2026-09-28: the pass ran all five stages before the test installed its wrapper, so the
first read after it found every buffer of 32 chips full (11,520 drop sites) on the FIRST attempt of
a freshly reset board. These pin: under the profiling env the pass reads every
<TT_PERF_FLUSH_EVERY> ops and after every stage, restores the ops it wrapped, and does nothing at
all outside a profiling run.
"""

import sys
import types

import pytest

from agent import probes
from agent import stage_marks as sm


class FastOperation:
    def __init__(self, sink):
        self._sink = sink

    def __call__(self, *a, **k):
        self._sink.append("op")
        return "r"


def _fake_ttnn(calls):
    t = types.ModuleType("ttnn")
    sub = types.ModuleType("ttnn.somewhere")
    t.somewhere = sub
    t.add = FastOperation(calls)
    sub.mul = FastOperation(calls)
    t.not_an_op = lambda: None
    t.ReadDeviceProfiler = lambda dev: calls.append("read")
    t.synchronize_device = lambda dev: None
    return t


class _Stage:
    def __init__(self, name, fn):
        self.name, self.step = name, fn


@pytest.fixture
def profiling(monkeypatch):
    for k, v in probes.PROFILING_ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv(probes.PERF_FLUSH_EVERY_ENV, "2")


def _run(monkeypatch, ops_per_stage=(3, 2)):
    calls = []
    t = _fake_ttnn(calls)
    monkeypatch.setitem(sys.modules, "ttnn", t)
    monkeypatch.setattr(sm, "signpost", lambda name: calls.append("mark:" + name))
    orig = (t.add, t.somewhere.mul)

    def stage(n):
        def step():
            for i in range(n):
                (t.add if i % 2 else t.somewhere.mul)()

        return step

    adapter = types.SimpleNamespace(stages=[_Stage("s%d" % i, stage(n)) for i, n in enumerate(ops_per_stage)])
    assert sm.mark_stages(adapter, object()) == len(ops_per_stage)
    return calls, t, orig


def test_the_pass_reads_at_the_tests_cadence_and_after_every_stage(profiling, monkeypatch):
    calls, _, _ = _run(monkeypatch)
    # s0: op op READ op | end READ ; s1: op READ op | end READ ; exit READ
    assert calls == [
        "mark:stage:s0", "op", "op", "read", "op", "mark:stage:s0:end", "read",
        "mark:stage:s1", "op", "read", "op", "mark:stage:s1:end", "read",
        "read",
    ]  # fmt: skip


def test_every_wrapped_op_is_put_back(profiling, monkeypatch):
    _, t, orig = _run(monkeypatch)
    assert (t.add, t.somewhere.mul) == orig


def test_outside_a_profiling_run_nothing_is_read_or_wrapped(monkeypatch):
    for k in probes.PROFILING_ENV:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv(probes.PERF_FLUSH_EVERY_ENV, "2")
    calls, t, orig = _run(monkeypatch)
    assert "read" not in calls and (t.add, t.somewhere.mul) == orig


def test_without_a_cadence_it_still_reads_after_every_stage(profiling, monkeypatch):
    monkeypatch.delenv(probes.PERF_FLUSH_EVERY_ENV)
    calls, _, _ = _run(monkeypatch)
    assert calls.count("read") == 3  # one per stage + one on exit


def test_a_failing_read_costs_nothing(profiling, monkeypatch):
    calls = []
    t = _fake_ttnn(calls)

    def boom(dev):
        raise RuntimeError("Event Synchronization is not supported during trace capture")

    t.ReadDeviceProfiler = boom
    monkeypatch.setitem(sys.modules, "ttnn", t)
    monkeypatch.setattr(sm, "signpost", lambda name: None)
    adapter = types.SimpleNamespace(stages=[_Stage("s", lambda: [t.add() for _ in range(4)])])
    assert sm.mark_stages(adapter, object()) == 1
