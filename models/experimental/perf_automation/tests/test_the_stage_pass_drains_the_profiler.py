"""The per-stage pass drains the device profiler as it runs, like the test's own op wrapper.

WH Galaxy, 2026-09-28: the pass ran all five stages before the test installed its wrapper, so the
first read after it found every buffer of 32 chips full (11,520 drop sites) on the FIRST attempt of
a freshly reset board. These pin: under the profiling env the pass reads every
<TT_PERF_FLUSH_EVERY> ops and after every stage, restores the ops it wrapped, and does nothing at
all outside a profiling run.
"""

import subprocess
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
    # a due read lands before the next op; the read after each stage restarts the count
    assert calls == [
        "mark:stage:s0", "op", "op", "read", "op", "mark:stage:s0:end", "read",
        "mark:stage:s1", "op", "op", "mark:stage:s1:end", "read",
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


# -- the session-wide drain: from the profiled process's first op ---------------------------------

from pathlib import Path  # noqa: E402

from agent import profiler_drain as pd  # noqa: E402


class _Mesh:
    def get_num_devices(self):
        return 32


class _Tensor:
    def __init__(self, dev):
        self._dev = dev

    def device(self):
        return self._dev


def test_the_session_drain_learns_the_device_from_the_ops_and_reads_from_the_first(profiling, monkeypatch):
    calls, mesh = [], _Mesh()
    t = _fake_ttnn(calls)
    t.from_torch = FastOperation(calls)
    reads = []
    t.ReadDeviceProfiler = lambda dev: reads.append(dev)
    monkeypatch.setitem(sys.modules, "ttnn", t)
    pd.pytest_configure(None)
    try:
        t.from_torch("w", device=mesh)  # a weight upload: the device arrives as a kwarg
        for _ in range(4):
            t.add()
    finally:
        pd.pytest_unconfigure(None)
    assert reads == [mesh, mesh], "a read every 2 ops, from the very first, on the op's device"
    assert type(t.add).__name__ == "FastOperation", "the session's ops are put back"


def test_a_device_is_found_on_a_returned_tensor():
    mesh = _Mesh()
    assert pd._device_of((), {}, _Tensor(mesh)) is mesh
    assert pd._device_of((_Tensor(None),), {}, "r") is None


def test_the_stage_pass_does_not_wrap_twice_under_the_session(profiling, monkeypatch):
    calls = []
    t = _fake_ttnn(calls)
    monkeypatch.setitem(sys.modules, "ttnn", t)
    pd.pytest_configure(None)
    try:
        wrapped = t.add
        with pd.ProfilerDrain(t, object()) as inner:
            assert t.add is wrapped and not inner._orig
    finally:
        pd.pytest_unconfigure(None)


def test_outside_a_profiling_run_the_plugin_does_nothing(monkeypatch):
    for k in probes.PROFILING_ENV:
        monkeypatch.delenv(k, raising=False)
    calls = []
    t = _fake_ttnn(calls)
    monkeypatch.setitem(sys.modules, "ttnn", t)
    orig = t.add
    pd.pytest_configure(None)
    assert t.add is orig and pd._session is None


def test_the_profiled_command_loads_the_drain():
    root = Path(pd.__file__).resolve().parents[4]
    name = probes.profiler_drain_plugin(root)
    assert name and name.endswith(".agent.profiler_drain")
    # importable the way the profiled run imports it: from the tree root, nothing else on the path
    probe = "import importlib.util, sys; sys.exit(importlib.util.find_spec(%r) is None)" % name
    assert subprocess.run([sys.executable, "-c", probe], cwd=root, env={}).returncode == 0
    cmd = probes.build_tracy_command("t.py", "S128", "/tmp/out", plugins=(name,))
    i = cmd.index("pytest")
    assert cmd[i + 1 :].index("-p") < cmd[i + 1 :].index("t.py") and name in cmd
    assert cmd[-1] == "-sv"


def test_a_tool_outside_the_profiled_tree_adds_no_plugin(tmp_path):
    assert probes.profiler_drain_plugin(tmp_path) is None
    assert "-p" not in probes.build_tracy_command("t.py", None, "/tmp/out")[6:]


def test_make_run_profiled_passes_it(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(probes, "profiler_drain_plugin", lambda root: "x.profiler_drain")

    def execute(cmd, cwd, env, timeout_s, log_path):
        seen.append(cmd)
        raise probes.TracyRunError("stop here")

    rp = probes.make_run_profiled(
        tmp_path,
        "t.py",
        "S128",
        execute=execute,
        collect_runner=lambda cmd, cwd, **k: subprocess.CompletedProcess(
            cmd, 0, "t.py::test_full[S128]\n1 test collected\n", ""
        ),
    )
    with pytest.raises(probes.TracyRunError):  # allow-pytest.raises: no expect_error fixture
        rp("e2e", 1, 128, tmp_path / "profiles", 0)
    assert seen and seen[0][seen[0].index("-p", 7) + 1] == "x.profiler_drain"


def test_a_read_by_the_tests_own_wrapper_is_not_repeated(profiling, monkeypatch):
    calls, mesh = [], _Mesh()
    t = _fake_ttnn(calls)
    reads = []
    t.ReadDeviceProfiler = lambda dev: reads.append(dev)
    monkeypatch.setitem(sys.modules, "ttnn", t)
    pd.pytest_configure(None)
    try:
        t.add(_Tensor(mesh))
        for _ in range(6):  # the test's wrapper: an op, and a read of its own every 2
            t.add()
            if len(calls) % 2 == 0:
                t.ReadDeviceProfiler(mesh)
    finally:
        pd.pytest_unconfigure(None)
    assert len(reads) == 3, "only the test's reads: the session drain found each interval already read"
    assert t.ReadDeviceProfiler is not None and not isinstance(t.ReadDeviceProfiler, FastOperation)
