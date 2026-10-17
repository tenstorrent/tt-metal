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
    monkeypatch.setenv(probes._SUPPORT_COUNT_ENV, "8")  # capacity interval 8 // 4 = 2


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


def test_a_big_buffer_still_reads_after_every_stage(profiling, monkeypatch):
    monkeypatch.delenv(probes._SUPPORT_COUNT_ENV)  # the default buffer: interval far above 5 ops
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


@pytest.fixture
def small_buffer(monkeypatch):
    # a buffer of 8 programs: the session reads every 8 // 4 = 2 ops
    monkeypatch.setenv(probes._SUPPORT_COUNT_ENV, "8")


def test_the_session_interval_is_the_buffers_capacity(monkeypatch):
    monkeypatch.delenv(probes._SUPPORT_COUNT_ENV, raising=False)
    assert pd.capacity_cadence() == probes._DEFAULT_SUPPORT_COUNT // pd._PROGRAMS_PER_OP_HEADROOM
    monkeypatch.setenv(probes._SUPPORT_COUNT_ENV, "8000")  # what one heal grows it to
    assert pd.capacity_cadence() == 8000 // pd._PROGRAMS_PER_OP_HEADROOM


def test_the_session_drain_learns_the_device_from_the_ops_and_reads_from_the_first(
    profiling, small_buffer, monkeypatch
):
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


def test_inside_the_session_the_stage_pass_reads_finer_and_nothing_is_read_twice(profiling, monkeypatch):
    monkeypatch.setenv(probes._SUPPORT_COUNT_ENV, "40")  # session every 10 ops
    calls, mesh, reads = [], _Mesh(), []
    t = _fake_ttnn(calls)
    t.ReadDeviceProfiler = lambda dev: reads.append("read")
    monkeypatch.setitem(sys.modules, "ttnn", t)
    pd.pytest_configure(None)
    try:
        t.add(_Tensor(mesh))
        session_add = t.add
        with pd.ProfilerDrain(t, mesh, final_read=False):  # the stage pass: every 2 ops
            assert t.add is not session_add, "the pass wraps on top of the session"
            for _ in range(12):
                t.add()
        assert t.add is session_add, "leaving the pass puts the session's wrappers back"
        # due reads land before the next op: before ops 3, 5, 7, 9 and 11 of the pass
        assert len(reads) == 5, "the pass's cadence, and the session never repeated one of its reads"
        # the session counted the pass's last 2 ops since its last read: 8 more reach 10, not yet read
        for _ in range(8):
            t.add()
        assert len(reads) == 5, "back at the session's interval"
        t.add()
        assert len(reads) == 6, "read before the op that follows a full interval"
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
    cmd = probes.build_tracy_command("t.py", None, "/tmp/out")
    assert [cmd[k + 1] for k, a in enumerate(cmd) if a == "-p" and k > 5] == ["no:timeout"], "no drain plugin"


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
    assert seen and seen[0][seen[0].index("x.profiler_drain") - 1] == "-p"
    assert "--dump-device-data-mid-run" in seen[0]
    assert "--disable-device-data-push-to-tracy" in seen[0], "device markers stay out of tracy-capture"


def test_a_read_by_the_tests_own_wrapper_is_not_repeated(profiling, small_buffer, monkeypatch):
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


def test_the_generated_tests_own_wrapper_still_finds_every_op(profiling, monkeypatch):
    """The generated test selects ops by type(op).__name__ == "FastOperation" (perf_test_gen); under
    the session drain it must still find all of them, or its forward loses its drain."""
    calls = []
    t = _fake_ttnn(calls)
    monkeypatch.setitem(sys.modules, "ttnn", t)
    before = sorted(n for n in dir(t) if type(getattr(t, n)).__name__ == "FastOperation")
    pd.pytest_configure(None)
    try:
        after = sorted(n for n in dir(t) if type(getattr(t, n)).__name__ == "FastOperation")
        assert after == before == ["add"]
        assert t.add() == "r" and calls == ["op"]
    finally:
        pd.pytest_unconfigure(None)


def test_the_drained_run_releases_each_read():
    cmd = probes.build_tracy_command("t.py", None, "/tmp/out", plugins=("x",), mid_run_dump=True)
    assert cmd.index("--dump-device-data-mid-run") < cmd.index("-m", 3), "a tracy option, before -m pytest"
    assert "--dump-device-data-mid-run" not in probes.build_tracy_command("t.py", None, "/tmp/out")


def test_a_half_built_proxy_does_not_recurse():
    import copy

    proxy = pd.FastOperation(lambda: "r", lambda: "r")
    assert copy.copy(proxy)() == "r"
    with pytest.raises(AttributeError):  # allow-pytest.raises: no expect_error fixture
        pd.FastOperation.__new__(pd.FastOperation).anything


def test_device_data_goes_to_tracy_unless_asked_not_to():
    assert "--disable-device-data-push-to-tracy" not in probes.build_tracy_command("t.py", None, "/tmp/out")
    cmd = probes.build_tracy_command("t.py", None, "/tmp/out", push_device_to_tracy=False)
    assert cmd.index("--disable-device-data-push-to-tracy") < cmd.index("-m", 3)
