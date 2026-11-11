"""failfast (rc1d robustness, tt/failfast.py): device-free tests by exception injection.

The real failure: the reader thread spins in ttnn.event_synchronize forever after tt-metal's completion-queue reader
threw TIMEOUT; ttnn.synchronize_device would rethrow it. Here the stuck wait is a blocked threading.Event inside
failfast.watch(), the probe is an injected function (raises = dead device, blocks = unresponsive, returns = busy device),
and the fatal action is captured instead of os._exit, except in the subprocess test, which checks the real os._exit and
that a parent waiting on the child's sentinel (vLLM's MultiprocWorkerMonitor mechanism) sees the death within seconds.

  pytest --noconftest -s models/demos/blackhole/qwen36/tests/test_failfast.py     (no device, no ttnn import)
"""
import importlib.util
import multiprocessing
import multiprocessing.connection
import os
import threading
import time

import pytest

_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tt", "failfast.py")


def _load():
    spec = importlib.util.spec_from_file_location("qwen36_failfast_test", _PATH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture
def ff(monkeypatch):
    m = _load()
    monkeypatch.setattr(m, "POLL_S", 0.05)
    monkeypatch.setattr(m, "GRACE_S", 0.3)
    monkeypatch.setenv("TT_METAL_OPERATION_TIMEOUT_SECONDS", "5.0")
    monkeypatch.setenv("QWEN36_FAILFAST_STALL_S", "0.5")
    monkeypatch.setenv("QWEN36_FAILFAST_HARD_S", "1.0")
    monkeypatch.delenv("QWEN36_FAILFAST", raising=False)
    wd = m._Watchdog()
    wd.exits, wd.probes = [], []
    wd.exit_fn = lambda code: wd.exits.append((code, time.monotonic()))
    monkeypatch.setattr(m, "_WD", wd)
    return m, wd


def _stuck_wait(m, release):
    """A reader thread inside watch() whose 'event wait' blocks until ``release`` is set."""
    def body():
        with m.watch("mesh", "fused step read"):
            release.wait()

    th = threading.Thread(target=body, daemon=True)
    th.start()
    return th


def _wait_for(pred, timeout):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if pred():
            return time.monotonic() - t0
        time.sleep(0.01)
    return None


def test_normal_waits_untouched(ff):
    m, wd = ff
    wd.probe_fn = lambda mesh: wd.probes.append(mesh)
    for _ in range(200):  # ms-scale waits, as in serving
        with m.watch("mesh", "fused step read"):
            time.sleep(0.001)
    time.sleep(0.8)
    assert wd.probes == [] and wd.exits == [] and wd.waits == {}
    n = 20000
    t0 = time.perf_counter()
    for _ in range(n):
        with m.watch("mesh", "x"):
            pass
    print(f"[failfast-test] watch() overhead {(time.perf_counter() - t0) / n * 1e6:.2f} us per wait")


def test_dead_device_exits_fast(ff):
    """TT_THROW TIMEOUT stored by tt-metal's reader thread: the probe (synchronize_device) rethrows it -> exit."""
    m, wd = ff

    def probe(mesh):
        wd.probes.append(mesh)
        raise RuntimeError("TT_THROW: TIMEOUT: device timeout, potential hang detected, the device is unrecoverable")

    wd.probe_fn = probe
    release = threading.Event()
    t0 = time.monotonic()
    _stuck_wait(m, release)
    dt = _wait_for(lambda: wd.exits, 5.0)
    release.set()
    assert dt is not None, "no exit within 5 s"
    assert wd.exits[0][0] == m.EXIT_CODE and wd.probes == ["mesh"]
    print(f"[failfast-test] dead device: exit after {wd.exits[0][1] - t0:.2f} s (stall threshold 0.5 s)")


def test_unresponsive_probe_exits(ff):
    m, wd = ff
    block = threading.Event()
    wd.probe_fn = lambda mesh: block.wait()
    release = threading.Event()
    t0 = time.monotonic()
    _stuck_wait(m, release)
    dt = _wait_for(lambda: wd.exits, 5.0)
    release.set()
    block.set()
    assert dt is not None and wd.exits[0][0] == m.EXIT_CODE
    assert 1.4 <= wd.exits[0][1] - t0 < 3.0, wd.exits[0][1] - t0  # stall 0.5 + hard 1.0


def test_busy_device_no_exit(ff):
    """A long but healthy wait: the probe returns and the wait finishes -> no exit."""
    m, wd = ff
    release = threading.Event()

    def probe(mesh):
        wd.probes.append(mesh)
        release.set()  # the device drains: the event completes

    wd.probe_fn = probe
    th = _stuck_wait(m, release)
    th.join(5.0)
    time.sleep(0.6)
    assert wd.probes == ["mesh"] and wd.exits == []


def test_event_never_completes_exits(ff):
    """The probe returns (device idle) but the wait stays stuck: the event can never complete -> exit."""
    m, wd = ff
    wd.probe_fn = lambda mesh: None
    release = threading.Event()
    _stuck_wait(m, release)
    dt = _wait_for(lambda: wd.exits, 5.0)
    release.set()
    assert dt is not None and wd.exits[0][0] == m.EXIT_CODE


@pytest.mark.parametrize("env", [{"QWEN36_FAILFAST": "report"}, {"TT_METAL_OPERATION_TIMEOUT_SECONDS": "0"}])
def test_report_mode_never_probes_or_exits(ff, monkeypatch, env):
    m, wd = ff
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    wd.probe_fn = lambda mesh: wd.probes.append(mesh)
    release = threading.Event()
    _stuck_wait(m, release)
    time.sleep(1.5)
    release.set()
    assert wd.probes == [] and wd.exits == []


def test_off_mode_no_thread(ff, monkeypatch):
    m, wd = ff
    monkeypatch.setenv("QWEN36_FAILFAST", "0")
    with m.watch("mesh", "x"):
        pass
    assert wd.thread is None and wd.waits == {}


def _child(path):
    """A stand-in TT worker: real failfast, real os._exit; the reader thread is stuck in its 'event wait'."""
    os.environ.update(TT_METAL_OPERATION_TIMEOUT_SECONDS="5.0", QWEN36_FAILFAST_STALL_S="1.0", QWEN36_FAILFAST_HARD_S="5")
    spec = importlib.util.spec_from_file_location("qwen36_failfast_child", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    def probe(mesh):
        raise RuntimeError("TT_THROW: TIMEOUT: device timeout, potential hang detected, the device is unrecoverable")

    m._WD.probe_fn = probe
    stuck = threading.Event()
    with m.watch("mesh", "fused step read"):
        stuck.wait()  # never returns: the process must die from the watchdog


def test_worker_process_dies_and_monitor_sees_it():
    ctx = multiprocessing.get_context("spawn")
    p = ctx.Process(target=_child, args=(_PATH,), name="VllmWorker-0")
    t0 = time.monotonic()
    p.start()
    died = multiprocessing.connection.wait([p.sentinel], timeout=30)  # vLLM MultiprocWorkerMonitor does exactly this
    dt = time.monotonic() - t0
    p.join(5)
    assert died, "worker did not exit within 30 s"
    print(f"[failfast-test] worker process exit code {p.exitcode} seen by the monitor after {dt:.2f} s "
          f"(spawn + import + stall 1.0 s + poll)")
    assert p.exitcode == 70 and dt < 15
