# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only tests of tests/measure: band math, estimators, the report's failure order, timing loops and the traced
step's lifecycle, with fake steps and a fake trace controller (no device)."""

import pytest

from models.common.ci_failure import PerfGateFailure
from models.demos.deepseek_v3_d_p.tests.measure import timing, trace
from models.demos.deepseek_v3_d_p.tests.measure.gate import PerfReport, PerfTarget, band_failure
from models.demos.deepseek_v3_d_p.tests.measure.timing import Probe, Samples, time_untraced
from models.demos.deepseek_v3_d_p.tests.measure.trace import TracedStep, time_traced


@pytest.fixture
def no_device(monkeypatch):
    """Device syncs are no-ops and record themselves."""
    calls = []
    monkeypatch.setattr(trace.ttnn, "synchronize_device", lambda *_: calls.append("sync"))
    return calls


@pytest.fixture
def clock(monkeypatch):
    """perf_counter advances 1 s per read."""
    now = [0.0]

    def tick():
        now[0] += 1.0
        return now[0]

    monkeypatch.setattr(timing.time, "perf_counter", tick)
    monkeypatch.setattr(trace.time, "perf_counter", tick)
    monkeypatch.setattr(timing, "current_cpu", lambda: 7)


def test_band_failure():
    target = PerfTarget(10.0, 0.05)
    assert band_failure("traced", 10.4, target) is None
    assert "slower than the target" in band_failure("traced", 10.6, target)
    assert "re-centre it" in band_failure("traced", 9.4, target)
    assert band_failure("traced", 99.0, PerfTarget(None, 0.05)) is None


def test_samples_estimators(expect_error):
    assert Samples([3.0, 1.0, 2.0]).estimate == 1.0
    assert Samples([3.0, 1.0, 2.0], "median").estimate == 2.0
    with expect_error(ValueError, "estimator"):
        Samples([1.0], "mean")


def test_report_correctness_before_band_and_skip(capsys, expect_error):
    emitted = []
    report = PerfReport("case", emitted.append)
    report.add("untraced", Samples([20.0]), PerfTarget(10.0, 0.05))
    report.add("traced", 5.0, PerfTarget(None, 0.03))
    assert [r.mode for r in emitted] == ["untraced", "traced"] and emitted[0].within is False
    assert "case untraced: 20.000 ms (max 20.000); target 10.0 ms ±5%; samples 20.000" in capsys.readouterr().out
    with expect_error(AssertionError, "traced output differs") as e:
        report.finish(["traced output differs"])
    assert not isinstance(e.value, PerfGateFailure) and "slower than the target" in str(e.value)
    with expect_error(PerfGateFailure, "slower than the target"):
        report.finish()


def test_report_skips_unmeasured_only_after_bands_pass(expect_error):
    report = PerfReport("case")
    report.add("untraced", Samples([10.0]), PerfTarget(10.0, 0.05))
    report.add("traced", Samples([5.0]), PerfTarget(None, 0.03))
    with expect_error(pytest.skip.Exception, "traced 5.000 ms"):
        report.finish()
    passing = PerfReport("case")
    passing.add("untraced", Samples([10.0]), PerfTarget(10.0, 0.05))
    passing.finish()


def test_time_untraced(clock):
    events = []

    class Recorder(Probe):
        def before(self):
            events.append("before")

        def enqueued(self):
            events.append("enqueued")

        def after(self):
            events.append("after")

    calls = []

    def step():
        calls.append("step")
        return f"out{len(calls)}"

    samples, kept = time_untraced(
        step,
        lambda: events.append("sync"),
        warmup=2,
        samples=3,
        calls_per_sample=2,
        prepare=lambda: events.append("prepare"),
        probe=Recorder(),
        keep_last=True,
    )
    assert len(calls) == 2 + 3 * 2
    assert kept == "out8"  # the last sample's last output
    # reads at t0, after the calls and after the sync, 1 s apart: (3 - 1) s over 2 calls
    assert samples.values == [1000.0] * 3 and samples.enqueue_ms == [500.0] * 3 and samples.cpus == [7] * 3
    first_sample = events[events.index("before") - 2 :][:6]
    assert first_sample == ["prepare", "sync", "before", "enqueued", "sync", "after"]


class FakeController:
    def __init__(self, log, fail_end=False):
        self.log, self.fail_end, self.capturing = log, fail_end, False

    def begin_capture(self):
        self.log.append("begin")
        self.capturing = True

    def end_capture(self):
        self.log.append("end")
        if self.fail_end:
            raise RuntimeError("end failed")
        self.capturing = False

    def replay(self, sync=True):
        self.log.append(f"replay sync={sync}")

    def release(self):
        assert not self.capturing
        self.log.append("release")


def _traced(log, step, **kw):
    controller = FakeController(log, kw.pop("fail_end", False))
    return TracedStep(
        "mesh",
        step,
        attach=lambda c: log.append("attach"),
        detach=lambda: log.append("detach"),
        controller_factory=lambda _: controller,
        **kw,
    )


def test_traced_step_lifecycle(no_device):
    log = []
    with _traced(log, lambda: log.append("step") or "out", compile_runs=2) as t:
        t.capture(before_capture=lambda out: log.append(f"before_capture {out}"))
        assert t.output == "out"
        t.replay(sync=False)
    assert log == [
        "attach",
        "step",
        "step",
        "before_capture out",
        "begin",
        "step",
        "end",
        "replay sync=False",
        "release",
        "detach",
    ]


def test_traced_step_failed_capture_ends_and_releases(no_device, expect_error):
    log, n = [], [0]

    def step():
        n[0] += 1
        if n[0] == 2:  # the captured call
            raise ValueError("capture failed")
        return "out"

    with expect_error(ValueError, "capture failed"):
        with _traced(log, step) as t:
            t.capture()
    assert log == ["attach", "begin", "end", "release", "detach"]


def test_traced_step_stuck_capture_keeps_first_error(no_device, expect_error):
    log, n = [], [0]

    def step():
        n[0] += 1
        if n[0] == 2:
            raise ValueError("capture failed")
        return "out"

    with expect_error(ValueError, "capture failed"):
        with _traced(log, step, fail_end=True) as t:
            t.capture()
    assert "release" not in log and log[-1] == "detach"


def test_time_traced(no_device, clock):
    log = []
    with _traced(log, lambda: "out") as t:
        t.capture()
        samples = time_traced(t, samples=2, calls_per_sample=4, estimator="median")
    assert log.count("replay sync=False") == 8
    assert samples.values == [250.0, 250.0] and samples.estimate == 250.0


def test_first_estimator_and_units():
    assert Samples([3.0, 1.0], "first").estimate == 3.0
    assert "84.000 us outside [94.050, 103.950] (target 99.000 us ±5%)" in band_failure(
        "per call", 84.0, PerfTarget(99.0, 0.05), unit="us"
    )


def test_report_passes_unmeasured_when_report_only():
    report = PerfReport("case")
    report.add("traced", Samples([5.0]), PerfTarget(None, 0.03))
    report.finish(skip_unmeasured=False)


def test_traced_step_take_output_and_release_hook(no_device):
    log, released = [], []
    with _traced(log, lambda: ("out", "state"), release=released.append) as t:
        t.capture()
        assert t.take_output() == ("out", "state")
    assert released == [("out", "state")]  # the compile run's output only; the taken one is the caller's
