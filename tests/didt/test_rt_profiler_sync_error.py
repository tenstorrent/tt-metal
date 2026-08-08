# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# Reads the sync error the real-time profiler publishes on every record while a didt power virus runs, so the clock
# mapping is measured against the AICLK steps a hot part actually takes. The other RT profiler stress tests load the
# record pipeline with a blank-kernel trace, which leaves the array idle.
#
# The workload is test_ff1_matmul.py::test_ff1_matmul, called directly rather than reimplemented: swapping in another
# didt workload is a one-line change here, and its shapes stay owned by its own file.
#
#   pytest tests/didt/test_rt_profiler_sync_error.py -k "8chips" --didt-workload-iterations 1000

import statistics
import threading

from loguru import logger
import pytest

from tests.didt.test_ff1_matmul import test_ff1_matmul as run_didt_power_workload
import ttnn

NUM_DEVICES = ttnn.distributed.get_num_devices()
MESH_X = NUM_DEVICES if NUM_DEVICES <= 8 else 8
MESH_Y = 1 if NUM_DEVICES <= 8 else int(NUM_DEVICES / MESH_X)

# The C++ sync suite (test_realtime_profiler_sync.cpp) bounds every claim at 15us with the array idle; the point of
# this test is that a power virus does not move the distribution.
MAX_SYNC_ERROR_P50_NS = 6_000
MAX_SYNC_ERROR_P90_NS = 10_000
MAX_SYNC_ERROR_P99_NS = 15_000

REPORT_INTERVAL_S = 5.0


class SyncErrorCollector:
    """Collects clock_sync.sync_error_ns from every delivered record and reports the running distribution. The callback
    runs on its own thread, so the samples are taken under a lock; unregistering on exit is what makes the final read
    safe."""

    def __init__(self, report_interval_s=REPORT_INTERVAL_S):
        self.errors = []
        self.dropped = 0
        self.report_interval_s = report_interval_s
        self._lock = threading.Lock()
        self._handle = None
        self._stop = threading.Event()
        self._reporter = None

    def __enter__(self):
        self._handle = ttnn.device.RegisterProgramRealtimeProfilerCallback(self._on_batch)
        self._reporter = threading.Thread(target=self._report_loop, daemon=True)
        self._reporter.start()
        return self

    def __exit__(self, *exc_info):
        self._stop.set()
        self._reporter.join()
        ttnn.device.UnregisterProgramRealtimeProfilerCallback(self._handle)

    def _on_batch(self, batch):
        with self._lock:
            self.dropped += batch.dropped
            self.errors.extend(record.clock_sync.sync_error_ns for record in batch.records)

    def _report_loop(self):
        while not self._stop.wait(self.report_interval_s):
            logger.info(self.summary())

    def snapshot(self):
        with self._lock:
            return sorted(self.errors), self.dropped

    def summary(self):
        ordered, dropped = self.snapshot()
        if not ordered:
            return "sync error: no records yet"

        def pct(p):
            return ordered[min(int(p * len(ordered)), len(ordered) - 1)]

        return (
            f"sync error over {len(ordered)} records ({dropped} dropped): "
            f"min={ordered[0]}ns p50={pct(0.50)}ns p90={pct(0.90)}ns p99={pct(0.99)}ns "
            f"max={ordered[-1]}ns mean={statistics.mean(ordered):.0f}ns"
        )


@pytest.mark.parametrize(
    "mesh_device",
    [
        pytest.param(1, id="1chips"),
        pytest.param(2, id="2chips"),
        pytest.param(8, id="8chips"),
        pytest.param((8, 4), id="galaxy"),
        pytest.param((MESH_X, MESH_Y), id="all"),  # run on all available devices
    ],
    indirect=["mesh_device"],
)
def test_rt_profiler_sync_error_under_didt_load(
    mesh_device,
    didt_workload_iterations,
    determinism_check_interval,
):
    if not ttnn.device.IsProgramRealtimeProfilerActive():
        pytest.skip("Real-time profiler not active on this configuration")

    # op_test_base logs per iteration and per device sync, which at these iteration counts buries everything else.
    logger.disable("tests.didt.op_test_base")
    try:
        with SyncErrorCollector() as collector:
            run_didt_power_workload(
                mesh_device,
                gelu=False,
                math_fidelity=ttnn.MathFidelity.LoFi,
                didt_workload_iterations=didt_workload_iterations,
                determinism_check_interval=determinism_check_interval,
            )
    finally:
        logger.enable("tests.didt.op_test_base")

    ordered, _ = collector.snapshot()
    assert ordered, "No real-time profiler records were delivered during the workload"
    logger.info(collector.summary())

    def pct(p):
        return ordered[min(int(p * len(ordered)), len(ordered) - 1)]

    assert ordered[0] > 0, "sync_error should be populated once the clock is anchored"
    assert pct(0.50) < MAX_SYNC_ERROR_P50_NS, "median sync error too high under compute load"
    assert pct(0.90) < MAX_SYNC_ERROR_P90_NS, "p90 sync error too high under compute load"
    assert pct(0.99) < MAX_SYNC_ERROR_P99_NS, "tail sync error too high under compute load"
