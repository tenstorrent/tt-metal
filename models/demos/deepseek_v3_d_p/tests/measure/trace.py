# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""A traced step through SubDeviceTraceController, which also captures models that load sub-device managers
mid-forward. capture(): attach the controller, run `compile_runs` untimed calls, call `before_capture` on the last
one's output (allocate replay copy buffers here: a buffer allocated while the trace is live can land on its scratch),
then capture one call into `output`. Leaving the `with` ends an open capture, syncs, releases and frees `output`; a
cleanup error never replaces the error in flight."""

import time

from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tests.measure.timing import free, time_untraced
from models.demos.deepseek_v3_d_p.utils.sub_device_trace import SubDeviceTraceController


def drain(mesh_device) -> None:
    """Best-effort sync before freeing; never raises."""
    try:
        ttnn.synchronize_device(mesh_device)
    except Exception as e:
        logger.debug(f"drain before cleanup failed: {e}")


class TracedStep:
    def __init__(self, mesh_device, step, *, attach=None, detach=None, compile_runs=1):
        self.mesh_device, self.step, self.compile_runs = mesh_device, step, compile_runs
        self._attach, self._detach = attach, detach
        self.controller = SubDeviceTraceController(mesh_device)
        self.output, self._capturing = None, False

    def __enter__(self):
        return self

    def capture(self, before_capture=None):
        if self._attach is not None:
            self._attach(self.controller)
        for i in range(self.compile_runs):
            out = self.step()
            try:
                if before_capture is not None and i == self.compile_runs - 1:
                    before_capture(out)
            finally:
                free(out)
        ttnn.synchronize_device(self.mesh_device)
        self._capturing = True
        self.controller.begin_capture()
        self.output = self.step()
        self.controller.end_capture()
        self._capturing = False
        return self

    def replay(self, sync=True):
        self.controller.replay(sync=sync)

    def __exit__(self, exc_type, exc, tb):
        errors = []
        if self._capturing:
            try:
                self.controller.end_capture()
                self._capturing = False
            except Exception as e:  # still capturing: release() would raise, so skip it
                logger.warning(f"ending the failed capture failed: {e}")
        cleanup = [self._detach, lambda: free(self.output)]
        if not self._capturing:
            drain(self.mesh_device)
            cleanup.insert(0, self.controller.release)
        for fn in cleanup:
            try:
                if fn is not None:
                    fn()
            except Exception as e:
                logger.warning(f"trace cleanup failed: {e}")
                errors.append(e)
        self.output = None
        if errors and exc is None:
            raise errors[0]
        return False


def time_traced(traced, samples=1):
    """Host ms of one replay through a sync, per sample. The perf gates time one replay, with no warm-up replay."""
    result = []
    for _ in range(samples):
        ttnn.synchronize_device(traced.mesh_device)
        t0 = time.perf_counter()
        traced.replay()
        result.append((time.perf_counter() - t0) * 1e3)
    return result


def measure_host(mesh_device, step, *, attach=None, detach=None):
    """Untraced and traced host ms of `step` (time_untraced, then one replay) and the minimum per-device PCC of the
    traced output against the last timed eager output."""
    from models.common.utility_functions import comp_pcc

    def per_device(t):
        return [ttnn.to_torch(x) for x in ttnn.get_device_tensors(ttnn.from_device(t))]

    untraced, out = time_untraced(step, lambda: ttnn.synchronize_device(mesh_device))
    try:
        eager = per_device(out)
    finally:
        ttnn.deallocate(out)
    with TracedStep(mesh_device, step, attach=attach, detach=detach) as traced:
        traced.capture()
        traced_ms = time_traced(traced)
        replayed = per_device(traced.output)
    return untraced, traced_ms, min(comp_pcc(a, b)[1] for a, b in zip(eager, replayed))
