# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""A traced step: compile, capture, replay and release one forward through SubDeviceTraceController.

The controller also captures models that load sub-device managers mid-forward (a raw ttnn trace cannot span a manager
swap); for a model that never does, it is one ordinary trace.

Lifecycle, in order:
  1. attach(controller): the model routes its sub-device loads through the controller (e.g. mla.set_trace_controller).
  2. compile: `compile_runs` untimed calls under the controller, each output freed.
  3. before_capture(output of the last compile run): allocate anything a replay writes into (copy buffers). A buffer
     allocated while the trace is live can land on its scratch and be overwritten by the next replay.
  3b. after_compile(): runs once the compile outputs are freed and the device is synced, for allocations or raw
     traces that must not see them (e.g. an input-refresh trace).
  4. capture: one call between begin_capture and end_capture; its output is `TracedStep.output`, rewritten by every
     replay.
  5. replay(sync).
  6. close (the context manager's exit): end a capture still open (the device cannot sync while capturing), sync,
     release, detach, free `output`. If ending the capture fails, the device is left capturing: release() would raise
     and replace the first error, so it is skipped and the original exception propagates.
"""

from __future__ import annotations

import time
from typing import Callable

from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tests.measure.timing import Samples, free
from models.demos.deepseek_v3_d_p.utils.sub_device_trace import SubDeviceTraceController


def drain(mesh_device) -> None:
    """Best-effort sync before freeing: leaves the device idle, and must not replace an error being raised."""
    try:
        ttnn.synchronize_device(mesh_device)
    except Exception as e:
        logger.debug(f"drain before cleanup failed: {e}")


class TracedStep:
    """See the module docstring. Several can be live at once; each owns its controller."""

    def __init__(
        self,
        mesh_device,
        step: Callable[[], object],
        *,
        attach: Callable[[SubDeviceTraceController], None] | None = None,
        detach: Callable[[], None] | None = None,
        compile_runs: int = 1,
        release: Callable[[object], None] = free,
        controller_factory: Callable[[object], SubDeviceTraceController] = SubDeviceTraceController,
    ):
        self.mesh_device = mesh_device
        self.step = step
        self._attach, self._detach = attach, detach
        self.compile_runs = compile_runs
        self.release = release  # frees a step's output (default: every device tensor in it)
        self.controller = controller_factory(mesh_device)
        self.output = None
        self._capturing = False
        self._attached = False

    def __enter__(self):
        return self

    def capture(
        self,
        before_capture: Callable[[object], None] | None = None,
        after_compile: Callable[[], None] | None = None,
    ) -> TracedStep:
        """See the module docstring. `before_capture(output)` runs on the last compile run's output, before it is freed;
        `after_compile()` runs once the compile outputs are freed and the device is synced, just before the capture."""
        assert before_capture is None or self.compile_runs >= 1, "before_capture needs a compile run's output"
        if self._attach is not None:
            self._attach(self.controller)
            self._attached = True
        for i in range(self.compile_runs):
            out = self.step()
            try:
                if before_capture is not None and i == self.compile_runs - 1:
                    before_capture(out)
            finally:
                self.release(out)
        ttnn.synchronize_device(self.mesh_device)
        if after_compile is not None:
            after_compile()
        self._capturing = True  # before begin_capture: a failed begin is ended too
        self.controller.begin_capture()
        self.output = self.step()
        self.controller.end_capture()
        self._capturing = False
        return self

    def replay(self, sync: bool = True) -> None:
        self.controller.replay(sync=sync)

    def take_output(self):
        """Hand the captured output to the caller, who then frees it (close() no longer does)."""
        output, self.output = self.output, None
        return output

    def __exit__(self, exc_type, exc, tb):
        cleanup_errors = []

        def attempt(what, fn):
            # Each step runs even if an earlier one failed; a cleanup error never replaces the error in flight.
            try:
                fn()
            except Exception as e:
                logger.warning(f"{what} failed: {e}")
                cleanup_errors.append(e)

        stuck = False
        if self._capturing:
            try:
                self.controller.end_capture()
                self._capturing = False
            except Exception as e:
                stuck = True
                logger.warning(f"ending the failed capture failed: {e}")
        if not stuck:
            drain(self.mesh_device)
            attempt("releasing the trace", self.controller.release)
        if self._attached and self._detach is not None:
            attempt("detaching the trace controller", self._detach)
        if self.output is not None:
            output, self.output = self.output, None
            attempt("freeing the captured output", lambda: self.release(output))
        if cleanup_errors and exc is None:
            raise cleanup_errors[0]
        return False


def time_traced(
    traced: TracedStep,
    *,
    samples: int = 1,
    calls_per_sample: int = 1,
    estimator: str = "fastest",
    prepare: Callable[[], None] | None = None,
) -> Samples:
    """Host ms of `calls_per_sample` back-to-back replays plus one sync, per sample. The perf gate's default is one
    replay, no warm-up replay. `prepare()` (e.g. refresh an input the forward consumed) runs and is synced outside the
    timed window."""
    result = Samples([], estimator)
    for _ in range(samples):
        if prepare is not None:
            prepare()
        ttnn.synchronize_device(traced.mesh_device)
        t0 = time.perf_counter()
        for _ in range(calls_per_sample):
            traced.replay(sync=False)
        ttnn.synchronize_device(traced.mesh_device)
        result.values.append((time.perf_counter() - t0) * 1e3 / calls_per_sample)
    return result
