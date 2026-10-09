# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import TypeVar

import torch

from models.experimental.chronos_forecast.tt.model import (
    TtChronos,
    TtChronosDeviceInputs,
    TtChronosPreparedInputs,
)

T = TypeVar("T")


@dataclass(frozen=True)
class TraceExecutionResult:
    quantile_preds: torch.Tensor
    prepared: TtChronosPreparedInputs


class TtChronosTraceRunner:
    """Fixed-shape, address-stable TTNN trace runner for one Chronos model."""

    def __init__(self, model: TtChronos, prepared: TtChronosPreparedInputs, *, cq_id: int = 0):
        import ttnn

        self.model = model
        self.device = model.device
        self.cq_id = cq_id
        self.inputs: TtChronosDeviceInputs = model.upload_inputs(prepared)
        self._prepared = prepared
        self._trace_id = None
        self._trace_output = None
        self._op_event = None
        self._write_event = None
        self._released = False
        self.device.enable_program_cache()

        # Compile every program and stabilize allocations before capture.
        for _ in range(2):
            output = self.model.forward_device(self.inputs)
            ttnn.synchronize_device(self.device)
            ttnn.deallocate(output)

    def _host_tensor(self, tensor: torch.Tensor, *, split_batch: bool = True):
        """Host tensor laid out like its device input, so refreshes land on the right chips."""
        return self.model.host_input(tensor, split_batch=split_batch)

    def capture(self):
        import ttnn

        if self._released:
            raise RuntimeError("Chronos trace runner has been released")
        if self._trace_id is not None:
            raise RuntimeError("Chronos trace is already captured")
        set_cache_misses_allowed = getattr(self.device, "set_program_cache_misses_allowed", None)
        if set_cache_misses_allowed is not None:
            set_cache_misses_allowed(False)
        trace_id = None
        try:
            trace_id = ttnn.begin_trace_capture(self.device, cq_id=self.cq_id)
            try:
                self._trace_output = self.model.forward_device(self.inputs)
            finally:
                # Always end capture, even on failure, so the CQ leaves bypass mode
                # and the trace gets registered before we try to release it.
                ttnn.end_trace_capture(self.device, trace_id, cq_id=self.cq_id)
            self._trace_id = trace_id
        except Exception:
            if trace_id is not None:
                ttnn.release_trace(self.device, trace_id)
            self._trace_output = None
            raise
        finally:
            if set_cache_misses_allowed is not None:
                set_cache_misses_allowed(True)
        ttnn.synchronize_device(self.device)
        self._op_event = ttnn.record_event(self.device, self.cq_id)
        return self._trace_output

    def update_inputs(self, prepared: TtChronosPreparedInputs, *, cq_id: int | None = None) -> None:
        import ttnn

        if prepared.patched_tokens.shape != self._prepared.patched_tokens.shape or (
            prepared.num_context_patches != self._prepared.num_context_patches
        ):
            raise ValueError(
                f"trace token shape changed: {tuple(prepared.patched_tokens.shape)} "
                f"({prepared.num_context_patches} context) != {tuple(self._prepared.patched_tokens.shape)} "
                f"({self._prepared.num_context_patches} context)"
            )
        same_groups = prepared.unique_groups == self._prepared.unique_groups and (
            prepared.unique_groups
            or (
                prepared.group_block == self._prepared.group_block
                and prepared.group_mask.shape == self._prepared.group_mask.shape
            )
        )
        if not same_groups:
            raise ValueError("trace group layout changed (unique groups, block size or mask shape)")
        target_cq = self.cq_id if cq_id is None else cq_id
        host_tokens = self.model.host_tokens(prepared.patched_tokens)
        assert (
            host_tokens.shape == self.inputs.patched_tokens.shape
        ), f"host tokens {host_tokens.shape} != persistent device tokens {self.inputs.patched_tokens.shape}"
        ttnn.copy_host_to_device_tensor(host_tokens, self.inputs.patched_tokens, cq_id=target_cq)
        if not prepared.unique_groups:
            host_mask = self._host_tensor(prepared.group_mask, split_batch=self.model.group_mask_is_split(prepared))
            assert (
                host_mask.shape == self.inputs.group_mask.shape
            ), f"host group mask {host_mask.shape} != persistent device mask {self.inputs.group_mask.shape}"
            ttnn.copy_host_to_device_tensor(host_mask, self.inputs.group_mask, cq_id=target_cq)
        self._prepared = prepared

    def execute(
        self,
        prepared: TtChronosPreparedInputs | None = None,
        *,
        blocking: bool = True,
        synchronize: bool = True,
        readback: bool = True,
    ):
        import ttnn

        if self._trace_id is None or self._trace_output is None:
            raise RuntimeError("capture() must be called before execute()")
        if prepared is not None:
            self.update_inputs(prepared)
        self._replay(blocking=blocking)
        if synchronize:
            ttnn.synchronize_device(self.device)
        if not readback:
            return self._trace_output
        return self.read_output()

    def _replay(self, *, blocking: bool = False) -> None:
        import ttnn

        ttnn.execute_trace(self.device, self._trace_id, cq_id=self.cq_id, blocking=blocking)
        # Recorded behind the replay on the same queue, so it fires only after the trace has
        # finished reading the persistent inputs. execute_pipelined() makes CQ1 wait on it.
        self._op_event = ttnn.record_event(self.device, self.cq_id)

    def read_output(self) -> TraceExecutionResult:
        """Download and unscale the last replay's output for the current inputs."""
        if self._trace_output is None:
            raise RuntimeError("capture() must be called before read_output()")
        return TraceExecutionResult(
            quantile_preds=self.model.postprocess_output(
                self._trace_output,
                self._prepared.loc_scale,
                num_output_patches=self._prepared.num_output_patches,
                output_rows=self._prepared.output_rows,
            ),
            prepared=self._prepared,
        )

    def execute_pipelined(self, prepared: TtChronosPreparedInputs, *, readback: bool = True):
        """Refresh the inputs on CQ1 and replay on CQ0, ordered with events.

        The device must be opened with two command queues. CQ1 waits for the
        previous replay to finish before overwriting the persistent inputs, and
        CQ0 waits for the write before replaying. Host-side preparation of the
        next batch still overlaps the replay.
        """
        import ttnn

        if self._trace_id is None or self._trace_output is None or self._op_event is None:
            raise RuntimeError("capture() must be called before execute_pipelined()")
        ttnn.wait_for_event(1, self._op_event)
        self.update_inputs(prepared, cq_id=1)
        self._write_event = ttnn.record_event(self.device, 1)
        ttnn.wait_for_event(self.cq_id, self._write_event)
        self._replay()
        if not readback:
            return self._trace_output
        ttnn.synchronize_device(self.device)
        return self.read_output()

    def stream(
        self, items: Iterable[T], prepare: Callable[[T], TtChronosPreparedInputs]
    ) -> Iterator[TraceExecutionResult]:
        """Yield one result per item, preparing item i+1 on the host while replay i runs.

        ``prepare`` must return inputs with the captured shapes and group layout.
        """
        import ttnn

        if self._trace_id is None or self._trace_output is None:
            raise RuntimeError("capture() must be called before stream()")
        items = iter(items)
        try:
            first = next(items)
        except StopIteration:
            return
        self.update_inputs(prepare(first))
        self._replay()
        for item in items:
            prepared = prepare(item)
            # The next replay overwrites the output, so read this one back first.
            ttnn.synchronize_device(self.device)
            yield self.read_output()
            self.update_inputs(prepared)
            self._replay()
        ttnn.synchronize_device(self.device)
        yield self.read_output()

    def release(self) -> None:
        import ttnn

        if self._released:
            return
        if self._trace_id is not None:
            ttnn.release_trace(self.device, self._trace_id)
            self._trace_id = None
        if self._trace_output is not None:
            ttnn.deallocate(self._trace_output)
            self._trace_output = None
        self.model.deallocate_inputs(self.inputs)
        self._op_event = None
        self._write_event = None
        self._released = True

    def __enter__(self):
        self.capture()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()
