# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import torch

from models.experimental.chronos_forecast.tt.model import (
    TtChronos,
    TtChronosDeviceInputs,
    TtChronosPreparedInputs,
)


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

    @staticmethod
    def _host_tensor(tensor: torch.Tensor):
        import ttnn

        return ttnn.from_torch(
            tensor.detach().to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )

    def capture(self):
        import ttnn

        if self._released:
            raise RuntimeError("Chronos trace runner has been released")
        if self._trace_id is not None:
            raise RuntimeError("Chronos trace is already captured")
        set_cache_misses_allowed = getattr(self.device, "set_program_cache_misses_allowed", None)
        if set_cache_misses_allowed is not None:
            set_cache_misses_allowed(False)
        try:
            self._trace_id = ttnn.begin_trace_capture(self.device, cq_id=self.cq_id)
            self._trace_output = self.model.forward_device(self.inputs)
            ttnn.end_trace_capture(self.device, self._trace_id, cq_id=self.cq_id)
        finally:
            if set_cache_misses_allowed is not None:
                set_cache_misses_allowed(True)
        ttnn.synchronize_device(self.device)
        self._op_event = ttnn.record_event(self.device, self.cq_id)
        return self._trace_output

    def update_inputs(self, prepared: TtChronosPreparedInputs, *, cq_id: int | None = None) -> None:
        import ttnn

        if prepared.patched_context.shape != self._prepared.patched_context.shape:
            raise ValueError(
                f"trace context shape changed: {tuple(prepared.patched_context.shape)} "
                f"!= {tuple(self._prepared.patched_context.shape)}"
            )
        if prepared.patched_future.shape != self._prepared.patched_future.shape:
            raise ValueError(
                f"trace future shape changed: {tuple(prepared.patched_future.shape)} "
                f"!= {tuple(self._prepared.patched_future.shape)}"
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
        ttnn.copy_host_to_device_tensor(
            self._host_tensor(prepared.patched_context),
            self.inputs.patched_context,
            cq_id=target_cq,
        )
        ttnn.copy_host_to_device_tensor(
            self._host_tensor(prepared.patched_future),
            self.inputs.patched_future,
            cq_id=target_cq,
        )
        if not prepared.unique_groups:
            ttnn.copy_host_to_device_tensor(
                self._host_tensor(prepared.group_mask),
                self.inputs.group_mask,
                cq_id=target_cq,
            )
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
        ttnn.execute_trace(self.device, self._trace_id, cq_id=self.cq_id, blocking=blocking)
        if synchronize:
            ttnn.synchronize_device(self.device)
        if not readback:
            return self._trace_output
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
        """Overlap CQ1 input refresh with CQ0 trace dispatch using events.

        The device must be opened with two command queues. This follows the
        established BGE-M3 fixed-input trace protocol.
        """
        import ttnn

        if self._trace_id is None or self._trace_output is None or self._op_event is None:
            raise RuntimeError("capture() must be called before execute_pipelined()")
        ttnn.wait_for_event(1, self._op_event)
        self.update_inputs(prepared, cq_id=1)
        self._write_event = ttnn.record_event(self.device, 1)
        ttnn.wait_for_event(self.cq_id, self._write_event)
        self._op_event = ttnn.record_event(self.device, self.cq_id)
        ttnn.execute_trace(self.device, self._trace_id, cq_id=self.cq_id, blocking=False)
        if not readback:
            return self._trace_output
        ttnn.synchronize_device(self.device)
        return TraceExecutionResult(
            quantile_preds=self.model.postprocess_output(
                self._trace_output,
                self._prepared.loc_scale,
                num_output_patches=self._prepared.num_output_patches,
                output_rows=self._prepared.output_rows,
            ),
            prepared=self._prepared,
        )

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
