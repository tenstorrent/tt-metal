# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Optional

import torch

try:
    import ttnn  # type: ignore
except Exception:  # pragma: no cover
    ttnn = None  # type: ignore


class MeshTraceDPExecutor:
    """Execute DP replicas in parallel by capturing a single trace on the mesh device.

    This follows the TT-NN mesh guidance: capture the model forward for each
    submesh while tracing on the *mesh_device*, then replay the trace on the
    mesh_device to dispatch work to all submeshes in parallel.
    """

    def __init__(self, *, mesh_device, tt_pipelines: list, execution_mode: str):
        if ttnn is None:
            raise RuntimeError("ttnn is required for MeshTraceDPExecutor")
        self.mesh_device = mesh_device
        self.tt_pipelines = list(tt_pipelines)
        self.execution_mode = str(execution_mode).lower()
        if self.execution_mode not in {"trace", "trace_2cq"}:
            raise ValueError(f"MeshTraceDPExecutor requires trace/trace_2cq, got {execution_mode!r}")

        self._trace_id = None
        self._trace_event = None
        self._persistent_inputs: list = []
        self._persistent_outputs: list = []

    def close(self):
        if self._trace_id is not None:
            try:
                ttnn.release_trace(self.mesh_device, self._trace_id)
            except Exception:
                pass
        self._trace_id = None
        self._trace_event = None
        self._persistent_inputs = []
        self._persistent_outputs = []

    def capture(self, tt_inputs_host: list):
        """Compile per-submesh pipelines, allocate persistent inputs, and capture a mesh trace."""
        if self._trace_id is not None:
            return
        if len(tt_inputs_host) != len(self.tt_pipelines):
            raise ValueError(
                f"Expected {len(self.tt_pipelines)} host inputs for capture, got {len(tt_inputs_host)}"
            )

        # Allocate persistent device input buffers on each submesh and compile once.
        self._persistent_inputs = []
        for pipe, host_in in zip(self.tt_pipelines, tt_inputs_host):
            submesh_dev = getattr(getattr(pipe, "backbone", None), "tt_device", None)
            if submesh_dev is None:
                raise RuntimeError("Pipeline has no TT device; cannot run mesh-trace DP")
            # Stable device address for trace replay.
            in_dev = host_in.to(submesh_dev)
            self._persistent_inputs.append(in_dev)
            # Compile.
            _ = pipe._tt_forward_core(in_dev)

        # Capture a single trace on the mesh device that runs every submesh forward once.
        trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        try:
            outs = []
            for pipe, in_dev in zip(self.tt_pipelines, self._persistent_inputs):
                outs.append(pipe._tt_forward_core(in_dev))
        finally:
            ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)

        self._trace_id = trace_id
        self._persistent_outputs = outs

        if self.execution_mode == "trace_2cq":
            # Event used to synchronize cq=1 host->device copies with cq=0 trace replay.
            self._trace_event = ttnn.record_event(self.mesh_device, 0)

    def _copy_inputs(self, tt_inputs_host: list, cq_id: int):
        for host_in, dev_in in zip(tt_inputs_host, self._persistent_inputs):
            ttnn.copy_host_to_device_tensor(host_in, dev_in, cq_id)

    def run(self, tt_inputs_host: list, *, normalize: bool = True) -> list:
        if self._trace_id is None:
            raise RuntimeError("MeshTraceDPExecutor.capture() must be called before run()")
        if len(tt_inputs_host) != len(self.tt_pipelines):
            raise ValueError(f"Expected {len(self.tt_pipelines)} host inputs, got {len(tt_inputs_host)}")

        if self.execution_mode == "trace_2cq":
            if self._trace_event is None:
                self._trace_event = ttnn.record_event(self.mesh_device, 0)
            ttnn.wait_for_event(1, self._trace_event)
            self._copy_inputs(tt_inputs_host, cq_id=1)
            write_event = ttnn.record_event(self.mesh_device, 1)
            ttnn.wait_for_event(0, write_event)
            ttnn.execute_trace(self.mesh_device, self._trace_id, cq_id=0, blocking=False)
            completion_event = ttnn.record_event(self.mesh_device, 0)
            ttnn.wait_for_event(0, completion_event)
            self._trace_event = completion_event
        else:
            self._copy_inputs(tt_inputs_host, cq_id=0)
            ttnn.execute_trace(self.mesh_device, self._trace_id, cq_id=0, blocking=True)

        # Read from persistent output buffers after trace execution completes.
        outs = []
        for pipe, out_dev in zip(self.tt_pipelines, self._persistent_outputs):
            out_host = out_dev.cpu()
            try:
                if hasattr(out_host, "layout") and out_host.layout == ttnn.TILE_LAYOUT:
                    out_host = out_host.to(ttnn.ROW_MAJOR_LAYOUT)
            except Exception:
                pass
            out_torch = out_host.to_torch()
            depth_t = torch.as_tensor(out_torch)
            if depth_t.dim() == 3:
                depth_t = depth_t.unsqueeze(1)
            if normalize:
                depth_t = pipe.fallback._normalize_depth(depth_t.float())
            else:
                depth_t = depth_t.float()
            outs.append(depth_t.cpu().numpy())
        return outs

