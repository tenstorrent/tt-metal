# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import time
import torch

try:
    import ttnn  # type: ignore
except Exception:  # pragma: no cover
    ttnn = None  # type: ignore


class SubmeshTraceDPExecutor:
    """DP executor that overlaps two submesh traces without Python threading.

    NOTE: MeshDevice-level trace capture/replay is the ideal DP mechanism per
    TT-NN mesh guidance, but some runtime builds can be unstable when tracing
    on the mesh device. This executor instead:
    - captures a full-model trace per submesh device (via each pipeline),
    - enqueues both traces non-blocking,
    - waits for completion events on each device,
    - reads from the persistent trace outputs.

    This keeps the host single-threaded while still allowing the two devices to
    run concurrently.
    """

    def __init__(self, *, tt_pipelines: list, execution_mode: str):
        if ttnn is None:
            raise RuntimeError("ttnn is required for SubmeshTraceDPExecutor")
        self.tt_pipelines = list(tt_pipelines)
        self.execution_mode = str(execution_mode).lower()
        if self.execution_mode not in {"trace", "trace_2cq"}:
            raise ValueError(f"SubmeshTraceDPExecutor requires trace/trace_2cq, got {execution_mode!r}")
        self._prepared = False
        self.last_perf = None

    def prepare(self, tt_inputs_host: list):
        if self._prepared:
            return
        if len(tt_inputs_host) != len(self.tt_pipelines):
            raise ValueError(f"Expected {len(self.tt_pipelines)} host inputs, got {len(tt_inputs_host)}")

        for pipe, host_in in zip(self.tt_pipelines, tt_inputs_host):
            # Use the pipeline's trace implementation (per-submesh device).
            pipe._ensure_full_trace(host_in, execution_mode=self.execution_mode)
            if pipe._full_trace_id is None or pipe._full_trace_input is None or pipe._full_trace_output is None:
                raise RuntimeError("Failed to prepare per-submesh full trace")
        self._prepared = True

    def run(self, tt_inputs_host: list, *, normalize: bool = True) -> list:
        if not self._prepared:
            raise RuntimeError("SubmeshTraceDPExecutor.prepare() must be called before run()")
        if len(tt_inputs_host) != len(self.tt_pipelines):
            raise ValueError(f"Expected {len(self.tt_pipelines)} host inputs, got {len(tt_inputs_host)}")

        total_start = time.perf_counter()
        trace_wall_start = time.perf_counter()

        completions = []
        for pipe, host_in in zip(self.tt_pipelines, tt_inputs_host):
            dev = pipe.backbone.tt_device
            if dev is None:
                raise RuntimeError("Pipeline has no TT device")

            if self.execution_mode == "trace_2cq":
                if pipe._trace_op_event is None:
                    pipe._trace_op_event = ttnn.record_event(dev, 0)
                ttnn.wait_for_event(1, pipe._trace_op_event)
                ttnn.copy_host_to_device_tensor(host_in, pipe._full_trace_input, 1)
                write_event = ttnn.record_event(dev, 1)
                ttnn.wait_for_event(0, write_event)
                t_exec = time.perf_counter()
                ttnn.execute_trace(dev, pipe._full_trace_id, cq_id=0, blocking=False)
                completion_event = ttnn.record_event(dev, 0)
                completions.append((pipe, completion_event, t_exec))
            else:
                ttnn.copy_host_to_device_tensor(host_in, pipe._full_trace_input, 0)
                t_exec = time.perf_counter()
                ttnn.execute_trace(dev, pipe._full_trace_id, cq_id=0, blocking=False)
                completion_event = ttnn.record_event(dev, 0)
                completions.append((pipe, completion_event, t_exec))

        # Wait for both devices to finish before reading from outputs.
        per_worker = []
        for pipe, ev, t_exec in completions:
            ttnn.wait_for_event(0, ev)
            trace_exec_ms = (time.perf_counter() - t_exec) * 1000.0
            if self.execution_mode == "trace_2cq":
                pipe._trace_op_event = ev
            entry = {
                "mode": "tt",
                "execution_mode": self.execution_mode,
                "effective_execution_mode": self.execution_mode,
                "requested_execution_mode": self.execution_mode,
                "num_images": 1,
                "trace_exec_ms": trace_exec_ms,
            }
            pipe.last_perf = dict(entry)
            per_worker.append(entry)

        trace_wall_ms = (time.perf_counter() - trace_wall_start) * 1000.0

        outs = []
        readback_ms = 0.0
        normalize_ms = 0.0
        for pipe in self.tt_pipelines:
            depth = pipe._full_trace_output
            rb_start = time.perf_counter()
            try:
                if isinstance(depth, ttnn.Tensor):
                    depth = depth.cpu().to_torch()
            except Exception:
                pass
            depth_t = torch.as_tensor(depth)
            if depth_t.dim() == 3:
                depth_t = depth_t.unsqueeze(1)
            readback_ms += (time.perf_counter() - rb_start) * 1000.0
            if normalize:
                norm_start = time.perf_counter()
                depth_t = pipe.fallback._normalize_depth(depth_t.float())
                normalize_ms += (time.perf_counter() - norm_start) * 1000.0
            else:
                depth_t = depth_t.float()
            outs.append(depth_t.cpu().numpy())

        total_wall_ms = (time.perf_counter() - total_start) * 1000.0
        self.last_perf = {
            "mode": "tt",
            "execution_mode": self.execution_mode,
            "effective_execution_mode": self.execution_mode,
            "requested_execution_mode": self.execution_mode,
            "num_images": len(self.tt_pipelines),
            "trace_wall_ms": trace_wall_ms,
            "readback_ms": readback_ms,
            "normalize_ms": normalize_ms,
            "total_wall_ms": total_wall_ms,
            "per_worker": per_worker,
        }

        # Make per-worker breakdowns visible through the pipeline objects as well,
        # since the runner already collects `pipeline.last_perf`.
        for pipe in self.tt_pipelines:
            try:
                pipe.last_perf = dict(pipe.last_perf or {})
                pipe.last_perf["trace_wall_ms"] = trace_wall_ms
                pipe.last_perf["readback_ms"] = readback_ms
                pipe.last_perf["normalize_ms"] = normalize_ms
                pipe.last_perf["total_wall_ms"] = total_wall_ms
            except Exception:
                pass
        return outs
