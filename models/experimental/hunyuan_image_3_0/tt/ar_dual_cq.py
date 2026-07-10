# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Two-command-queue (2CQ) coordination for HunyuanImage-3.0 AR recaption.
#
# CQ0: backbone forward + LM head (compute)
# CQ1: logits D2H (and optional H2D for forced stage tokens)
#
# Pattern follows Whisper ``WhisperGenerator`` (CQ0 trace/compute, CQ1 I/O) and ViT
# 2CQ trace demos (``test_demo_vit_ttnn_inference_perf_e2e_2cq_trace.py``).

from __future__ import annotations


import torch

import ttnn

COMPUTE_CQ = 0
IO_CQ = 1

from models.experimental.hunyuan_image_3_0.tt.trace_config import (
    hy_trace_enabled,
    recaption_2cq_enabled as _recaption_2cq_enabled,
    trace_region_size,
)


def recaption_trace_region_size() -> int:
    return trace_region_size()


def device_num_command_queues(device) -> int:
    if hasattr(device, "num_command_queues"):
        return int(device.num_command_queues)
    getter = getattr(device, "get_num_command_queues", None)
    if getter is not None:
        return int(getter())
    return 1


def _stash_mesh_command_queues(mesh, num_cq: int) -> None:
    """MeshDevice has no public CQ count; record what ``open_mesh_device`` requested."""
    mesh.num_command_queues = num_cq


def recaption_2cq_enabled(device) -> bool:
    return _recaption_2cq_enabled(device)


def open_recaption_mesh(mesh_shape, *, l1_small_size: int = 32768, enable_2cq: bool | None = None):
    """Open a mesh for AR recaption with optional 2 command queues and trace region."""
    if enable_2cq is None:
        enable_2cq = hy_trace_enabled()
    num_cq = 2 if enable_2cq else 1
    trace_region = trace_region_size() if hy_trace_enabled() else ttnn._ttnn.device.DEFAULT_TRACE_REGION_SIZE
    mesh = ttnn.open_mesh_device(
        mesh_shape,
        l1_small_size=l1_small_size,
        trace_region_size=trace_region,
        num_command_queues=num_cq,
    )
    _stash_mesh_command_queues(mesh, num_cq)
    if hy_trace_enabled():
        print(
            f"[recaption] HY_TRACE=1 trace region {trace_region // (1024 * 1024)} MiB "
            f"(num_command_queues={device_num_command_queues(mesh)})",
            flush=True,
        )
    if enable_2cq and device_num_command_queues(mesh) < 2:
        print("[recaption] warning: requested 2CQ but mesh opened with 1 CQ", flush=True)
    elif enable_2cq:
        print(f"[recaption] 2CQ enabled (num_command_queues={device_num_command_queues(mesh)})", flush=True)
    return mesh


def logits_host_to_torch(logits_host, device, batch_size: int) -> torch.Tensor:
    """Convert a host tensor from ``from_device`` into float logits ``[B, V]``."""
    if hasattr(device, "get_num_devices") and device.get_num_devices() > 1:
        logits = ttnn.to_torch(logits_host, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0))
        logits = logits[:batch_size]
    else:
        logits = ttnn.to_torch(logits_host)
    return logits.float().squeeze(1)


class ArDualCQCoordinator:
    """Fence and async D2H for AR decode: compute on CQ0, logits read on CQ1."""

    def __init__(self, device):
        self.device = device
        if device_num_command_queues(device) < 2:
            raise ValueError(
                f"ArDualCQCoordinator requires num_command_queues>=2, got {device_num_command_queues(device)}"
            )
        self._read_event = None
        self._write_event = None
        self._pending_logits_host = None
        self.steps = 0

    def fence_compute_before_io(self) -> None:
        """Ensure the previous CQ1 D2H finished before reusing the logits buffer."""
        if self._read_event is not None:
            ttnn.wait_for_event(IO_CQ, self._read_event)
            self._read_event = None

    def fence_compute_before_forward(self) -> None:
        """CQ0 waits for an in-flight CQ1 H2D (forced-token path) before forward."""
        if self._write_event is not None:
            ttnn.wait_for_event(COMPUTE_CQ, self._write_event)
            self._write_event = None

    def launch_logits_d2h(self, logits_tt: ttnn.Tensor) -> None:
        """After forward on CQ0, enqueue async logits D2H on CQ1."""
        if self._pending_logits_host is not None:
            raise RuntimeError("launch_logits_d2h called before consume_logits")
        self.fence_compute_before_io()
        self.fence_compute_before_forward()
        compute_done = ttnn.record_event(self.device, COMPUTE_CQ)
        ttnn.wait_for_event(IO_CQ, compute_done)
        self._pending_logits_host = ttnn.from_device(logits_tt, blocking=False, cq_id=IO_CQ)
        self._read_event = ttnn.record_event(self.device, IO_CQ)
        self.steps += 1

    def consume_logits(self, batch_size: int) -> torch.Tensor:
        """Synchronize CQ1 D2H and return float logits ``[B, V]``."""
        if self._pending_logits_host is None:
            raise RuntimeError("consume_logits called with no pending D2H")
        ttnn.event_synchronize(self._read_event)
        self._read_event = None
        host = self._pending_logits_host
        self._pending_logits_host = None
        return logits_host_to_torch(host, self.device, batch_size)

    def copy_host_to_device_async(self, host_tensor, device_tensor) -> None:
        """H2D on CQ1 after CQ0 compute completes (forced stage tokens)."""
        compute_done = ttnn.record_event(self.device, COMPUTE_CQ)
        ttnn.wait_for_event(IO_CQ, compute_done)
        ttnn.copy_host_to_device_tensor(host_tensor, device_tensor, IO_CQ)
        self._write_event = ttnn.record_event(self.device, IO_CQ)
