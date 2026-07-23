# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Two-command-queue (2CQ) coordination for HunyuanImage-3.0 DiT denoise loop.
#
# CQ0: patch_embed + backbone + final_layer + scheduler Euler update
# CQ1: async latent D2H after each step (legacy ``HY_LATENT_RESIDENT=0`` only;
#      overlaps host distill-scatter prep on the next step). With resident latents
#      the loop stays on-device until a single final D2H for VAE.

from __future__ import annotations


import torch
import ttnn

from models.experimental.hunyuan_image_3_0.tt.ar_dual_cq import (
    COMPUTE_CQ,
    IO_CQ,
    device_num_command_queues,
)


def denoise_2cq_enabled(device) -> bool:
    from models.experimental.hunyuan_image_3_0.tt.trace_config import denoise_2cq_enabled as _denoise_2cq

    return _denoise_2cq(device)


def open_denoise_mesh(mesh_shape, *, l1_small_size: int = 32768, enable_2cq: bool | None = None):
    """Open a mesh for DiT denoise with optional two command queues."""
    from models.experimental.hunyuan_image_3_0.tt.trace_config import hy_trace_enabled, open_traced_mesh

    if enable_2cq is None:
        enable_2cq = hy_trace_enabled()
    num_cq = 2 if enable_2cq else 1
    mesh = open_traced_mesh(mesh_shape, l1_small_size=l1_small_size, num_cq=num_cq)
    if enable_2cq and device_num_command_queues(mesh) < 2:
        print("[denoise] warning: requested 2CQ but mesh opened with 1 CQ", flush=True)
    elif enable_2cq:
        print(
            f"[denoise] 2CQ enabled (HY_TRACE=1, num_command_queues={device_num_command_queues(mesh)})",
            flush=True,
        )
    return mesh


def latent_tt_to_torch(latent_host, mesh_device, *, batch: int, channels: int, h: int, w: int) -> torch.Tensor:
    """Device NHWC-flat host tensor -> torch ``[B, C, h, w]``."""
    if mesh_device is not None and hasattr(mesh_device, "get_num_devices") and mesh_device.get_num_devices() > 1:
        out = ttnn.to_torch(latent_host, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
        out = out[:batch]
    else:
        out = ttnn.to_torch(latent_host)
    return out.reshape(batch, h, w, channels).permute(0, 3, 1, 2).contiguous()


class DenoiseDualCQCoordinator:
    """CQ0 denoise compute + CQ1 async latent D2H between Euler steps."""

    def __init__(self, device):
        self.device = device
        if device_num_command_queues(device) < 2:
            raise ValueError(
                f"DenoiseDualCQCoordinator requires num_command_queues>=2, " f"got {device_num_command_queues(device)}"
            )
        self._read_event = None
        self._pending_latent_host = None
        self._pending_nxt_tt = None
        self.steps = 0

    def fence_compute_before_io(self) -> None:
        """CQ1: wait for the previous async D2H before reusing the host buffer."""
        if self._read_event is not None:
            ttnn.wait_for_event(IO_CQ, self._read_event)
            self._read_event = None

    def launch_latent_d2h(self, nxt_tt: ttnn.Tensor) -> None:
        """After scheduler.step on CQ0, enqueue async latent D2H on CQ1."""
        if self._pending_latent_host is not None:
            raise RuntimeError("launch_latent_d2h called before consume_latent_torch")
        self.fence_compute_before_io()
        compute_done = ttnn.record_event(self.device, COMPUTE_CQ)
        ttnn.wait_for_event(IO_CQ, compute_done)
        self._pending_latent_host = ttnn.from_device(nxt_tt, blocking=False, cq_id=IO_CQ)
        self._pending_nxt_tt = nxt_tt
        self._read_event = ttnn.record_event(self.device, IO_CQ)
        self.steps += 1

    def consume_latent_torch(
        self,
        mesh_device,
        *,
        batch: int,
        channels: int,
        h: int,
        w: int,
    ) -> torch.Tensor:
        """Synchronize CQ1 D2H and return the updated latent on host."""
        if self._pending_latent_host is None:
            raise RuntimeError("consume_latent_torch called with no pending D2H")
        ttnn.event_synchronize(self._read_event)
        self._read_event = None
        host = self._pending_latent_host
        self._pending_latent_host = None
        latent = latent_tt_to_torch(host, mesh_device, batch=batch, channels=channels, h=h, w=w)
        if self._pending_nxt_tt is not None:
            ttnn.deallocate(self._pending_nxt_tt)
            self._pending_nxt_tt = None
        return latent
