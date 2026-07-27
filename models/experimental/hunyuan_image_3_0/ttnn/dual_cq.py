# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Two-command-queue (2CQ) coordination for HunyuanImage-3.0.
#
# One module for the three 2CQ stages; all share CQ0=compute / CQ1=async I/O and the
# ``COMPUTE_CQ`` / ``IO_CQ`` / ``device_num_command_queues`` helpers below.
#
#   AR recaption : CQ0 backbone + LM head; CQ1 logits D2H (+ optional forced-token H2D)
#   DiT denoise  : CQ0 patch_embed + backbone + final_layer + Euler step; CQ1 latent D2H
#   VAE decode   : CQ0 decoder forward; CQ1 RGB output D2H (1024² image is the big transfer)
#
# Pattern follows Whisper ``WhisperGenerator`` (CQ0 trace/compute, CQ1 I/O) and ViT
# 2CQ trace demos (``test_demo_vit_ttnn_inference_perf_e2e_2cq_trace.py``).

from __future__ import annotations


import torch

import ttnn

COMPUTE_CQ = 0
IO_CQ = 1

from models.experimental.hunyuan_image_3_0.ttnn.trace_config import (
    denoise_2cq_enabled as _denoise_2cq_enabled,
    hy_trace_enabled,
    open_traced_mesh,
    recaption_2cq_enabled as _recaption_2cq_enabled,
    trace_region_size,
    vae_2cq_enabled as _vae_2cq_enabled,
)

# ---------------------------------------------------------------------------------------
# Shared: command-queue count
# ---------------------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------------------
# AR recaption: CQ0 backbone + LM head, CQ1 logits D2H
# ---------------------------------------------------------------------------------------


def recaption_trace_region_size() -> int:
    return trace_region_size()


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


def logits_host_to_torch(logits_host, device, batch_size: int, *, vocab_parallel: bool = False) -> torch.Tensor:
    """Convert a host tensor from ``from_device`` into float logits ``[B, V]``.

    When ``vocab_parallel`` the lm_head sharded V across the mesh, so concatenate the
    per-device vocab slices along the last dim to rebuild the full ``[B, 1, V]``.
    Otherwise the logits are replicated: concat along the batch dim and keep one copy.
    """
    if hasattr(device, "get_num_devices") and device.get_num_devices() > 1:
        if vocab_parallel:
            logits = ttnn.to_torch(logits_host, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=-1))
        else:
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
        # Set by the caller once the lm_head is known: True when the vocab is
        # sharded across the mesh so ``consume_logits`` concatenates vocab slices.
        self.vocab_parallel = False

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
        return logits_host_to_torch(host, self.device, batch_size, vocab_parallel=self.vocab_parallel)

    def copy_host_to_device_async(self, host_tensor, device_tensor) -> None:
        """H2D on CQ1 after CQ0 compute completes (forced stage tokens)."""
        compute_done = ttnn.record_event(self.device, COMPUTE_CQ)
        ttnn.wait_for_event(IO_CQ, compute_done)
        ttnn.copy_host_to_device_tensor(host_tensor, device_tensor, IO_CQ)
        self._write_event = ttnn.record_event(self.device, IO_CQ)


# ---------------------------------------------------------------------------------------
# DiT denoise: CQ0 patch_embed + backbone + final_layer + Euler step, CQ1 latent D2H
#
# The latent D2H is legacy ``HY_LATENT_RESIDENT=0`` only (it overlaps host
# distill-scatter prep on the next step). With resident latents the loop stays
# on-device until a single final D2H for VAE.
# ---------------------------------------------------------------------------------------


def denoise_2cq_enabled(device) -> bool:
    return _denoise_2cq_enabled(device)


def open_denoise_mesh(mesh_shape, *, l1_small_size: int = 32768, enable_2cq: bool | None = None):
    """Open a mesh for DiT denoise with optional two command queues."""
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


# ---------------------------------------------------------------------------------------
# VAE decode: CQ0 decoder forward, CQ1 async RGB output D2H
# ---------------------------------------------------------------------------------------


def vae_2cq_enabled(device) -> bool:
    return _vae_2cq_enabled(device)


def open_vae_mesh(mesh_shape, *, l1_small_size: int = 32768, enable_2cq: bool | None = None):
    """Open a mesh for VAE decode with optional two command queues."""
    if enable_2cq is None:
        enable_2cq = hy_trace_enabled()
    num_cq = 2 if enable_2cq else 1
    mesh = open_traced_mesh(mesh_shape, l1_small_size=l1_small_size, num_cq=num_cq)
    if enable_2cq and device_num_command_queues(mesh) < 2:
        print("[vae] warning: requested 2CQ but mesh opened with 1 CQ", flush=True)
    elif enable_2cq:
        print(
            f"[vae] 2CQ enabled (HY_TRACE=1, num_command_queues={device_num_command_queues(mesh)})",
            flush=True,
        )
    return mesh


class VaeDualCQCoordinator:
    """CQ0 VAE decode + CQ1 async output D2H."""

    def __init__(self, device):
        self.device = device
        if device_num_command_queues(device) < 2:
            raise ValueError(
                f"VaeDualCQCoordinator requires num_command_queues>=2, got {device_num_command_queues(device)}"
            )
        self._read_event = None
        self._write_event = None
        self._pending_host = None
        self._pending_dev_tt = None
        self.d2h_transfers = 0

    def fence_compute_before_forward(self) -> None:
        """CQ0: wait for any prior CQ1 work before decoder forward."""
        if self._write_event is not None:
            ttnn.wait_for_event(COMPUTE_CQ, self._write_event)
            self._write_event = None

    def fence_io_before_launch(self) -> None:
        """CQ1: wait for the previous async D2H before reusing the host buffer."""
        if self._read_event is not None:
            ttnn.wait_for_event(IO_CQ, self._read_event)
            self._read_event = None

    def launch_output_d2h(self, output_tt: ttnn.Tensor) -> None:
        """After decoder forward on CQ0, enqueue async output D2H on CQ1."""
        if self._pending_host is not None:
            raise RuntimeError("launch_output_d2h called before consume_output_host")
        self.fence_io_before_launch()
        compute_done = ttnn.record_event(self.device, COMPUTE_CQ)
        ttnn.wait_for_event(IO_CQ, compute_done)
        self._pending_host = ttnn.from_device(output_tt, blocking=False, cq_id=IO_CQ)
        self._pending_dev_tt = output_tt
        self._read_event = ttnn.record_event(self.device, IO_CQ)
        self.d2h_transfers += 1

    def consume_output_host(self) -> ttnn.Tensor:
        """Synchronize CQ1 D2H and return the host output tensor."""
        if self._pending_host is None:
            raise RuntimeError("consume_output_host called with no pending D2H")
        ttnn.event_synchronize(self._read_event)
        self._read_event = None
        host = self._pending_host
        self._pending_host = None
        if self._pending_dev_tt is not None:
            ttnn.deallocate(self._pending_dev_tt, force=False)
            self._pending_dev_tt = None
        return host
