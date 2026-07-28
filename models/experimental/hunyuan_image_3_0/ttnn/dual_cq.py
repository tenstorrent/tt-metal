# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Trace / two-command-queue (2CQ) policy and coordination for HunyuanImage-3.0.
#
# Policy
# ------
# ``HY_TRACE=1`` (default): 2CQ mesh + denoise CFG ``execute_trace`` + recaption AR trace.
#
# Denoise ``execute_trace`` auto-disables when step count is at or below
# ``HY_DENOISE_TRACE_MIN_STEPS`` (default 8): capture overhead does not amortize on
# short loops (e.g. Instruct-Distil). Override with ``HY_DENOISE_TRACE=1`` / ``0``.
#
# VAE decode and I2I cond encode (VAE encoder + ViT/aligner) are **opt-in** sub-flags
# (default OFF) — see ``HY_VAE_DECODE_TRACE`` and ``HY_COND_ENCODE_TRACE``.
#
# ``HY_TRACE=0``: eager single-CQ path everywhere (sub-flags ignored).
#
# Coordination
# ------------
# All three 2CQ stages share CQ0=compute / CQ1=async I/O and the ``COMPUTE_CQ`` /
# ``IO_CQ`` / ``device_num_command_queues`` helpers below.
#
#   AR recaption : CQ0 backbone + LM head; CQ1 logits D2H (+ optional forced-token H2D)
#   DiT denoise  : CQ0 patch_embed + backbone + final_layer + Euler step; CQ1 latent D2H
#   VAE decode   : CQ0 decoder forward; CQ1 RGB output D2H (1024² image is the big transfer)
#
# Pattern follows Whisper ``WhisperGenerator`` (CQ0 trace/compute, CQ1 I/O) and ViT
# 2CQ trace demos (``test_demo_vit_ttnn_inference_perf_e2e_2cq_trace.py``).

from __future__ import annotations

import os

import torch

import ttnn

from models.experimental.hunyuan_image_3_0.ref.model_config import NUM_HIDDEN_LAYERS

COMPUTE_CQ = 0
IO_CQ = 1

_TRACE_REGION_MB_MIN = 128
_TRACE_REGION_MB_MAX = 512
_TRACE_REGION_MB_PER_LAYER = 8

# Set by demos before mesh open so startup logs match the denoise loop.
_denoise_trace_steps: int | None = None


# ---------------------------------------------------------------------------------------
# Trace policy: env flags
# ---------------------------------------------------------------------------------------


def set_denoise_trace_steps(steps: int | None) -> None:
    """Register planned denoise step count for auto trace policy (``print_trace_policy``)."""
    global _denoise_trace_steps
    _denoise_trace_steps = steps


def _denoise_trace_min_steps() -> int:
    return int(os.environ.get("HY_DENOISE_TRACE_MIN_STEPS", "8"))


def hy_trace_enabled() -> bool:
    """Master on/off for trace + 2CQ (default ON to match prior 2CQ defaults)."""
    return os.environ.get("HY_TRACE", "1") != "0"


def _sub_trace_enabled(env_var: str) -> bool:
    """Sub-flag: requires ``HY_TRACE=1`` and ``env_var=1`` (default off when unset)."""
    return hy_trace_enabled() and os.environ.get(env_var, "0") != "0"


def denoise_execute_trace_enabled(*, steps: int | None = None) -> bool:
    """Denoise CFG ``execute_trace`` — auto-off when steps <= ``HY_DENOISE_TRACE_MIN_STEPS``."""
    if not hy_trace_enabled():
        return False
    override = os.environ.get("HY_DENOISE_TRACE")
    if override == "0":
        return False
    if override == "1":
        return True
    n = steps if steps is not None else _denoise_trace_steps
    if n is not None and n <= _denoise_trace_min_steps():
        return False
    return True


def vae_execute_trace_enabled() -> bool:
    """Final RGB VAE decode ``execute_trace`` (opt-in via ``HY_VAE_DECODE_TRACE=1``)."""
    return _sub_trace_enabled("HY_VAE_DECODE_TRACE")


def cond_encode_trace_enabled() -> bool:
    """I2I cond VAE encoder + ViT/aligner trace (opt-in via ``HY_COND_ENCODE_TRACE=1``)."""
    return _sub_trace_enabled("HY_COND_ENCODE_TRACE")


def recaption_trace_enabled(*, sp_factor: int = 1, use_kv_cache: bool = True) -> bool:
    if not hy_trace_enabled():
        return False
    if not use_kv_cache:
        print("[trace] HY_TRACE=1 recaption trace requires KV cache; trace disabled", flush=True)
        return False
    if sp_factor > 1:
        print(
            f"[trace] HY_TRACE=1 recaption trace requires sp_factor=1, got {sp_factor}; trace disabled",
            flush=True,
        )
        return False
    return True


def print_trace_policy(*, prefix: str = "[trace]", denoise_steps: int | None = None) -> None:
    """Log active trace policy at demo startup."""
    if denoise_steps is not None:
        set_denoise_trace_steps(denoise_steps)
    if not hy_trace_enabled():
        print(f"{prefix} HY_TRACE=0: eager 1CQ, no denoise/recaption trace, no 2CQ", flush=True)
        return
    vae_dec = vae_execute_trace_enabled()
    cond = cond_encode_trace_enabled()
    denoise = denoise_execute_trace_enabled()
    min_steps = _denoise_trace_min_steps()
    if denoise:
        denoise_note = "on"
    elif _denoise_trace_steps is not None and _denoise_trace_steps <= min_steps:
        denoise_note = f"off (steps={_denoise_trace_steps} <= {min_steps}; " f"set HY_DENOISE_TRACE=1 to force)"
    else:
        denoise_note = "off (set HY_DENOISE_TRACE=1)"
    print(
        f"{prefix} HY_TRACE=1: denoise CFG trace={denoise_note}; recaption AR trace + 2CQ mesh; "
        f"VAE decode trace={'on' if vae_dec else 'off (set HY_VAE_DECODE_TRACE=1)'}; "
        f"cond VAE+ViT trace={'on' if cond else 'off (set HY_COND_ENCODE_TRACE=1)'}",
        flush=True,
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


def _stage_2cq_enabled(device, stage: str) -> bool:
    """``HY_TRACE=1`` and the device actually opened with >= 2 CQs."""
    if not hy_trace_enabled():
        return False
    n = device_num_command_queues(device)
    if n < 2:
        print(f"[trace] HY_TRACE=1 but num_command_queues={n}; {stage} 2CQ disabled", flush=True)
        return False
    return True


def recaption_2cq_enabled(device) -> bool:
    return _stage_2cq_enabled(device, "recaption")


def denoise_2cq_enabled(device) -> bool:
    return _stage_2cq_enabled(device, "denoise")


def vae_2cq_enabled(device) -> bool:
    return _stage_2cq_enabled(device, "VAE")


# ---------------------------------------------------------------------------------------
# Shared: mesh open / teardown
# ---------------------------------------------------------------------------------------


def trace_region_size() -> int:
    override_mb = os.environ.get("HY_TRACE_REGION_MB")
    if override_mb:
        return int(override_mb) * 1024 * 1024
    num_layers = int(os.environ.get("HY_NUM_LAYERS", str(NUM_HIDDEN_LAYERS)))
    size_mb = min(
        _TRACE_REGION_MB_MAX,
        max(_TRACE_REGION_MB_MIN, _TRACE_REGION_MB_MIN + num_layers * _TRACE_REGION_MB_PER_LAYER),
    )
    return size_mb * 1024 * 1024


def open_traced_mesh(mesh_shape, *, l1_small_size: int = 32768, num_cq: int | None = None):
    """Open a 2x2 mesh with optional trace region and 2 command queues."""
    trace_on = hy_trace_enabled()
    if num_cq is None:
        num_cq = 2 if trace_on else 1
    trace_region = trace_region_size() if trace_on else ttnn._ttnn.device.DEFAULT_TRACE_REGION_SIZE
    mesh = ttnn.open_mesh_device(
        mesh_shape,
        l1_small_size=l1_small_size,
        trace_region_size=trace_region,
        num_command_queues=num_cq,
    )
    _stash_mesh_command_queues(mesh, num_cq)
    if trace_on:
        print(
            f"[trace] HY_TRACE=1 mesh: trace_region={trace_region // (1024 * 1024)} MiB "
            f"num_command_queues={device_num_command_queues(mesh)}",
            flush=True,
        )
    else:
        print("[trace] HY_TRACE=0 mesh: eager 1CQ (no trace region, no 2CQ)", flush=True)
    return mesh


def open_pipeline_mesh(mesh_shape, *, l1_small_size: int = 32768):
    """Open one mesh for the full T2I pipeline (recaption, denoise, VAE)."""
    return open_traced_mesh(mesh_shape, l1_small_size=l1_small_size)


def _open_stage_mesh(mesh_shape, stage: str, *, l1_small_size: int, enable_2cq: bool | None):
    """Open a per-stage mesh; ``enable_2cq=None`` follows ``HY_TRACE``."""
    if enable_2cq is None:
        enable_2cq = hy_trace_enabled()
    mesh = open_traced_mesh(mesh_shape, l1_small_size=l1_small_size, num_cq=2 if enable_2cq else 1)
    if enable_2cq and device_num_command_queues(mesh) < 2:
        print(f"[{stage}] warning: requested 2CQ but mesh opened with 1 CQ", flush=True)
    elif enable_2cq:
        print(f"[{stage}] 2CQ enabled (num_command_queues={device_num_command_queues(mesh)})", flush=True)
    return mesh


def release_stage_resources(mesh_device) -> None:
    """Sync device and collect host refs between pipeline stages (caller must del stage objects)."""
    import gc

    ttnn.synchronize_device(mesh_device)
    gc.collect()


def release_pipeline_traces(mesh_device) -> None:
    """Release cached execute_trace handles at pipeline teardown."""
    from models.experimental.hunyuan_image_3_0.ttnn.cond_encode_trace import release_cond_encode_tracers

    ttnn.synchronize_device(mesh_device)
    release_cond_encode_tracers()


def invalidate_cond_encode_traces(mesh_device) -> None:
    """Drop cond-encode traces before backbone load (trace DRAM is not stable across backbone)."""
    from models.experimental.hunyuan_image_3_0.ttnn.cond_encode_trace import (
        invalidate_cond_encode_traces as _invalidate,
    )

    ttnn.synchronize_device(mesh_device)
    _invalidate()


# ---------------------------------------------------------------------------------------
# AR recaption: CQ0 backbone + LM head, CQ1 logits D2H
# ---------------------------------------------------------------------------------------


def open_recaption_mesh(mesh_shape, *, l1_small_size: int = 32768, enable_2cq: bool | None = None):
    """Open a mesh for AR recaption with optional 2 command queues and trace region."""
    return _open_stage_mesh(mesh_shape, "recaption", l1_small_size=l1_small_size, enable_2cq=enable_2cq)


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


def open_denoise_mesh(mesh_shape, *, l1_small_size: int = 32768, enable_2cq: bool | None = None):
    """Open a mesh for DiT denoise with optional two command queues."""
    return _open_stage_mesh(mesh_shape, "denoise", l1_small_size=l1_small_size, enable_2cq=enable_2cq)


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


def open_vae_mesh(mesh_shape, *, l1_small_size: int = 32768, enable_2cq: bool | None = None):
    """Open a mesh for VAE decode with optional two command queues."""
    return _open_stage_mesh(mesh_shape, "vae", l1_small_size=l1_small_size, enable_2cq=enable_2cq)


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
