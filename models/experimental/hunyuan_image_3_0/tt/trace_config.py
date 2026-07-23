# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Trace / 2CQ policy for HunyuanImage-3.0.
#
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

from __future__ import annotations

import os

from models.experimental.hunyuan_image_3_0.ref.model_config import NUM_HIDDEN_LAYERS

import ttnn

_TRACE_REGION_MB_MIN = 128
_TRACE_REGION_MB_MAX = 512
_TRACE_REGION_MB_PER_LAYER = 8

# Set by demos before mesh open so startup logs match the denoise loop.
_denoise_trace_steps: int | None = None


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


def vision_encode_trace_enabled() -> bool:
    """Alias for ViT/aligner — gated with cond encode (no separate capture path)."""
    return cond_encode_trace_enabled()


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


def recaption_2cq_enabled(device) -> bool:
    if not hy_trace_enabled():
        return False
    from models.experimental.hunyuan_image_3_0.tt.ar_dual_cq import device_num_command_queues

    n = device_num_command_queues(device)
    if n < 2:
        print(f"[trace] HY_TRACE=1 but num_command_queues={n}; recaption 2CQ disabled", flush=True)
        return False
    return True


def denoise_2cq_enabled(device) -> bool:
    if not hy_trace_enabled():
        return False
    from models.experimental.hunyuan_image_3_0.tt.ar_dual_cq import device_num_command_queues

    n = device_num_command_queues(device)
    if n < 2:
        print(f"[trace] HY_TRACE=1 but num_command_queues={n}; denoise 2CQ disabled", flush=True)
        return False
    return True


def vae_2cq_enabled(device) -> bool:
    if not hy_trace_enabled():
        return False
    from models.experimental.hunyuan_image_3_0.tt.ar_dual_cq import device_num_command_queues

    n = device_num_command_queues(device)
    if n < 2:
        print(f"[trace] HY_TRACE=1 but num_command_queues={n}; VAE 2CQ disabled", flush=True)
        return False
    return True


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
    from models.experimental.hunyuan_image_3_0.tt.ar_dual_cq import (
        _stash_mesh_command_queues,
        device_num_command_queues,
    )

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


def release_stage_resources(mesh_device) -> None:
    """Sync device and collect host refs between pipeline stages (caller must del stage objects)."""
    import gc

    ttnn.synchronize_device(mesh_device)
    gc.collect()


def release_pipeline_traces(mesh_device) -> None:
    """Release cached execute_trace handles at pipeline teardown."""
    from models.experimental.hunyuan_image_3_0.tt.cond_encode_trace import release_cond_encode_tracers

    ttnn.synchronize_device(mesh_device)
    release_cond_encode_tracers()


def invalidate_cond_encode_traces(mesh_device) -> None:
    """Drop cond-encode traces before backbone load (trace DRAM is not stable across backbone)."""
    from models.experimental.hunyuan_image_3_0.tt.cond_encode_trace import invalidate_cond_encode_traces as _invalidate

    ttnn.synchronize_device(mesh_device)
    _invalidate()
