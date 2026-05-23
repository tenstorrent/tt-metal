# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""ACE-Step TT device helpers: mesh SKU resolution, split-device lifecycle, mesh readback, CFG DP."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import numpy as np
import torch

import ttnn

# Logical mesh SKUs → (rows, cols). Preprocess (Qwen / 5 Hz LM) must stay on 1×1; DiT/VAE use the full mesh.
_MESH_SKU_SHAPES: dict[str, Tuple[int, int]] = {
    "P100": (1, 1),
    "P150": (1, 1),
    "N150": (1, 1),
    "N300": (1, 2),
    "P300": (1, 2),
    "BH_QB": (2, 2),
    "P150x4": (2, 2),
    "BH_LB": (2, 4),
    "T3K": (1, 8),
    "N150x4": (1, 4),
    "TG": (8, 4),
    "BHGLX": (4, 8),
}


def resolve_ace_step_mesh_sku(*, cli_value: str | None = None) -> str | None:
    """Resolve mesh SKU from CLI, ``ACE_STEP_MESH_DEVICE``, then ``MESH_DEVICE``."""
    for raw in (
        cli_value,
        os.environ.get("ACE_STEP_MESH_DEVICE"),
        os.environ.get("MESH_DEVICE"),
    ):
        if raw is None:
            continue
        key = str(raw).strip()
        if not key:
            continue
        upper = key.upper()
        if upper in _MESH_SKU_SHAPES:
            return upper
        raise ValueError(f"Unknown ACE-Step mesh SKU {key!r}. Supported: {', '.join(sorted(_MESH_SKU_SHAPES))}.")
    return None


def ace_step_mesh_shape(mesh_sku: str | None) -> Tuple[int, int]:
    if mesh_sku is None:
        return (1, 1)
    if mesh_sku not in _MESH_SKU_SHAPES:
        raise ValueError(f"Unknown mesh SKU {mesh_sku!r}")
    return _MESH_SKU_SHAPES[mesh_sku]


def ace_step_needs_split_device(mesh_sku: str | None) -> bool:
    """True when preprocess must run on card 0 and DiT on a multi-device mesh."""
    rows, cols = ace_step_mesh_shape(mesh_sku)
    return int(rows) * int(cols) > 1


def ace_step_mesh_use_split_ttnn_preprocess(mesh_sku: str | None) -> bool:
    """Run LM + Qwen + condition on a 1×1 TTNN device before opening the DiT mesh (BH_QB Phase A).

    Set ``ACE_STEP_MESH_HOST_PREPROCESS=1`` to restore legacy CPU ``prepare_condition`` on mesh.
    """
    if os.environ.get("ACE_STEP_MESH_HOST_PREPROCESS", "").lower() in ("1", "true", "yes", "on"):
        return False
    return ace_step_needs_split_device(mesh_sku)


def ace_step_mesh_use_host_temb_precompute(device: Any) -> bool:
    """Precompute timestep embeddings on CPU; device ``time_embed`` linears stall on BH 2×2."""
    return ace_step_device_num_chips(device) > 1


def ace_step_mesh_use_host_cfg_euler(device: Any) -> bool:
    """Run APG/ADG + Euler on CPU after each DiT forward (trace or eager) on multi-device meshes.

    Device-side guidance ops on BH 2×2 can add noise vs the torch reference; host math matches
    P150 while DiT stays on the mesh (optionally traced). Set ``ACE_STEP_MESH_HOST_CFG_EULER=0``
    to restore the legacy all-device denoise loop.
    """
    if os.environ.get("ACE_STEP_MESH_HOST_CFG_EULER", "").lower() in ("0", "false", "no", "off"):
        return False
    return ace_step_device_num_chips(device) > 1


def ace_step_resolve_vae_tiling(
    *,
    frames: int,
    mesh_sku: str | None,
    chunk_cli: int,
    overlap_cli: int,
) -> tuple[int, int]:
    """Pick TTNN VAE ``decode_tiled`` chunk/overlap; mesh long clips use wider overlap (fewer seams).

    Env ``ACE_STEP_VAE_CHUNK_LATENTS`` / ``ACE_STEP_VAE_OVERLAP_LATENTS`` override CLI when set.
    """
    chunk = int(os.environ.get("ACE_STEP_VAE_CHUNK_LATENTS", str(int(chunk_cli))))
    overlap = int(os.environ.get("ACE_STEP_VAE_OVERLAP_LATENTS", str(int(overlap_cli))))
    frames_i = int(frames)
    on_mesh = mesh_sku is not None and ace_step_needs_split_device(mesh_sku)
    if on_mesh:
        # 15 s @ 25 Hz ≈ 375 frames → ~16 tiles at overlap=4; wider overlap reduces boundary artifacts.
        if frames_i >= 400 and overlap < 12:
            overlap = 12
        elif frames_i >= 200 and overlap < 8:
            overlap = 8
    elif frames_i > 500 and overlap < 8:
        overlap = 8
    # decode_tiled requires chunk_size > 2 * overlap (stride > 0).
    while chunk - 2 * overlap <= 0 and overlap > 4:
        overlap //= 2
    # Multi-device mesh: each overlap-add tile runs a full VAE forward; large chunk windows
    # (e.g. 48 latents) need ~23 MiB L1 per conv and OOM after many tiles without per-tile free.
    if on_mesh:
        max_chunk = int(os.environ.get("ACE_STEP_VAE_MAX_CHUNK_LATENTS", "32"))
        allow_large = os.environ.get("ACE_STEP_VAE_ALLOW_LARGE_CHUNK", "").lower() in ("1", "true", "yes", "on")
        if chunk > max_chunk and not allow_large:
            prev_chunk, prev_overlap = int(chunk), int(overlap)
            chunk = max_chunk
            while chunk - 2 * overlap <= 0 and overlap > 4:
                overlap //= 2
            if chunk - 2 * overlap <= 0:
                overlap = max(4, (chunk - 4) // 2)
            print(
                f"[ace_step_v1_5] VAE: mesh chunk/overlap {prev_chunk}/{prev_overlap} "
                f"-> {chunk}/{overlap} (L1-safe; set ACE_STEP_VAE_ALLOW_LARGE_CHUNK=1 to keep "
                f"{prev_chunk}/{prev_overlap})",
                flush=True,
            )
    return chunk, overlap


def ace_step_mesh_perf_log_default(*, mesh_sku: str | None) -> bool:
    """Enable wall-clock perf logging by default on multi-device mesh runs."""
    env = os.environ.get("ACE_STEP_DEMO_PERF_LOG", os.environ.get("ACE_STEP_PERF_LOG", "")).lower()
    if env in ("0", "false", "no", "off"):
        return False
    if env in ("1", "true", "yes", "on"):
        return True
    return mesh_sku is not None and ace_step_needs_split_device(mesh_sku)


def ace_step_mesh_use_adg(*, mesh_sku: str | None, variant: str, cli_use_adg: bool | None) -> bool:
    """CFG guidance on multi-device meshes defaults to APG (host/device); ADG can sound harsh on BH.

    Single-chip P150 keeps ADG for base. Set ``ACE_STEP_MESH_USE_ADG=1`` or pass ``--use-adg`` to force ADG on mesh.
    """
    is_turbo = "turbo" in str(variant).lower()
    if is_turbo:
        return False
    is_base = "base" in str(variant).lower() or "sft" in str(variant).lower()
    if not is_base:
        return bool(cli_use_adg)
    if cli_use_adg is not None:
        return bool(cli_use_adg)
    if mesh_sku is not None and ace_step_needs_split_device(mesh_sku):
        return os.environ.get("ACE_STEP_MESH_USE_ADG", "").lower() in ("1", "true", "yes", "on")
    return True


def ace_step_mesh_use_host_latent_sampler(device: Any, *, use_trace: bool) -> bool:
    """Keep latents + Euler/CFG on host for eager multi-device runs (BH_QB).

    When ``use_trace`` is set, latents stay on device and the trace denoise loop runs instead.
    """
    if bool(use_trace):
        return False
    return ace_step_device_num_chips(device) > 1


def ace_step_mesh_use_sequential_cfg(device: Any, *, do_cfg: bool) -> bool:
    """Run CFG as two B=1 forwards on multi-device meshes (batch=2 single forward can stall on BH)."""
    return bool(do_cfg) and ace_step_device_num_chips(device) > 1


def ace_step_dit_pipe_batch_size(device: Any, *, do_cfg: bool) -> int:
    """Effective DiT ``fuse_batch`` M for matmul/trace gates (B=1 under mesh sequential CFG)."""
    if ace_step_mesh_use_sequential_cfg(device, do_cfg=do_cfg):
        return 1
    return 2 if bool(do_cfg) else 1


def run_mesh_sequential_cfg_forwards(
    *,
    pipe: Any,
    xt_b1: ttnn.Tensor,
    enc_tt_pipe: ttnn.Tensor,
    ctx_tt_pipe: ttnn.Tensor,
    temb_bd: ttnn.Tensor,
    timestep_proj_b6d: ttnn.Tensor,
    encoder_attention_mask_1d_bk: Any | None,
    device: Any,
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """Two sequential B=1 DiT forwards on the full mesh (cond then uncond)."""
    enc_cond = slice_batch_dim0(enc_tt_pipe, 0, 1)
    enc_uncond = slice_batch_dim0(enc_tt_pipe, 1, 2)
    ctx_cond = slice_batch_dim0(ctx_tt_pipe, 0, 1)
    ctx_uncond = slice_batch_dim0(ctx_tt_pipe, 1, 2)

    enc_mask_cond = None
    enc_mask_uncond = None
    if encoder_attention_mask_1d_bk is not None:
        enc_mask_cond = encoder_attention_mask_1d_bk[0:1]
        enc_mask_uncond = encoder_attention_mask_1d_bk[1:2]

    vpc_rm = pipe.forward_with_temb_tp(
        xt_bt64=xt_b1,
        context_latents_bt128=ctx_cond,
        encoder_hidden_states_btd=enc_cond,
        temb_bd=temb_bd,
        timestep_proj_b6d=timestep_proj_b6d,
        attention_mask_1d_bt=None,
        encoder_attention_mask_1d_bk=enc_mask_cond,
    )
    ace_step_synchronize_device(ttnn, device)

    vpu_rm = pipe.forward_with_temb_tp(
        xt_bt64=xt_b1,
        context_latents_bt128=ctx_uncond,
        encoder_hidden_states_btd=enc_uncond,
        temb_bd=temb_bd,
        timestep_proj_b6d=timestep_proj_b6d,
        attention_mask_1d_bt=None,
        encoder_attention_mask_1d_bk=enc_mask_uncond,
    )
    ace_step_synchronize_device(ttnn, device)
    return vpc_rm, vpu_rm


def ace_step_preprocess_num_command_queues(*, use_trace: bool) -> int:
    """Command queues for the 1×1 preprocess device (Qwen, detokenizer, optional 5 Hz LM).

    Trace replay overlaps host→device copies on CQ 1 with ``execute_trace`` on CQ 0; opening with
    a single CQ while ``use_trace`` is enabled triggers ``cq_id 1 is out of range``.
    """
    return 2 if bool(use_trace) else 1


def ace_step_device_num_command_queues(device: Any) -> int:
    """Best-effort CQ count for runtime guards (trace helpers fall back to eager when < 2)."""
    n = getattr(device, "num_command_queues", None)
    if n is not None:
        return max(1, int(n))
    fn = getattr(device, "get_num_command_queues", None)
    if callable(fn):
        return max(1, int(fn()))
    cqs = getattr(device, "mesh_command_queues_", None)
    if cqs is not None:
        return max(1, len(cqs))
    return 1


def ace_step_open_kwargs(*, num_command_queues: int = 1) -> dict[str, Any]:
    kw: dict[str, Any] = dict(
        l1_small_size=int(os.environ.get("ACE_STEP_L1_SMALL_SIZE", "98304")),
        trace_region_size=128 << 20,
    )
    if int(num_command_queues) > 1:
        kw["num_command_queues"] = int(num_command_queues)
    return kw


def open_preprocess_device(
    ttnn_mod: Any,
    *,
    device_id: int = 0,
    num_command_queues: int = 1,
) -> Any:
    """Open a 1×1 device for Qwen / 5 Hz LM / detokenizer (never a multi-device mesh).

    Use :func:`ace_step_preprocess_num_command_queues` when trace replay is enabled.
    """
    dev = ttnn_mod.open_device(
        device_id=int(device_id),
        **ace_step_open_kwargs(num_command_queues=num_command_queues),
    )
    if hasattr(dev, "enable_program_cache"):
        dev.enable_program_cache()
    return dev


def open_dit_device(
    ttnn_mod: Any,
    *,
    mesh_sku: str | None,
    device_id: int = 0,
    num_command_queues: int = 1,
) -> Any:
    """Open the DiT/VAE device: 1×1 for single-chip SKUs, ``open_mesh_device`` otherwise."""
    rows, cols = ace_step_mesh_shape(mesh_sku)
    open_kw = ace_step_open_kwargs(num_command_queues=num_command_queues)
    if rows * cols == 1:
        dev = ttnn_mod.open_device(device_id=int(device_id), **open_kw)
    else:
        if not hasattr(ttnn_mod, "open_mesh_device") or not hasattr(ttnn_mod, "MeshShape"):
            raise RuntimeError(
                f"Mesh SKU {mesh_sku!r} needs ttnn.open_mesh_device / MeshShape; build may be single-device only."
            )
        mesh_kw = {k: v for k, v in open_kw.items()}
        dev = ttnn_mod.open_mesh_device(ttnn_mod.MeshShape(int(rows), int(cols)), **mesh_kw)
    if hasattr(dev, "enable_program_cache"):
        dev.enable_program_cache()
    return dev


def close_ace_step_device(ttnn_mod: Any, device: Any) -> None:
    if device is None:
        return
    if isinstance(device, ttnn_mod.MeshDevice):
        ttnn_mod.close_mesh_device(device)
    else:
        ttnn_mod.close_device(device)


def transition_preprocess_to_dit_device(
    ttnn_mod: Any,
    preprocess_dev: Any,
    *,
    mesh_sku: str | None,
    device_id: int = 0,
    num_command_queues: int = 1,
) -> Any:
    """Close preprocess 1×1 device, then open the DiT mesh (avoids parent/submesh CQ conflicts)."""
    close_ace_step_device(ttnn_mod, preprocess_dev)
    return open_dit_device(
        ttnn_mod,
        mesh_sku=mesh_sku,
        device_id=device_id,
        num_command_queues=num_command_queues,
    )


def ace_step_is_mesh_device(device: Any) -> bool:
    return isinstance(device, ttnn.MeshDevice)


def ace_step_device_num_chips(device: Any) -> int:
    if hasattr(device, "get_num_devices"):
        return int(device.get_num_devices())
    return 1


def ace_step_mesh_is_2d(mesh_device: Any) -> bool:
    """True when ``mesh_device`` is a multi-chip 2-D mesh (e.g. BH_QB 2×2)."""
    if ace_step_device_num_chips(mesh_device) <= 1:
        return False
    if not hasattr(mesh_device, "shape"):
        return False
    shape = tuple(int(x) for x in mesh_device.shape)
    return len(shape) == 2 and int(shape[0]) > 1 and int(shape[1]) > 1


def ace_step_replicate_mesh_mapper(mesh_device: Any) -> Any | None:
    """Return a mapper that replicates host tensors to every chip in a mesh.

    On 2-D meshes (e.g. BH_QB 2×2) use ``ShardTensor2dMesh(..., dims=(None, None))`` for
    replicate semantics. ``ReplicateTensorToMesh`` can stall on large DiT uploads on Blackhole.
    """
    if mesh_device is None or not ace_step_is_mesh_device(mesh_device):
        return None
    if not hasattr(ttnn, "ReplicateTensorToMesh"):
        return None
    if ace_step_device_num_chips(mesh_device) <= 1:
        return None
    if ace_step_mesh_is_2d(mesh_device) and hasattr(ttnn, "ShardTensor2dMesh"):
        shape = tuple(int(x) for x in mesh_device.shape)
        return ttnn.ShardTensor2dMesh(mesh_device, dims=(None, None), mesh_shape=shape)
    return ttnn.ReplicateTensorToMesh(mesh_device)


def ace_step_synchronize_device(ttnn_mod: Any, device: Any) -> None:
    """Drain pending mesh/device work (no-op on failure)."""
    if device is None:
        return
    try:
        ttnn_mod.synchronize_device(device)
    except Exception:
        pass


def ace_step_dit_weight_mesh_mapper(_mesh_device: Any) -> Any | None:
    """Mesh mapper for DiT **weight** upload.

    Omit explicit mappers (same as patchify / output_head). ``ReplicateTensorToMesh`` and
    ``ShardTensor2dMesh`` on BH 2×2 can stall during large decoder-layer uploads.
    """
    return None


def ace_step_dit_rope_max_seq_len(
    *,
    expected_input_length: int | None,
    patch_size: int,
    hf_max: int,
) -> int:
    """Cap RoPE cache length to the padded patch sequence for this run (not full 4096)."""
    hf_max_i = int(hf_max)
    if expected_input_length is None or int(patch_size) <= 0:
        return hf_max_i
    frames = int(expected_input_length)
    ps = int(patch_size)
    remainder = frames % ps
    pad = 0 if remainder == 0 else (ps - remainder)
    patch_seq = (frames + pad) // ps
    margin = int(os.environ.get("ACE_STEP_ROPE_SEQ_MARGIN", "128"))
    cap = max(int(patch_seq) + margin, 128)
    out = min(hf_max_i, cap)
    if out < hf_max_i:
        print(
            f"[ace_step_v1_5] DiT RoPE max_seq_len={out} (capped from {hf_max_i} for T={frames})",
            flush=True,
        )
    return out


def ace_step_slice_encoder_mask_b1qk(
    mask: ttnn.Tensor | None,
    b0: int,
    b1_exc: int,
) -> ttnn.Tensor | None:
    """Slice batch on ``[B,1,S_q,S_k]`` encoder SDPA masks for sequential B=1 CFG."""
    if mask is None:
        return None
    s_q = int(mask.shape[2])
    s_k = int(mask.shape[3])
    return ttnn.slice(mask, (int(b0), 0, 0, 0), (int(b1_exc), 1, s_q, s_k))


def ace_step_log_mesh_quality_hints(
    *,
    mesh_sku: str | None,
    variant: str,
    infer_steps: int,
    guidance_scale: float,
    use_trace: bool,
    torch_vae: bool,
    use_adg: bool = True,
) -> None:
    """Log P150-equivalent quality settings for multi-device BH runs."""
    if mesh_sku is None:
        return
    rows, cols = ace_step_mesh_shape(mesh_sku)
    if int(rows) * int(cols) <= 1:
        return
    is_base = "base" in str(variant).lower() or "sft" in str(variant).lower()
    is_turbo = "turbo" in str(variant).lower()
    print(
        f"[ace_step_v1_5] mesh quality path: trace={'on' if use_trace else 'off'}, "
        f"VAE={'torch' if torch_vae else 'ttnn'}, CFG gs={guidance_scale:g}, steps={infer_steps}, "
        f"guidance={'ADG' if use_adg else 'APG'}",
        flush=True,
    )
    rows, cols = ace_step_mesh_shape(mesh_sku)
    if int(rows) * int(cols) > 1:
        if ace_step_mesh_use_split_ttnn_preprocess(mesh_sku):
            print(
                "[ace_step_v1_5] mesh: split TTNN preprocess (1×1 LM/Qwen/condition) → full mesh DiT/VAE; "
                "set ACE_STEP_MESH_HOST_PREPROCESS=1 for legacy CPU preprocess",
                flush=True,
            )
        print(
            "[ace_step_v1_5] mesh: host APG/ADG + Euler after each DiT step "
            "(DiT trace/eager on device; set ACE_STEP_MESH_HOST_CFG_EULER=0 to disable)",
            flush=True,
        )
    if is_base and not is_turbo and float(guidance_scale) < 6.0:
        print(
            "[ace_step_v1_5] warning: base/sft expects --guidance_scale 7 for clean audio "
            "(low CFG sounds noisy/hollow on BH_QB).",
            flush=True,
        )
    if is_base and not is_turbo and int(infer_steps) < 20:
        print(
            f"[ace_step_v1_5] warning: --infer_steps {int(infer_steps)} is very low for base "
            "(recommended 50; fewer steps increase noise/artifacts).",
            flush=True,
        )
    if is_base and not is_turbo and int(rows) * int(cols) > 1 and not use_adg:
        print(
            "[ace_step_v1_5] mesh tip: using APG (not ADG) on BH by default. If still noisy, A/B with "
            "``--torch-vae`` to isolate TTNN VAE or ``ACE_STEP_MESH_USE_ADG=1 --use-adg``.",
            flush=True,
        )
    elif is_base and not is_turbo and int(rows) * int(cols) > 1 and use_adg:
        print(
            "[ace_step_v1_5] mesh tip: ADG enabled. If audio is harsh/noisy try ``--no-use-adg`` (APG).",
            flush=True,
        )
    if not use_trace:
        print(
            "[ace_step_v1_5] warning: --no-use-trace on mesh uses the slower eager host-latent path; "
            "omit it for P150-equivalent trace + TTNN VAE on BH_QB.",
            flush=True,
        )
    if torch_vae:
        print(
            "[ace_step_v1_5] note: --torch-vae skips TTNN VAE; omit for on-device decode parity with P150.",
            flush=True,
        )


def ace_step_ttnn_to_torch(
    tensor: ttnn.Tensor,
    *,
    dtype: torch.dtype | None = None,
    mesh_device: Any = None,
) -> torch.Tensor:
    """Read back a TTNN tensor from single- or multi-device meshes.

    Uses ``to_torch_auto_compose`` so **replicated** activations (ACE-Step DiT/VAE on
    BH 2×2) are not concatenated across chips. Blind ``ConcatMesh2dToTensor`` on every
    tensor duplicates/garbles data and produces noisy audio.
    """
    from models.common.auto_compose import to_torch_auto_compose

    torch_dtype = dtype if dtype is not None else torch.float32
    out = to_torch_auto_compose(tensor, device=mesh_device)
    if out.dtype != torch_dtype:
        out = out.to(torch_dtype)
    return out.contiguous()


def ace_step_cfg_data_parallel_requested(*, cli_flag: bool = False) -> bool:
    if cli_flag:
        return True
    env = os.environ.get("ACE_STEP_CFG_DATA_PARALLEL", "")
    return str(env).lower() in ("1", "true", "yes", "on")


def ace_step_cfg_data_parallel_available(device: Any, *, do_cfg: bool, requested: bool) -> bool:
    if not requested or not do_cfg:
        return False
    n = ace_step_device_num_chips(device)
    return n >= 2 and (n % 2) == 0


def ace_step_cfg_dp_use_submeshes() -> bool:
    """True to split the parent mesh into submeshes (experimental; can hang on BH_QB during DiT init).

    Default is sequential cond/uncond forwards on the **full** mesh with a single DiT pipeline
    (``--cfg-data-parallel`` still splits CFG across two B=1 forwards, just not on separate submeshes).
    Set ``ACE_STEP_CFG_DP_SUBMESHES=1`` to opt into the submesh path.
    """
    return os.environ.get("ACE_STEP_CFG_DP_SUBMESHES", "").lower() in ("1", "true", "yes", "on")


def prepare_cfg_dp_submeshes(mesh_device: ttnn.MeshDevice) -> Tuple[ttnn.MeshDevice, ttnn.MeshDevice]:
    """Split an even-size mesh into two DP groups (cond / uncond), each of shape ``(1, N/2)``.

    Must be called **before** any kernels run on ``mesh_device``. While submeshes are
    open the **parent mesh must stay idle** (no ``ttnn.randn``, APG/Euler, or other parent
    dispatch); otherwise BH_QB hits CQ ownership hangs. CFG-DP keeps latents and guidance
    on the host and only runs DiT on the submeshes until they are closed.
    """
    n = int(mesh_device.get_num_devices())
    if n < 2 or (n % 2) != 0:
        raise ValueError(f"CFG data parallelism needs an even device count >= 2, got {n}")
    print(f"[ace_step_v1_5] CFG-DP: create_submeshes (1, {n // 2}) on parent mesh …", flush=True)
    submeshes = mesh_device.create_submeshes(ttnn.MeshShape(1, n // 2))
    if len(submeshes) < 2:
        raise RuntimeError(f"create_submeshes returned {len(submeshes)} groups, expected >= 2")
    print("[ace_step_v1_5] CFG-DP: submeshes open (cond / uncond)", flush=True)
    return submeshes[0], submeshes[1]


def slice_batch_dim0(t: ttnn.Tensor, b0: int, b1_exc: int) -> ttnn.Tensor:
    """Slice batch dimension 0 of a rank-3 tensor ``[B, T, C]``."""
    t_dim = int(t.shape[1])
    c_dim = int(t.shape[2])
    return ttnn.slice(t, (int(b0), 0, 0), (int(b1_exc), t_dim, c_dim))


def stage_row_tensor_to_device(
    t: ttnn.Tensor,
    *,
    target_device: Any,
    mem: Any,
) -> ttnn.Tensor:
    """Host-roundtrip staging so ``t`` is valid on ``target_device`` (parent mesh → submesh safe)."""
    from models.demos.ace_step_v1_5.ttnn_impl.dit_sampling_ttnn import bf16_row_from_numpy_bc

    try:
        src_dev = t.device()
    except Exception:
        src_dev = None
    host = ace_step_ttnn_to_torch(t, mesh_device=src_dev, dtype=torch.float32).detach().cpu().numpy()
    return bf16_row_from_numpy_bc(host.astype(np.float32, copy=False), device=target_device, dram=mem)


@dataclass
class AceStepCfgDpRuntime:
    """CFG runtime: sequential B=1 cond/uncond forwards (full mesh) or optional submesh split."""

    submesh_cond: ttnn.MeshDevice
    submesh_uncond: ttnn.MeshDevice
    parent_mesh: ttnn.MeshDevice
    pipe_cond: Any
    pipe_uncond: Any
    enc_cond: ttnn.Tensor
    enc_uncond: ttnn.Tensor
    ctx_cond: ttnn.Tensor
    ctx_uncond: ttnn.Tensor
    temb_cond_per_step: list[ttnn.Tensor]
    tp_cond_per_step: list[ttnn.Tensor]
    temb_uncond_per_step: list[ttnn.Tensor]
    tp_uncond_per_step: list[ttnn.Tensor]
    encoder_attention_mask_cond: ttnn.Tensor | None = None
    encoder_attention_mask_uncond: ttnn.Tensor | None = None
    owns_submeshes: bool = False

    def close_submeshes(self, ttnn_mod: Any) -> None:
        if not self.owns_submeshes:
            return
        for sub in (self.submesh_cond, self.submesh_uncond):
            if sub is self.parent_mesh:
                continue
            try:
                ttnn_mod.close_mesh_device(sub)
            except Exception:
                pass


def build_cfg_dp_runtime(
    *,
    parent_mesh: ttnn.MeshDevice,
    submesh_cond: ttnn.MeshDevice | None = None,
    submesh_uncond: ttnn.MeshDevice | None = None,
    pipe_factory: Callable[[ttnn.MeshDevice], Any],
    enc_tt_pipe: ttnn.Tensor | None = None,
    ctx_tt_pipe: ttnn.Tensor | None = None,
    enc_host_np: np.ndarray | None = None,
    ctx_host_np: np.ndarray | None = None,
    encoder_mask_host_np: np.ndarray | None = None,
    t_schedule: list[float],
    mem: Any,
    encoder_attention_mask_b1qk: ttnn.Tensor | None,
) -> AceStepCfgDpRuntime:
    """Build CFG state: one shared DiT pipeline on the full mesh (default) or two submesh pipelines."""
    from models.demos.ace_step_v1_5.ttnn_impl.dit_sampling_ttnn import bf16_row_from_numpy_bc

    if enc_host_np is None and enc_tt_pipe is None:
        raise ValueError("build_cfg_dp_runtime requires enc_host_np or enc_tt_pipe.")
    if ctx_host_np is None and ctx_tt_pipe is None:
        raise ValueError("build_cfg_dp_runtime requires ctx_host_np or ctx_tt_pipe.")
    if enc_host_np is not None:
        enc_chk = np.asarray(enc_host_np, dtype=np.float32)
        if enc_chk.ndim != 3 or int(enc_chk.shape[0]) < 2:
            raise ValueError(f"enc_host_np must be [2, S, D] for CFG, got {enc_chk.shape}")
    if ctx_host_np is not None:
        ctx_chk = np.asarray(ctx_host_np, dtype=np.float32)
        if ctx_chk.ndim != 3 or int(ctx_chk.shape[0]) < 2:
            raise ValueError(f"ctx_host_np must be [2, T, C] for CFG, got {ctx_chk.shape}")

    owns_submeshes = submesh_cond is not None and submesh_uncond is not None
    if not owns_submeshes:
        submesh_cond = parent_mesh
        submesh_uncond = parent_mesh

    def _stage_enc(row: int, target: Any) -> ttnn.Tensor:
        if enc_host_np is not None:
            enc_host = np.asarray(enc_host_np, dtype=np.float32)
            return bf16_row_from_numpy_bc(enc_host[row : row + 1], device=target, dram=mem)
        assert enc_tt_pipe is not None
        return stage_row_tensor_to_device(slice_batch_dim0(enc_tt_pipe, row, row + 1), target_device=target, mem=mem)

    def _stage_ctx(row: int, target: Any) -> ttnn.Tensor:
        if ctx_host_np is not None:
            ctx_host = np.asarray(ctx_host_np, dtype=np.float32)
            return bf16_row_from_numpy_bc(ctx_host[row : row + 1], device=target, dram=mem)
        assert ctx_tt_pipe is not None
        return stage_row_tensor_to_device(slice_batch_dim0(ctx_tt_pipe, row, row + 1), target_device=target, mem=mem)

    def _stage_cond_ctx_and_masks() -> (
        tuple[
            ttnn.Tensor,
            ttnn.Tensor,
            ttnn.Tensor,
            ttnn.Tensor,
            ttnn.Tensor | None,
            ttnn.Tensor | None,
        ]
    ):
        print("[ace_step_v1_5] CFG: staging enc_cond …", flush=True)
        enc_c = _stage_enc(0, submesh_cond)
        print("[ace_step_v1_5] CFG: staging enc_uncond …", flush=True)
        enc_u = _stage_enc(1, submesh_uncond)
        print("[ace_step_v1_5] CFG: staging ctx_cond …", flush=True)
        ctx_c = _stage_ctx(0, submesh_cond)
        print("[ace_step_v1_5] CFG: staging ctx_uncond …", flush=True)
        ctx_u = _stage_ctx(1, submesh_uncond)
        mask_c = None
        mask_u = None
        if encoder_mask_host_np is not None:
            mask_host = np.asarray(encoder_mask_host_np, dtype=np.float32)
            mask_c = bf16_row_from_numpy_bc(mask_host[0:1], device=submesh_cond, dram=mem)
            mask_u = bf16_row_from_numpy_bc(mask_host[1:2], device=submesh_uncond, dram=mem)
        elif encoder_attention_mask_b1qk is not None:
            mask_c = stage_row_tensor_to_device(
                slice_batch_dim0(encoder_attention_mask_b1qk, 0, 1),
                target_device=submesh_cond,
                mem=mem,
            )
            mask_u = stage_row_tensor_to_device(
                slice_batch_dim0(encoder_attention_mask_b1qk, 1, 2),
                target_device=submesh_uncond,
                mem=mem,
            )
        ace_step_synchronize_device(ttnn, parent_mesh)
        print("[ace_step_v1_5] CFG: cond/ctx staged", flush=True)
        return enc_c, enc_u, ctx_c, ctx_u, mask_c, mask_u

    def _load_pipes() -> tuple[Any, Any]:
        if owns_submeshes:
            print("[ace_step_v1_5] CFG-DP: loading cond DiT pipeline on submesh …", flush=True)
            p_cond = pipe_factory(submesh_cond)
            print("[ace_step_v1_5] CFG-DP: cond DiT pipeline ready", flush=True)
            print("[ace_step_v1_5] CFG-DP: loading uncond DiT pipeline on submesh …", flush=True)
            p_uncond = pipe_factory(submesh_uncond)
            print("[ace_step_v1_5] CFG-DP: uncond DiT pipeline ready", flush=True)
            return p_cond, p_uncond
        print(
            "[ace_step_v1_5] CFG: loading DiT pipeline on full mesh "
            "(sequential cond/uncond B=1 forwards; set ACE_STEP_CFG_DP_SUBMESHES=1 for submesh split)",
            flush=True,
        )
        p_cond = pipe_factory(parent_mesh)
        print("[ace_step_v1_5] CFG: DiT pipeline ready", flush=True)
        return p_cond, p_cond

    if owns_submeshes:
        enc_cond, enc_uncond, ctx_cond, ctx_uncond, mask_cond, mask_uncond = _stage_cond_ctx_and_masks()
        pipe_cond, pipe_uncond = _load_pipes()
    else:
        pipe_cond, pipe_uncond = _load_pipes()
        enc_cond, enc_uncond, ctx_cond, ctx_uncond, mask_cond, mask_uncond = _stage_cond_ctx_and_masks()
    num_steps = len(t_schedule)
    temb_cond_per_step: list[ttnn.Tensor] = []
    tp_cond_per_step: list[ttnn.Tensor] = []
    temb_uncond_per_step: list[ttnn.Tensor] = []
    tp_uncond_per_step: list[ttnn.Tensor] = []
    for idx in range(num_steps):
        temb_c, tp_c = pipe_cond.compute_temb_tp(int(idx), target_batch=1)
        temb_u, tp_u = pipe_uncond.compute_temb_tp(int(idx), target_batch=1)
        temb_cond_per_step.append(temb_c)
        tp_cond_per_step.append(tp_c)
        temb_uncond_per_step.append(temb_u)
        tp_uncond_per_step.append(tp_u)

    return AceStepCfgDpRuntime(
        submesh_cond=submesh_cond,
        submesh_uncond=submesh_uncond,
        parent_mesh=parent_mesh,
        pipe_cond=pipe_cond,
        pipe_uncond=pipe_uncond,
        enc_cond=enc_cond,
        enc_uncond=enc_uncond,
        ctx_cond=ctx_cond,
        ctx_uncond=ctx_uncond,
        temb_cond_per_step=temb_cond_per_step,
        tp_cond_per_step=tp_cond_per_step,
        temb_uncond_per_step=temb_uncond_per_step,
        tp_uncond_per_step=tp_uncond_per_step,
        encoder_attention_mask_cond=mask_cond,
        encoder_attention_mask_uncond=mask_uncond,
        owns_submeshes=owns_submeshes,
    )


def run_cfg_dp_dit_forwards(
    *,
    cfg_dp: AceStepCfgDpRuntime,
    step_idx: int,
    xt_cond: ttnn.Tensor,
    xt_uncond: ttnn.Tensor,
    encoder_attn_1d_bk_np: Optional[Any] = None,
) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
    """Run cond then uncond DiT forwards on separate submeshes (single-threaded dispatch)."""
    vpc_rm = cfg_dp.pipe_cond.forward_with_temb_tp(
        xt_bt64=xt_cond,
        context_latents_bt128=cfg_dp.ctx_cond,
        encoder_hidden_states_btd=cfg_dp.enc_cond,
        temb_bd=cfg_dp.temb_cond_per_step[int(step_idx)],
        timestep_proj_b6d=cfg_dp.tp_cond_per_step[int(step_idx)],
        attention_mask_1d_bt=None,
        encoder_attention_mask_1d_bk=None if cfg_dp.encoder_attention_mask_cond is not None else encoder_attn_1d_bk_np,
        encoder_attention_mask_b1qk=cfg_dp.encoder_attention_mask_cond,
    )
    ttnn.synchronize_device(cfg_dp.submesh_cond)

    vpu_rm = cfg_dp.pipe_uncond.forward_with_temb_tp(
        xt_bt64=xt_uncond,
        context_latents_bt128=cfg_dp.ctx_uncond,
        encoder_hidden_states_btd=cfg_dp.enc_uncond,
        temb_bd=cfg_dp.temb_uncond_per_step[int(step_idx)],
        timestep_proj_b6d=cfg_dp.tp_uncond_per_step[int(step_idx)],
        attention_mask_1d_bt=None,
        encoder_attention_mask_1d_bk=None
        if cfg_dp.encoder_attention_mask_uncond is not None
        else encoder_attn_1d_bk_np,
        encoder_attention_mask_b1qk=cfg_dp.encoder_attention_mask_uncond,
    )
    ttnn.synchronize_device(cfg_dp.submesh_uncond)
    return vpc_rm, vpu_rm
