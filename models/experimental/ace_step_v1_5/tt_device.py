# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""ACE-Step TT device helpers: mesh SKU resolution, split-device lifecycle, mesh readback."""

from __future__ import annotations

import os
from typing import Any, Tuple

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
    """Run LM + Qwen + condition on a 1×1 TTNN device before opening the DiT mesh (BH_QB Phase A)."""
    return ace_step_needs_split_device(mesh_sku)


def ace_step_mesh_use_host_temb_precompute(device: Any) -> bool:
    """Precompute timestep embeddings on CPU; device ``time_embed`` linears stall on BH 2×2."""
    return ace_step_device_num_chips(device) > 1


def ace_step_mesh_use_host_cfg_euler(device: Any) -> bool:
    """Run APG/ADG + Euler on CPU after each DiT forward (trace or eager) on multi-device meshes."""
    return ace_step_device_num_chips(device) > 1


def ace_step_resolve_vae_tiling(
    *,
    frames: int,
    mesh_sku: str | None,
    chunk_cli: int,
    overlap_cli: int,
) -> tuple[int, int]:
    """Pick TTNN VAE ``decode_tiled`` chunk/overlap; mesh long clips use wider overlap (fewer seams)."""
    chunk = int(chunk_cli)
    overlap = int(overlap_cli)
    frames_i = int(frames)
    on_mesh = mesh_sku is not None and ace_step_needs_split_device(mesh_sku)
    if on_mesh:
        # 15 s @ 25 Hz ≈ 375 frames → ~16 tiles at overlap=4; wider overlap reduces boundary artifacts.
        if frames_i >= 1000 and overlap < 14:
            overlap = 14
        elif frames_i >= 750 and overlap < 14:
            overlap = 14
        elif frames_i >= 400 and overlap < 12:
            overlap = 12
        elif frames_i >= 200 and overlap < 8:
            overlap = 8
    elif frames_i > 500 and overlap < 8:
        overlap = 8
    # decode_tiled requires chunk_size > 2 * overlap (stride > 0).
    while chunk - 2 * overlap <= 0 and overlap > 4:
        overlap //= 2
    # Multi-device mesh: each overlap-add tile runs a full VAE forward; cap chunk for L1 safety.
    if on_mesh and chunk > 32:
        prev_chunk, prev_overlap = int(chunk), int(overlap)
        chunk = 32
        while chunk - 2 * overlap <= 0 and overlap > 4:
            overlap //= 2
        if chunk - 2 * overlap <= 0:
            overlap = max(4, (chunk - 4) // 2)
        print(
            f"[ace_step_v1_5] VAE: mesh chunk/overlap {prev_chunk}/{prev_overlap} -> {chunk}/{overlap} (L1-safe)",
            flush=True,
        )
    return chunk, overlap


def ace_step_mesh_perf_log_default(*, mesh_sku: str | None) -> bool:
    """Wall-clock perf logging default (same as :func:`ace_step_perf_logging_enabled`)."""
    del mesh_sku  # kept for call-site compatibility
    from models.experimental.ace_step_v1_5.ace_step_perf_log import ace_step_perf_logging_enabled

    return ace_step_perf_logging_enabled()


def ace_step_mesh_use_adg(*, mesh_sku: str | None, variant: str, cli_use_adg: bool | None) -> bool:
    """CFG guidance: ADG for base/sft (single-chip and mesh); APG only when ``--no-use-adg``."""
    is_turbo = "turbo" in str(variant).lower()
    if is_turbo:
        return False
    is_base = "base" in str(variant).lower() or "sft" in str(variant).lower()
    if not is_base:
        return bool(cli_use_adg)
    if cli_use_adg is not None:
        return bool(cli_use_adg)
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
    margin = 128
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
                "[ace_step_v1_5] mesh: split TTNN preprocess (1×1 LM/Qwen/condition) → full mesh DiT/VAE",
                flush=True,
            )
        print(
            "[ace_step_v1_5] mesh: host APG/ADG + Euler after each DiT step (DiT trace/eager on device)",
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
            "[ace_step_v1_5] mesh tip: using APG (--no-use-adg). If harsh/noisy try default ADG (omit --no-use-adg).",
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


def slice_batch_dim0(t: ttnn.Tensor, b0: int, b1_exc: int) -> ttnn.Tensor:
    """Slice batch dimension 0 of a rank-3 tensor ``[B, T, C]``."""
    t_dim = int(t.shape[1])
    c_dim = int(t.shape[2])
    return ttnn.slice(t, (int(b0), 0, 0), (int(b1_exc), t_dim, c_dim))
