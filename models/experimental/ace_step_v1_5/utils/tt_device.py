# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""ACE-Step TT device helpers: mesh SKU resolution, split-device lifecycle, mesh readback."""

from __future__ import annotations

import os
from pathlib import Path
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


def ace_step_mesh_use_host_preprocess(mesh_sku: str | None) -> bool:
    """Run preprocess on host PyTorch (skip TTNN Phase A on mesh)."""
    if not ace_step_needs_split_device(mesh_sku):
        return False
    raw = os.environ.get("ACE_STEP_MESH_HOST_PREPROCESS", "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def ace_step_mesh_use_split_ttnn_preprocess(mesh_sku: str | None) -> bool:
    """Run LM + Qwen + condition on a 1×1 TTNN device before opening the DiT mesh (BH_QB Phase A)."""
    if ace_step_mesh_use_host_preprocess(mesh_sku):
        return False
    return ace_step_needs_split_device(mesh_sku)


_ACE_STEP_VISIBLE_DEVICES_SAVED_ATTR = "_ace_step_tt_visible_devices_saved"
_TT_VISIBLE_DEVICES_ENV = "TT_VISIBLE_DEVICES"
_TT_MESH_GRAPH_DESC_PATH_ENV = "TT_MESH_GRAPH_DESC_PATH"
_TT_METAL_FORCE_REINIT_ENV = "TT_METAL_FORCE_REINIT"
_PREPROCESS_MGD_REL = "tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto"


def _tt_metal_repo_root() -> str:
    env_root = os.environ.get("TT_METAL_HOME")
    if env_root:
        return str(env_root)
    # utils/ -> ace_step_v1_5/ -> experimental/ -> models/ -> repo root
    return str(Path(__file__).resolve().parents[4])


def _preprocess_mesh_graph_descriptor_path() -> str:
    return str(Path(_tt_metal_repo_root()) / _PREPROCESS_MGD_REL)


_DIT_MGD_BY_SKU: dict[str, str] = {
    "BH_QB": "tt_metal/fabric/mesh_graph_descriptors/p300_x2_mesh_graph_descriptor.textproto",
    "P150x4": "tt_metal/fabric/mesh_graph_descriptors/p150_x4_mesh_graph_descriptor.textproto",
}


def _dit_mesh_graph_descriptor_path(mesh_sku: str | None) -> str | None:
    if mesh_sku is None:
        return None
    rel = _DIT_MGD_BY_SKU.get(str(mesh_sku).upper())
    if rel is None:
        return None
    return str(Path(_tt_metal_repo_root()) / rel)


def _apply_single_chip_open_env(device_id: int) -> dict[str, str | None] | None:
    """On multi-PCIe hosts, open one chip with P150 MGD (fabric auto-discovery breaks open_device)."""
    if os.environ.get(_TT_VISIBLE_DEVICES_ENV) is not None:
        return None
    try:
        n_pcie = int(ttnn.GetNumPCIeDevices())
    except Exception:
        return None
    if n_pcie <= 1:
        return None
    chip = str(int(device_id))
    saved: dict[str, str | None] = {
        _TT_VISIBLE_DEVICES_ENV: os.environ.get(_TT_VISIBLE_DEVICES_ENV),
        _TT_MESH_GRAPH_DESC_PATH_ENV: os.environ.get(_TT_MESH_GRAPH_DESC_PATH_ENV),
    }
    os.environ[_TT_VISIBLE_DEVICES_ENV] = chip
    if saved[_TT_MESH_GRAPH_DESC_PATH_ENV] is None:
        os.environ[_TT_MESH_GRAPH_DESC_PATH_ENV] = _preprocess_mesh_graph_descriptor_path()
    return saved


def _restrict_cluster_to_preprocess_chip(mesh_sku: str | None, device_id: int) -> dict[str, str | None] | None:
    """Limit UMD to one PCIe device for Phase A on multi-chip SKUs (default on for BH_QB).

    Sets ``TT_VISIBLE_DEVICES`` and a 1×1 P150 MGD so ``open_device`` does not require the
    full fabric mesh. DiT transition clears this and forces MetalContext re-init (see
    :func:`_ensure_full_cluster_env_for_dit`).

    Set ``ACE_STEP_PREPROCESS_SINGLE_CHIP=0`` to disable (not recommended on BH_QB).
    """
    if os.environ.get("ACE_STEP_PREPROCESS_SINGLE_CHIP", "").strip().lower() in (
        "0",
        "false",
        "no",
        "off",
    ):
        return _apply_single_chip_open_env(device_id)
    if mesh_sku is None or not ace_step_needs_split_device(mesh_sku):
        return None
    chip = str(int(device_id))
    saved: dict[str, str | None] = {
        _TT_VISIBLE_DEVICES_ENV: os.environ.get(_TT_VISIBLE_DEVICES_ENV),
        _TT_MESH_GRAPH_DESC_PATH_ENV: os.environ.get(_TT_MESH_GRAPH_DESC_PATH_ENV),
    }
    os.environ[_TT_VISIBLE_DEVICES_ENV] = chip
    if saved[_TT_MESH_GRAPH_DESC_PATH_ENV] is None:
        os.environ[_TT_MESH_GRAPH_DESC_PATH_ENV] = _preprocess_mesh_graph_descriptor_path()
    print(
        f"[ace_step_v1_5] preprocess: {_TT_VISIBLE_DEVICES_ENV}={chip} "
        f"(single-chip UMD cluster for Phase A; full mesh after preprocess close)",
        flush=True,
    )
    return saved


def _apply_preprocess_cluster_env(mesh_sku: str | None, device_id: int) -> dict[str, str | None] | None:
    """Env overrides for Phase A device open on multi-chip SKUs."""
    return _restrict_cluster_to_preprocess_chip(mesh_sku, device_id)


def _restore_cluster_visibility(saved: dict[str, str | None] | None) -> None:
    if saved is None:
        return
    for key in (_TT_VISIBLE_DEVICES_ENV, _TT_MESH_GRAPH_DESC_PATH_ENV, _TT_METAL_FORCE_REINIT_ENV):
        prior = saved.get(key)
        if prior is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = prior


def open_single_tt_device(
    ttnn_mod: Any,
    *,
    device_id: int = 0,
    num_command_queues: int = 1,
    mesh_sku: str | None = None,
) -> Any:
    """Open one TTNN device; applies single-chip env on multi-PCIe Blackhole hosts."""
    ace_step_preflight_devices_available(mesh_sku=mesh_sku, device_id=device_id, single_chip_only=True)
    visible_saved = _apply_preprocess_cluster_env(mesh_sku, device_id)
    if visible_saved is None:
        visible_saved = _apply_single_chip_open_env(device_id)
    try:
        dev = ttnn_mod.open_device(
            device_id=int(device_id),
            **ace_step_open_kwargs(num_command_queues=num_command_queues, ttnn_mod=ttnn_mod),
        )
    except RuntimeError as exc:
        _restore_cluster_visibility(visible_saved)
        msg = str(exc)
        if "tt_tlb_alloc" in msg or "TLB window" in msg:
            hint = _format_device_busy_hint(mesh_sku=mesh_sku, device_id=device_id, single_chip_only=True)
            if hint:
                raise RuntimeError(f"{msg}{hint}") from exc
        raise
    except Exception:
        _restore_cluster_visibility(visible_saved)
        raise
    if hasattr(dev, "enable_program_cache"):
        dev.enable_program_cache()
    if visible_saved is not None:
        setattr(dev, _ACE_STEP_VISIBLE_DEVICES_SAVED_ATTR, visible_saved)
    return dev


def _read_proc_cmdline(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
        return raw.replace(b"\0", b" ").decode(errors="replace").strip() or "?"
    except OSError:
        return "?"


def _is_live_userspace_pid(pid: int) -> bool:
    """True for a real process directory under ``/proc`` (skip kernel pid 0 placeholders)."""
    return int(pid) > 0 and Path(f"/proc/{int(pid)}").is_dir()


def ace_step_device_holders(device_ids: list[int] | None = None) -> list[tuple[int, int, str]]:
    """Return ``(chip_id, pid, cmdline)`` for foreign processes holding TT device TLBs."""
    if device_ids is None:
        device_ids = list(range(4))
    my_pid = os.getpid()
    holders: list[tuple[int, int, str]] = []
    seen: set[tuple[int, int]] = set()
    for chip_id in device_ids:
        pids_path = Path(f"/proc/driver/tenstorrent/{chip_id}/pids")
        if not pids_path.is_file():
            continue
        for tok in pids_path.read_text().split():
            if not tok.isdigit():
                continue
            pid = int(tok)
            if pid == my_pid or not _is_live_userspace_pid(pid) or (chip_id, pid) in seen:
                continue
            seen.add((chip_id, pid))
            holders.append((chip_id, pid, _read_proc_cmdline(pid)))
    return holders


def ace_step_preflight_devices_available(
    *,
    mesh_sku: str | None,
    device_id: int = 0,
    single_chip_only: bool = False,
) -> None:
    """Fail fast when another process holds the chip(s) we need (avoids opaque TLB -12 errors)."""
    chip_ids = [int(device_id)]
    if not single_chip_only and mesh_sku is not None and ace_step_needs_split_device(mesh_sku):
        rows, cols = ace_step_mesh_shape(mesh_sku)
        if int(rows) * int(cols) > 1:
            chip_ids = list(range(int(rows) * int(cols)))
    holders = ace_step_device_holders(chip_ids)
    if not holders:
        return
    lines = "\n".join(f"  chip {chip}: pid {pid} — {cmd}" for chip, pid, cmd in holders)
    sku_label = str(mesh_sku) if mesh_sku else f"device_id={device_id}"
    stale_pids = sorted({pid for _, pid, _ in holders})
    kill_hint = " ".join(f"kill {pid}" for pid in stale_pids)
    reset_chips = " ".join(f"tt-smi -r {chip}" for chip in sorted({chip for chip, _, _ in holders}))
    raise RuntimeError(
        "Tenstorrent device(s) busy — another process holds TLB/sysmem (tt_tlb_alloc -12 if ignored):\n"
        f"{lines}\n"
        f"Free the device, then rerun ACE-Step {sku_label}:\n"
        f"  {kill_hint}\n"
        f"  # if still busy: {reset_chips}\n"
        "Or set ACE_STEP_MESH_HOST_PREPROCESS=1 to skip TTNN Phase A (DiT mesh still needs all chips)."
    )


def _format_device_busy_hint(*, mesh_sku: str | None, device_id: int, single_chip_only: bool = False) -> str:
    chip_ids = [int(device_id)]
    if not single_chip_only and mesh_sku is not None and ace_step_needs_split_device(mesh_sku):
        rows, cols = ace_step_mesh_shape(mesh_sku)
        if int(rows) * int(cols) > 1:
            chip_ids = list(range(int(rows) * int(cols)))
    holders = ace_step_device_holders(chip_ids)
    if not holders:
        return ""
    lines = "\n".join(f"  chip {chip}: pid {pid} — {cmd}" for chip, pid, cmd in holders)
    return (
        "\nTenstorrent device(s) appear busy:\n"
        f"{lines}\n"
        "Stop those processes or reset chips (`tt-smi -r`) before retrying."
    )


def ace_step_mesh_use_host_temb_precompute(device: Any) -> bool:
    """Precompute timestep embeddings on CPU; device ``time_embed`` linears stall on BH 2×2."""
    return ace_step_device_num_chips(device) > 1


def ace_step_mesh_use_host_cfg_euler(device: Any) -> bool:
    """Run APG/ADG + Euler on CPU after each DiT forward (trace or eager) on multi-device meshes."""
    return ace_step_device_num_chips(device) > 1


def ace_step_mesh_use_pytorch_dit(
    *,
    mesh_sku: str | None,
    duration_sec: float,
    latent_frames: int,
) -> bool:
    """Opt-in HF PyTorch DiT denoise (``ACE_STEP_PYTORCH_DIT=1``) for A/B vs TTNN.

    Default demo path uses TTNN DiT with long/ultra-long clip quality presets on mesh.
    """
    _ = mesh_sku, duration_sec, latent_frames
    import os

    env = os.environ.get("ACE_STEP_PYTORCH_DIT", "")
    return env.lower() in ("1", "true", "yes", "on")


def ace_step_mesh_use_pytorch_condition(
    *,
    mesh_sku: str | None,
    duration_sec: float,
    latent_frames: int,
) -> bool:
    """Opt-in HF ``prepare_condition`` (``ACE_STEP_PYTORCH_CONDITION=1``) for A/B vs TTNN."""
    _ = mesh_sku, duration_sec, latent_frames
    import os

    env = os.environ.get("ACE_STEP_PYTORCH_CONDITION", "")
    return env.lower() in ("1", "true", "yes", "on")


def ace_step_torch_condition_handoff() -> bool:
    """Opt-in: keep host torch enc/ctx through re-exec and denoise staging (``ACE_STEP_TORCH_CONDITION_HANDOFF=1``)."""
    import os

    return os.environ.get("ACE_STEP_TORCH_CONDITION_HANDOFF", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def ace_step_mesh_use_device_condition_handoff(*, latent_frames: int) -> bool:
    """Restage enc/ctx on device for DiT after mesh condition encode (default for long clips).

    Skips keeping host torch enc/ctx for denoise staging. Tensors are still normalized through
    the validated f32 readback → BF16 TILE L1 upload path inside
    :func:`restage_condition_tensors_for_dit_mesh` — raw condition-encoder outputs are not fed
    directly to DiT (that mismatch caused the first P4 60s regression).
    """
    if ace_step_torch_condition_handoff():
        return False
    return int(latent_frames) >= 750


def ace_step_lm_hint_ctx_blend(*, latent_frames: int) -> float:
    """Deprecated: hint blending removed; use ``audio_cover_strength`` denoise switch instead."""
    _ = latent_frames
    return 1.0


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
        if frames_i >= 1000 and overlap < 15:
            overlap = 15
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
    _ = mesh_sku  # kept for call-site compatibility
    from models.experimental.ace_step_v1_5.utils.ace_step_perf_log import ace_step_perf_logging_enabled

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


def ace_step_open_kwargs(*, num_command_queues: int = 1, ttnn_mod: Any | None = None) -> dict[str, Any]:
    kw: dict[str, Any] = dict(
        l1_small_size=int(os.environ.get("ACE_STEP_L1_SMALL_SIZE", "98304")),
        trace_region_size=128 << 20,
    )
    if ttnn_mod is not None:
        kw["dispatch_core_config"] = ace_step_dit_dispatch_core_config(ttnn_mod)
    if int(num_command_queues) > 1:
        kw["num_command_queues"] = int(num_command_queues)
    return kw


def ace_step_dit_dispatch_core_config(ttnn_mod: Any) -> Any:
    """Explicit dispatch cores for DiT mesh open (avoids stale cluster type after preprocess reset)."""
    return ttnn_mod.DispatchCoreConfig(
        type=ttnn_mod.DispatchCoreType.WORKER,
        axis=ttnn_mod.DispatchCoreAxis.COL,
    )


def ace_step_preprocess_on_mesh() -> bool:
    """Phase-A-on-mesh: run preprocess (5 Hz LM / Qwen / condition / detok) on the FULL mesh
    instead of a 1×1 chip, so their TP shards. Opt-in via ``ACE_STEP_PREPROCESS_ON_MESH``;
    default off → the validated 1×1 preprocess + DiT-reexec path is unchanged."""
    return os.environ.get("ACE_STEP_PREPROCESS_ON_MESH", "").strip().lower() in ("1", "on", "true", "yes")


def open_preprocess_device(
    ttnn_mod: Any,
    *,
    device_id: int = 0,
    num_command_queues: int = 1,
    mesh_sku: str | None = None,
) -> Any:
    """Open the preprocess device: a 1×1 chip by default (Qwen / 5 Hz LM / detokenizer).

    When ``ACE_STEP_PREPROCESS_ON_MESH`` is set on a multi-device SKU, open the **full mesh**
    instead (same path as the DiT mesh: fabric + DiT MGD), so Phase-A modules run on the mesh and
    their TP activates. The demo then reuses this mesh for DiT (no reexec). Default: 1×1.

    When ``mesh_sku`` is a multi-device SKU (e.g. ``BH_QB``), the 1×1 path sets
    ``TT_VISIBLE_DEVICES`` to ``device_id`` so UMD does not start all chips during Phase A.
    """
    if ace_step_preprocess_on_mesh() and mesh_sku is not None and ace_step_needs_split_device(mesh_sku):
        print("[ace_step_v1_5] Phase-A-on-mesh: opening full mesh for preprocess (TP active)", flush=True)
        _ensure_full_cluster_env_for_dit(mesh_sku)
        return open_dit_device(
            ttnn_mod,
            mesh_sku=mesh_sku,
            device_id=device_id,
            num_command_queues=num_command_queues,
        )
    return open_single_tt_device(
        ttnn_mod,
        device_id=device_id,
        num_command_queues=num_command_queues,
        mesh_sku=mesh_sku,
    )


def open_dit_device(
    ttnn_mod: Any,
    *,
    mesh_sku: str | None,
    device_id: int = 0,
    num_command_queues: int = 1,
) -> Any:
    """Open the DiT/VAE device: 1×1 for single-chip SKUs, ``open_mesh_device`` otherwise."""
    ace_step_preflight_devices_available(mesh_sku=mesh_sku, device_id=device_id)
    rows, cols = ace_step_mesh_shape(mesh_sku)
    open_kw = ace_step_open_kwargs(num_command_queues=num_command_queues, ttnn_mod=ttnn_mod)
    if rows * cols == 1:
        dev = ttnn_mod.open_device(device_id=int(device_id), **open_kw)
    else:
        if not hasattr(ttnn_mod, "open_mesh_device") or not hasattr(ttnn_mod, "MeshShape"):
            raise RuntimeError(
                f"Mesh SKU {mesh_sku!r} needs ttnn.open_mesh_device / MeshShape; build may be single-device only."
            )
        mesh_kw = {k: v for k, v in open_kw.items()}
        dit_mgd = _dit_mesh_graph_descriptor_path(mesh_sku)
        # TP: CCL collectives require the fabric context up BEFORE the mesh opens. Gated on the
        # ACE_STEP_TP env so the legacy replicate path (fabric off) is unchanged.
        _tp_fabric = False
        from models.experimental.ace_step_v1_5.ttnn_impl.tp_config import ace_step_tp_env_requested

        if ace_step_tp_env_requested() and hasattr(ttnn_mod, "set_fabric_config") and hasattr(ttnn_mod, "FabricConfig"):
            from models.experimental.ace_step_v1_5.ttnn_impl.tp_config import ace_step_tp_full_requested

            ttnn_mod.set_fabric_config(ttnn_mod.FabricConfig.FABRIC_1D)
            _tp_fabric = True
            _mode = "4-way (all chips)" if ace_step_tp_full_requested() else "per-axis (2-way)"
            print(f"[ace_step_v1_5] TP: enabled FABRIC_1D for DiT mesh CCL ({_mode})", flush=True)
            print(
                "[ace_step_v1_5] TP: run WITH --use-trace. Eager TP is ~2.5x slower on the DiT "
                "(un-amortised CCL); trace replay recovers latency parity. See docs/TP4_PLAN.md.",
                flush=True,
            )
        try:
            dev = ttnn_mod.open_mesh_device(ttnn_mod.MeshShape(int(rows), int(cols)), **mesh_kw)
        except Exception as exc:
            hint = _format_mesh_open_failure_hint(
                mesh_sku=mesh_sku,
                rows=int(rows),
                cols=int(cols),
                mgd_path=dit_mgd,
            )
            raise RuntimeError(f"{exc}{hint}") from exc
        if _tp_fabric:
            try:
                setattr(dev, "_ace_tp_fabric_enabled", True)
            except Exception:
                pass
    if hasattr(dev, "enable_program_cache"):
        dev.enable_program_cache()
    return dev


def _format_mesh_open_failure_hint(
    *,
    mesh_sku: str | None,
    rows: int,
    cols: int,
    mgd_path: str | None,
) -> str:
    """Actionable context when ``open_mesh_device`` fails (MGD vs physical fabric topology)."""
    lines = [
        "",
        f"ACE-Step could not open DiT mesh {mesh_sku!r} ({rows}×{cols}).",
    ]
    if mgd_path:
        lines.append(f"MGD: {mgd_path}")
    lines.extend(
        [
            "This is usually fabric topology (not missing Python/checkpoint files):",
            "  • BH_QB needs a full 2×2 fabric mesh across 4 Blackhole chips (WARP400 links).",
            "  • If logs show 'Downgrading to mesh shape 2x1', only 2 chips are fabric-connected.",
            "  • Check WARP cabling / run `tt-smi -r` on all chips; verify with a bare 2×2 open_mesh_device test.",
            "  • On a single P150/P300 card use `--mesh-device P150` instead of BH_QB.",
            "  • Unknown motherboards (e.g. B850M-C) need an entry in "
            "tt_metal/fabric/physical_system_discovery.cpp (rebuild tt_metal after editing).",
        ]
    )
    return "\n".join(lines)


def close_ace_step_device(ttnn_mod: Any, device: Any, *, restore_cluster_env: bool = True) -> None:
    if device is None:
        return
    try:
        if hasattr(device, "is_remote_only") and device.is_remote_only():
            return
    except Exception:
        pass
    visible_saved = getattr(device, _ACE_STEP_VISIBLE_DEVICES_SAVED_ATTR, None)
    tp_fabric = bool(getattr(device, "_ace_tp_fabric_enabled", False))
    try:
        if isinstance(device, ttnn_mod.MeshDevice):
            ttnn_mod.close_mesh_device(device)
        else:
            ttnn_mod.close_device(device)
    except Exception:
        # Stale/remote-only meshes can abort pytest teardown (SubDeviceManagerTracker).
        pass
    if tp_fabric and hasattr(ttnn_mod, "set_fabric_config") and hasattr(ttnn_mod, "FabricConfig"):
        # Reset fabric so a subsequent (non-TP) open on this process starts clean.
        try:
            ttnn_mod.set_fabric_config(ttnn_mod.FabricConfig.DISABLED)
        except Exception:
            pass
    if restore_cluster_env and visible_saved is not None:
        _restore_cluster_visibility(visible_saved)


def _ensure_full_cluster_env_for_dit(mesh_sku: str | None) -> dict[str, str | None] | None:
    """Clear single-chip preprocess env and set DiT MGD before opening the full mesh."""
    if mesh_sku is None or not ace_step_needs_split_device(mesh_sku):
        return None
    saved: dict[str, str | None] = {
        _TT_VISIBLE_DEVICES_ENV: os.environ.get(_TT_VISIBLE_DEVICES_ENV),
        _TT_MESH_GRAPH_DESC_PATH_ENV: os.environ.get(_TT_MESH_GRAPH_DESC_PATH_ENV),
        _TT_METAL_FORCE_REINIT_ENV: os.environ.get(_TT_METAL_FORCE_REINIT_ENV),
    }
    os.environ.pop(_TT_VISIBLE_DEVICES_ENV, None)
    dit_mgd = _dit_mesh_graph_descriptor_path(mesh_sku)
    if dit_mgd is not None:
        os.environ[_TT_MESH_GRAPH_DESC_PATH_ENV] = dit_mgd
    os.environ[_TT_METAL_FORCE_REINIT_ENV] = "1"
    print(
        f"[ace_step_v1_5] DiT: full cluster ({mesh_sku}), {_TT_MESH_GRAPH_DESC_PATH_ENV} set",
        flush=True,
    )
    return saved


def transition_preprocess_to_dit_device(
    ttnn_mod: Any,
    preprocess_dev: Any,
    *,
    mesh_sku: str | None,
    device_id: int = 0,
    num_command_queues: int = 1,
) -> Any:
    """Close preprocess 1×1 device, then open the DiT mesh (same process; prefer :func:`ace_step_reexec_for_dit_mesh`)."""
    close_ace_step_device(ttnn_mod, preprocess_dev)
    dit_env_saved = _ensure_full_cluster_env_for_dit(mesh_sku)
    try:
        dev = open_dit_device(
            ttnn_mod,
            mesh_sku=mesh_sku,
            device_id=device_id,
            num_command_queues=num_command_queues,
        )
    except Exception:
        _restore_cluster_visibility(dit_env_saved)
        raise
    if dit_env_saved is not None:
        setattr(dev, _ACE_STEP_VISIBLE_DEVICES_SAVED_ATTR, dit_env_saved)
    return dev


def _resolve_demo_reexec_argv(argv: list[str]) -> list[str]:
    """Return ``[script_path, *args]`` for DiT mesh ``os.execv``.

    ``run_prompt_to_wav.main()`` under pytest often sets ``sys.argv[0]`` to a bare
    ``run_prompt_to_wav`` token; re-exec must use the real ``.py`` path under ``/work``.
    """
    if not argv:
        raise RuntimeError("ace_step_reexec_for_dit_mesh: empty argv")
    script, *rest = argv
    if script and os.path.isfile(script):
        return [script, *rest]
    demo_py = Path(__file__).resolve().parent.parent / "demo" / "run_prompt_to_wav.py"
    if demo_py.is_file():
        return [str(demo_py), *rest]
    raise RuntimeError(
        f"ace_step_reexec_for_dit_mesh: cannot resolve demo script (argv[0]={script!r}, " f"expected {demo_py})"
    )


def ace_step_reexec_for_dit_mesh(
    ttnn_mod: Any,
    *,
    preprocess_dev: Any,
    mesh_sku: str | None,
    argv: list[str],
    cached_preprocess: Any = None,
    deferred_condition_payload: Any = None,
    frames: int | None = None,
    preprocess_perf: dict | None = None,
) -> None:
    """Re-exec demo in a fresh process so DiT opens the full mesh after single-chip preprocess."""
    import pickle
    import sys
    import tempfile

    if cached_preprocess is None and deferred_condition_payload is None:
        raise RuntimeError("ace_step_reexec_for_dit_mesh requires cached_preprocess or deferred_condition_payload")

    close_ace_step_device(ttnn_mod, preprocess_dev, restore_cluster_env=False)

    fd, handoff_path = tempfile.mkstemp(prefix="ace_step_dit_handoff_", suffix=".pkl")
    os.close(fd)
    handoff_payload: dict = {
        "cached_preprocess": cached_preprocess,
        "deferred_condition_payload": deferred_condition_payload,
        "frames": frames,
        "mesh_sku": mesh_sku,
    }
    if preprocess_perf is not None:
        handoff_payload["preprocess_perf"] = preprocess_perf
    with open(handoff_path, "wb") as f:
        pickle.dump(handoff_payload, f)

    os.environ.pop(_TT_VISIBLE_DEVICES_ENV, None)
    dit_mgd = _dit_mesh_graph_descriptor_path(mesh_sku)
    if dit_mgd is not None:
        os.environ[_TT_MESH_GRAPH_DESC_PATH_ENV] = dit_mgd
    os.environ[_TT_METAL_FORCE_REINIT_ENV] = "1"
    print(
        f"[ace_step_v1_5] DiT: re-exec for full mesh ({mesh_sku}), handoff={handoff_path}",
        flush=True,
    )
    if preprocess_perf is not None:
        _hp = preprocess_perf.get("params") or {}
        _ht = preprocess_perf.get("timings_ms") or []
        _pa_s = float(preprocess_perf.get("phase_a_wall_ms") or 0.0) / 1000.0
        print(
            f"[ace_step_v1_5][perf] handoff pickle includes Phase-A stats: "
            f"phase_a_wall_s={_pa_s:.2f} "
            f"lm_gen_time_s={_hp.get('lm_gen_time_s', 'n/a')} "
            f"tokens={_hp.get('lm_num_tokens', 'n/a')} "
            f"modules={len(_ht)}",
            flush=True,
        )

    new_argv = [sys.executable]
    skip_next = False
    for tok in _resolve_demo_reexec_argv(argv):
        if skip_next:
            skip_next = False
            continue
        if tok == "--ace-step-dit-handoff":
            skip_next = True
            continue
        new_argv.append(tok)
    new_argv.extend(["--ace-step-dit-handoff", handoff_path])
    os.execv(sys.executable, new_argv)


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

    Under **TP** every host readback is replicated (DiT/VAE outputs), but a tensor that
    passed through a CCL all-reduce carries topology metadata that makes
    ``to_torch_auto_compose`` assert (``dims must be unique``). Read device-0's shard
    directly in that case — it holds the full (replicated) answer.
    """
    from models.common.auto_compose import to_torch_auto_compose

    torch_dtype = dtype if dtype is not None else torch.float32

    def _finish(out: torch.Tensor) -> torch.Tensor:
        if out.dtype != torch_dtype:
            out = out.to(torch_dtype)
        return out.contiguous()

    from models.experimental.ace_step_v1_5.ttnn_impl.tp_config import ace_step_tp_enabled

    if ace_step_tp_enabled(mesh_device):
        return _finish(ttnn.to_torch(ttnn.get_device_tensors(tensor)[0]))
    try:
        return _finish(to_torch_auto_compose(tensor, device=mesh_device))
    except Exception:
        # Fallback: a replicated tensor whose post-CCL topology confuses auto_compose.
        return _finish(ttnn.to_torch(ttnn.get_device_tensors(tensor)[0]))


def slice_batch_dim0(t: ttnn.Tensor, b0: int, b1_exc: int) -> ttnn.Tensor:
    """Slice batch dimension 0 of a rank-3 tensor ``[B, T, C]``."""
    t_dim = int(t.shape[1])
    c_dim = int(t.shape[2])
    return ttnn.slice(t, (int(b0), 0, 0), (int(b1_exc), t_dim, c_dim))
