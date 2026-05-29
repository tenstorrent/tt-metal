# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""ACE-Step TTNN throughput helpers (alignment with ``tt-perf-report`` / ``perf*.txt`` stacks).

Stacked E2E summaries often show large DRAM-interleaved shares for:

- ``PermuteDeviceOperation`` (~26 %)
- ``ReshapeViewDeviceOperation`` (~22 %)

Both ``ttnn.reshape`` and ``ttnn.permute`` accept ``memory_config``; this module **always** requests
L1 outputs where supported so reshape/permute chains avoid unnecessary DRAM round-trips.

E2E Tracy ``BinaryNgDeviceOperation (in0:dram_interleaved)`` (~24% device time) was dominated by
**VAE Snake** in FP32 (``BF16→FP32`` typecast, ~784 μs FP32 ``multiply``/``add`` per layer).
``TtSnake1d`` now always uses BF16 compute to match conv activations.

Production defaults (PCC + perf + E2E — no env toggle):

- **LoFi** matmul / RMSNorm / SDPA via :func:`ace_step_init_dit_linear_compute_kernel_config`,
  :func:`ace_step_init_cond_linear_compute_kernel_config`, :func:`ace_step_qwen3_optimizations`,
  :func:`ace_step_five_hz_lm_optimizations`
- **``bfloat8_b``** linear projection weights via :func:`ace_step_linear_weight_dtype`
  (attn ``q_proj`` / ``o_proj`` use ``bfloat4_b`` when available — :func:`ace_step_attn_qo_weight_dtype`)
  (embedding tables / norm scales / conv kernels stay BF16 unless noted)
- **L1 interleaved** activations via :func:`ace_step_linear_l1_memory_config` /
  :func:`ace_step_ensure_l1_activation` / :func:`ace_step_ensure_cond_activation`
- Long clips: ``ACE_STEP_DIT_MAX_FUSED_M`` (default 16) caps 1D-mcast ``per_core_M``;
  ``ACE_STEP_DIT_FORCE_L1_MATMUL=1`` keeps L1 in0/out (A/B; validate PCC on submodule first)

Remaining **DRAM ``in0``** buckets in Tracy (``perf_dit_4`` / conditioning) may still appear for:

- Qwen **SDPA attn masks**, **KV paged cache** (``PagedFillCache``), and **weight** tensors (DRAM by design)
- **Embedding lookup** token indices (``EmbeddingsDeviceOperation in0:dram`` — uint32 ids stay DRAM)

Other expected DRAM in DiT/VAE traces:

- ``proj_in`` uses **L1 TILE linear** patch embed (not ``conv1d``) to avoid ``Tilize`` / ``Copy`` / im2col DRAM matmul.
- Denoise feeds **TILE BF16 L1** ``xt``/``ctx`` (not ROW_MAJOR DRAM) so Tracy drops front ``Tilize``/``CopyDevice``.
- 5 Hz LM decoder **matmuls**: default **BF16** weights + ``accuracy_decoder_config.json`` (HiFi4) for HF logits PCC; opt-in ``ACE_STEP_LM_BFLOAT8_WEIGHTS=1`` for HiFi2 + ``bfloat8_b`` (via :func:`ace_step_five_hz_lm_optimizations`). DiT decoder **matmuls**: LoFi + ``bfloat8_b`` weights + L1 activations; **rms_norm**: LoFi + L1 (not default HiFi4).
- **SDPA attn masks**: DRAM-only (TTNN requirement).
- **RoPE / norm / linear weights**: DRAM storage; matmul reads weights from DRAM while ``in0`` activations are L1.
- Residual **BinaryNg (in0:dram)** (~0.3%): usually scalar-broadcast or slice outputs — call sites use :func:`ace_step_ensure_l1_activation` after ``ace_step_add_one``.

DiT linears are often DRAM-bound at HiFi4 without tuning (reference path only):

- ``256×1024×N`` @ 32c — attn ``q_proj`` / ``wkv`` via 2D 8×4 mcast (``in0_block_w=4``, latest sweep winner)
- ``256×1024×1024`` @ 64c — legacy 1D 8×4 baseline if shape does not match prefill ``M=256``
- ``256×2048×2048`` @ 32c — attn ``o_proj`` (``bfloat4_b`` weights when available)
- ``256×1024×1024`` @ 96c — MLP fused ``gate_up`` (``bfloat4_b`` gate weights; ``down_proj`` stays ``bfloat8_b``)
- ``256×3072×3072`` @ 32c — MLP ``down_proj`` (``bfloat4_b`` weights when available)

VAE decode exposes large-M matmuls inside ``conv1d`` / ``conv_transpose2d`` im2col (e.g.
``1920×512×512``, ``30720×128×128``, ``61440×128×128``). Production VAE uses **LoFi** conv compute + **BF16** activations; **``bfloat8_b``** weights on
``k>1`` conv / conv-transpose im2col (DRAM BW). ``k=1`` projections use ``ttnn.conv1d`` with tuned
1D mcast matmul configs on large im2col ``M`` (``61440`` / ``30720`` / ``7680`` buckets via
:func:`ace_step_vae_conv1d_im2col_matmul_program_config`).
**``bfloat8_b`` activation compute** (conv output dtype + Snake TILE chain) is **on by default**;
inter-op buffers stay BF16 ``ROW_MAJOR`` (TTNN layout limit). Set ``ACE_STEP_VAE_BFLOAT8_ACTIVATIONS=0`` to disable.

Memory policy (VAE):

- **L1 interleaved** on ``1×1`` conv + conv-transpose activations and outputs.
- **Snake eltwise chain** (multiply, sin, square, multiply, add + Tilize/Untilize) runs in L1
  (params in L1, intermediates in L1) — eliminates ~5 DRAM round-trips per call vs the
  original all-DRAM path.  Snake **output** is staged back to DRAM inside ``TtSnake1d``
  before returning so the caller always receives a DRAM tensor — **except** residual
  ``snake2`` which uses ``return_tile=True`` to hand **L1 TILE** activations to ``conv2``
  (k=1), skipping snake Untilize and conv2 Tilize on the linear path.
- **conv1(k=7)→snake2 TILE contract:** ``TtConv1d(return_sharded=True)`` tries HEIGHT_SHARDED
  L1 when ``ACE_STEP_VAE_K7_SHARDED_OUTPUT=1``; otherwise falls back to **DRAM TILE**
  (``return_tile`` behaviour) so snake2 avoids Tilize on ROW_MAJOR DRAM.
- **k>1 ``conv1d`` input/output** stays in DRAM (static CB region extends to ~180 KiB on
  Blackhole — any live L1 activation in that band fails program validation at compile time;
  k>1 output also exceeds per-bank L1 budget).

Condition encoder linears (lyric/timbre, ``hidden_size=2048``) are often DRAM-bound:

- ``32×2048×2048`` — attn ``q``/``k``/``v``/``o`` (short packed sequences)
- ``32×6144×6144`` — MLP ``gate``/``up``
- ``288×2048×2048`` — longer lyric windows
"""

from __future__ import annotations

import os
from typing import Any

# Lyric/timbre encoders use intermediate ≥ 6144; gate/up L1 CBs need in0_block_w=1 there.
_WIDE_MLP_INTERMEDIATE_THRESHOLD = 4608


def _l1_memory_kwargs(ttnn: Any) -> dict:
    mc = getattr(ttnn, "L1_MEMORY_CONFIG", None)
    return {"memory_config": mc} if mc is not None else {}


def ace_step_reshape_kwargs(ttnn: Any) -> dict:
    """Keyword args for ``ttnn.reshape`` to place outputs in L1 when ``L1_MEMORY_CONFIG`` exists."""
    return _l1_memory_kwargs(ttnn)


def ace_step_permute_kwargs(ttnn: Any) -> dict:
    """Keyword args for ``ttnn.permute`` to place outputs in L1 when ``L1_MEMORY_CONFIG`` exists."""
    return _l1_memory_kwargs(ttnn)


def ace_step_to_layout_kwargs(ttnn: Any, l1_mc: Any | None = None) -> dict:
    """Keyword args for ``ttnn.to_layout`` so ``Tilize*`` outputs land in L1 (not DRAM)."""
    mc = l1_mc if l1_mc is not None else ace_step_linear_l1_memory_config(ttnn)
    return {"memory_config": mc} if mc is not None else {}


def ace_step_concat_kwargs(ttnn: Any, l1_mc: Any | None = None) -> dict:
    """Keyword args for ``ttnn.concat`` on activation tensors."""
    mc = l1_mc if l1_mc is not None else ace_step_linear_l1_memory_config(ttnn)
    return {"memory_config": mc} if mc is not None else {}


def ace_step_lofi_bfloat8_enabled() -> bool:
    """ACE-Step production path: LoFi compute + ``bfloat8_b`` linear weights (always on, no env)."""
    return True


def ace_step_dit_lofi_bfloat8_enabled() -> bool:
    """Alias for :func:`ace_step_lofi_bfloat8_enabled` (DiT call sites)."""
    return ace_step_lofi_bfloat8_enabled()


def ace_step_use_manual_concat_heads() -> bool:
    """Use permute+reshape instead of fused ``nlp_concat_heads`` (A/B only — fused is faster on DiT traces)."""
    return os.environ.get("ACE_STEP_MANUAL_CONCAT_HEADS", "").lower() in ("1", "true", "yes")


def ace_step_use_nlp_create_qkv_heads() -> bool:
    """Opt-in ``nlp_create_qkv_heads`` (default off — manual reshape+permute is faster on DiT traces)."""
    return os.environ.get("ACE_STEP_USE_NLP_CREATE_QKV_HEADS", "").lower() in ("1", "true", "yes")


def ace_step_use_bfloat4_weights() -> bool:
    """Opt-in ``bfloat4_b`` linear weights (validate PCC on submodule before full DiT)."""
    return os.environ.get("ACE_STEP_DIT_BFLOAT4_WEIGHTS", "").lower() in ("1", "true", "yes")


def ace_step_bfloat8_attn_qo_weights() -> bool:
    """Force attn ``q_proj`` / ``o_proj`` weights to stay ``bfloat8_b`` (opt-out of default ``bfloat4_b``)."""
    return os.environ.get("ACE_STEP_DIT_BFLOAT8_ATTN_QO", "").lower() in ("1", "true", "yes")


def ace_step_attn_qo_weight_dtype(ttnn: Any, default_dtype: Any) -> Any:
    """Weight dtype for attn ``q_proj`` / ``o_proj`` (e.g. DiT/Qwen ``256×1024×1024``).

    Defaults to ``bfloat4_b`` when the build exposes it (halves DRAM weight BW vs ``bfloat8_b``).
    Set ``ACE_STEP_DIT_BFLOAT8_ATTN_QO=1`` to keep BFP8; ``ACE_STEP_DIT_BFLOAT4_WEIGHTS=1`` forces
    bf4 on all linears via :func:`ace_step_linear_weight_dtype`.
    """
    if ace_step_use_bfloat4_weights():
        return ace_step_linear_weight_dtype(ttnn, default_dtype)
    if not ace_step_bfloat8_attn_qo_weights():
        bf4 = getattr(ttnn, "bfloat4_b", None)
        if bf4 is not None:
            return bf4
    return ace_step_linear_weight_dtype(ttnn, default_dtype)


def ace_step_linear_weight_dtype(ttnn: Any, default_dtype: Any) -> Any:
    """Weight storage dtype for **linear** projections (activations stay ``default_dtype``, usually BF16)."""
    if ace_step_use_bfloat4_weights():
        return getattr(ttnn, "bfloat4_b", None) or getattr(ttnn, "bfloat8_b", None) or default_dtype
    if ace_step_lofi_bfloat8_enabled():
        return getattr(ttnn, "bfloat8_b", None) or default_dtype
    return default_dtype


def ace_step_dit_weight_dtype(ttnn: Any, default_dtype: Any) -> Any:
    """DiT linear weights — alias for :func:`ace_step_linear_weight_dtype`."""
    return ace_step_linear_weight_dtype(ttnn, default_dtype)


def ace_step_dit_conv_weight_dtype(ttnn: Any, default_dtype: Any) -> Any:
    """Legacy conv weight dtype helper (``proj_in`` is linear TILE now; kept for callers)."""
    return ace_step_dit_weight_dtype(ttnn, default_dtype)


def ace_step_dit_weight_layout(ttnn: Any, weight_dtype: Any, *, default_layout: Any) -> Any:
    """Layout for weight upload: ``bfloat8_b`` / ``bfloat4_b`` must be TILE (TT_FATAL otherwise)."""
    bf8 = getattr(ttnn, "bfloat8_b", None)
    bf4 = getattr(ttnn, "bfloat4_b", None)
    if weight_dtype in (bf8, bf4):
        return ttnn.TILE_LAYOUT
    return default_layout


def ace_step_is_tile_layout(ttnn: Any, tensor: Any) -> bool:
    tile = getattr(ttnn, "TILE_LAYOUT", None)
    if tile is None or tensor is None:
        return False
    layout = getattr(tensor, "layout", None)
    return layout == tile


def ace_step_ensure_tile_layout(
    ttnn: Any, tensor: Any, l1_mc: Any | None = None, *, out_memory_config: Any | None = None
) -> Any:
    """Return *tensor* unchanged when already TILE — avoids redundant ``TilizeDeviceOperation``.

    When a Tilize is required the output is placed in L1 (via ``L1_MEMORY_CONFIG``) so the
    caller gets an L1-resident tile instead of a DRAM-resident one.  Pass *out_memory_config*
    (e.g. ``DRAM_MEMORY_CONFIG`` for SDPA masks) to override the default L1 placement.
    """
    if tensor is None:
        return tensor
    out_mc = out_memory_config if out_memory_config is not None else l1_mc
    if out_mc is None and out_memory_config is None:
        out_mc = ace_step_linear_l1_memory_config(ttnn)
    if ace_step_is_tile_layout(ttnn, tensor):
        if out_mc is not None and hasattr(tensor, "memory_config") and hasattr(ttnn, "to_memory_config"):
            if tensor.memory_config() != out_mc:
                return ttnn.to_memory_config(tensor, out_mc)
        return tensor
    kw = {"memory_config": out_mc} if out_mc is not None else {}
    return ttnn.to_layout(tensor, ttnn.TILE_LAYOUT, **kw)


def ace_step_upload_f32_np_as_bf16_tile(
    ttnn: Any,
    host_f32: Any,
    *,
    device: Any,
    dtype: Any | None = None,
    memory_config: Any | None = None,
    mesh_mapper: Any | None = None,
) -> Any:
    """Upload host float32 bias/mask numpy as BF16 TILE without on-device FP32 tilize + typecast.

    ``ttnn.as_tensor(fp32_np, dtype=bfloat16, layout=TILE)`` runs Tilize FP32→FP32 then
    Typecast FP32→BF16 on device. Converting on host and uploading ROW_MAJOR BF16 needs one
    BF16 tilize only.
    """
    import numpy as np
    import torch

    bf16 = dtype if dtype is not None else getattr(ttnn, "bfloat16", None)
    if bf16 is None:
        raise RuntimeError("bfloat16 required for ace_step_upload_f32_np_as_bf16_tile")
    mc = memory_config if memory_config is not None else ace_step_sdpa_mask_memory_config(ttnn)
    host = np.ascontiguousarray(host_f32, dtype=np.float32)
    torch_bf16 = torch.from_numpy(host).to(torch.bfloat16)
    kw: dict = {}
    if mesh_mapper is not None:
        kw["mesh_mapper"] = mesh_mapper
    if mc is not None:
        kw["memory_config"] = mc
    tt_rm = ttnn.from_torch(
        torch_bf16,
        device=device,
        dtype=bf16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        **kw,
    )
    return ace_step_ensure_tile_layout(ttnn, tt_rm, out_memory_config=mc)


def ace_step_ensure_row_major_layout(ttnn: Any, tensor: Any) -> Any:
    """Return *tensor* unchanged when already ROW_MAJOR."""
    if tensor is None:
        return tensor
    rm = getattr(ttnn, "ROW_MAJOR_LAYOUT", None)
    if rm is not None and getattr(tensor, "layout", None) == rm:
        return tensor
    return ttnn.to_layout(tensor, ttnn.ROW_MAJOR_LAYOUT)


def ace_step_dit_linear_perf_enabled() -> bool:
    """Deprecated: perf kwargs are always enabled. Kept for call-site compatibility."""
    return True


def ace_step_cond_linear_perf_enabled() -> bool:
    """Deprecated: perf kwargs are always enabled. Kept for call-site compatibility."""
    return True


def ace_step_vae_conv_perf_enabled() -> bool:
    """Deprecated: VAE conv perf path is always enabled. Kept for call-site compatibility."""
    return True


def ace_step_init_vae_conv_compute_kernel_config(device: Any):
    """LoFi compute kernel for Oobleck VAE ``conv1d`` / ``conv_transpose2d`` im2col matmuls."""
    return ace_step_init_lofi_linear_compute_kernel_config(device)


def ace_step_vae_quality_decode_enabled(
    *,
    latent_frames: int | None = None,
    mesh_sku: str | None = None,
    duration_sec: float | None = None,
) -> bool:
    """Prefer BF16 VAE compute/weights for long tiled decode (reduces hiss on 60s+ mesh runs)."""
    from models.demos.ace_step_v1_5.tt_device import ace_step_needs_split_device

    env = os.environ.get("ACE_STEP_VAE_QUALITY", "")
    if env.lower() in ("1", "true", "yes", "on"):
        return True
    if env.lower() in ("0", "false", "no", "off"):
        return False
    if latent_frames is None:
        lf = os.environ.get("ACE_STEP_VAE_LATENT_FRAMES", "")
        if lf.strip().isdigit():
            latent_frames = int(lf)
    if duration_sec is None:
        ds = os.environ.get("ACE_STEP_VAE_DURATION_SEC", "")
        try:
            if ds.strip():
                duration_sec = float(ds)
        except ValueError:
            pass
    if mesh_sku is None:
        mesh_sku = os.environ.get("ACE_STEP_VAE_MESH_SKU") or None
    on_mesh = mesh_sku is not None and ace_step_needs_split_device(mesh_sku)
    if not on_mesh:
        return False
    if latent_frames is not None and int(latent_frames) >= 400:
        return True
    if duration_sec is not None and float(duration_sec) >= 30.0:
        return True
    return False


def ace_step_vae_bfloat8_activations_enabled(
    *,
    latent_frames: int | None = None,
    mesh_sku: str | None = None,
    duration_sec: float | None = None,
) -> bool:
    """``bfloat8_b`` for VAE conv im2col + Snake eltwise compute — **on by default** for short clips.

    TTNN does not support ``bfloat8_b`` with ``ROW_MAJOR`` activations (conv1d / slice / residual
    trim). Inter-op buffers stay **BF16 ROW_MAJOR**; :func:`ace_step_vae_activation_compute_dtype`
    applies inside conv/Snake kernels only.

    Long overlap-add decode on mesh defaults to BF16 compute (see :func:`ace_step_vae_quality_decode_enabled`).
    Set ``ACE_STEP_VAE_BFLOAT8_ACTIVATIONS=0`` to force BF16, or ``=1`` to force BFP8.
    """
    if ace_step_vae_quality_decode_enabled(latent_frames=latent_frames, mesh_sku=mesh_sku, duration_sec=duration_sec):
        return False
    return os.environ.get("ACE_STEP_VAE_BFLOAT8_ACTIVATIONS", "1").lower() not in ("0", "false", "no", "off")


def ace_step_vae_activation_storage_dtype(ttnn: Any) -> Any:
    """VAE activation buffers between ops: **BF16** ``ROW_MAJOR`` (slice / residual / chunking)."""
    dtype = getattr(ttnn, "bfloat16", None)
    if dtype is None:
        raise RuntimeError("TTNN build missing bfloat16; VAE activation storage requires BF16")
    return dtype


def ace_step_vae_activation_compute_dtype(ttnn: Any) -> Any:
    """VAE conv im2col + Snake internal compute dtype (``bfloat8_b`` when opt-in env is set)."""
    if ace_step_vae_bfloat8_activations_enabled():
        bf8 = getattr(ttnn, "bfloat8_b", None)
        if bf8 is not None:
            return bf8
    return ace_step_vae_activation_storage_dtype(ttnn)


def ace_step_vae_activation_dtype(ttnn: Any) -> Any:
    """Alias for :func:`ace_step_vae_activation_compute_dtype` (Tracy / demo-facing dtype)."""
    return ace_step_vae_activation_compute_dtype(ttnn)


def ace_step_vae_host_weight_staging_dtype(ttnn: Any) -> Any:
    """Host ROW_MAJOR dtype before ``prepare_conv_weights`` (always BF16)."""
    return ace_step_vae_activation_storage_dtype(ttnn)


def ace_step_vae_ensure_interleaved(
    ttnn: Any,
    tensor: Any,
    *,
    memory_config: Any | None = None,
) -> Any:
    """Convert sharded VAE activations to interleaved before unsqueeze/reshape/slice."""
    if tensor is None:
        return tensor
    is_sharded_fn = getattr(ttnn, "is_sharded", None)
    s2i = getattr(ttnn, "sharded_to_interleaved", None)
    if not callable(is_sharded_fn) or not callable(s2i) or not is_sharded_fn(tensor):
        return tensor
    mc = memory_config if memory_config is not None else getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
    kw = {"memory_config": mc} if mc is not None else {}
    return s2i(tensor, **kw)


def ace_step_vae_normalize_activation_output(ttnn: Any, tensor: Any, *, storage_dtype: Any, compute_dtype: Any) -> Any:
    """Return a ``ROW_MAJOR`` tensor in *storage_dtype* (post-conv / post-Snake boundary contract)."""
    if tensor is None:
        return tensor
    dram = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
    tensor = ace_step_vae_ensure_interleaved(ttnn, tensor, memory_config=dram)
    if compute_dtype != storage_dtype and getattr(tensor, "dtype", None) != storage_dtype:
        kw = {"memory_config": dram} if dram is not None else {}
        tensor = ttnn.typecast(tensor, storage_dtype, **kw)
    kw = {"memory_config": dram} if dram is not None else {}
    return ttnn.to_layout(tensor, ttnn.ROW_MAJOR_LAYOUT, **kw)


def ace_step_vae_activation_memory_config(ttnn: Any):
    """L1 interleaved activations for VAE Snake / ``1×1`` conv / conv-transpose glue ops."""
    return ace_step_linear_l1_memory_config(ttnn)


def ace_step_vae_conv1d_memory_config(ttnn: Any, *, kernel_size: int):
    """Memory config for ``TtConv1d``: L1 for ``kernel_size==1``, DRAM for ``k>1``.

    Wide ``k>1`` conv programs allocate multi-MB L1 output tensors (``conv2d_L1``) that exceed
    per-bank budget on Blackhole when activations are forced into L1 interleaved.
    """
    if int(kernel_size) == 1:
        return ace_step_vae_activation_memory_config(ttnn)
    return getattr(ttnn, "DRAM_MEMORY_CONFIG", None)


def ace_step_vae_synchronize(ttnn: Any, device: Any) -> None:
    """Drain the device queue so ``deallocate`` / ``deallocate_activation`` frees L1 before k>7 conv compile."""
    if device is None:
        return
    try:
        ttnn.synchronize_device(device)
    except Exception:
        pass


def ace_step_vae_k7_sharded_output_config(ttnn: Any, device: Any, out_length: int, out_channels: int) -> Any:
    """HEIGHT_SHARDED L1 memory config for the k=7 conv1d output (``return_sharded=True``).

    Distributes ``[out_length, out_channels]`` TILE across all cores so each core holds
    ``ceil(T_tiles / num_cores) × C_tiles`` tiles — well within each core's 1.5 MB L1 budget
    on Blackhole.  Returns ``None`` if the config cannot be constructed (caller falls back to
    ``return_tile=True`` DRAM output).

    Gated by ``ACE_STEP_VAE_K7_SHARDED_OUTPUT=1`` (default off).  The env var lets users
    opt in only after verifying that ``ttnn.conv1d`` honours a HEIGHT_SHARDED output
    memory_config on the target firmware.
    """
    if os.environ.get("ACE_STEP_VAE_K7_SHARDED_OUTPUT", "").strip() != "1":
        return None
    import math as _math

    try:
        grid = device.compute_with_storage_grid_size()
        num_cores = int(grid.x) * int(grid.y)
        T_tiles = _math.ceil(int(out_length) / 32)
        C_tiles = _math.ceil(int(out_channels) / 32)
        shard_T_tiles = _math.ceil(T_tiles / num_cores)
        shard_H = max(1, shard_T_tiles) * 32  # in elements
        shard_W = max(1, C_tiles) * 32  # in elements
        core_grid = ttnn.CoreRangeSet(
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(int(grid.x) - 1, int(grid.y) - 1),
            )
        )
        return ttnn.create_sharded_memory_config(
            shape=(shard_H, shard_W),
            core_grid=core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
    except Exception:
        return None


def ace_step_vae_conv_weight_dtype(ttnn: Any, default_dtype: Any, *, kernel_size: int) -> Any:
    """VAE conv weights: ``bfloat8_b`` for ``k>1`` DRAM im2col (halves weight BW); BF16 for ``k==1``."""
    if int(kernel_size) > 1:
        if os.environ.get("ACE_STEP_VAE_BF16_CONV_WEIGHTS", "").lower() in ("1", "true", "yes", "on"):
            return default_dtype
        return ace_step_linear_weight_dtype(ttnn, default_dtype)
    return default_dtype


def ace_step_vae_typecast_kwargs(ttnn: Any, l1_mc: Any | None = None) -> dict:
    """``memory_config`` for VAE ``typecast`` (latent FP32 → BF16)."""
    mc = l1_mc if l1_mc is not None else ace_step_vae_activation_memory_config(ttnn)
    return {"memory_config": mc} if mc is not None else {}


def ace_step_vae_eltwise_kwargs(ttnn: Any, *, device: Any = None, l1_mc: Any | None = None) -> dict:
    """Keyword args for VAE Snake ``multiply`` / ``add`` — L1 output only.

    ``ttnn.multiply`` / ``ttnn.add`` accept ``memory_config`` but not ``compute_kernel_config`` or
    ``fast_and_approximate_mode`` on all builds. LoFi fidelity is applied on conv im2col matmuls.
    """
    _ = device
    return ace_step_binary_kwargs(ttnn, l1_mc)


def ace_step_device_profiler_enabled() -> bool:
    return (
        os.environ.get("TTNN_OP_PROFILER", "").strip() == "1"
        or os.environ.get("TT_METAL_DEVICE_PROFILER", "").strip() == "1"
    )


def ace_step_profiler_flush_every_layer() -> int:
    """Drain device profiler every N VAE/DiT layers (0=off). Default ``1`` when profiling."""
    if not ace_step_device_profiler_enabled():
        return 0
    try:
        return max(0, int(os.environ.get("ACE_STEP_PROFILER_FLUSH_EVERY_LAYER", "1")))
    except ValueError:
        return 1


def ace_step_enable_tracy_profiler_env() -> None:
    """Match TTNN + metal profilers so Tracy host/device op IDs align in ``cpp_device_perf_report.csv``."""
    if os.environ.get("TT_METAL_DEVICE_PROFILER", "").strip() == "1":
        os.environ.setdefault("TTNN_OP_PROFILER", "1")


def ace_step_flush_device_profiler(device) -> None:
    """Sync and drain per-RISC device profiler rings (avoids Tracy 'Device data missing' on merge)."""
    if not ace_step_device_profiler_enabled():
        return
    if os.environ.get("ACE_STEP_USE_TRACE", "").lower() in ("1", "true", "yes"):
        return
    try:
        import ttnn

        ttnn.synchronize_device(device)
        ttnn.ReadDeviceProfiler(device)
    except Exception:
        pass


def ace_step_vae_conv1d_im2col_matmul_enabled() -> bool:
    """Whether ``1×1`` conv may bypass ``ttnn.conv1d`` with tuned ``ttnn.linear`` matmul configs.

    - ``ACE_STEP_VAE_LARGE_M_MATMUL=0`` — always off.
    - ``ACE_STEP_VAE_LARGE_M_MATMUL=1`` — always on.
    - Default — on. **Stays on under profilers** so large im2col ``M`` uses a clamped full-grid
      program config instead of ``ttnn.conv1d`` matmul probing 640 M-tiles as 640 cores.
    """
    flag = os.environ.get("ACE_STEP_VAE_LARGE_M_MATMUL")
    if flag is not None:
        return flag.lower() not in ("0", "false", "no", "off")
    return True


def ace_step_vae_sharded_matmul_enabled() -> bool:
    """Whether midsize wide ``1×1`` conv (e.g. ``1920×512×512``) skips ``ttnn.linear`` for ``ttnn.conv1d`` L1.

    ``ttnn.linear`` with L1 interleaved ``in0`` overflows Blackhole L1 / probes invalid core counts
    at ``C≥512``; the validated path is ``conv1d`` ``k=1`` with L1 activations (snake2→conv2 chain).
    Set ``ACE_STEP_VAE_SHARDED_MATMUL=1`` to enable.
    """
    flag = os.environ.get("ACE_STEP_VAE_SHARDED_MATMUL")
    if flag is not None:
        return flag.lower() not in ("0", "false", "no", "off")
    return False


def ace_step_vae_k1_prefer_conv1d_l1(
    *,
    m_dim: int,
    k_dim: int,
    n_dim: int,
) -> bool:
    """True when ``ACE_STEP_VAE_SHARDED_MATMUL`` should route ``k=1`` wide conv through ``conv1d`` L1."""
    if not ace_step_vae_sharded_matmul_enabled():
        return False
    return ace_step_vae_k1_mid_m_eligible(m_dim=m_dim, k_dim=k_dim, n_dim=n_dim)


def ace_step_vae_k1_mid_m_eligible(*, m_dim: int, k_dim: int, n_dim: int) -> bool:
    """``M`` in ``[512, 7680)`` with wide ``K/N`` — Tracy ``1920×512`` / ``320×1024`` buckets."""
    m = int(m_dim)
    if m < 512 or m >= 7680:
        return False
    return max(int(k_dim), int(n_dim)) >= 512


def ace_step_vae_k1_height_sharded_eligible(*, m_dim: int, k_dim: int, n_dim: int) -> bool:
    """Alias for :func:`ace_step_vae_k1_mid_m_eligible` (residual add / legacy name)."""
    return ace_step_vae_k1_mid_m_eligible(m_dim=m_dim, k_dim=k_dim, n_dim=n_dim)


def ace_step_vae_k1_mid_m_matmul_program_config(
    device: Any,
    *,
    m_dim: int,
    k_dim: int,
    n_dim: int,
):
    """Tall-M ``MultiCast1D`` (``mcast_in0=False``) with L1 **interleaved** ``in0``."""
    if not ace_step_vae_k1_mid_m_eligible(m_dim=m_dim, k_dim=k_dim, n_dim=n_dim):
        return None
    return _mcast_1d_vae_im2col_tall_m_program_config(
        device,
        m_dim=int(m_dim),
        k_dim=int(k_dim),
        n_dim=int(n_dim),
        in0_block_w_cap=2,
        out_subblock_h_cap=4,
        out_subblock_w=1,
    )


def ace_step_vae_k1_height_sharded_program_config(
    device: Any,
    *,
    m_dim: int,
    k_dim: int,
    n_dim: int,
    compute_grid_size: tuple[int, int] | None = None,
):
    """Alias for :func:`ace_step_vae_k1_mid_m_matmul_program_config` (ignores ``compute_grid_size``)."""
    _ = compute_grid_size
    return ace_step_vae_k1_mid_m_matmul_program_config(
        device,
        m_dim=m_dim,
        k_dim=k_dim,
        n_dim=n_dim,
    )


def ace_step_vae_max_per_core_m_tiles() -> int:
    """Max ``per_core_M`` (M tiles) for VAE im2col ``MultiCast1D`` before falling back to ``ttnn.conv1d``.

    Default ``8`` for normal perf. While profiling, default ``20`` so ``61440×128`` im2col
    (``per_core_M≈18`` on 110 cores) still uses tuned matmul. Override via env.
    """
    default = "20" if ace_step_device_profiler_enabled() else "8"
    return max(1, int(os.environ.get("ACE_STEP_VAE_MAX_PER_CORE_M_TILES", default)))


def ace_step_vae_conv1d_im2col_matmul_program_config(
    device: Any,
    *,
    m_dim: int,
    k_dim: int,
    n_dim: int,
):
    """Tuned 1D reuse config for VAE ``1×1`` ``ttnn.conv1d`` im2col matmuls.

    Applies to large-M buckets from Tracy (``61440×128``, ``30720×128``, ``7680×256``, ``1920×512``) using
    **tall-M** ``MultiCast1D`` (``mcast_in0=False``, ``M`` split across cores). Skips ``M < 7680`` only.
    """
    m = int(m_dim)
    k = int(k_dim)
    n = int(n_dim)
    if m < 7680:
        return None
    if not ace_step_vae_conv1d_im2col_matmul_enabled():
        # Large im2col still needs a clamped program config — ``ttnn.conv1d`` probes M-tiles as cores.
        pass
    return ace_step_vae_large_m_matmul_program_config(
        device,
        m_dim=m,
        k_dim=k,
        n_dim=n,
    )


def _mcast_1d_vae_im2col_tall_m_program_config(
    device: Any,
    *,
    m_dim: int,
    k_dim: int,
    n_dim: int,
    in0_block_w_cap: int = 2,
    out_subblock_h_cap: int = 4,
    out_subblock_w: int = 1,
):
    """1D reuse matmul for **tall** VAE ``1×1`` conv im2col (``M >> N``).

    Uses ``mcast_in0=False`` and splits ``M`` across the compute grid. ``mcast_in0=True`` would
    require every core to buffer the **full** ``M`` strip in L1 (see TT_FATAL in
    ``matmul_multicore_reuse_mcast_1d``) and overflows Blackhole for production audio lengths
    (``61440×128`` im2col).
    """
    import ttnn

    cfg_cls = getattr(ttnn, "MatmulMultiCoreReuseMultiCast1DProgramConfig", None)
    if cfg_cls is None or not hasattr(device, "compute_with_storage_grid_size"):
        return None

    grid = device.compute_with_storage_grid_size()
    gx = max(1, int(grid.x))
    gy = max(1, int(grid.y))
    num_cores = gx * gy
    tile = int(getattr(ttnn, "TILE_SIZE", 32))
    m_tiles = max(1, (int(m_dim) + tile - 1) // tile)
    n_tiles = max(1, (int(n_dim) + tile - 1) // tile)
    k = max(tile, int(k_dim))
    k_tiles = max(1, k // tile)

    per_core_m = max(1, (m_tiles + num_cores - 1) // num_cores)
    if per_core_m > ace_step_vae_max_per_core_m_tiles() and not ace_step_device_profiler_enabled():
        return None
    num_blocks_m = (m_tiles + per_core_m - 1) // per_core_m
    if num_blocks_m > num_cores:
        return None

    in0_block_w = min(int(in0_block_w_cap), k_tiles)
    while k_tiles % in0_block_w != 0 and in0_block_w > 1:
        in0_block_w -= 1

    # Use the full device grid — a 1-row subgrid (e.g. 11×1) makes matmul output-spec probing
    # request hundreds of cores against only 11 available (TT_FATAL spam / Tracy 32K limit).
    per_core_n = max(1, (n_tiles + num_cores - 1) // num_cores)

    out_subblock_h = min(int(out_subblock_h_cap), per_core_m)
    while per_core_m % out_subblock_h != 0 and out_subblock_h > 1:
        out_subblock_h -= 1

    out_subblock_w_target = min(int(out_subblock_w), max(1, int(per_core_n)))
    if out_subblock_w_target > 1 and per_core_n % out_subblock_w_target != 0:
        per_core_n = ((per_core_n + out_subblock_w_target - 1) // out_subblock_w_target) * out_subblock_w_target

    out_subblock_w_eff = min(int(out_subblock_w), max(1, int(per_core_n)))
    while per_core_n % out_subblock_w_eff != 0 and out_subblock_w_eff > 1:
        out_subblock_w_eff -= 1

    return cfg_cls(
        compute_with_storage_grid_size=(gx, gy),
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w_eff,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=False,
    )


def ace_step_vae_large_m_matmul_program_config(
    device: Any,
    *,
    m_dim: int,
    k_dim: int,
    n_dim: int,
):
    """Matmul program config for VAE conv im2col shapes (e.g. 1920×512, 30720×128, 61440×128)."""
    tile = 32
    m = max(1, int(m_dim))
    k = max(tile, int(k_dim))
    n = max(tile, int(n_dim))
    if m >= 61440:
        return _mcast_1d_vae_im2col_tall_m_program_config(
            device,
            m_dim=m,
            k_dim=k,
            n_dim=n,
            in0_block_w_cap=2,
            out_subblock_h_cap=2,
            out_subblock_w=2,
        )
    if m >= 7680:
        return _mcast_1d_vae_im2col_tall_m_program_config(
            device,
            m_dim=m,
            k_dim=k,
            n_dim=n,
            in0_block_w_cap=2,
            out_subblock_h_cap=1,
            out_subblock_w=4,
        )
    return _mcast_1d_vae_im2col_tall_m_program_config(
        device,
        m_dim=m,
        k_dim=k,
        n_dim=n,
        in0_block_w_cap=2,
        out_subblock_h_cap=4,
        out_subblock_w=1,
    )


def ace_step_linear_l1_memory_config(ttnn: Any):
    """L1 interleaved buffer config for linear activations / outputs."""
    return getattr(ttnn, "L1_MEMORY_CONFIG", None)


def ace_step_dit_weight_memory_config(ttnn: Any):
    """DRAM interleaved for DiT **weights** (linears, norms, RoPE tables, conv kernels)."""
    return getattr(ttnn, "DRAM_MEMORY_CONFIG", None)


def ace_step_dit_linear_l1_memory_config(ttnn: Any):
    """Alias for :func:`ace_step_linear_l1_memory_config`."""
    return ace_step_linear_l1_memory_config(ttnn)


def ace_step_dit_force_l1_matmul() -> bool:
    """Keep DiT matmul ``in0``/output in L1 even when fused-M exceeds the default CB cap (A/B only)."""
    return os.environ.get("ACE_STEP_DIT_FORCE_L1_MATMUL", "").lower() in ("1", "true", "yes")


def ace_step_dit_max_fused_m_tiles() -> int:
    """Max fused batch×seq tile rows for 1D-mcast matmul (``ACE_STEP_DIT_MAX_FUSED_M``, default 16)."""
    try:
        return max(1, int(os.environ.get("ACE_STEP_DIT_MAX_FUSED_M", "16")))
    except ValueError:
        return 16


def ace_step_linear_kwargs_memory_config(
    program_config: Any | None,
    *,
    linear_out_l1: Any | None,
    dram: Any | None,
) -> Any | None:
    """L1 linear outputs when a 1D/2D-mcast program is active, or when ``ACE_STEP_DIT_FORCE_L1_MATMUL=1``."""
    if linear_out_l1 is not None and (program_config is not None or ace_step_dit_force_l1_matmul()):
        return linear_out_l1
    return dram


def ace_step_sdpa_activation_kwargs(ttnn: Any, l1_mc: Any | None = None) -> dict:
    """Keyword args for fused SDPA so attention output stays in L1 (mask stays DRAM)."""
    mc = l1_mc if l1_mc is not None else ace_step_linear_l1_memory_config(ttnn)
    return {"memory_config": mc} if mc is not None else {}


def ace_step_safe_deallocate(ttnn: Any, *tensors: Any) -> None:
    """Best-effort ``ttnn.deallocate`` for optional / already-freed tensors."""
    for t in tensors:
        if t is None:
            continue
        try:
            ttnn.deallocate(t)
        except Exception:
            pass


def ace_step_add_one(ttnn: Any, tensor: Any, **kwargs: Any) -> Any:
    """``tensor + 1`` via ``ttnn.add(tensor, 1.0)`` — avoids per-call ``ones_like`` / ``full``."""
    return ttnn.add(tensor, 1.0, **kwargs)


def _ace_step_split_q_bhsd_manual(
    ttnn: Any,
    q_b1sd: Any,
    *,
    b: int,
    s_q: int,
    h: int,
    dh: int,
    l1_mc: Any | None,
) -> Any:
    _sr = ace_step_reshape_kwargs(ttnn)
    _pk = ace_step_permute_kwargs(ttnn)
    _kw = {"memory_config": l1_mc} if l1_mc is not None else {}
    q = ttnn.reshape(q_b1sd, (b, s_q, h, dh), **_sr)
    return ttnn.permute(q, (0, 2, 1, 3), **_pk)


def _ace_step_split_kv_bhsd_manual(
    ttnn: Any,
    kv_b1sd: Any,
    *,
    b: int,
    s_k: int,
    kv_h: int,
    dh: int,
    l1_mc: Any | None,
) -> tuple[Any, Any]:
    _sr = ace_step_reshape_kwargs(ttnn)
    _pk = ace_step_permute_kwargs(ttnn)
    kv_dim = kv_h * dh
    k4 = ttnn.slice(kv_b1sd, (0, 0, 0, 0), (b, 1, s_k, kv_dim))
    v4 = ttnn.slice(kv_b1sd, (0, 0, 0, kv_dim), (b, 1, s_k, 2 * kv_dim))
    k = ttnn.reshape(k4, (b, s_k, kv_h, dh), **_sr)
    v = ttnn.reshape(v4, (b, s_k, kv_h, dh), **_sr)
    k = ttnn.permute(k, (0, 2, 1, 3), **_pk)
    v = ttnn.permute(v, (0, 2, 1, 3), **_pk)
    return k, v


def ace_step_split_qkv_heads_bhsd(
    ttnn: Any,
    q_b1sd: Any,
    kv_b1sd: Any,
    *,
    num_heads: int,
    num_kv_heads: int,
    l1_mc: Any | None = None,
    transpose_k_heads: bool = False,
) -> tuple[Any, Any, Any]:
    """``[B,1,S,*]`` linear outputs → ``q,k,v`` each ``[B,H,S,Dh]`` (or ``[B,kv_h,S_k,Dh]``).

    **Default:** slice (K/V) + ``reshape`` + ``permute`` with L1 ``memory_config`` — cheaper than
    ``NlpCreateHeadsDeviceOperation`` on ACE-Step denoise traces (~7% device time at 96 calls).

    Set ``ACE_STEP_USE_NLP_CREATE_QKV_HEADS=1`` only for A/B (self-attn requires ``S_q == S_k``;
    cross-attn always uses the manual K/V path).
    """
    mc = l1_mc if l1_mc is not None else ace_step_linear_l1_memory_config(ttnn)
    h = int(num_heads)
    kv_h = int(num_kv_heads)
    b = int(q_b1sd.shape[0])
    s_q = int(q_b1sd.shape[2])
    s_k = int(kv_b1sd.shape[2])
    dh = int(q_b1sd.shape[3]) // h

    if ace_step_use_nlp_create_qkv_heads():
        experimental = getattr(ttnn, "experimental", None)
        nlp = getattr(experimental, "nlp_create_qkv_heads", None) if experimental is not None else None
        if nlp is not None:
            kw = {"memory_config": mc} if mc is not None else {}
            if s_q == s_k:
                return nlp(
                    q_b1sd,
                    kv_b1sd,
                    num_heads=h,
                    num_kv_heads=kv_h,
                    transpose_k_heads=transpose_k_heads,
                    **kw,
                )
            q, _, _ = nlp(
                q_b1sd,
                q_b1sd,
                num_heads=h,
                num_kv_heads=kv_h,
                transpose_k_heads=transpose_k_heads,
                **kw,
            )
            k, v = _ace_step_split_kv_bhsd_manual(ttnn, kv_b1sd, b=b, s_k=s_k, kv_h=kv_h, dh=dh, l1_mc=mc)
            return q, k, v

    q = _ace_step_split_q_bhsd_manual(ttnn, q_b1sd, b=b, s_q=s_q, h=h, dh=dh, l1_mc=mc)
    k, v = _ace_step_split_kv_bhsd_manual(ttnn, kv_b1sd, b=b, s_k=s_k, kv_h=kv_h, dh=dh, l1_mc=mc)
    return q, k, v


def ace_step_nlp_concat_heads(ttnn: Any, ctx: Any, *, l1_mc: Any | None = None) -> Any:
    """``[B,H,S,Dh]`` → ``[B,1,S,H*Dh]`` for the output projection linear.

    **Default:** ``ttnn.experimental.nlp_concat_heads`` (~3.9% device time, faster than manual
    permute+reshape on ACE-Step denoise traces).

    Set ``ACE_STEP_MANUAL_CONCAT_HEADS=1`` to force permute+reshape (view ops only).
    """
    mc = l1_mc if l1_mc is not None else ace_step_linear_l1_memory_config(ttnn)

    if not ace_step_use_manual_concat_heads():
        experimental = getattr(ttnn, "experimental", None)
        nlp_concat = getattr(experimental, "nlp_concat_heads", None) if experimental is not None else None
        if nlp_concat is not None:
            if mc is not None and hasattr(ctx, "memory_config") and hasattr(ttnn, "to_memory_config"):
                if ctx.memory_config() != mc:
                    ctx = ace_step_ensure_l1_activation(ttnn, ctx, mc)
            kw = {"memory_config": mc} if mc is not None else {}
            return nlp_concat(ctx, **kw)

    b, h, s, dh = (int(ctx.shape[0]), int(ctx.shape[1]), int(ctx.shape[2]), int(ctx.shape[3]))
    out_shape = (b, 1, s, h * dh)
    _pk = ace_step_permute_kwargs(ttnn)
    _sr = ace_step_reshape_kwargs(ttnn)
    ctx = ttnn.permute(ctx, (0, 2, 1, 3), **_pk)
    return ttnn.reshape(ctx, out_shape, **_sr)


def ace_step_try_nlp_qkv_heads_split(
    ttnn: Any,
    *,
    q_b1sd: Any,
    kv_b1sd: Any | None = None,
    num_heads: int,
    num_kv_heads: int,
    memory_config: Any | None = None,
) -> tuple[Any, Any, Any] | None:
    """Split ``[B,1,S,H*Dh]`` Q (and optional fused ``[B,1,S,2*kv_h*Dh]`` KV) into ``[B,H,S,Dh]`` heads.

    Replaces three ``reshape`` + three ``permute`` (~205 μs/layer) with one
    ``nlp_create_qkv_heads`` call when available. Returns ``None`` on missing op or signature mismatch.
    """
    experimental = getattr(ttnn, "experimental", None)
    nlp_heads = getattr(experimental, "nlp_create_qkv_heads", None) if experimental is not None else None
    if nlp_heads is None:
        return None
    mc = memory_config if memory_config is not None else ace_step_linear_l1_memory_config(ttnn)
    try:
        kw: dict = {
            "num_heads": int(num_heads),
            "num_kv_heads": int(num_kv_heads),
            "transpose_k_heads": False,
        }
        if mc is not None:
            kw["memory_config"] = mc
        if kv_b1sd is not None:
            return nlp_heads(q_b1sd, kv_b1sd, **kw)
        return nlp_heads(q_b1sd, **kw)
    except Exception:
        return None


def ace_step_eltwise_l1_memory_config(ttnn: Any):
    """L1 config for BinaryNg / ``add`` / ``multiply`` / ``softmax`` activations (Tracy ``in0`` bucket)."""
    return ace_step_linear_l1_memory_config(ttnn)


def ace_step_sdpa_mask_memory_config(ttnn: Any):
    """SDPA requires ``attn_mask`` buffers in DRAM (see ``sdpa_device_operation.cpp``)."""
    return getattr(ttnn, "DRAM_MEMORY_CONFIG", None)


def ace_step_ensure_l1_activation(ttnn: Any, tensor: Any, l1_mc: Any | None = None) -> Any:
    """Move a device tensor to L1 interleaved so BinaryNg tags ``in0:l1_interleaved``."""
    if tensor is None:
        return tensor
    mc = l1_mc if l1_mc is not None else ace_step_eltwise_l1_memory_config(ttnn)
    if mc is None or not hasattr(ttnn, "to_memory_config"):
        return tensor
    if hasattr(tensor, "memory_config") and tensor.memory_config() == mc:
        return tensor
    return ttnn.to_memory_config(tensor, mc)


def ace_step_ensure_l1_if_dram(ttnn: Any, tensor: Any, l1_mc: Any | None = None) -> Any:
    """Move a DRAM tensor to L1 interleaved; leave any L1-resident tensor (sharded or not) unchanged.

    Unlike ``ace_step_ensure_l1_activation``, this does not convert L1-sharded tensors to
    L1-interleaved, so ops that accept sharded input (e.g. ``nlp_create_qkv_heads``) receive
    the shard layout intact and avoid a spurious ShardedToInterleaved.
    """
    if tensor is None or not hasattr(tensor, "memory_config"):
        return tensor
    try:
        buf_type = getattr(tensor.memory_config(), "buffer_type", None)
        dram_buf = getattr(getattr(ttnn, "BufferType", None), "DRAM", None)
        if dram_buf is not None and buf_type != dram_buf:
            return tensor  # already in L1 (sharded or interleaved) — leave it
    except Exception:
        pass
    return ace_step_ensure_l1_activation(ttnn, tensor, l1_mc)


def ace_step_ensure_dram_activation(ttnn: Any, tensor: Any, dram_mc: Any | None = None) -> Any:
    """Move a device tensor to DRAM so matmul programs do not clash with L1 activation buffers."""
    if tensor is None:
        return tensor
    mc = dram_mc if dram_mc is not None else getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
    if mc is None or not hasattr(ttnn, "to_memory_config"):
        return tensor
    if hasattr(tensor, "memory_config") and tensor.memory_config() == mc:
        return tensor
    return ttnn.to_memory_config(tensor, mc)


def ace_step_dit_fused_m_tiles(*, batch_size: int, seq_len: int, tile: int = 32) -> int:
    """Fused batch×seq tile rows used by 1D-mcast matmul (``fuse_batch=True``)."""
    tile_size = max(1, int(tile))
    s_tiles = (max(1, int(seq_len)) + tile_size - 1) // tile_size
    return max(1, int(batch_size)) * s_tiles


def ace_step_dit_prefers_dram_activations(
    *,
    batch_size: int,
    seq_len: int,
    max_fused_m: int | None = None,
) -> bool:
    """True when DiT should keep activations in DRAM (long clips; avoids L1 CB overflow).

    Set ``ACE_STEP_DIT_FORCE_L1_MATMUL=1`` to keep L1 in0 + DRAM weights on long clips
    (raise ``ACE_STEP_DIT_MAX_FUSED_M`` if 1D-mcast program config is still needed).
    """
    if ace_step_dit_force_l1_matmul():
        return False
    cap = ace_step_dit_max_fused_m_tiles() if max_fused_m is None else int(max_fused_m)
    return ace_step_dit_fused_m_tiles(batch_size=int(batch_size), seq_len=int(seq_len)) > cap


def ace_step_dit_body_trace_safe(
    *,
    batch_size: int,
    patch_seq_len: int,
    max_fused_m: int | None = None,
) -> bool:
    """Return False when DiT body trace replay is known to drift vs eager (audible noise).

    Long clips fall back to DRAM matmul (``per_core_M`` > 16) with ``to_memory_config`` in the
    graph; body trace capture/replay is not bit-accurate in that regime (same class of issue as
    traced VAE tiles / ``DitCfgPrepTrace``).
    """
    cap = ace_step_dit_max_fused_m_tiles() if max_fused_m is None else int(max_fused_m)
    return ace_step_dit_fused_m_tiles(batch_size=int(batch_size), seq_len=int(patch_seq_len)) <= cap


def ace_step_matmul_activation(
    ttnn: Any,
    tensor: Any,
    linear_kwargs: dict,
    *,
    l1_fn,
    dram_mc: Any | None = None,
) -> Any:
    """Place matmul ``in0`` in L1 when 1D-mcast (or force-L1) is active; weights stay DRAM."""
    if "program_config" in linear_kwargs or ace_step_dit_force_l1_matmul():
        return l1_fn(tensor)
    return ace_step_ensure_dram_activation(ttnn, tensor, dram_mc)


def ace_step_from_torch_activation(
    ttnn: Any,
    tensor: Any,
    *,
    device: Any,
    dtype: Any,
    layout: Any | None = None,
    l1_mc: Any | None = None,
) -> Any:
    """Upload a host tensor directly into L1 (avoids default DRAM ``from_torch``)."""
    tile = getattr(ttnn, "TILE_LAYOUT", None)
    use_layout = layout if layout is not None else tile
    mc = l1_mc if l1_mc is not None else ace_step_linear_l1_memory_config(ttnn)
    kw: dict = {"device": device, "dtype": dtype}
    if use_layout is not None:
        kw["layout"] = use_layout
    if mc is not None:
        kw["memory_config"] = mc
    return ttnn.from_torch(tensor, **kw)


def ace_step_pad_activation_kwargs(ttnn: Any, l1_mc: Any | None = None) -> dict:
    """``memory_config`` for ``ttnn.pad`` / fill-pad style ops on activations."""
    mc = l1_mc if l1_mc is not None else ace_step_linear_l1_memory_config(ttnn)
    return {"memory_config": mc} if mc is not None else {}


def ace_step_binary_kwargs(ttnn: Any, l1_mc: Any | None = None) -> dict:
    """``memory_config`` for ``add`` / ``multiply`` / ``softmax`` with L1 output."""
    mc = l1_mc if l1_mc is not None else ace_step_eltwise_l1_memory_config(ttnn)
    return {"memory_config": mc} if mc is not None else {}


def ace_step_init_hifi2_linear_compute_kernel_config(device: Any):
    """HiFi2 linear config for DRAM-bound projections (DiT + condition encoder)."""
    import ttnn

    init_ck = getattr(ttnn, "init_device_compute_kernel_config", None)
    if not callable(init_ck):
        return None
    return init_ck(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )


def ace_step_init_hifi4_linear_compute_kernel_config(device: Any):
    """HiFi4 linear config for FLOP-bound projections (condition encoder small-seq linears)."""
    import ttnn

    init_ck = getattr(ttnn, "init_device_compute_kernel_config", None)
    if not callable(init_ck):
        return None
    return init_ck(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )


def ace_step_init_lofi_linear_compute_kernel_config(device: Any):
    """LoFi linear config for DiT throughput (paired with ``bfloat8_b`` weights).

    Matches matrix-engine guidance: ``math_approx_mode=True``, ``fp32_dest_acc_en=False``,
    ``packer_l1_acc=False`` (HiFi paths may use ``packer_l1_acc=True``).
    """
    import ttnn

    init_ck = getattr(ttnn, "init_device_compute_kernel_config", None)
    if not callable(init_ck):
        return None
    return init_ck(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )


def ace_step_init_dit_linear_compute_kernel_config(device: Any):
    """DiT linear compute kernel: LoFi (production default, matches perf traces)."""
    return ace_step_init_lofi_linear_compute_kernel_config(device)


def ace_step_init_cond_linear_compute_kernel_config(device: Any):
    """Condition encoder / lyric / timbre linears: LoFi (same default as DiT)."""
    return ace_step_init_lofi_linear_compute_kernel_config(device)


def ace_step_init_cond_sdpa_compute_kernel_config(device: Any):
    """SDPA compute kernel for condition lyric/timbre encoders: LoFi."""
    return ace_step_init_lofi_linear_compute_kernel_config(device)


def ace_step_init_cond_sdpa_program_config(device: Any, *, seq_len: int, num_heads: int, batch_size: int = 1):
    """Explicit SDPA program config for the short fixed-length caption-encoder attention.

    SDPA parallelizes over ``batch * heads * num_q_chunks``. For the encoder (B=1, S=256,
    H≈16) the default heuristic leaves cores idle once it picks a large q-chunk, so we size
    ``q_chunk_size`` so ``B*H*(S/q_chunk)`` covers the worker grid — small q-chunks (down to
    one tile) keep all cores busy on this short sequence. ``k_chunk_size`` stays the full seq
    (one flash pass, no inter-chunk rescale). Env overrides:
    ``ACE_STEP_SDPA_Q_CHUNK`` / ``ACE_STEP_SDPA_K_CHUNK``; ``ACE_STEP_SDPA_PROGRAM_CONFIG=0``
    disables (fall back to the heuristic). Returns ``None`` when unsupported/disabled.
    """
    import ttnn

    if os.environ.get("ACE_STEP_SDPA_PROGRAM_CONFIG", "1").lower() in ("0", "false", "no"):
        return None
    sdpa_pc_cls = getattr(ttnn, "SDPAProgramConfig", None)
    if sdpa_pc_cls is None or not hasattr(device, "compute_with_storage_grid_size"):
        return None

    tile = int(getattr(ttnn, "TILE_SIZE", 32))
    s = max(tile, int(seq_len))
    grid = device.compute_with_storage_grid_size()
    num_cores = int(grid.x) * int(grid.y)
    work = max(1, int(batch_size)) * max(1, int(num_heads))

    # Smallest tile-multiple q-chunk that divides S and yields >= num_cores work units
    # (capped at S). More q-chunks → more parallel work; 1 tile is the finest.
    q_chunk = s
    for q_tiles in range(1, s // tile + 1):
        cand = q_tiles * tile
        if s % cand != 0:
            continue
        if work * (s // cand) >= num_cores:
            q_chunk = cand
            break
    else:
        q_chunk = tile  # finest available if none reaches num_cores

    def _env_chunk(name: str, default: int) -> int:
        v = os.environ.get(name)
        if not v:
            return default
        try:
            iv = int(v)
        except ValueError:
            return default
        if iv <= 0 or iv % tile != 0 or s % iv != 0:
            return default
        return iv

    q_chunk = _env_chunk("ACE_STEP_SDPA_Q_CHUNK", q_chunk)
    k_chunk = _env_chunk("ACE_STEP_SDPA_K_CHUNK", s)
    try:
        return sdpa_pc_cls(
            compute_with_storage_grid_size=grid,
            q_chunk_size=int(q_chunk),
            k_chunk_size=int(k_chunk),
            exp_approx_mode=False,
        )
    except Exception:
        return None


def ace_step_qwen3_optimizations(model_args: Any):
    """``tt_transformers`` decoder config for ACE Qwen3 caption encoder: LoFi + ``bfloat8_b`` weights.

    Passed to :func:`models.tt_transformers.tt.common.create_tt_model` as ``optimizations=``.
    Sets per-op math fidelity and ``bfloat8_b`` projection dtypes (replacing the default
    ``accuracy`` path that uses HiFi4 on prefill attention). Prefill activations are moved to
    L1 via :mod:`qwen_prefill_l1` (``ace_step_apply_qwen_prefill_l1``).
    """
    from models.tt_transformers.tt.model_config import (
        DecodersPrecision,
        MathFidelitySetting,
        ModelOptimizations,
        OpGroup,
        PrecisionSetting,
        TensorGroup,
    )

    conf = ModelOptimizations(
        {
            "TensorPrecision": {
                TensorGroup.FF1_FF3: PrecisionSetting.BFP8,
                TensorGroup.FF2: PrecisionSetting.BFP8,
                TensorGroup.WQKV: PrecisionSetting.BFP8,
                TensorGroup.WO: PrecisionSetting.BFP4,
                TensorGroup.KV_CACHE: PrecisionSetting.BF16,
            },
            "OpFidelity": {
                OpGroup.LI_FF1_FF3: MathFidelitySetting.LOFI,
                OpGroup.LI_FF2: MathFidelitySetting.LOFI,
                OpGroup.LI_QKV_DECODE: MathFidelitySetting.LOFI,
                OpGroup.LI_QKV_PREFILL: MathFidelitySetting.LOFI,
                OpGroup.SDPA_DECODE: MathFidelitySetting.LOFI,
                OpGroup.SDPA_PREFILL: MathFidelitySetting.LOFI,
                OpGroup.LI_O_DECODE: MathFidelitySetting.LOFI,
                OpGroup.LI_O_PREFILL: MathFidelitySetting.LOFI,
            },
        }
    )
    return DecodersPrecision(model_args.n_layers, model_args.model_name, decoder_conf=conf)


def ace_step_five_hz_lm_bfloat8_weights_enabled() -> bool:
    """Use ``bfloat8_b`` for all 5 Hz causal-LM decoder weights (opt-in for perf).

    Default **off** — production uses ``accuracy_decoder_config.json`` (BF16 weights + HiFi4)
    via ``optimizations=None`` in :mod:`qwen_tt_transformers_lm` for HF logits PCC (~0.98).
    Set ``ACE_STEP_LM_BFLOAT8_WEIGHTS=1`` for HiFi2 + ``bfloat8_b`` (~0.90 HF PCC).
    """
    return os.environ.get("ACE_STEP_LM_BFLOAT8_WEIGHTS", "0").lower() in ("1", "true", "yes", "on")


def ace_step_lm_prefill_l1_enabled() -> bool:
    """Keep 5 Hz LM prefill activations in L1 (``ace_step_apply_qwen_prefill_l1``).

    Default **off** — ``ACE_STEP_LM_PREFILL_L1=1`` can L1 circular-buffer clash on P150/BH
    during prefill matmul (see ``qwen_prefill_l1``). Tracy harness sets ``1`` explicitly.
    """
    return os.environ.get("ACE_STEP_LM_PREFILL_L1", "0").lower() in ("1", "true", "yes", "on")


def ace_step_lm_unified_decode_shard_enabled() -> bool:
    """Unify decode WIDTH_SHARDED specs to residual grid (fewer ``ReshardDeviceOperation``). Default on."""
    return os.environ.get("ACE_STEP_LM_UNIFIED_DECODE_SHARD", "1").lower() not in ("0", "false", "no", "off")


def ace_step_lm_decode_qk_norm_sharded_enabled() -> bool:
    """Sharded Q/K head RMSNorm on decode (no L1 interleaved ping-pong). Default on."""
    return os.environ.get("ACE_STEP_LM_DECODE_QK_NORM_SHARDED", "1").lower() not in ("0", "false", "no", "off")


def ace_step_lm_sdpa_gather_unified_enabled() -> bool:
    """Align post-SDPA ``gather_users`` WIDTH with residual grid. Default on."""
    return os.environ.get("ACE_STEP_LM_SDPA_GATHER_UNIFIED", "1").lower() not in ("0", "false", "no", "off")


def ace_step_lm_sdpa_concat_width_enabled() -> bool:
    """Deprecated alias for :func:`ace_step_lm_sdpa_gather_unified_enabled`."""
    if os.environ.get("ACE_STEP_LM_SDPA_CONCAT_WIDTH") is not None:
        return os.environ.get("ACE_STEP_LM_SDPA_CONCAT_WIDTH", "1").lower() not in ("0", "false", "no", "off")
    return ace_step_lm_sdpa_gather_unified_enabled()


def ace_step_lm_narrow_audio_vocab_enabled() -> bool:
    """Narrow ``LMHead`` matmul band during audio-code generation. Default on."""
    return os.environ.get("ACE_STEP_LM_NARROW_AUDIO_VOCAB", "1").lower() not in ("0", "false", "no", "off")


def ace_step_five_hz_lm_optimizations(model_args: Any):
    """``tt_transformers`` decoder config for ACE 5 Hz causal LM: HiFi2 + ``bfloat8_b`` on all weights.

    Replaces the default ``accuracy_decoder_config.json`` path (BF16 weights) when
    :func:`ace_step_five_hz_lm_bfloat8_weights_enabled` is true. Embedding / ``lm_head`` use
    ``dtype=ttnn.bfloat8_b`` from :func:`create_tt_model`; ``LMHead`` keeps the stock HiFi2
    compute kernel (see :mod:`qwen_tt_transformers_lm`).
    """
    from models.tt_transformers.tt.model_config import (
        DecodersPrecision,
        MathFidelitySetting,
        ModelOptimizations,
        OpGroup,
        PrecisionSetting,
        TensorGroup,
    )

    conf = ModelOptimizations(
        {
            "TensorPrecision": {
                TensorGroup.FF1_FF3: PrecisionSetting.BFP8,
                TensorGroup.FF2: PrecisionSetting.BFP8,
                TensorGroup.WQKV: PrecisionSetting.BFP8,
                TensorGroup.WO: PrecisionSetting.BFP8,
                TensorGroup.KV_CACHE: PrecisionSetting.BFP8,
            },
            "OpFidelity": {
                OpGroup.LI_FF1_FF3: MathFidelitySetting.HIFI2,
                OpGroup.LI_FF2: MathFidelitySetting.HIFI2,
                OpGroup.LI_QKV_DECODE: MathFidelitySetting.HIFI2,
                OpGroup.LI_QKV_PREFILL: MathFidelitySetting.HIFI2,
                OpGroup.SDPA_DECODE: MathFidelitySetting.HIFI2,
                OpGroup.SDPA_PREFILL: MathFidelitySetting.HIFI2,
                OpGroup.LI_O_DECODE: MathFidelitySetting.HIFI2,
                OpGroup.LI_O_PREFILL: MathFidelitySetting.HIFI2,
            },
        }
    )
    return DecodersPrecision(model_args.n_layers, model_args.model_name, decoder_conf=conf)


def ace_step_init_dit_rmsnorm_compute_kernel_config(device: Any):
    """RMSNorm compute kernel for DiT blocks: LoFi (default TTNN rmsnorm uses HiFi4)."""
    return ace_step_init_lofi_linear_compute_kernel_config(device)


def ace_step_init_cond_rmsnorm_compute_kernel_config(device: Any):
    """RMSNorm compute kernel for condition encoder blocks: LoFi."""
    return ace_step_init_lofi_linear_compute_kernel_config(device)


def ace_step_dit_rms_norm_kwargs(ttnn: Any, l1_mc: Any | None = None, *, device: Any = None) -> dict:
    """``memory_config`` + LoFi ``compute_kernel_config`` for DiT ``ttnn.rms_norm``."""
    return ace_step_cond_rms_norm_kwargs(ttnn, l1_mc, device=device)


def ace_step_cond_rms_norm_kwargs(ttnn: Any, l1_mc: Any | None = None, *, device: Any = None) -> dict:
    """``memory_config`` + LoFi ``compute_kernel_config`` for ``ttnn.rms_norm`` (keeps TILE/L1)."""
    mc = l1_mc if l1_mc is not None else ace_step_linear_l1_memory_config(ttnn)
    kw: dict = {}
    if mc is not None:
        kw["memory_config"] = mc
    if device is not None:
        ck = ace_step_init_cond_rmsnorm_compute_kernel_config(device)
        if ck is not None:
            kw["compute_kernel_config"] = ck
    return kw


def ace_step_ensure_dit_activation(ttnn: Any, tensor: Any, l1_mc: Any | None = None) -> Any:
    """TILE layout + L1 interleaved — DiT matmul/binary/rms_norm input contract."""
    return ace_step_ensure_cond_activation(ttnn, tensor, l1_mc)


def ace_step_ensure_cond_activation(ttnn: Any, tensor: Any, l1_mc: Any | None = None) -> Any:
    """TILE layout + L1 interleaved — condition encoder matmul/binary/rms_norm contract."""
    if tensor is None:
        return tensor
    t = ace_step_ensure_tile_layout(ttnn, tensor)
    return ace_step_ensure_l1_activation(ttnn, t, l1_mc)


def ace_step_dit_activation_from_torch(
    ttnn: Any,
    host_tensor: Any,
    *,
    device: Any,
    dtype: Any | None = None,
) -> Any:
    """Upload a host tensor as TILE BF16 in L1 — same contract as perf/E2E (no env toggle)."""
    import numpy as np

    try:
        import torch
    except ImportError:
        torch = None  # type: ignore

    use_dtype = dtype if dtype is not None else getattr(ttnn, "bfloat16", None)
    if use_dtype is None:
        raise RuntimeError("bfloat16 required for activations")
    if torch is not None and isinstance(host_tensor, torch.Tensor):
        arr = host_tensor.detach().to(dtype=torch.float32, device="cpu").numpy()
    else:
        arr = np.asarray(host_tensor, dtype=np.float32)
    return ace_step_from_torch_activation(
        ttnn,
        arr,
        device=device,
        dtype=use_dtype,
        l1_mc=ace_step_linear_l1_memory_config(ttnn),
    )


def _mcast_2d_linear_program_config(
    device: Any,
    *,
    m_dim: int,
    k_dim: int,
    n_dim: int,
    grid_size: tuple[int, int] | None = None,
    in0_block_w: int = 4,
    out_subblock_h: int = 1,
    out_subblock_w: int = 4,
    fuse_batch: bool = False,
):
    """2D mcast matmul program config (``MatmulMultiCoreReuseMultiCastProgramConfig``).

    Parallelizes ``M`` over grid height and ``N`` over grid width. Used for DiT prefill
    linears (e.g. ``256×2048×2048`` fused ``wkv``, ``256×3072×3072`` MLP gate/up) where
    1D mcast only spreads ``N`` and leaves ~16 cores active on Blackhole.

    **``in0_block_w`` WARNING**: default is 4 (suited for BF16 weights where accumulation order
    does not affect output quality). For **BFP8 weight** linears, pass ``in0_block_w=2`` —
    the larger value changes K-tile accumulation order and produces audible WAV noise across
    28+ DiT encoder layers (same root cause as the reverted ``in0_block_w_cap=4`` in 1D mcast).
    """
    import ttnn

    cfg_cls = getattr(ttnn, "MatmulMultiCoreReuseMultiCastProgramConfig", None)
    if cfg_cls is None or not hasattr(device, "compute_with_storage_grid_size"):
        return None

    dev_grid = device.compute_with_storage_grid_size()
    if grid_size is None:
        gx = min(8, max(1, int(dev_grid.x)))
        gy = min(4, max(1, int(dev_grid.y)))
    else:
        gx = min(int(grid_size[0]), max(1, int(dev_grid.x)))
        gy = min(int(grid_size[1]), max(1, int(dev_grid.y)))

    tile = int(getattr(ttnn, "TILE_SIZE", 32))
    m = max(tile, int(m_dim))
    k = max(tile, int(k_dim))
    n = max(tile, int(n_dim))

    m_tiles = (m + tile - 1) // tile
    k_tiles = max(1, k // tile)
    n_tiles = max(1, (n + tile - 1) // tile)

    per_core_m = max(1, (m_tiles + gy - 1) // gy)
    per_core_n = max(1, (n_tiles + gx - 1) // gx)

    in0_w = min(int(in0_block_w), k_tiles)
    while k_tiles % in0_w != 0 and in0_w > 1:
        in0_w -= 1

    osh = min(int(out_subblock_h), per_core_m)
    while per_core_m % osh != 0 and osh > 1:
        osh -= 1
    osw = min(int(out_subblock_w), per_core_n)
    while per_core_n % osw != 0 and osw > 1:
        osw -= 1
    while osh * osw > 4 and (osh > 1 or osw > 1):
        if osw > osh and osw > 1:
            osw -= 1
        elif osh > 1:
            osh -= 1
        else:
            break

    return cfg_cls(
        compute_with_storage_grid_size=(gx, gy),
        in0_block_w=in0_w,
        out_subblock_h=osh,
        out_subblock_w=osw,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=bool(fuse_batch),
    )


def _ace_step_dit_prefill_m_dim(*, batch_size: int, seq_len: int) -> int:
    return max(1, int(batch_size)) * max(1, int(seq_len))


def _mcast_1d_linear_program_config(
    device: Any,
    *,
    seq_len: int,
    in_dim: int,
    out_dim: int,
    batch_size: int = 1,
    in0_block_w_cap: int = 2,
    out_subblock_h_cap: int = 4,
    out_subblock_w: int = 1,
    max_per_core_m: int | None = None,
):
    """Shared 1D mcast matmul program config builder for ACE-Step linears.

    With ``fuse_batch=True``, TTNN fuses batch into the ``M`` dim (tile rows) for tensors
    shaped ``[B, 1, S, K]``. TILE layout pads ``S`` to the tile height, so runtime
    ``M`` (tiles) is ``B * ceil(seq_len / tile)`` once ``S`` is padded to TILE height.
    ``seq_len`` without batch—or ignoring TILE padding along ``S``—can under-report
    ``per_core_M``. For ``N`` (tiles across the output inner dim), derive ``per_core_N`` from
    ``ceil(out_dim / tile)`` spread across ``grid.x`` so ``num_blocks_x`` stays within core count.
    ``out_subblock_w`` is clipped so ``per_core_N % out_subblock_w == 0`` (default ``out_block_w``
    equals ``per_core_N``).
    """
    import ttnn

    cfg_cls = getattr(ttnn, "MatmulMultiCoreReuseMultiCast1DProgramConfig", None)
    if cfg_cls is None or not hasattr(device, "compute_with_storage_grid_size"):
        return None

    grid = device.compute_with_storage_grid_size()
    tile = int(getattr(ttnn, "TILE_SIZE", 32))
    bsz = max(1, int(batch_size))
    seq = max(1, int(seq_len))
    # mcast_in0 requires num_blocks_y == 1: every core must hold all M tiles (per_core_M = total M).
    # Do NOT reduce per_core_M to split M across y-rows — that violates the constraint.
    # Instead, use extra y-rows as additional N-workers so the full grid participates.
    s_tiles = (seq + tile - 1) // tile
    per_core_m = max(1, bsz * s_tiles)

    # Guard against L1 CB overflow on WH cores (~1.5 MB each).
    # Static CBs grow ~28 KB per per_core_M tile row; at per_core_M=48 they reach 1339 KB,
    # leaving no room for L1 activation tensors (observed clash at 865 KB with duration=30).
    # per_core_M cap avoids L1 CB overflow on WH (~1.5 MB/core). Override via ACE_STEP_DIT_MAX_FUSED_M.
    _m_cap = ace_step_dit_max_fused_m_tiles() if max_per_core_m is None else int(max_per_core_m)
    if per_core_m > _m_cap:
        return None

    k = max(tile, int(in_dim))

    k_tiles = max(1, k // tile)
    in0_block_w = min(int(in0_block_w_cap), k_tiles)

    n_width_tiles = max(1, (int(out_dim) + tile - 1) // tile)
    gx = max(1, int(grid.x))
    gy = max(1, int(grid.y))

    # Use y-rows as additional N-workers: with (gx, y_rows) grid, N is distributed across
    # gx*y_rows cores.  When N tiles fit within gx*gy we can set per_core_n=1 and just pick
    # enough rows; when N tiles exceed total cores we use the full grid with per_core_n>1.
    if n_width_tiles <= gx * gy:
        y_rows = min(gy, max(1, (n_width_tiles + gx - 1) // gx))
        per_core_n = max(1, (n_width_tiles + gx * y_rows - 1) // (gx * y_rows))
    else:
        y_rows = gy
        per_core_n = max(1, (n_width_tiles + gx * gy - 1) // (gx * gy))

    # When M=1 tile (e.g. seq_len=32), out_subblock_h is forced to 1.  TTNN recommends
    # out_subblock_h * out_subblock_w >= 2, so we need per_core_n >= 2 for out_subblock_w=2.
    # Reduce y_rows (fewer cores) until each core handles at least 2 N tiles.
    if per_core_m == 1 and per_core_n < 2 and n_width_tiles >= 2:
        for cand_y in range(y_rows - 1, 0, -1):
            cand_pcn = max(1, (n_width_tiles + gx * cand_y - 1) // (gx * cand_y))
            if cand_pcn >= 2:
                y_rows = cand_y
                per_core_n = cand_pcn
                break

    out_subblock_h = min(int(out_subblock_h_cap), per_core_m)
    while per_core_m % out_subblock_h != 0 and out_subblock_h > 1:
        out_subblock_h -= 1

    # Default ``out_block_w`` is ``per_core_N``; TTNN requires ``out_block_w % out_subblock_w == 0``.
    # Round per_core_N up to a multiple of the requested subblock width instead of shrinking the
    # subblock to 1×1 (e.g. 64 N-tiles / 10 cores → 7 → snap to 8 so 1×2 subblocks are valid).
    out_subblock_w_target = min(int(out_subblock_w), max(1, int(per_core_n)))
    if out_subblock_w_target > 1 and per_core_n % out_subblock_w_target != 0:
        per_core_n = ((per_core_n + out_subblock_w_target - 1) // out_subblock_w_target) * out_subblock_w_target

    out_subblock_w_eff = min(int(out_subblock_w), max(1, int(per_core_n)))
    while per_core_n % out_subblock_w_eff != 0 and out_subblock_w_eff > 1:
        out_subblock_w_eff -= 1

    return cfg_cls(
        compute_with_storage_grid_size=(gx, y_rows),
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w_eff,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )


def ace_step_dit_attn_linear_program_config(
    device: Any,
    *,
    seq_len: int,
    in_dim: int,
    out_dim: int,
    batch_size: int = 1,
):
    """``MatmulMultiCoreReuseMultiCast1DProgramConfig`` for square DiT ``q`` / ``o`` (e.g. 256×1024×1024)."""
    pc = _mcast_1d_linear_program_config(
        device,
        seq_len=seq_len,
        in_dim=in_dim,
        out_dim=out_dim,
        batch_size=batch_size,
        in0_block_w_cap=2,  # BFP8: cap=4 changes K-tile accumulation → audible noise on DiT
        out_subblock_h_cap=4,
        out_subblock_w=2,
    )
    if pc is not None:
        return pc
    return _mcast_1d_linear_program_config(
        device,
        seq_len=seq_len,
        in_dim=in_dim,
        out_dim=out_dim,
        batch_size=batch_size,
        in0_block_w_cap=2,
        out_subblock_h_cap=4,
        out_subblock_w=1,
    )


def _ace_step_cond_in0_block_w_cap(*, intermediate_size: int | None = None) -> int:
    """``in0_block_w`` cap for Qwen3 / condition linears (not DiT BFP8 2D mcast paths)."""
    if intermediate_size is not None and int(intermediate_size) >= _WIDE_MLP_INTERMEDIATE_THRESHOLD:
        return 1
    return 4


def _ace_step_pick_2d_out_subblock(per_core_m: int, per_core_n: int, *, out_sharded: bool) -> tuple[int, int]:
    """Subblock sizing for 2D mcast (block-sharded out requires ``out_subblock_h==1``)."""
    if out_sharded:
        for w in (4, 3, 2, 1):
            if per_core_n % w == 0:
                return 1, w
        return 1, 1
    osh = min(2, per_core_m)
    while per_core_m % osh != 0 and osh > 1:
        osh -= 1
    osw = min(2, per_core_n)
    while per_core_n % osw != 0 and osw > 1:
        osw -= 1
    return osh, osw


def _ace_step_cond_256x1024_2d_program_config(
    device: Any,
    *,
    seq_len: int,
    in_dim: int,
    out_dim: int,
    batch_size: int = 1,
    in0_sharded: bool = False,
    out_sharded: bool = False,
):
    """Pinned 2D mcast winner for Qwen3 ``256×1024×N`` (sweep: 8×4 grid, ``in0_block_w=4``).

    Latest device-profiler sweep winner on ``M=256, K=1024, N=1024`` with ``bs/dram/bs``:
    ``10.56us, 50.82 TFLOPs, PCC=0.9931`` vs baseline ``18.00us`` (~1.70x).
    Height-shards ``M`` across 4 rows (``per_core_M=2``), ``N`` across 8 cols (``per_core_N=N/8`` tiles).
    Block-sharded ``in0``/``out`` use ``out_subblock_h=1`` (matmul kernel requirement).
    """
    import ttnn

    cfg_cls = getattr(ttnn, "MatmulMultiCoreReuseMultiCastProgramConfig", None)
    if cfg_cls is None or not hasattr(device, "compute_with_storage_grid_size"):
        return None

    tile = int(getattr(ttnn, "TILE_SIZE", 32))
    m_dim = max(1, int(batch_size)) * max(1, int(seq_len))
    k_dim = int(in_dim)
    n_dim = int(out_dim)
    if k_dim != 1024 or m_dim != 256:
        return None

    m_tiles = m_dim // tile  # 8
    k_tiles = k_dim // tile  # 32
    n_tiles = (n_dim + tile - 1) // tile

    gx, gy = 8, 4
    dev_grid = device.compute_with_storage_grid_size()
    if gx > int(dev_grid.x) or gy > int(dev_grid.y):
        return None
    if m_tiles % gy != 0 or n_tiles % gx != 0:
        return None

    per_core_m = m_tiles // gy
    per_core_n = n_tiles // gx
    in0_block_w = 4
    if k_tiles % in0_block_w != 0:
        return None
    out_subblock_h, out_subblock_w = _ace_step_pick_2d_out_subblock(
        per_core_m, per_core_n, out_sharded=bool(out_sharded)
    )
    if not out_sharded and (out_subblock_h * out_subblock_w > 4):
        return None

    return cfg_cls(
        compute_with_storage_grid_size=(gx, gy),
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        out_block_h=per_core_m,
        out_block_w=per_core_n,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        transpose_mcast=False,
        fused_activation=None,
    )


def ace_step_cond_256x1024_block_sharded_memory_config(
    ttnn: Any,
    device: Any,
    *,
    seq_len: int,
    in_dim: int,
    out_dim: int,
    batch_size: int = 1,
    for_output: bool = False,
):
    """Block-sharded L1 memory config for pinned ``256×1024×N`` 2D-mcast winner.

    Returns ``None`` when shape/device/capability does not match the promoted path.
    """
    create_sharded = getattr(ttnn, "create_sharded_memory_config", None)
    shard_strategy = getattr(ttnn, "ShardStrategy", None)
    shard_orientation = getattr(ttnn, "ShardOrientation", None)
    core_grid_cls = getattr(ttnn, "CoreGrid", None)
    if not callable(create_sharded) or shard_strategy is None or shard_orientation is None or core_grid_cls is None:
        return None
    if not hasattr(device, "compute_with_storage_grid_size"):
        return None

    m_dim = max(1, int(batch_size)) * max(1, int(seq_len))
    k_dim = int(in_dim)
    n_dim = int(out_dim)
    if m_dim != 256 or k_dim != 1024:
        return None

    dev_grid = device.compute_with_storage_grid_size()
    gx, gy = 8, 4
    if gx > int(dev_grid.x) or gy > int(dev_grid.y):
        return None

    # in0 shape = [B,1,M,K], out shape = [B,1,M,N]
    w_dim = n_dim if for_output else k_dim
    try:
        return create_sharded(
            (1, 1, m_dim, w_dim),
            core_grid=core_grid_cls(y=gy, x=gx),
            strategy=shard_strategy.BLOCK,
            orientation=shard_orientation.ROW_MAJOR,
        )
    except Exception:
        return None


def ace_step_cond_256x1024_width_sharded_memory_config(
    ttnn: Any,
    device: Any,
    *,
    seq_len: int,
    in_dim: int,
    out_dim: int,
    batch_size: int = 1,
    for_output: bool = False,
):
    """Width-sharded L1 memory config for 1D ws/dram/ws q/wkv path."""
    create_sharded = getattr(ttnn, "create_sharded_memory_config", None)
    shard_strategy = getattr(ttnn, "ShardStrategy", None)
    shard_orientation = getattr(ttnn, "ShardOrientation", None)
    core_grid_cls = getattr(ttnn, "CoreGrid", None)
    if not callable(create_sharded) or shard_strategy is None or shard_orientation is None or core_grid_cls is None:
        return None
    if not hasattr(device, "compute_with_storage_grid_size"):
        return None

    m_dim = max(1, int(batch_size)) * max(1, int(seq_len))
    k_dim = int(in_dim)
    n_dim = int(out_dim)
    if m_dim != 256 or k_dim != 1024:
        return None

    # 1D width-sharded tuned on 4x4 in sweep for this bucket.
    gx, gy = 4, 4
    dev_grid = device.compute_with_storage_grid_size()
    if gx > int(dev_grid.x) or gy > int(dev_grid.y):
        return None

    w_dim = n_dim if for_output else k_dim
    try:
        return create_sharded(
            (1, 1, m_dim, w_dim),
            core_grid=core_grid_cls(y=gy, x=gx),
            strategy=shard_strategy.WIDTH,
            orientation=shard_orientation.ROW_MAJOR,
        )
    except Exception:
        return None


def _ace_step_cond_256x1024_1d_width_program_config(
    device: Any,
    *,
    seq_len: int,
    in_dim: int,
    out_dim: int,
    batch_size: int = 1,
):
    """Pinned 1D ws/dram/ws config from local sweep for ``M=256, K=1024``."""
    import ttnn

    cfg_cls = getattr(ttnn, "MatmulMultiCoreReuseMultiCast1DProgramConfig", None)
    if cfg_cls is None or not hasattr(device, "compute_with_storage_grid_size"):
        return None

    tile = int(getattr(ttnn, "TILE_SIZE", 32))
    m_dim = max(1, int(batch_size)) * max(1, int(seq_len))
    k_dim = int(in_dim)
    n_dim = int(out_dim)
    if m_dim != 256 or k_dim != 1024:
        return None

    gx, gy = 4, 4
    dev_grid = device.compute_with_storage_grid_size()
    if gx > int(dev_grid.x) or gy > int(dev_grid.y):
        return None

    n_tiles = (n_dim + tile - 1) // tile
    num_cores = gx * gy
    if n_tiles % num_cores != 0:
        return None

    return cfg_cls(
        compute_with_storage_grid_size=(gx, gy),
        in0_block_w=2,
        out_subblock_h=1,
        out_subblock_w=2,
        per_core_M=8,
        per_core_N=(n_tiles // num_cores),
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )


def ace_step_cond_linear_program_config(
    device: Any,
    *,
    seq_len: int,
    in_dim: int,
    out_dim: int,
    batch_size: int = 1,
    in0_sharded: bool = False,
    out_sharded: bool = False,
):
    """Program config for condition / Qwen3 linears (e.g. ``256×1024×1024`` attn Q/K)."""
    pc = _ace_step_cond_256x1024_1d_width_program_config(
        device,
        seq_len=seq_len,
        in_dim=in_dim,
        out_dim=out_dim,
        batch_size=batch_size,
    )
    if pc is not None:
        return pc

    pc = _ace_step_cond_256x1024_2d_program_config(
        device,
        seq_len=seq_len,
        in_dim=in_dim,
        out_dim=out_dim,
        batch_size=batch_size,
        in0_sharded=in0_sharded,
        out_sharded=out_sharded,
    )
    if pc is not None:
        return pc
    short = int(seq_len) <= 64
    return _mcast_1d_linear_program_config(
        device,
        seq_len=seq_len,
        in_dim=in_dim,
        out_dim=out_dim,
        batch_size=batch_size,
        in0_block_w_cap=_ace_step_cond_in0_block_w_cap(),
        out_subblock_h_cap=2 if short else 4,
        out_subblock_w=2 if short else 1,
    )


def ace_step_cond_mlp_gate_up_linear_program_config(
    device: Any,
    *,
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,
    batch_size: int = 1,
    out_dim: int | None = None,
):
    """Program config for condition MLP gate/up (e.g. 32×6144×6144).

    Lyric/timbre encoders use intermediate 6144×2048; ``in0_block_w=2`` plus L1-hosted
    activations can overrun per-core circular-buffer budget (static CB vs tensor L1).
    Use ``in0_block_w_cap=1`` when ``intermediate_size >= 4608``; ``4`` for Qwen3/DiT-scale MLPs.
    """
    short = int(seq_len) <= 64
    n_out = int(out_dim) if out_dim is not None else int(intermediate_size)
    return _mcast_1d_linear_program_config(
        device,
        seq_len=seq_len,
        in_dim=hidden_size,
        out_dim=n_out,
        batch_size=batch_size,
        in0_block_w_cap=_ace_step_cond_in0_block_w_cap(intermediate_size=intermediate_size),
        out_subblock_h_cap=2 if short else 4,
        out_subblock_w=2 if short else 1,
    )


def ace_step_dit_fused_qwkv_linear_program_config(
    device: Any,
    *,
    seq_len: int,
    in_dim: int,
    hidden_size: int,
    fused_kv_dim: int,
    batch_size: int = 1,
):
    """Program config for fused self-attn ``q`` + ``wkv`` (one matmul vs two)."""
    return _mcast_1d_linear_program_config(
        device,
        seq_len=seq_len,
        in_dim=in_dim,
        out_dim=int(hidden_size) + int(fused_kv_dim),
        batch_size=batch_size,
        in0_block_w_cap=2,
        out_subblock_h_cap=4,
        out_subblock_w=1,
    )


def ace_step_dit_mlp_fused_gate_up_linear_program_config(
    device: Any,
    *,
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,
    batch_size: int = 1,
):
    """Program config for fused MLP ``gate_proj`` + ``up_proj`` (``2×intermediate`` output)."""
    return _mcast_1d_linear_program_config(
        device,
        seq_len=seq_len,
        in_dim=hidden_size,
        out_dim=2 * int(intermediate_size),
        batch_size=batch_size,
        in0_block_w_cap=2,
        out_subblock_h_cap=4,
        out_subblock_w=1,
    )


def ace_step_dit_fused_wkv_linear_program_config(
    device: Any,
    *,
    seq_len: int,
    hidden_size: int,
    fused_kv_dim: int,
    batch_size: int = 1,
):
    """Program config for fused ``wkv`` (e.g. 256×2048×2048 when ``hidden_size=1024``, GQA ``fused_kv_dim=2048``)."""
    m_dim = _ace_step_dit_prefill_m_dim(batch_size=batch_size, seq_len=seq_len)
    if m_dim >= 128:
        pc = _mcast_2d_linear_program_config(
            device,
            m_dim=m_dim,
            k_dim=int(hidden_size),
            n_dim=int(fused_kv_dim),
            in0_block_w=2,  # cap=4 causes BFP8 K-tile accumulation order change → WAV noise
        )
        if pc is not None:
            return pc
    return _mcast_1d_linear_program_config(
        device,
        seq_len=seq_len,
        in_dim=hidden_size,
        out_dim=fused_kv_dim,
        batch_size=batch_size,
        in0_block_w_cap=2,
        out_subblock_h_cap=4,
        out_subblock_w=1,
    )


def ace_step_dit_mlp_gate_up_linear_program_config(
    device: Any,
    *,
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,
    batch_size: int = 1,
):
    """Program config for MLP ``gate_proj`` / ``up_proj`` (e.g. 256×3072×3072)."""
    m_dim = _ace_step_dit_prefill_m_dim(batch_size=batch_size, seq_len=seq_len)
    if m_dim >= 128:
        pc = _mcast_2d_linear_program_config(
            device,
            m_dim=m_dim,
            k_dim=int(hidden_size),
            n_dim=int(intermediate_size),
            in0_block_w=2,  # cap=4 causes BFP8 K-tile accumulation order change → WAV noise
        )
        if pc is not None:
            return pc
    return _mcast_1d_linear_program_config(
        device,
        seq_len=seq_len,
        in_dim=hidden_size,
        out_dim=intermediate_size,
        batch_size=batch_size,
        in0_block_w_cap=2,
        out_subblock_h_cap=4,
        out_subblock_w=1,
    )


def ace_step_dit_mlp_down_proj_linear_program_config(
    device: Any,
    *,
    seq_len: int,
    intermediate_size: int,
    hidden_size: int,
    batch_size: int = 1,
):
    """Program config for MLP ``down_proj`` (e.g. 256×3072×1024)."""
    m_dim = _ace_step_dit_prefill_m_dim(batch_size=batch_size, seq_len=seq_len)
    if m_dim >= 128:
        pc = _mcast_2d_linear_program_config(
            device,
            m_dim=m_dim,
            k_dim=int(intermediate_size),
            n_dim=int(hidden_size),
            in0_block_w=2,  # cap=4 causes BFP8 K-tile accumulation order change → WAV noise
        )
        if pc is not None:
            return pc
    return _mcast_1d_linear_program_config(
        device,
        seq_len=seq_len,
        in_dim=intermediate_size,
        out_dim=hidden_size,
        batch_size=batch_size,
        in0_block_w_cap=2,
        out_subblock_h_cap=4,
        out_subblock_w=1,
    )


def ace_step_dense_linear_program_config(
    device: Any,
    *,
    seq_len: int,
    in_dim: int,
    out_dim: int,
    batch_size: int = 1,
):
    """General 1D mcast program config for dense linears not covered by a dedicated helper.

    Used for: condition_embedder (256×2048×1024), depatchify proj_out (256×1024×4096),
    text projector, and any other linear where shapes are not known at module init time.
    """
    return _mcast_1d_linear_program_config(
        device,
        seq_len=seq_len,
        in_dim=in_dim,
        out_dim=out_dim,
        batch_size=batch_size,
        in0_block_w_cap=2,
        out_subblock_h_cap=4,
        out_subblock_w=1,
    )


# ---------------------------------------------------------------------------
# WIDTH_SHARDED 1D-mcast RMSNorm helper
# ---------------------------------------------------------------------------

_ACE_STEP_WIDTH_SHARDED_RMSNORM_MAX_BLOCK_H = 16


def ace_step_tile_physical_m_dim(
    *,
    batch: int,
    one: int,
    seq: int,
    tile: int = 32,
) -> int:
    """Physical M (elements) for TILE ``[B,1,S,K]`` tensors used in width-sharded RMSNorm.

    TILE layout pads ``S`` to a tile multiple, so logical ``B*1*S`` can understate the
    physical height seen by sharded memory configs (e.g. ``[128,1,5,H]`` → 4096, not 640).
    """
    s_tiles = (max(1, int(seq)) + tile - 1) // tile
    return max(1, int(batch)) * max(1, int(one)) * s_tiles * tile


def _ace_step_pick_width_shard_cores(*, k_tiles: int, device: Any) -> "tuple[int, int] | None":
    """Return ``(cx, cy)`` for a WIDTH_SHARDED rms_norm across ``num_cores = cx * cy`` cores."""
    if not hasattr(device, "compute_with_storage_grid_size"):
        return None
    g = device.compute_with_storage_grid_size()
    max_cx = max(1, int(g.x))
    max_cy = max(1, int(g.y))

    for nc in range(k_tiles, 0, -1):
        if k_tiles % nc != 0:
            continue
        for cx in range(min(nc, max_cx), 0, -1):
            if nc % cx == 0:
                cy = nc // cx
                if cy <= max_cy:
                    return int(cx), int(cy)
    return None


def ace_step_rms_norm_width_sharded(
    ttnn: Any,
    x: Any,
    weight: Any,
    epsilon: float,
    *,
    device: Any,
    l1_mc: Any | None = None,
    compute_kernel_config: Any | None = None,
    activation_dtype: Any | None = None,
) -> Any:
    """WIDTH_SHARDED ``ttnn.rms_norm`` using ``LayerNormShardedMultiCoreProgramConfig``.

    For block norms ``[B, 1, S, K]`` (e.g. ``[1, 1, 256, 1024]``): shards K across cores,
    runs sharded RMSNorm, then converts back to L1 interleaved. Falls back to interleaved
    ``rms_norm`` on shape or capability mismatch.

    When ``activation_dtype`` is set (e.g. ``bfloat8_b`` for LoFi Q/K linears), activations are
    typecast before RMSNorm so the norm output matches that dtype (``ttnn.rms_norm`` output dtype
    equals input). Gamma (``weight``) stays BF16 per TTNN API.
    """
    tile = 32
    lnpc_cls = getattr(ttnn, "LayerNormShardedMultiCoreProgramConfig", None)
    create_shard = getattr(ttnn, "create_sharded_memory_config", None)
    i2s = getattr(ttnn, "interleaved_to_sharded", None)
    s2i = getattr(ttnn, "sharded_to_interleaved", None)
    shard_strat = getattr(ttnn, "ShardStrategy", None)
    shard_ori = getattr(ttnn, "ShardOrientation", None)

    _fb_kw: dict = {}
    if l1_mc is not None:
        _fb_kw["memory_config"] = l1_mc
    if compute_kernel_config is not None:
        _fb_kw["compute_kernel_config"] = compute_kernel_config

    def _maybe_typecast_act(t: Any) -> Any:
        if activation_dtype is None or t.dtype == activation_dtype:
            return t
        tc_kw = {"memory_config": l1_mc} if l1_mc is not None else {}
        return ttnn.typecast(t, dtype=activation_dtype, **tc_kw)

    def _fallback(t: Any) -> Any:
        return ttnn.rms_norm(_maybe_typecast_act(t), weight=weight, epsilon=epsilon, **_fb_kw)

    if any(v is None for v in (lnpc_cls, create_shard, i2s, s2i, shard_strat, shard_ori)):
        return _fallback(x)

    shape = x.shape
    if len(shape) != 4:
        return _fallback(x)

    b_dim = int(shape[0])
    one_dim = int(shape[1])
    s_dim = int(shape[2])
    k_dim = int(shape[3])

    k_pad = (k_dim + tile - 1) // tile * tile
    did_k_pad = k_pad != k_dim
    x_work = x
    if did_k_pad:
        pad4 = ((0, 0), (0, 0), (0, 0), (0, k_pad - k_dim))
        kw_pad = {"memory_config": l1_mc} if l1_mc is not None else {}
        x_work = ttnn.pad(x_work, padding=pad4, value=0.0, **kw_pad)

    m_physical = ace_step_tile_physical_m_dim(batch=b_dim, one=one_dim, seq=s_dim, tile=tile)
    k_tiles = k_pad // tile
    grid_pair = _ace_step_pick_width_shard_cores(k_tiles=k_tiles, device=device)
    if grid_pair is None:
        if did_k_pad:
            ace_step_safe_deallocate(ttnn, x_work)
        return _fallback(x)

    cx, cy = grid_pair
    shard_width = k_pad // (cx * cy)
    if shard_width % tile != 0 or shard_width == 0:
        if did_k_pad:
            ace_step_safe_deallocate(ttnn, x_work)
        return _fallback(x)

    try:
        sharded_mc = create_shard(
            shape=(m_physical, shard_width),
            core_grid=ttnn.CoreGrid(y=cy, x=cx),
            strategy=shard_strat.WIDTH,
            orientation=shard_ori.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
    except Exception:
        if did_k_pad:
            ace_step_safe_deallocate(ttnn, x_work)
        return _fallback(x)

    x_l1 = ace_step_ensure_l1_activation(ttnn, x_work, l1_mc)
    x_l1 = _maybe_typecast_act(x_l1)
    try:
        x_sharded = i2s(x_l1, sharded_mc)
    except Exception:
        return _fallback(x)

    block_h = m_physical // tile
    block_w = shard_width // tile

    if block_h > _ACE_STEP_WIDTH_SHARDED_RMSNORM_MAX_BLOCK_H:
        ace_step_safe_deallocate(ttnn, x_sharded)
        if did_k_pad:
            ace_step_safe_deallocate(ttnn, x_work)
        return _fallback(x)

    subblock_w = 1
    for sw in range(min(block_w, 4), 0, -1):
        if block_w % sw == 0:
            subblock_w = sw
            break

    try:
        prog_cfg = lnpc_cls(
            compute_with_storage_grid_size=(cx, cy),
            subblock_w=subblock_w,
            block_h=block_h,
            block_w=block_w,
            inplace=False,
        )
    except Exception:
        ace_step_safe_deallocate(ttnn, x_sharded)
        return _fallback(x)

    rn_kw: dict = {"memory_config": sharded_mc, "program_config": prog_cfg}
    if compute_kernel_config is not None:
        rn_kw["compute_kernel_config"] = compute_kernel_config
    try:
        out_sharded = ttnn.rms_norm(x_sharded, weight=weight, epsilon=epsilon, **rn_kw)
        ace_step_safe_deallocate(ttnn, x_sharded)
    except Exception:
        ace_step_safe_deallocate(ttnn, x_sharded)
        return _fallback(x)

    out_kw = {"memory_config": l1_mc} if l1_mc is not None else {}
    try:
        out = s2i(out_sharded, **out_kw)
        ace_step_safe_deallocate(ttnn, out_sharded)
    except Exception:
        ace_step_safe_deallocate(ttnn, out_sharded)
        return _fallback(x)

    if did_k_pad:
        try:
            out = ttnn.slice(out, (0, 0, 0, 0), (b_dim, one_dim, s_dim, k_dim), **out_kw)
        except Exception:
            pass

    return out
