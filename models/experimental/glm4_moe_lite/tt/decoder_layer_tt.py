# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import os
import time
from typing import Any

import torch
from loguru import logger

import ttnn
from models.experimental.glm4_moe_lite.tt.attention_decode import flash_mla_and_output, kv_cache_update, q_projection
from models.experimental.glm4_moe_lite.tt.config import Glm4MoeLiteHParams
from models.experimental.glm4_moe_lite.tt.linear_helpers import (
    _is_l1_interleaved,
    _to_l1_if_needed,
    prefill_linear_ws_out,
    prefill_norm_config_from_width_sharded_tensor,
    prefill_per_head_linear,
    prefill_width_sharded_norm_config,
    sharded_decode_norm,
)
from models.experimental.glm4_moe_lite.tt.mlp_decode import dense_mlp_forward, moe_mlp_forward
from models.experimental.glm4_moe_lite.tt.runtime_config import Glm4RuntimeConfig
from models.experimental.glm4_moe_lite.fused_ops.q_split_heads import (
    prefill_q_heads_memory_config,
    prefill_q_nlp_input_memory_config,
    prefill_q_slice_memory_config,
    q_split_heads,
)

_SIGNPOST_ENABLED = os.environ.get("GLM4_MOE_LITE_SIGNPOST", "").strip() == "1"
if _SIGNPOST_ENABLED:
    from tracy import signpost


def _profile_add(profile: dict[str, float] | None, key: str, elapsed_s: float) -> None:
    if profile is None:
        return
    profile[key] = float(profile.get(key, 0.0)) + float(elapsed_s)


def prepare_decode_rope_inputs_for_rotary_llama_decode_mode_tt(
    *,
    device: Any,
    cos_batch: ttnn.Tensor,  # [1,1,B,rope_dim] TILE
    sin_batch: ttnn.Tensor,  # [1,1,B,rope_dim] TILE
    trans_matrix: ttnn.Tensor,  # [1,1,32,32] TILE
    batch: int,
    rope_dim: int,
) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.MemoryConfig]:
    """Prepare HEIGHT_SHARDED cos/sin/trans inputs for decode-mode rotary_embedding_llama."""
    batch = int(batch)
    rope_dim = int(rope_dim)
    if batch <= 0:
        raise ValueError("batch must be > 0")
    if rope_dim <= 0:
        raise ValueError("rope_dim must be > 0")

    grid_size = device.compute_with_storage_grid_size()
    user_grid = ttnn.num_cores_to_corerangeset(int(batch), grid_size, row_wise=True)

    rope_sharded_cfg = ttnn.create_sharded_memory_config(
        shape=(ttnn.TILE_SIZE, rope_dim),
        core_grid=user_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    # Match the DeepSeek decode pattern:
    # - Input cos/sin from gather/embedding is [1, 1, B, rope_dim]
    # - Decode RoPE kernel expects batch in dim=1: [1, B, 1[32], rope_dim]
    #
    # IMPORTANT: Use interleaved_to_sharded (not to_memory_config) to avoid
    # sharding alignment failures for small batches.
    cos_decode = ttnn.transpose(cos_batch, 1, 2)  # [1, B, 1[32], rope_dim]
    sin_decode = ttnn.transpose(sin_batch, 1, 2)  # [1, B, 1[32], rope_dim]
    cos_decode = ttnn.interleaved_to_sharded(cos_decode, rope_sharded_cfg)
    sin_decode = ttnn.interleaved_to_sharded(sin_decode, rope_sharded_cfg)

    trans_mat_mem_config = ttnn.create_sharded_memory_config(
        shape=(ttnn.TILE_SIZE, ttnn.TILE_SIZE),
        core_grid=user_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    trans_decode = ttnn.repeat(trans_matrix, ttnn.Shape((1, 1, batch, 1)))
    trans_decode = ttnn.interleaved_to_sharded(trans_decode, trans_mat_mem_config)

    return cos_decode, sin_decode, trans_decode, rope_sharded_cfg


def _mesh_shape(device: Any) -> tuple[int, int]:
    if device.__class__.__name__ != "MeshDevice":
        return (1, 1)
    return (int(device.shape[0]), int(device.shape[1]))


def _tp_cluster_axis(device: Any) -> int | None:
    """Return the mesh axis used for TP-style sharding (preferred: cols)."""
    if device.__class__.__name__ != "MeshDevice":
        return None
    mesh_rows, mesh_cols = _mesh_shape(device)
    if mesh_cols > 1:
        return 1
    if mesh_rows > 1:
        return 0
    return None


def _moe_sparse_tokens_multiple(*, device: Any, moe_runtime: Any) -> int:
    """Return the minimum per-device token multiple required by sparse MoE.

    Sparse experts require global token count (`tokens_per_device * num_dispatch_devices`)
    divisible by `sparsity_block_size`. This computes the minimal per-device multiple.
    """
    block = max(1, int(getattr(moe_runtime, "sparsity_block_size", 32)))
    dispatch_axis = int(getattr(moe_runtime, "dispatch_cluster_axis", 0))
    mesh_rows, mesh_cols = _mesh_shape(device)
    dispatch_devices = int((mesh_rows, mesh_cols)[dispatch_axis])
    dispatch_devices = max(1, dispatch_devices)
    return max(1, block // math.gcd(block, dispatch_devices))


def _parse_math_fidelity(value: str, *, default: ttnn.MathFidelity) -> ttnn.MathFidelity:
    raw = value.strip().lower()
    if not raw:
        return default
    table = {
        "lofi": ttnn.MathFidelity.LoFi,
        "hifi2": ttnn.MathFidelity.HiFi2,
        "hifi3": ttnn.MathFidelity.HiFi3,
        "hifi4": ttnn.MathFidelity.HiFi4,
    }
    return table.get(raw, default)


def _env_bool(name: str, *, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return bool(default)
    return raw not in {"0", "false", "no", "off"}


@torch.no_grad()
def prepare_decode_rope_and_positions_tt(
    *,
    device: Any,
    rope: dict[str, ttnn.Tensor],
    positions: torch.Tensor,
) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
    """Prepare shared decode inputs used across all layers for a decode step.

    Returns:
    - tt_positions: TT int32 [B]
    - cos_batch: TT bf16 [1, 1, B, rope_dim]
    - sin_batch: TT bf16 [1, 1, B, rope_dim]
    """
    if positions.dtype not in (torch.int32, torch.int64):
        positions = positions.to(torch.int32)
    if positions.ndim != 1:
        raise ValueError(f"expected positions shape [B], got {tuple(positions.shape)}")

    is_mesh_device = device.__class__.__name__ == "MeshDevice"
    mesh_mapper = ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None

    tt_positions = ttnn.from_torch(
        positions.to(torch.int32),
        device=device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper,
    )

    # Fetch per-position RoPE cos/sin rows.
    #
    # Important for multi-device correctness:
    # - `ttnn.gather` on MeshDevice has been observed to be fragile for small decode
    #   batches, and it also builds a large repeated index tensor on host.
    # - `ttnn.embedding` is the DeepSeek implementation pattern and yields the exact
    #   [1, batch, rope_dim] rows we need without host-side index expansion.
    positions_clamped = positions.to(torch.int32).clamp_min(0)
    rope_dim = int(rope["cos_matrix"].shape[3])

    batch = int(positions_clamped.shape[0])
    padded_batch = ((batch + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
    if padded_batch != batch:
        positions_padded = torch.nn.functional.pad(
            positions_clamped.view(1, batch),
            (0, padded_batch - batch),
            "constant",
            0,
        )
    else:
        positions_padded = positions_clamped.view(1, batch)

    rot_idxs = ttnn.from_torch(
        positions_padded.to(torch.int32),
        device=device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper,
    )

    # embedding([1, B], [1,1,S,D]) -> [1, B, D]
    cos_rows = ttnn.embedding(rot_idxs, rope["cos_matrix"], layout=ttnn.TILE_LAYOUT)
    sin_rows = ttnn.embedding(rot_idxs, rope["sin_matrix"], layout=ttnn.TILE_LAYOUT)

    # `ttnn.unsqueeze_to_4D` is a reshape which can behave like a view. Some
    # TTNN view-like ops do not refcount underlying buffers, so aggressively
    # deallocating the source tensor can lead to use-after-free and corrupt
    # RoPE inputs (observed as nondeterministic / garbled text output).
    # Materialize (clone) the final cos/sin batches before freeing intermediates.
    cos_batch_view = ttnn.unsqueeze_to_4D(cos_rows)  # [1,1,B_pad,D]
    sin_batch_view = ttnn.unsqueeze_to_4D(sin_rows)  # [1,1,B_pad,D]

    if padded_batch != batch:
        cos_batch_view = ttnn.slice(cos_batch_view, [0, 0, 0, 0], [1, 1, batch, rope_dim])
        sin_batch_view = ttnn.slice(sin_batch_view, [0, 0, 0, 0], [1, 1, batch, rope_dim])

    cos_batch = ttnn.clone(cos_batch_view, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    sin_batch = ttnn.clone(sin_batch_view, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # Best-effort cleanup of intermediate buffers (do not deallocate the cloned outputs).
    for t in (rot_idxs, cos_rows, sin_rows, cos_batch_view, sin_batch_view):
        try:
            ttnn.deallocate(t, force=False)
        except Exception:
            pass

    return tt_positions, cos_batch, sin_batch


def _shard_kvpe_update_tensor(
    *,
    device: Any,
    kvpe_new: ttnn.Tensor,
    batch: int,
    kvpe_dim: int,
    skip_defensive_clones: bool = False,
) -> ttnn.Tensor:
    """Transform KVPE update tensor into the sharded layout required by paged_update_cache."""
    # `paged_update_cache` expects the update tensor to be sharded.
    #
    # Correctness: `ttnn.pad` and some view-like ops can alias the input buffer without
    # increasing refcounts. Make the padded/permuted update tensor own its buffer to
    # avoid intermittent use-after-free corruption in decode.
    # Pad in ROW_MAJOR to avoid FillPad on TILE input; TilizeWithValPadding zeros padding.
    kvpe_rm = ttnn.to_layout(kvpe_new, ttnn.ROW_MAJOR_LAYOUT)
    kvpe_padded_rm = ttnn.pad(kvpe_rm, [(0, 0), (0, ttnn.TILE_SIZE - 1), (0, 0), (0, 0)], 0)  # [1,32,B,kvpe_dim]
    ttnn.deallocate(kvpe_rm, force=False)
    kvpe_perm_rm = ttnn.permute(kvpe_padded_rm, (0, 2, 1, 3))  # [1,B,32,kvpe_dim]
    ttnn.deallocate(kvpe_padded_rm, force=False)
    kvpe_perm = ttnn.to_layout(kvpe_perm_rm, ttnn.TILE_LAYOUT)
    ttnn.deallocate(kvpe_perm_rm, force=False)

    # Shard across the (B*32) height dimension so each user gets one 32xkvpe_dim shard.
    grid_size = device.compute_with_storage_grid_size()
    user_grid = ttnn.num_cores_to_corerangeset(int(batch), grid_size, row_wise=True)
    kvpe_sharded_cfg = ttnn.create_sharded_memory_config(
        shape=(ttnn.TILE_SIZE, int(kvpe_dim)),
        core_grid=user_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    if skip_defensive_clones:
        kvpe_sharded = ttnn.to_memory_config(kvpe_perm, kvpe_sharded_cfg)
    else:
        kvpe_sharded_view = ttnn.to_memory_config(kvpe_perm, kvpe_sharded_cfg)
        kvpe_sharded = ttnn.clone(kvpe_sharded_view, memory_config=kvpe_sharded_cfg)
        # NOTE: kvpe_sharded_view may alias kvpe_perm; do not deallocate it separately.
        ttnn.deallocate(kvpe_perm, force=False)
    return kvpe_sharded


def _fused_kv_branch_forward(
    *,
    device: Any,
    x: ttnn.Tensor,
    fused_kv: dict,
    cos_batch: ttnn.Tensor,
    sin_batch: ttnn.Tensor,
) -> ttnn.Tensor:
    """Execute fused KV cache branch: matmul + gather + RMSNorm + RoPE in one dispatch.

    The kernel reads x, cos, sin directly from DRAM and writes nope+rope output
    directly to a pre-allocated DRAM tensor. Only format conversions and weight
    resharding remain as Python-side ops.

    Input x: [1,1,1,2048] TILE DRAM (batch=1 only).
    Returns kvpe_new: [1,1,1,576] TILE DRAM.
    """
    from models.experimental.glm4_moe_lite.fused_ops.kv_cache_branch.op import GLMKVCacheBranch

    # 1. Convert x to ROW_MAJOR DRAM (kernel reads contiguous bytes as TILE_1x32 tiles)
    x_rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)

    # 2. Convert cos/sin to ROW_MAJOR DRAM (kernel reads contiguous bytes)
    cos_rm = ttnn.to_layout(cos_batch, ttnn.ROW_MAJOR_LAYOUT)
    cos_dram = ttnn.to_memory_config(cos_rm, ttnn.DRAM_MEMORY_CONFIG)
    sin_rm = ttnn.to_layout(sin_batch, ttnn.ROW_MAJOR_LAYOUT)
    sin_dram = ttnn.to_memory_config(sin_rm, ttnn.DRAM_MEMORY_CONFIG)

    # 3. Reshard weight from DRAM to L1 for this layer
    w_kv_a_l1 = ttnn.to_memory_config(fused_kv["w_kv_a"], fused_kv["w_kv_a_l1_config"])

    # 4. Fused kernel: reads x/cos/sin from DRAM, writes nope+rope to kvpe_output DRAM
    kvpe_output = GLMKVCacheBranch.op(
        input_tensor=x_rm,
        dkv_matmul_weights_tensor=w_kv_a_l1,
        gamma_tensor=fused_kv["gamma"],
        cos_tensor=cos_dram,
        sin_tensor=sin_dram,
        trans_mat_tensor=fused_kv["trans_mat"],
        kvpe_output_tensor=fused_kv["kvpe_output"],
        rope_core_grid=fused_kv["rope_crs"],
        epsilon=fused_kv["epsilon"],
    )

    ttnn.deallocate(w_kv_a_l1, force=True)

    # 5. Convert output to standard TILE format for downstream ops
    kvpe_dim = int(kvpe_output.shape[-1])
    kvpe_4d = ttnn.reshape(kvpe_output, [1, 1, 1, kvpe_dim])
    kvpe_new = ttnn.to_layout(kvpe_4d, ttnn.TILE_LAYOUT)

    # Cleanup temporaries
    ttnn.deallocate(x_rm, force=False)
    ttnn.deallocate(cos_dram, force=False)
    ttnn.deallocate(sin_dram, force=False)

    return kvpe_new


@torch.no_grad()
def run_decoder_layer_decode_one_step_update_cache_tt(
    *,
    device: Any,
    x_embed_tok: ttnn.Tensor,
    tt_positions: ttnn.Tensor,
    page_table_tt: ttnn.Tensor,
    kvpe_cache: ttnn.Tensor,
    cos_batch: ttnn.Tensor,
    sin_batch: ttnn.Tensor,
    trans_matrix: ttnn.Tensor,
    cos_decode: ttnn.Tensor | None = None,
    sin_decode: ttnn.Tensor | None = None,
    trans_decode: ttnn.Tensor | None = None,
    rope_sharded_cfg: ttnn.MemoryConfig | None = None,
    w: Any,
    hparams: Glm4MoeLiteHParams,
    moe_runtime: Any | None = None,
    profile: dict[str, float] | None = None,
    use_decode_rope: bool = False,
    positions_main_tt: ttnn.Tensor | None = None,
    positions_draft_tt: ttnn.Tensor | None = None,
    layer_idx: int = -1,
    use_signpost: bool = False,
) -> ttnn.Tensor:
    """Run one decode step for a single decoder layer and update its KVPE cache.

    Inputs:
    - x_embed_tok: [1, 1, B, hidden] tiled, batch is in dim=2 (seq axis)
    - tt_positions: TT int32 [B] row-major, absolute positions for each batch slot
      (used for RoPE and attention; all lanes have real positions)
    - page_table_tt: [B, max_num_blocks_per_req] int32 row-major on device
    - kvpe_cache: [num_blocks, 1, block_size, kvpe_dim] on device
    - cos_batch/sin_batch/trans_matrix: RoPE tensors for the per-user positions
    - w: weights object with RMSNorms and projection weights (duck-typed)
    - positions_main_tt: optional TT int32 [B] with draft lanes = -1.
    - positions_draft_tt: optional TT int32 [B] with main lanes = -1.
      When both positions_main_tt and positions_draft_tt are provided,
      paged_update_cache is called twice with alternating -1 masks to serialize
      writes for aliased page_table entries (batch-expansion spec decode).

    Returns:
    - x_out: [1, 1, B, hidden] tiled
    """
    batch = int(x_embed_tok.shape[2])
    if batch <= 0:
        raise ValueError("batch must be > 0")

    cfg = Glm4RuntimeConfig.from_env(device=device)

    if cfg.layer_identity:
        return ttnn.clone(x_embed_tok, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    t_layer0 = time.perf_counter() if profile is not None else 0.0

    assert tt_positions.layout == ttnn.ROW_MAJOR_LAYOUT, "tt_positions must be ROW_MAJOR_LAYOUT"
    rope_dim = int(hparams.qk_rope_head_dim)
    if rope_dim <= 0:
        raise ValueError("rope_dim must be > 0")
    if use_decode_rope and rope_dim % ttnn.TILE_SIZE != 0:
        raise ValueError(f"decode RoPE requires rope_dim divisible by {ttnn.TILE_SIZE}, got rope_dim={rope_dim}")

    # Build the decode-mode RoPE closure (captures sharded cos/sin/trans tensors)
    owns_decode_rope_inputs = False
    rope_decode_fn = None
    if use_decode_rope:
        any_provided = cos_decode is not None or sin_decode is not None or trans_decode is not None
        if any_provided:
            if cos_decode is None or sin_decode is None or trans_decode is None:
                raise ValueError("cos_decode, sin_decode, and trans_decode must be provided together")
            if rope_sharded_cfg is None:
                rope_sharded_cfg = cos_decode.memory_config()
        else:
            (
                cos_decode,
                sin_decode,
                trans_decode,
                rope_sharded_cfg,
            ) = prepare_decode_rope_inputs_for_rotary_llama_decode_mode_tt(
                device=device,
                cos_batch=cos_batch,
                sin_batch=sin_batch,
                trans_matrix=trans_matrix,
                batch=batch,
                rope_dim=rope_dim,
            )
            owns_decode_rope_inputs = True

        def rope_decode_fn(t: ttnn.Tensor, *, heads: int) -> ttnn.Tensor:
            nonlocal rope_sharded_cfg
            assert rope_sharded_cfg is not None
            if int(t.shape[-1]) != rope_dim:
                raise ValueError(f"rope tensor dim mismatch: expected {rope_dim}, got {int(t.shape[-1])}")
            t = ttnn.permute(t, (0, 2, 1, 3))
            heads = int(heads)
            pad_h = ttnn.TILE_SIZE - heads
            if pad_h:
                t = ttnn.pad(t, [(0, 0), (0, 0), (0, pad_h), (0, 0)], 0)
            t = ttnn.to_memory_config(t, rope_sharded_cfg)
            t = ttnn.experimental.rotary_embedding_llama(t, cos_decode, sin_decode, trans_decode, is_decode_mode=True)
            t = ttnn.to_memory_config(t, memory_config=cfg.decode_act_mc or ttnn.DRAM_MEMORY_CONFIG)
            if pad_h:
                t = ttnn.slice(t, [0, 0, 0, 0], [1, batch, heads, rope_dim])
            t = ttnn.permute(t, (0, 2, 1, 3))
            return t

    # ---- Input LayerNorm ----
    if use_signpost:
        signpost(f"L{layer_idx}_attn-start")
    residual = x_embed_tok
    t0 = time.perf_counter() if profile is not None else 0.0
    if cfg.sharded_decode_norm:
        x = sharded_decode_norm(
            w.input_layernorm,
            x_embed_tok,
            device=device,
            width=int(hparams.hidden_size),
            downstream_mc=cfg.decode_act_mc or ttnn.DRAM_MEMORY_CONFIG,
        )
    else:
        x = w.input_layernorm(x_embed_tok, mode="decode")
    _profile_add(profile, "norm_s", time.perf_counter() - t0 if profile is not None else 0.0)

    # ---- KV Cache Update ----
    if use_signpost:
        signpost(f"L{layer_idx}_kv_update-start")
    fused_kv_branch = getattr(w, "fused_kv_branch", None)
    q_a_from_kv = kv_cache_update(
        device=device,
        x=x,
        w=w,
        hparams=hparams,
        cfg=cfg,
        batch=batch,
        cos_batch=cos_batch,
        sin_batch=sin_batch,
        trans_matrix=trans_matrix,
        kvpe_cache=kvpe_cache,
        page_table_tt=page_table_tt,
        tt_positions=tt_positions,
        positions_main_tt=positions_main_tt,
        positions_draft_tt=positions_draft_tt,
        use_decode_rope=use_decode_rope,
        rope_decode_fn=rope_decode_fn,
        shard_kvpe_fn=_shard_kvpe_update_tensor,
        fused_kv_branch_fn=_fused_kv_branch_forward if fused_kv_branch is not None and batch == 1 else None,
        profile=profile,
    )
    if use_signpost:
        signpost(f"L{layer_idx}_kv_update-end")

    # ---- Q Projection ----
    if use_signpost:
        signpost(f"L{layer_idx}_q_proj-start")
    q_kvpe = q_projection(
        device=device,
        x=x,
        w=w,
        hparams=hparams,
        cfg=cfg,
        batch=batch,
        cos_batch=cos_batch,
        sin_batch=sin_batch,
        trans_matrix=trans_matrix,
        q_a_from_kv=q_a_from_kv,
        use_decode_rope=use_decode_rope,
        rope_decode_fn=rope_decode_fn,
        profile=profile,
    )
    ttnn.deallocate(x, force=False)
    if use_signpost:
        signpost(f"L{layer_idx}_q_proj-end")

    # Clean up decode RoPE inputs if we allocated them
    if use_decode_rope and owns_decode_rope_inputs:
        for t in (cos_decode, sin_decode, trans_decode):
            if t is not None:
                ttnn.deallocate(t, force=False)

    # ---- FlashMLA + Output Projection ----
    if use_signpost:
        signpost(f"L{layer_idx}_flash_mla-start")
    attn_out = flash_mla_and_output(
        device=device,
        q_kvpe=q_kvpe,
        w=w,
        hparams=hparams,
        cfg=cfg,
        batch=batch,
        kvpe_cache=kvpe_cache,
        page_table_tt=page_table_tt,
        tt_positions=tt_positions,
        profile=profile,
    )
    if use_signpost:
        signpost(f"L{layer_idx}_flash_mla-end")

    x_attn_out = ttnn.add(residual, attn_out, memory_config=cfg.decode_act_mc or ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(attn_out, force=False)
    if use_signpost:
        signpost(f"L{layer_idx}_attn-end")

    if cfg.disable_mlp:
        _profile_add(profile, "total_s", time.perf_counter() - t_layer0 if profile is not None else 0.0)
        return x_attn_out

    # ---- MLP ----
    if use_signpost:
        signpost(f"L{layer_idx}_moe-start")
    residual = x_attn_out
    t0 = time.perf_counter() if profile is not None else 0.0
    if cfg.sharded_decode_norm:
        x = sharded_decode_norm(
            w.post_attention_layernorm,
            x_attn_out,
            device=device,
            width=int(hparams.hidden_size),
            downstream_mc=cfg.decode_act_mc or ttnn.DRAM_MEMORY_CONFIG,
        )
    else:
        x = w.post_attention_layernorm(x_attn_out, mode="decode")
    _profile_add(profile, "mlp_norm_s", time.perf_counter() - t0 if profile is not None else 0.0)

    use_moe = moe_runtime is not None and getattr(w, "moe", None) is not None
    if use_moe:
        mlp_out = moe_mlp_forward(
            x,
            w,
            device=device,
            cfg=cfg,
            hparams=hparams,
            moe_runtime=moe_runtime,
            profile=profile,
            layer_idx=layer_idx,
            use_signpost=use_signpost,
        )
    else:
        mlp_out = dense_mlp_forward(x, w, device=device, cfg=cfg, profile=profile)

    x_out = ttnn.add(residual, mlp_out, memory_config=cfg.decode_act_mc or ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(mlp_out, force=False)
    ttnn.deallocate(residual, force=False)
    if use_signpost:
        signpost(f"L{layer_idx}_moe-end")
    _profile_add(profile, "total_s", time.perf_counter() - t_layer0 if profile is not None else 0.0)
    return x_out


@torch.no_grad()
def run_decoder_layer_prefill_update_cache_tt(
    *,
    device: Any,
    x_embed: ttnn.Tensor,  # [1, 1, S, hidden] tiled (S = B*S_pad when batched)
    page_table_tt: ttnn.Tensor,  # [B, max_num_blocks_per_req] int32 row-major on device
    kvpe_cache: ttnn.Tensor,  # [num_blocks, 1, block_size, kvpe_dim] on device
    cos_matrix: ttnn.Tensor,  # [1, 1, S_pad, rope_dim] bf16
    sin_matrix: ttnn.Tensor,  # [1, 1, S_pad, rope_dim] bf16
    trans_matrix: ttnn.Tensor,
    w: Any,
    hparams: Glm4MoeLiteHParams,
    prompt_len: int,  # number of valid tokens to write into KV cache (<= S_pad)
    batch: int = 1,  # number of requests batched in the S dimension
    prompt_lens: list[int] | None = None,  # per-request prompt lengths (required when batch > 1)
    moe_runtime: Any | None = None,
    profile: dict[str, float] | None = None,
    chunk_page_table: ttnn.Tensor | None = None,  # page table slice for this chunk (chunked prefill)
    chunk_start_idx: int | None = None,  # absolute token position of this chunk's start
) -> ttnn.Tensor:
    """Run prefill for a single decoder layer and fill its paged KVPE cache.

    This is the sequence-length (S>1) counterpart to
    `run_decoder_layer_decode_one_step_update_cache_tt`.

    When ``batch > 1``, the token dimension of ``x_embed`` is ``B * S_pad``
    (all requests concatenated along dim-2).  Token-wise ops (norms, linears,
    MoE) operate on the flat ``[1,1,B*S_pad,hidden]`` shape.  For RoPE and
    FlashMLA the tensors are reshaped to ``[B,...,S_pad,...]`` so that
    positional encoding and causal masking are per-request.

    For chunked prefill (``chunk_start_idx is not None``), the KV cache is
    filled using ``chunk_page_table`` and attention reads from the full cache
    via ``page_table_tt`` using ``chunked_flash_mla_prefill``.
    """
    if len(x_embed.shape) != 4:
        raise ValueError(f"expected x_embed rank 4, got shape={tuple(x_embed.shape)}")
    total_seq = int(x_embed.shape[2])
    if total_seq <= 0:
        raise ValueError("prefill seq_len must be > 0")

    batch = int(batch)
    if batch < 1:
        raise ValueError(f"batch must be >= 1, got {batch}")

    if batch > 1:
        if total_seq % batch != 0:
            raise ValueError(f"x_embed dim-2 ({total_seq}) must be divisible by batch ({batch})")
        seq_len = total_seq // batch
        if prompt_lens is None:
            raise ValueError("prompt_lens required when batch > 1")
        if len(prompt_lens) != batch:
            raise ValueError(f"prompt_lens length {len(prompt_lens)} != batch {batch}")
    else:
        seq_len = total_seq
        if prompt_lens is None:
            prompt_lens = [int(prompt_len)]

    prompt_len = int(prompt_len)
    if prompt_len <= 0 or prompt_len > seq_len:
        raise ValueError(f"invalid prompt_len={prompt_len} for seq_len={seq_len}")

    kvpe_dim = int(hparams.kv_lora_rank + hparams.qk_rope_head_dim)
    num_heads = int(hparams.num_attention_heads)
    t_layer0 = time.perf_counter() if profile is not None else 0.0

    # Optional precision knob for MLP/router matmuls during bring-up. Controlled
    # by the same env var as decode.
    mlp_compute_kernel_config = None
    if os.environ.get("GLM4_MOE_LITE_MOE_FP32_ACC", "").strip() == "1":
        mlp_compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    fuse_shared_gate_up = os.environ.get("GLM4_MOE_LITE_FUSE_SHARED_GATE_UP", "1").strip() != "0"
    prefill_sharded_qkv = os.environ.get("GLM4_MOE_LITE_PREFILL_SHARDED_QKV", "1").strip() != "0"

    def _mlp_linear(
        a: ttnn.Tensor,
        b: ttnn.Tensor,
        memory_config: ttnn.MemoryConfig | None = None,
        skip_gather: bool = False,
        sharded_in0: bool = False,
    ) -> ttnn.Tensor:
        m_total = 1
        for i in range(len(a.shape) - 1):
            m_total *= int(a.shape[i])
        b_batch = 1
        for i in range(len(b.shape) - 2):
            b_batch *= int(b.shape[i])
        if b_batch == 1 and m_total > ttnn.TILE_SIZE:
            return prefill_linear_ws_out(
                a,
                b,
                device=device,
                compute_kernel_config=mlp_compute_kernel_config,
                memory_config=memory_config,
                return_sharded=skip_gather,
                sharded_in0=sharded_in0,
            )
        if b_batch > 1 and m_total > ttnn.TILE_SIZE:
            return prefill_per_head_linear(
                a,
                b,
                device=device,
                compute_kernel_config=mlp_compute_kernel_config,
                memory_config=memory_config,
            )
        if mlp_compute_kernel_config is None:
            return ttnn.linear(a, b, memory_config=memory_config)
        return ttnn.linear(a, b, compute_kernel_config=mlp_compute_kernel_config, memory_config=memory_config)

    _interleaved_cache: dict[int, ttnn.Tensor] = {}

    def _ensure_interleaved(weight: ttnn.Tensor) -> ttnn.Tensor:
        """If weight is DRAM WIDTH_SHARDED, convert to DRAM interleaved via host round-trip.

        DRAM-sharded weights are optimized for decode (m=32) matmuls but incompatible
        with regular ttnn.linear used in prefill.  The host round-trip avoids the
        sharded_to_interleaved kernel which has the same DRAM→TENSIX coordinate bug.
        Returns the original tensor unchanged if already interleaved.
        Caches results keyed by tensor id to avoid repeated host round-trips.
        """
        mc = weight.memory_config()
        if mc.memory_layout != ttnn.TensorMemoryLayout.WIDTH_SHARDED:
            return weight
        tid = id(weight)
        cached = _interleaved_cache.get(tid)
        if cached is not None:
            return cached
        host_weight = ttnn.from_device(weight)
        interleaved = host_weight.to(device, ttnn.DRAM_MEMORY_CONFIG)
        _interleaved_cache[tid] = interleaved
        return interleaved

    tp_axis = _tp_cluster_axis(device)
    tp_enabled = tp_axis is not None and os.environ.get("GLM4_MOE_LITE_TP", "").strip() == "1"
    mesh_rows, mesh_cols = _mesh_shape(device)
    tp_size = int((mesh_rows, mesh_cols)[tp_axis]) if tp_axis is not None else 1
    attn_dp = _env_bool("GLM4_MOE_LITE_ATTN_DP")
    ccl_num_links = int(os.environ.get("GLM4_MOE_LITE_CCL_NUM_LINKS", "1").strip() or "1")
    ccl_topology_str = os.environ.get("GLM4_MOE_LITE_CCL_TOPOLOGY", "linear").strip().lower()
    ccl_topology = ttnn.Topology.Ring if ccl_topology_str == "ring" else ttnn.Topology.Linear
    fuse_mlp_moe_reduce = os.environ.get("GLM4_MOE_LITE_FUSE_MLP_MOE_REDUCE", "").strip() == "1"
    _skip_shared_reduce = fuse_mlp_moe_reduce and tp_enabled

    def _tp_row_parallel_linear_from_replicated(a: ttnn.Tensor, b: ttnn.Tensor) -> ttnn.Tensor:
        a_tp = ttnn.mesh_partition(a, dim=3, cluster_axis=tp_axis)
        out = _mlp_linear(a_tp, b)
        ttnn.deallocate(a_tp, force=False)
        out_reduced = ttnn.all_reduce(
            out,
            num_links=ccl_num_links,
            topology=ccl_topology,
            cluster_axis=tp_axis,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(out, force=False)
        return out_reduced

    x_embed, _ = _to_l1_if_needed(x_embed)
    residual = x_embed
    t0 = time.perf_counter() if profile is not None else 0.0
    x = w.input_layernorm(x_embed, mode="prefill")  # [1,1,S,hidden]
    x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)  # x reused for q and kv projections
    _profile_add(profile, "norm_s", time.perf_counter() - t0 if profile is not None else 0.0)

    # ---- Q path ----
    t0 = time.perf_counter() if profile is not None else 0.0
    q_a = None
    kv = None
    # Token-wise linears operate on [1,1,B*S_pad,...] (total_seq tokens).
    w_q_kv_a = getattr(w, "w_q_kv_a", None)
    use_sharded_q_a = prefill_sharded_qkv and not (tp_enabled and not attn_dp)
    q_lora = int(hparams.q_lora_rank)

    def _run_sharded_q_a_norm(q_a_in: ttnn.Tensor, *, from_fused_slice: bool) -> ttnn.Tensor:
        qa_norm_cfg = prefill_width_sharded_norm_config(device, total_seq, q_lora)
        if from_fused_slice:
            # Slice of fused [T, q_lora+kvpe] may use a sub-grid of the parent shard layout.
            q_a_in = ttnn.to_memory_config(q_a_in, qa_norm_cfg["sharded_output_config"])
        elif q_a_in.memory_config().memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED:
            # Standalone w_q_a matmul: run norm on the matmul's native shard spec (no gather).
            tensor_norm_cfg = prefill_norm_config_from_width_sharded_tensor(q_a_in)
            qa_norm_cfg["sharded_program_config"] = tensor_norm_cfg["sharded_program_config"]
            qa_norm_cfg["sharded_output_config"] = tensor_norm_cfg["sharded_output_config"]
        else:
            q_a_in = ttnn.to_memory_config(q_a_in, qa_norm_cfg["sharded_output_config"])
        return w.q_a_layernorm(
            q_a_in,
            mode="prefill",
            in_sharded=True,
            out_sharded=True,
            norm_config=qa_norm_cfg,
        )

    if w_q_kv_a is not None:
        if tp_enabled and not attn_dp:
            qkv = _tp_row_parallel_linear_from_replicated(x, w_q_kv_a)  # [1,1,T,q_lora_rank+kvpe_dim]
        else:
            qkv = _mlp_linear(x, w_q_kv_a, skip_gather=use_sharded_q_a)  # [1,1,T,q_lora_rank+kvpe_dim]
        q_a = ttnn.slice(qkv, [0, 0, 0, 0], [1, 1, total_seq, q_lora])
        kv = ttnn.slice(qkv, [0, 0, 0, q_lora], [1, 1, total_seq, q_lora + kvpe_dim])
        ttnn.deallocate(qkv, force=False)
        if use_sharded_q_a:
            q_a = _run_sharded_q_a_norm(q_a, from_fused_slice=True)
            kv = ttnn.to_memory_config(kv, ttnn.DRAM_MEMORY_CONFIG)
    else:
        if tp_enabled and not attn_dp:
            q_a = _tp_row_parallel_linear_from_replicated(x, w.w_q_a)  # [1,1,T,q_lora_rank]
        else:
            q_a = _mlp_linear(
                x,
                w.w_q_a,
                skip_gather=use_sharded_q_a,
                memory_config=ttnn.L1_MEMORY_CONFIG if not use_sharded_q_a else None,
            )
    if not use_sharded_q_a:
        q_a = w.q_a_layernorm(q_a, mode="prefill")
    elif w_q_kv_a is None:
        q_a = _run_sharded_q_a_norm(q_a, from_fused_slice=False)
    qk_head_dim = int(hparams.qk_head_dim)
    if tp_enabled and not attn_dp:
        q = _tp_row_parallel_linear_from_replicated(q_a, w.w_q_b)  # [1,1,T,H*qk_head_dim]
    elif use_sharded_q_a:
        # Fast 80-core w_q_b beats sharded-in0 on 8-core grid (~15 μs vs ~57 μs device).
        # Gather q_a in (~1 μs), matmul, then gather output once into nlp in0 MC.
        q = _mlp_linear(
            q_a,
            w.w_q_b,
            skip_gather=False,
            memory_config=prefill_q_nlp_input_memory_config(
                seq_tokens=total_seq,
                fused_q_dim=num_heads * qk_head_dim,
            ),
        )
    else:
        q = _mlp_linear(q_a, w.w_q_b)  # [1,1,T,H*qk_head_dim]
    ttnn.deallocate(q_a, force=False)

    q_heads_mc = prefill_q_heads_memory_config(
        num_heads=num_heads,
        seq_tokens=total_seq,
        head_dim=qk_head_dim,
    )
    q = q_split_heads(
        q,
        num_heads=num_heads,
        head_dim=qk_head_dim,
        total_seq=total_seq,
        batch=batch,
        seq_len=seq_len,
        device=device,
        memory_config=q_heads_mc,
    )  # [B,H,S_pad,qk_head_dim]
    # Full Q [1,H,S,576] exceeds 1 MiB L1 budget; slices fit (q_nope ~960 KiB, q_rope ~320 KiB).
    q_nope_mc = prefill_q_slice_memory_config(
        num_heads=num_heads,
        seq_tokens=seq_len,
        slice_dim=int(hparams.qk_nope_head_dim),
    )
    q_rope_mc = prefill_q_slice_memory_config(
        num_heads=num_heads,
        seq_tokens=seq_len,
        slice_dim=int(hparams.qk_rope_head_dim),
    )
    q_nope = ttnn.slice(
        q,
        [0, 0, 0, 0],
        [batch, num_heads, seq_len, int(hparams.qk_nope_head_dim)],
        memory_config=q_nope_mc,
    )
    q_rope = ttnn.slice(
        q,
        [0, 0, 0, int(hparams.qk_nope_head_dim)],
        [batch, num_heads, seq_len, qk_head_dim],
        memory_config=q_rope_mc,
    )
    ttnn.deallocate(q, force=False)

    # Project q_nope into KV latent space (per-head).
    # TTNN non-bcast matmul requires dim-0==1 for 4D×2D. When batch>1, reshape
    # [B,H,S,D] → [1,B*H,S,D] so dim-0 is 1, then reshape back after.
    # w_kv_b1 stays row-parallel under TP regardless of ATTN_DP.
    use_tp_kv_b1 = tp_enabled
    if use_tp_kv_b1:
        qk_nope = int(hparams.qk_nope_head_dim)
        qk_nope_per_shard = qk_nope // max(1, int(tp_size))
        if qk_nope % max(1, int(tp_size)) != 0 or qk_nope_per_shard % int(ttnn.TILE_SIZE) != 0:
            use_tp_kv_b1 = False
    # w_kv_b1 is a small per-head matmul (qk_nope_head_dim → kv_lora_rank).
    # TTNN non-bcast matmul doesn't broadcast across dim-0 when B>1.
    # Loop over batch: each [1,H,S,D] matmul matches the serial-path shape exactly.
    # This is fast because the matmul is small — the big batching wins come from
    # the token-wise linears and MoE which stay on flat [1,1,B*S,hidden].
    # Keep kv_b1 output in L1: prefill_per_head_linear writes L1 then gathers to
    # memory_config; DRAM default causes an extra CopyDeviceOperation (~1.7 ms).
    kv_b1_out_mc = q_nope_mc if _is_l1_interleaved(q_nope_mc) else ttnn.L1_MEMORY_CONFIG
    if batch > 1:
        if use_tp_kv_b1:
            kv_b1_fn = _tp_row_parallel_linear_from_replicated
        else:
            kv_b1_fn = lambda a, b: _mlp_linear(a, b, memory_config=kv_b1_out_mc)
        q_nope_parts = []
        for bi in range(batch):
            q_bi = ttnn.slice(q_nope, [bi, 0, 0, 0], [bi + 1, num_heads, seq_len, int(hparams.qk_nope_head_dim)])
            q_bi = kv_b1_fn(q_bi, w.w_kv_b1)
            q_nope_parts.append(q_bi)
        ttnn.deallocate(q_nope, force=False)
        q_nope = ttnn.concat(q_nope_parts, dim=0)  # [B,H,S,kv_lora_rank]
        for p in q_nope_parts:
            ttnn.deallocate(p, force=False)
    else:
        if use_tp_kv_b1:
            q_nope = _tp_row_parallel_linear_from_replicated(q_nope, w.w_kv_b1)
        else:
            q_nope = _mlp_linear(q_nope, w.w_kv_b1, memory_config=kv_b1_out_mc)
    _profile_add(profile, "q_path_s", time.perf_counter() - t0 if profile is not None else 0.0)

    # ---- KVPE for the prompt -> fill cache ----
    t0 = time.perf_counter() if profile is not None else 0.0
    if kv is None:
        if tp_enabled and not attn_dp:
            kv = _tp_row_parallel_linear_from_replicated(x, w.w_kv_a)  # [1,1,B*S_pad,kvpe_dim]
        else:
            kv = _mlp_linear(x, w.w_kv_a, memory_config=ttnn.L1_MEMORY_CONFIG)  # [1,1,B*S_pad,kvpe_dim]
    ttnn.deallocate(x, force=False)

    # Reshape KV from flat [1,1,B*S_pad,...] to [B,1,S_pad,...] for per-request
    # RoPE and KV cache fill.
    kv = ttnn.reshape(kv, (batch, 1, seq_len, kvpe_dim))

    kv_l1_mc = ttnn.L1_MEMORY_CONFIG
    kv_nope = ttnn.slice(
        kv,
        [0, 0, 0, 0],
        [batch, 1, seq_len, int(hparams.kv_lora_rank)],
        memory_config=kv_l1_mc,
    )
    kv_rope = ttnn.slice(
        kv,
        [0, 0, 0, int(hparams.kv_lora_rank)],
        [batch, 1, seq_len, kvpe_dim],
        memory_config=kv_l1_mc,
    )
    ttnn.deallocate(kv, force=False)

    kv_nope_in, _kv_nope_copied = _to_l1_if_needed(kv_nope)
    if _kv_nope_copied:
        ttnn.deallocate(kv_nope, force=False)
    kv_nope = w.kv_a_layernorm(kv_nope_in, mode="prefill")
    ttnn.deallocate(kv_nope_in, force=False)

    # RoPE ops require BF16.
    if q_rope.dtype != ttnn.bfloat16:
        q_rope = ttnn.typecast(q_rope, dtype=ttnn.bfloat16)
    if kv_rope.dtype != ttnn.bfloat16:
        kv_rope = ttnn.typecast(kv_rope, dtype=ttnn.bfloat16)

    # RoPE: cos/sin are [1,1,S_pad,rope_dim] and broadcast across batch.
    q_rope = ttnn.experimental.rotary_embedding_llama(
        q_rope,
        cos_matrix,
        sin_matrix,
        trans_matrix,
        is_decode_mode=False,
    )  # [B,H,S_pad,rope_dim]
    kv_rope = ttnn.experimental.rotary_embedding_llama(
        kv_rope,
        cos_matrix,
        sin_matrix,
        trans_matrix,
        is_decode_mode=False,
    )  # [B,1,S_pad,rope_dim]

    q_kvpe = ttnn.concat([q_nope, q_rope], dim=-1)  # [B,H,S_pad,kvpe_dim]
    ttnn.deallocate(q_nope, force=False)
    ttnn.deallocate(q_rope, force=False)

    # kvpe is small (~144 KiB at S=128); keep L1 interleaved for flash_mla_prefill K reads.
    kvpe = ttnn.concat([kv_nope, kv_rope], dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(kv_nope, force=False)
    ttnn.deallocate(kv_rope, force=False)

    # Fill KV cache for each request. paged_fill_cache operates on one batch
    # item at a time (no batched API), so we loop over the batch dimension.
    # For chunked prefill, use chunk_page_table so data lands at the correct
    # physical pages for this chunk.
    fill_page_table = chunk_page_table if chunk_page_table is not None else page_table_tt
    for bi in range(batch):
        plen = int(prompt_lens[bi])
        kvpe_bi = ttnn.slice(kvpe, [bi, 0, 0, 0], [bi + 1, 1, plen, kvpe_dim])

        if kvpe_bi.dtype != kvpe_cache.dtype:
            kvpe_bi_cast = ttnn.typecast(kvpe_bi, dtype=kvpe_cache.dtype)
        else:
            kvpe_bi_cast = kvpe_bi

        ttnn.experimental.paged_fill_cache(kvpe_cache, kvpe_bi_cast, page_table=fill_page_table, batch_idx=bi)

        if kvpe_bi_cast is not kvpe_bi:
            ttnn.deallocate(kvpe_bi_cast, force=False)
        ttnn.deallocate(kvpe_bi, force=False)

    _profile_add(profile, "kv_cache_fill_s", time.perf_counter() - t0 if profile is not None else 0.0)

    # ---- FlashMLA prefill ----
    # MLA attention scores are computed from dot(q_kvpe, kvpe) with dot-product
    # dimension `kvpe_dim`.
    #
    # Correctness: match HF DeepseekV3Attention default scaling of 1/sqrt(qk_head_dim).
    # Keep `GLM4_MOE_LITE_MLA_SCALE_MODE=kvpe` as an escape hatch for experiments.
    scale_mode = os.environ.get("GLM4_MOE_LITE_MLA_SCALE_MODE", "qk").strip().lower()
    if scale_mode == "kvpe":
        scale = float(int(kvpe_dim) ** -0.5)
    else:
        scale = float(int(hparams.qk_head_dim) ** -0.5)
    sdpa_program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=32,  # heads padded up to 32
        k_chunk_size=128,
        exp_approx_mode=False,
    )
    mla_fidelity = _parse_math_fidelity(
        os.environ.get("GLM4_MOE_LITE_MLA_FIDELITY", ""),
        default=ttnn.MathFidelity.HiFi4,
    )
    mla_approx = os.environ.get("GLM4_MOE_LITE_MLA_APPROX", "0").strip() != "0"
    mla_fp32_acc_req = os.environ.get("GLM4_MOE_LITE_MLA_FP32_ACC", "").strip() == "1"
    mla_fp32_acc = mla_fp32_acc_req
    if mla_fp32_acc_req and os.environ.get("GLM4_MOE_LITE_UNSAFE_ALLOW_FP32_MLA", "").strip() != "1":
        logger.warning(
            "GLM4_MOE_LITE_MLA_FP32_ACC=1 is currently unsafe for FlashMLA prefill/bring-up. "
            "Forcing fp32_dest_acc_en=0. Set GLM4_MOE_LITE_UNSAFE_ALLOW_FP32_MLA=1 to override."
        )
        mla_fp32_acc = False
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=mla_fidelity,
        math_approx_mode=mla_approx,
        fp32_dest_acc_en=mla_fp32_acc,
        packer_l1_acc=False,
    )

    # FlashMLA prefill with batch dimension: q_kvpe [B,H,S_pad,kvpe_dim],
    # kvpe [B,1,S_pad,kvpe_dim].  is_causal=True applies per-batch causal mask.
    # For chunked prefill, use chunked_flash_mla_prefill which reads K/V from
    # the paged cache (including all previously filled chunks) via page_table_tt.
    t0 = time.perf_counter() if profile is not None else 0.0
    if chunk_start_idx is not None:
        ttnn.deallocate(kvpe, force=False)
        attn_latent = ttnn.transformer.chunked_flash_mla_prefill(
            q_kvpe,
            kvpe_cache,
            int(hparams.kv_lora_rank),
            page_table_tt,
            chunk_start_idx=chunk_start_idx,
            scale=scale,
            program_config=sdpa_program_config,
            compute_kernel_config=compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [B,H_padded,S_pad,kv_lora_rank]
        ttnn.deallocate(q_kvpe, force=False)
    else:
        attn_latent = ttnn.transformer.flash_mla_prefill(
            q_kvpe,
            kvpe,
            head_dim_v=int(hparams.kv_lora_rank),
            scale=scale,
            program_config=sdpa_program_config,
            compute_kernel_config=compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            attn_mask=None,
            is_causal=True,
        )  # [B,H_padded,S_pad,kv_lora_rank]
        ttnn.deallocate(q_kvpe, force=False)
        ttnn.deallocate(kvpe, force=False)
    _profile_add(profile, "flash_mla_prefill_s", time.perf_counter() - t0 if profile is not None else 0.0)

    # flash_mla_prefill pads heads up to q_chunk_size. Slice back to num_heads.
    attn_latent = ttnn.slice(attn_latent, [0, 0, 0, 0], [batch, num_heads, seq_len, int(hparams.kv_lora_rank)])

    t0 = time.perf_counter() if profile is not None else 0.0
    use_nlp_concat = os.environ.get("GLM4_MOE_LITE_NLP_CONCAT_HEADS", "1").strip() != "0"
    head_parallel_kvb2 = (
        os.environ.get("GLM4_MOE_LITE_HEAD_PARALLEL_KVB2", "").strip() == "1" and tp_enabled and tp_size > 1
    )

    if head_parallel_kvb2:
        # Shard heads across TP; fuse kv_b2 + w_o partials into one all_reduce (matches decode).
        heads_per_dev = num_heads // tp_size
        flat_dim = heads_per_dev * int(hparams.v_head_dim)

        kv_b2_out_mc = ttnn.L1_MEMORY_CONFIG

        def _kvb2_head_parallel_prefill(a_latent: ttnn.Tensor) -> ttnn.Tensor:
            # a_latent: [1, H, S, kv_lora]
            # Land the partition output directly in L1 so the kv_b2 matmul's input
            # staging copy (CopyDeviceOperation) is avoided (~640 KiB fits L1).
            a_part = ttnn.mesh_partition(a_latent, dim=1, cluster_axis=tp_axis, memory_config=ttnn.L1_MEMORY_CONFIG)
            v_local = _mlp_linear(a_part, w.w_kv_b2, memory_config=kv_b2_out_mc)
            ttnn.deallocate(a_part, force=False)
            if use_nlp_concat:
                v_local = ttnn.experimental.nlp_concat_heads(v_local)
                return ttnn.reshape(v_local, (1, 1, seq_len, flat_dim))
            v_local = ttnn.permute(v_local, (0, 2, 1, 3))
            return ttnn.reshape(v_local, (1, 1, seq_len, flat_dim))

        if batch > 1:
            v_parts = []
            for bi in range(batch):
                a_bi = ttnn.slice(attn_latent, [bi, 0, 0, 0], [bi + 1, num_heads, seq_len, int(hparams.kv_lora_rank)])
                v_parts.append(_kvb2_head_parallel_prefill(a_bi))
            ttnn.deallocate(attn_latent, force=False)
            v = ttnn.concat(v_parts, dim=0)  # [B,1,S,flat_dim]
            for p in v_parts:
                ttnn.deallocate(p, force=False)
            v = ttnn.permute(v, (0, 2, 1, 3))
            v = ttnn.reshape(v, (1, 1, total_seq, flat_dim))
        else:
            v = _kvb2_head_parallel_prefill(attn_latent)
            ttnn.deallocate(attn_latent, force=False)

        attn_out_partial = _mlp_linear(v, w.w_o, skip_gather=True)
        ttnn.deallocate(v, force=False)
        attn_out_reduced = ttnn.all_reduce(
            attn_out_partial,
            num_links=ccl_num_links,
            topology=ccl_topology,
            cluster_axis=tp_axis,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(attn_out_partial, force=False)
        attn_out = attn_out_reduced
    else:
        # Same per-batch loop as w_kv_b1 for w_kv_b2 (small per-head matmul).
        if batch > 1:
            kv_b2_fn = _tp_row_parallel_linear_from_replicated if (tp_enabled and not attn_dp) else _mlp_linear
            v_parts = []
            for bi in range(batch):
                a_bi = ttnn.slice(attn_latent, [bi, 0, 0, 0], [bi + 1, num_heads, seq_len, int(hparams.kv_lora_rank)])
                v_bi = kv_b2_fn(a_bi, w.w_kv_b2)
                ttnn.deallocate(a_bi, force=False)
                v_parts.append(v_bi)
            ttnn.deallocate(attn_latent, force=False)
            v = ttnn.concat(v_parts, dim=0)  # [B,H,S,v_head_dim]
            for p in v_parts:
                ttnn.deallocate(p, force=False)
        else:
            if tp_enabled and not attn_dp:
                v = _tp_row_parallel_linear_from_replicated(attn_latent, w.w_kv_b2)
            else:
                v = _mlp_linear(attn_latent, w.w_kv_b2, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(attn_latent, force=False)

        # Flatten back from [B,H,S_pad,v_head_dim] to [1,1,B*S_pad,H*v_head_dim]
        # for the output projection (token-wise linear).
        if use_nlp_concat:
            v = ttnn.experimental.nlp_concat_heads(v)  # [1,B,S_pad,H*v_head_dim]
            v = ttnn.reshape(v, (1, 1, total_seq, int(num_heads * hparams.v_head_dim)))  # [1,1,B*S_pad,H*v_head_dim]
        else:
            v = ttnn.permute(v, (0, 2, 1, 3))  # [B,S_pad,H,v_head_dim]
            v = ttnn.reshape(v, (1, 1, total_seq, int(num_heads * hparams.v_head_dim)))  # [1,1,B*S_pad,H*v_head_dim]
        if tp_enabled:
            attn_out = _tp_row_parallel_linear_from_replicated(
                v, w.w_o
            )  # [1,1,B*S_pad,hidden] — w_o stays row-parallel even with ATTN_DP
        else:
            attn_out = _mlp_linear(v, w.w_o)  # [1,1,B*S_pad,hidden]
        ttnn.deallocate(v, force=False)

    x_attn_out = residual + attn_out
    ttnn.deallocate(attn_out, force=False)
    _profile_add(profile, "attn_out_s", time.perf_counter() - t0 if profile is not None else 0.0)

    # ---- MLP (dense or shared-expert-as-dense) ----
    x_attn_out = ttnn.to_memory_config(x_attn_out, ttnn.L1_MEMORY_CONFIG)
    residual = x_attn_out
    t0 = time.perf_counter() if profile is not None else 0.0
    x = w.post_attention_layernorm(x_attn_out, mode="prefill")
    x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)  # x reused for gate and up projections
    _profile_add(profile, "mlp_norm_s", time.perf_counter() - t0 if profile is not None else 0.0)

    use_moe = moe_runtime is not None and getattr(w, "moe", None) is not None
    if not use_moe:
        t0 = time.perf_counter() if profile is not None else 0.0
        # DRAM-sharded weights are incompatible with regular ttnn.linear (prefill).
        # Convert to interleaved via host round-trip if needed.
        _down_w = _ensure_interleaved(w.w_mlp_down)
        if fuse_shared_gate_up and w.w_mlp_gate_up is not None:
            _gate_up_w = _ensure_interleaved(w.w_mlp_gate_up)
            gate_up = _mlp_linear(x, _gate_up_w, skip_gather=True)
            _batch = int(gate_up.shape[2])
            _inter = int(gate_up.shape[3]) // 2
            gate = ttnn.slice(gate_up, [0, 0, 0, 0], [1, 1, _batch, _inter])
            up = ttnn.slice(gate_up, [0, 0, 0, _inter], [1, 1, _batch, _inter * 2])
            ttnn.deallocate(gate_up, force=False)
        else:
            _gate_w = _ensure_interleaved(w.w_mlp_gate)
            _up_w = _ensure_interleaved(w.w_mlp_up)
            gate = _mlp_linear(x, _gate_w, skip_gather=True)
            up = _mlp_linear(x, _up_w, skip_gather=True)
        ttnn.deallocate(x, force=False)

        # gate and up are WIDTH_SHARDED L1; ttnn.mul with fused silu works on
        # same-shard-spec inputs, avoiding two DRAM gathers.  x_ff stays in
        # WIDTH_SHARDED L1; _prefill_linear_ws_out's _to_l1_if_needed gathers
        # it L1→L1 (not DRAM→L1) before the down matmul.
        x_ff = ttnn.mul(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        ttnn.deallocate(gate, force=False)
        ttnn.deallocate(up, force=False)

        mlp_out = _mlp_linear(x_ff, _down_w)
        ttnn.deallocate(x_ff, force=False)
        if tp_enabled:
            mlp_out_reduced = ttnn.all_reduce(
                mlp_out,
                num_links=ccl_num_links,
                topology=ccl_topology,
                cluster_axis=tp_axis,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            ttnn.deallocate(mlp_out, force=False)
            mlp_out = mlp_out_reduced
        _profile_add(profile, "mlp_dense_s", time.perf_counter() - t0 if profile is not None else 0.0)
    else:
        # MoE path:
        # - shared_experts MLP (dense) + routed experts MLP
        from models.experimental.glm4_moe_lite.tt.moe_tt import (
            moe_dense_experts_forward_prefill_tt,
            moe_packed_experts_forward_prefill_tt,
            moe_sparse_experts_forward_tt,
            moe_topk_cpu_reference,
            moe_topk_tt,
        )

        dense_prefill = _env_bool("GLM4_MOE_LITE_MOE_DENSE_PREFILL", default=False)
        packed_prefill = _env_bool("GLM4_MOE_LITE_MOE_PACKED_PREFILL", default=False)
        skip_defensive_clones = _env_bool("GLM4_MOE_LITE_SKIP_DEFENSIVE_CLONES")

        # Pad tokens to the minimum legal sparse multiple for this mesh.
        # Dense/packed prefill paths use ttnn.linear (no block alignment needed), so skip padding.
        tokens = int(x.shape[2])
        _PACKED_PREFILL_MIN_TOKENS = 33
        use_packed_prefill_here = packed_prefill and tokens >= _PACKED_PREFILL_MIN_TOKENS
        pad_tokens = 0
        if not dense_prefill and not use_packed_prefill_here:
            sparse_multiple = _moe_sparse_tokens_multiple(device=device, moe_runtime=moe_runtime)
            pad_tokens = (-tokens) % sparse_multiple
            if pad_tokens:
                # IMPORTANT: `ttnn.pad` can return a view that aliases the input buffer.
                # Materialize with `ttnn.clone` before deallocating the original tensor.
                x_padded_view = ttnn.pad(x, [(0, 0), (0, 0), (0, pad_tokens), (0, 0)], 0.0)
                if skip_defensive_clones:
                    x_padded = x_padded_view
                else:
                    x_padded = ttnn.clone(x_padded_view, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                # NOTE: x_padded_view may alias `x`; do not deallocate it separately.
                ttnn.deallocate(x, force=False)
                x = x_padded

        # Shared expert (dense MLP).
        # DRAM-sharded weights are incompatible with regular ttnn.linear (prefill).
        _down_w = _ensure_interleaved(w.w_mlp_down)
        t0 = time.perf_counter() if profile is not None else 0.0
        if fuse_shared_gate_up and w.w_mlp_gate_up is not None:
            _gate_up_w = _ensure_interleaved(w.w_mlp_gate_up)
            # Keep WIDTH_SHARDED L1 (skip gather) so silu*mul matches the unfused path
            # and layer-0 dense MLP; gathering to DRAM before slice breaks TP column shards.
            gate_up = _mlp_linear(x, _gate_up_w, skip_gather=True)
            _batch = int(gate_up.shape[2])
            _inter_tp = int(gate_up.shape[3]) // 2
            gate_shared = ttnn.slice(gate_up, [0, 0, 0, 0], [1, 1, _batch, _inter_tp])
            up_shared = ttnn.slice(gate_up, [0, 0, 0, _inter_tp], [1, 1, _batch, _inter_tp * 2])
            ttnn.deallocate(gate_up, force=False)
        else:
            _gate_w = _ensure_interleaved(w.w_mlp_gate)
            _up_w = _ensure_interleaved(w.w_mlp_up)
            gate_shared = _mlp_linear(x, _gate_w, skip_gather=True)
            up_shared = _mlp_linear(x, _up_w, skip_gather=True)
        x_ff_shared = ttnn.mul(gate_shared, up_shared, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        ttnn.deallocate(gate_shared, force=False)
        ttnn.deallocate(up_shared, force=False)
        shared_out = _mlp_linear(x_ff_shared, _down_w)
        ttnn.deallocate(x_ff_shared, force=False)
        if tp_enabled and not _skip_shared_reduce:
            shared_out_reduced = ttnn.all_reduce(
                shared_out,
                num_links=ccl_num_links,
                topology=ccl_topology,
                cluster_axis=tp_axis,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            ttnn.deallocate(shared_out, force=False)
            shared_out = shared_out_reduced
        _profile_add(profile, "moe_shared_s", time.perf_counter() - t0 if profile is not None else 0.0)

        # Routed experts.
        t0 = time.perf_counter() if profile is not None else 0.0
        router_impl = os.environ.get("GLM4_MOE_LITE_MOE_ROUTER_IMPL", "tt").strip().lower()
        if router_impl == "cpu":
            topk_weights, topk_indices = moe_topk_cpu_reference(device=device, x=x, moe_w=w.moe, hparams=hparams)
        else:
            # Router pins its own LoFi config internally; no longer inherits the over-precise shared HiFi4 MLP config.
            topk_weights, topk_indices = moe_topk_tt(
                x=x,
                moe_w=w.moe,
                hparams=hparams,
            )
        _profile_add(profile, "moe_router_s", time.perf_counter() - t0 if profile is not None else 0.0)
        t0 = time.perf_counter() if profile is not None else 0.0
        if use_packed_prefill_here:
            routed_out = moe_packed_experts_forward_prefill_tt(
                device=device,
                hidden_states=x,  # consumed
                topk_expert_indices=topk_indices,  # consumed
                topk_expert_weights=topk_weights,  # consumed
                moe_w=w.moe,
                hparams=hparams,
                rt=moe_runtime,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=mlp_compute_kernel_config,
                skip_defensive_clones=skip_defensive_clones,
            )
        elif dense_prefill:
            routed_out = moe_dense_experts_forward_prefill_tt(
                device=device,
                hidden_states=x,  # consumed
                topk_expert_indices=topk_indices,  # consumed
                topk_expert_weights=topk_weights,  # consumed
                moe_w=w.moe,
                hparams=hparams,
                rt=moe_runtime,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=mlp_compute_kernel_config,
                skip_defensive_clones=skip_defensive_clones,
            )
        else:
            routed_out = moe_sparse_experts_forward_tt(
                device=device,
                hidden_states=x,  # consumed
                topk_expert_indices=topk_indices,  # consumed
                topk_expert_weights=topk_weights,  # consumed
                moe_w=w.moe,
                rt=moe_runtime,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                skip_final_reduce=_skip_shared_reduce,
                skip_defensive_clones=skip_defensive_clones,
            )
        _profile_add(profile, "moe_experts_s", time.perf_counter() - t0 if profile is not None else 0.0)

        t0 = time.perf_counter() if profile is not None else 0.0
        # Match decode: explicit add with DRAM MC. Python `+` on L1 shared + DRAM routed
        # can produce wrong results after TP all_reduce leaves shared_out in L1.
        shared_dram = ttnn.to_memory_config(shared_out, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(shared_out, force=False)
        mlp_out = ttnn.add(shared_dram, routed_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(shared_dram, force=False)
        ttnn.deallocate(routed_out, force=False)
        if tp_enabled and _skip_shared_reduce:
            mlp_out_reduced = ttnn.all_reduce(
                mlp_out,
                num_links=ccl_num_links,
                topology=ccl_topology,
                cluster_axis=tp_axis,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            ttnn.deallocate(mlp_out, force=False)
            mlp_out = mlp_out_reduced

        # Slice back to the real token count if we padded.
        if pad_tokens:
            # `slice` may return a view that aliases `mlp_out` (no refcounting).
            # Materialize before freeing the padded tensor to avoid prefill corruption.
            mlp_out_view = ttnn.slice(mlp_out, [0, 0, 0, 0], [1, 1, tokens, int(hparams.hidden_size)])
            if skip_defensive_clones:
                mlp_out = mlp_out_view
            else:
                mlp_out_sliced = ttnn.clone(mlp_out_view, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                ttnn.deallocate(mlp_out, force=False)
                mlp_out = mlp_out_sliced
        _profile_add(profile, "moe_merge_s", time.perf_counter() - t0 if profile is not None else 0.0)

    x_mlp_out = residual + mlp_out
    ttnn.deallocate(mlp_out, force=False)
    ttnn.deallocate(residual, force=False)
    _profile_add(profile, "total_s", time.perf_counter() - t_layer0 if profile is not None else 0.0)

    return x_mlp_out


__all__ = [
    "prepare_decode_rope_and_positions_tt",
    "run_decoder_layer_decode_one_step_update_cache_tt",
    "run_decoder_layer_prefill_update_cache_tt",
]
