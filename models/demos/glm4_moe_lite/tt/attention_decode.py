# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""MLA attention decode path for GLM-4.7-Flash.

Extracted from decoder_layer_tt.py lines 819-1318. Three functions
corresponding to the three phases of a decode attention step:
  1. kv_cache_update — project KV, RoPE, write to paged cache
  2. q_projection — project Q, RoPE, produce q_kvpe
  3. flash_mla_and_output — run FlashMLA, kv_b2, flatten heads, w_o
"""

from __future__ import annotations

import time
from typing import Any

import torch

import ttnn
from models.demos.glm4_moe_lite.tt.config import Glm4MoeLiteHParams
from models.demos.glm4_moe_lite.tt.linear_helpers import attn_linear, mlp_linear, tp_row_parallel_linear
from models.demos.glm4_moe_lite.tt.runtime_config import Glm4RuntimeConfig


def _profile_add(profile: dict[str, float] | None, key: str, elapsed_s: float) -> None:
    if profile is None:
        return
    profile[key] = float(profile.get(key, 0.0)) + float(elapsed_s)


def _safe_slice(
    tensor: ttnn.Tensor,
    starts: list[int],
    ends: list[int],
    *,
    skip_clones: bool,
) -> ttnn.Tensor:
    """Slice with optional defensive clone to avoid aliasing bugs."""
    result = ttnn.slice(tensor, starts, ends)
    if not skip_clones:
        cloned = ttnn.clone(result, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return cloned
    return result


def kv_cache_update(
    *,
    device: Any,
    x: ttnn.Tensor,
    w: Any,
    hparams: Glm4MoeLiteHParams,
    cfg: Glm4RuntimeConfig,
    batch: int,
    cos_batch: ttnn.Tensor,
    sin_batch: ttnn.Tensor,
    trans_matrix: ttnn.Tensor,
    kvpe_cache: ttnn.Tensor,
    page_table_tt: ttnn.Tensor,
    tt_positions: ttnn.Tensor,
    positions_main_tt: ttnn.Tensor | None,
    positions_draft_tt: ttnn.Tensor | None,
    use_decode_rope: bool,
    rope_decode_fn: Any | None,
    shard_kvpe_fn: Any,
    fused_kv_branch_fn: Any | None,
    profile: dict[str, float] | None = None,
) -> ttnn.Tensor | None:
    """Project KV for the new token and update the paged KVPE cache.

    Returns q_a if it was computed as a side effect of a fused QKV projection,
    or None if q_a still needs to be computed separately.
    """
    kvpe_dim = int(hparams.kv_lora_rank + hparams.qk_rope_head_dim)
    q_a = None

    if cfg.skip_kv_update:
        return None

    t0 = time.perf_counter() if profile is not None else 0.0

    if fused_kv_branch_fn is not None and batch == 1:
        q_a = attn_linear(x, w.w_q_a, device=device, cfg=cfg, force_no_tp=cfg.attn_dp)
        kvpe_new = fused_kv_branch_fn(
            device=device,
            x=x,
            fused_kv=w.fused_kv_branch,
            cos_batch=cos_batch,
            sin_batch=sin_batch,
        )
    else:
        kv = None
        qkv = None
        w_q_kv_a = getattr(w, "w_q_kv_a", None)
        if w_q_kv_a is not None:
            qkv = attn_linear(x, w_q_kv_a, device=device, cfg=cfg, force_no_tp=cfg.attn_dp)
            q_a = _safe_slice(
                qkv, [0, 0, 0, 0], [1, 1, batch, int(hparams.q_lora_rank)], skip_clones=cfg.skip_defensive_clones
            )
            kv = _safe_slice(
                qkv,
                [0, 0, 0, int(hparams.q_lora_rank)],
                [1, 1, batch, int(hparams.q_lora_rank) + kvpe_dim],
                skip_clones=cfg.skip_defensive_clones,
            )
            if not cfg.skip_defensive_clones:
                ttnn.deallocate(qkv, force=False)
                qkv = None
        else:
            kv = attn_linear(x, w.w_kv_a, device=device, cfg=cfg, force_no_tp=cfg.attn_dp)

        kv_nope = _safe_slice(
            kv, [0, 0, 0, 0], [1, 1, batch, int(hparams.kv_lora_rank)], skip_clones=cfg.skip_defensive_clones
        )
        kv_rope = _safe_slice(
            kv, [0, 0, 0, int(hparams.kv_lora_rank)], [1, 1, batch, kvpe_dim], skip_clones=cfg.skip_defensive_clones
        )
        if not cfg.skip_defensive_clones and kv is not None:
            ttnn.deallocate(kv, force=False)
            kv = None

        kv_nope = w.kv_a_layernorm(kv_nope, mode="decode")

        if not cfg.skip_typecast and kv_rope.dtype != ttnn.bfloat16:
            kv_rope = ttnn.typecast(kv_rope, dtype=ttnn.bfloat16)
        if use_decode_rope and rope_decode_fn is not None:
            kv_rope = rope_decode_fn(kv_rope, heads=1)
        else:
            kv_rope = ttnn.experimental.rotary_embedding_llama(
                kv_rope,
                cos_batch,
                sin_batch,
                trans_matrix,
                is_decode_mode=False,
            )

        if kv is not None:
            ttnn.deallocate(kv, force=False)

        kvpe_new = ttnn.concat([kv_nope, kv_rope], dim=-1)
        ttnn.deallocate(kv_nope, force=False)
        ttnn.deallocate(kv_rope, force=False)

        # Free qkv after all slices are consumed
        if qkv is not None:
            ttnn.deallocate(qkv, force=False)

    kvpe_new_sharded = shard_kvpe_fn(
        device=device,
        kvpe_new=kvpe_new,
        batch=batch,
        kvpe_dim=kvpe_dim,
        skip_defensive_clones=cfg.skip_defensive_clones,
    )

    mesh_coords = None
    if device.__class__.__name__ == "MeshDevice":
        try:
            mesh_rows, mesh_cols = int(device.shape[0]), int(device.shape[1])
            mesh_coords = {ttnn.MeshCoordinate(r, c) for r in range(mesh_rows) for c in range(mesh_cols)}
        except Exception:
            mesh_coords = None

    _puc_kwargs = dict(page_table=page_table_tt)
    if mesh_coords is not None:
        _puc_kwargs["mesh_coords"] = mesh_coords

    if positions_main_tt is not None and positions_draft_tt is not None:
        ttnn.experimental.paged_update_cache(
            kvpe_cache, kvpe_new_sharded, update_idxs_tensor=positions_main_tt, **_puc_kwargs
        )
        ttnn.experimental.paged_update_cache(
            kvpe_cache, kvpe_new_sharded, update_idxs_tensor=positions_draft_tt, **_puc_kwargs
        )
    else:
        ttnn.experimental.paged_update_cache(
            kvpe_cache, kvpe_new_sharded, update_idxs_tensor=tt_positions, **_puc_kwargs
        )
    if cfg.sync_after_kv_update:
        ttnn.synchronize_device(device)
    ttnn.deallocate(kvpe_new_sharded, force=False)
    ttnn.deallocate(kvpe_new, force=False)
    _profile_add(profile, "kv_cache_update_s", time.perf_counter() - t0 if profile is not None else 0.0)

    return q_a


def q_projection(
    *,
    device: Any,
    x: ttnn.Tensor,
    w: Any,
    hparams: Glm4MoeLiteHParams,
    cfg: Glm4RuntimeConfig,
    batch: int,
    cos_batch: ttnn.Tensor,
    sin_batch: ttnn.Tensor,
    trans_matrix: ttnn.Tensor,
    q_a_from_kv: ttnn.Tensor | None,
    use_decode_rope: bool,
    rope_decode_fn: Any | None,
    profile: dict[str, float] | None = None,
) -> ttnn.Tensor:
    """Compute Q projection, RoPE, and produce q_kvpe [1,H,B,kvpe_dim].

    If q_a_from_kv is provided (from fused QKV in kv_cache_update), uses it
    instead of computing q_a from scratch.
    """
    t0 = time.perf_counter() if profile is not None else 0.0
    kvpe_dim = int(hparams.kv_lora_rank + hparams.qk_rope_head_dim)

    if q_a_from_kv is not None:
        q_a = q_a_from_kv
    else:
        q_a = attn_linear(x, w.w_q_a, device=device, cfg=cfg, force_no_tp=cfg.attn_dp)

    q_a = w.q_a_layernorm(q_a, mode="decode")
    q = attn_linear(q_a, w.w_q_b, device=device, cfg=cfg, force_no_tp=cfg.attn_dp)
    ttnn.deallocate(q_a, force=False)

    q = ttnn.reshape(q, (1, batch, int(hparams.num_attention_heads), int(hparams.qk_head_dim)))
    q = ttnn.permute(q, (0, 2, 1, 3))  # [1,H,B,qk_head_dim]

    q_nope = _safe_slice(
        q,
        [0, 0, 0, 0],
        [1, int(hparams.num_attention_heads), batch, int(hparams.qk_nope_head_dim)],
        skip_clones=cfg.skip_defensive_clones,
    )
    q_rope = _safe_slice(
        q,
        [0, 0, 0, int(hparams.qk_nope_head_dim)],
        [1, int(hparams.num_attention_heads), batch, int(hparams.qk_head_dim)],
        skip_clones=cfg.skip_defensive_clones,
    )
    if not cfg.skip_defensive_clones:
        ttnn.deallocate(q, force=False)
        q = None

    # kv_b1: project q_nope into KV latent space
    use_tp_kv_b1 = cfg.tp_enabled
    if use_tp_kv_b1:
        qk_nope = int(hparams.qk_nope_head_dim)
        qk_nope_per_shard = qk_nope // max(1, cfg.tp_size)
        if qk_nope % max(1, cfg.tp_size) != 0 or qk_nope_per_shard % int(ttnn.TILE_SIZE) != 0:
            use_tp_kv_b1 = False
    if use_tp_kv_b1:
        q_nope = tp_row_parallel_linear(q_nope, w.w_kv_b1, device=device, cfg=cfg)
    else:
        q_nope = mlp_linear(q_nope, w.w_kv_b1, device=device, cfg=cfg)

    if not cfg.skip_typecast and q_rope.dtype != ttnn.bfloat16:
        q_rope = ttnn.typecast(q_rope, dtype=ttnn.bfloat16)
    if use_decode_rope and rope_decode_fn is not None:
        q_rope = rope_decode_fn(q_rope, heads=int(hparams.num_attention_heads))
    else:
        q_rope = ttnn.experimental.rotary_embedding_llama(
            q_rope,
            cos_batch,
            sin_batch,
            trans_matrix,
            is_decode_mode=False,
        )

    if q is not None:
        ttnn.deallocate(q, force=False)

    q_kvpe = ttnn.concat([q_nope, q_rope], dim=-1)  # [1,H,B,kvpe_dim]
    ttnn.deallocate(q_nope, force=False)
    ttnn.deallocate(q_rope, force=False)

    _profile_add(profile, "q_path_s", time.perf_counter() - t0 if profile is not None else 0.0)
    return q_kvpe


def flash_mla_and_output(
    *,
    device: Any,
    q_kvpe: ttnn.Tensor,
    w: Any,
    hparams: Glm4MoeLiteHParams,
    cfg: Glm4RuntimeConfig,
    batch: int,
    kvpe_cache: ttnn.Tensor,
    page_table_tt: ttnn.Tensor,
    tt_positions: ttnn.Tensor,
    profile: dict[str, float] | None = None,
) -> ttnn.Tensor:
    """Run FlashMLA decode, then kv_b2 + head-flatten + w_o.

    Returns attn_out [1,1,B,hidden].
    """
    kvpe_dim = int(hparams.kv_lora_rank + hparams.qk_rope_head_dim)
    num_heads = int(hparams.num_attention_heads)

    # Prepare Q for decode: [1,H,B,kvpe_dim] -> [1,B,H,kvpe_dim]
    if cfg.skip_defensive_clones:
        q_for_decode = ttnn.permute(q_kvpe, (0, 2, 1, 3))
    else:
        q_for_decode_view = ttnn.permute(q_kvpe, (0, 2, 1, 3))
        q_for_decode = ttnn.clone(q_for_decode_view, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(q_kvpe, force=False)
        q_kvpe = None

    # Attention scale
    if cfg.mla_scale_mode == "kvpe":
        scale = float(kvpe_dim**-0.5)
    else:
        scale = float(int(hparams.qk_head_dim) ** -0.5)

    sdpa_program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=0,
        k_chunk_size=cfg.mla_k_chunk_size,
        exp_approx_mode=False,
    )
    compute_kernel_config = cfg.mla_compute_kernel_config()
    flash_mla_memcfg = ttnn.DRAM_MEMORY_CONFIG

    # Optional Q sharding
    if cfg.shard_q:
        grid_size = device.compute_with_storage_grid_size()
        num_cores = int(grid_size.x) * int(grid_size.y)
        height = batch * num_heads
        tiles_h = max(1, (height + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE)
        q_num_cores = min(tiles_h, max(1, num_cores))
        shard_h = (height + q_num_cores - 1) // q_num_cores
        shard_h = ((shard_h + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
        q_core_grid = ttnn.num_cores_to_corerangeset(q_num_cores, grid_size, row_wise=True)
        q_mem_config = ttnn.create_sharded_memory_config(
            shape=(int(shard_h), kvpe_dim),
            core_grid=q_core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        flash_mla_out_memcfg = ttnn.create_sharded_memory_config(
            shape=(int(shard_h), int(hparams.kv_lora_rank)),
            core_grid=q_core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        if cfg.skip_defensive_clones:
            q_for_decode = ttnn.to_memory_config(q_for_decode, q_mem_config)
        else:
            q_view = ttnn.to_memory_config(q_for_decode, q_mem_config)
            q_sharded = ttnn.clone(q_view, memory_config=q_mem_config)
            ttnn.deallocate(q_for_decode, force=False)
            q_for_decode = q_sharded
        flash_mla_memcfg = flash_mla_out_memcfg

    # Run FlashMLA
    t0 = time.perf_counter() if profile is not None else 0.0
    if cfg.disable_flash_mla_decode:
        is_mesh = device.__class__.__name__ == "MeshDevice"
        mapper = ttnn.ReplicateTensorToMesh(device) if is_mesh else None
        heads_padded = ((num_heads + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
        attn_latent = ttnn.from_torch(
            torch.zeros((1, batch, heads_padded, int(hparams.kv_lora_rank)), dtype=torch.bfloat16),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )
    elif cfg.use_v_cache_slice:
        v_cache = ttnn.slice(
            kvpe_cache, [0, 0, 0, 0], [int(kvpe_cache.shape[0]), 1, int(kvpe_cache.shape[2]), int(hparams.kv_lora_rank)]
        )
        attn_latent = ttnn.transformer.paged_flash_multi_latent_attention_decode(
            q_for_decode,
            kvpe_cache,
            v_cache,
            head_dim_v=int(hparams.kv_lora_rank),
            page_table_tensor=page_table_tt,
            cur_pos_tensor=tt_positions,
            scale=scale,
            program_config=sdpa_program_config,
            compute_kernel_config=compute_kernel_config,
            memory_config=flash_mla_memcfg,
        )
        ttnn.deallocate(v_cache, force=False)
    else:
        attn_latent = ttnn.transformer.paged_flash_multi_latent_attention_decode(
            q_for_decode,
            kvpe_cache,
            head_dim_v=int(hparams.kv_lora_rank),
            page_table_tensor=page_table_tt,
            cur_pos_tensor=tt_positions,
            scale=scale,
            program_config=sdpa_program_config,
            compute_kernel_config=compute_kernel_config,
            memory_config=flash_mla_memcfg,
        )
    ttnn.deallocate(q_for_decode, force=False)
    if q_kvpe is not None:
        ttnn.deallocate(q_kvpe, force=False)
    _profile_add(profile, "flash_mla_decode_s", time.perf_counter() - t0 if profile is not None else 0.0)

    # Reshard from L1 to DRAM if Q was sharded
    if cfg.shard_q and not cfg.disable_flash_mla_decode:
        if cfg.skip_defensive_clones:
            attn_latent = ttnn.to_memory_config(attn_latent, memory_config=cfg.decode_act_mc or ttnn.DRAM_MEMORY_CONFIG)
        else:
            view = ttnn.to_memory_config(attn_latent, memory_config=cfg.decode_act_mc or ttnn.DRAM_MEMORY_CONFIG)
            attn_latent_dram = ttnn.clone(view, memory_config=cfg.decode_act_mc or ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(attn_latent, force=False)
            attn_latent = attn_latent_dram

    # Slice padded heads
    attn_latent_padded = attn_latent
    attn_latent = _safe_slice(
        attn_latent_padded,
        [0, 0, 0, 0],
        [1, batch, num_heads, int(hparams.kv_lora_rank)],
        skip_clones=cfg.skip_defensive_clones,
    )
    if not cfg.skip_defensive_clones:
        ttnn.deallocate(attn_latent_padded, force=False)
    attn_latent = ttnn.permute(attn_latent, (0, 2, 1, 3))  # [1,H,B,kv_lora_rank]

    # kv_b2 + output projection
    t0 = time.perf_counter() if profile is not None else 0.0
    if cfg.head_parallel_kvb2:
        attn_latent = ttnn.mesh_partition(attn_latent, dim=1, cluster_axis=cfg.tp_axis)
        v = mlp_linear(attn_latent, w.w_kv_b2, device=device, cfg=cfg)
        ttnn.deallocate(attn_latent, force=False)
        if cfg.skip_defensive_clones:
            try:
                ttnn.deallocate(attn_latent_padded, force=False)
            except Exception:
                pass
        heads_per_dev = num_heads // cfg.tp_size
        flat_dim = heads_per_dev * int(hparams.v_head_dim)
        if cfg.concat_heads:
            v = ttnn.transformer.concatenate_heads(v)
            v = ttnn.reshape(v, (1, 1, batch, flat_dim))
        else:
            v = ttnn.permute(v, (0, 2, 1, 3))
            v = ttnn.reshape(v, (1, batch, 1, flat_dim))
            v = ttnn.permute(v, (0, 2, 1, 3))
        attn_out_partial = mlp_linear(v, w.w_o, device=device, cfg=cfg)
        ttnn.deallocate(v, force=False)
        attn_out = ttnn.all_reduce(
            attn_out_partial,
            num_links=1,
            topology=ttnn.Topology.Linear,
            cluster_axis=cfg.tp_axis,
            memory_config=cfg.decode_act_mc or ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(attn_out_partial, force=False)
    else:
        if cfg.tp_enabled and not cfg.attn_dp:
            v = tp_row_parallel_linear(attn_latent, w.w_kv_b2, device=device, cfg=cfg)
        else:
            v = mlp_linear(attn_latent, w.w_kv_b2, device=device, cfg=cfg)
        ttnn.deallocate(attn_latent, force=False)
        if cfg.skip_defensive_clones:
            try:
                ttnn.deallocate(attn_latent_padded, force=False)
            except Exception:
                pass
        if cfg.concat_heads:
            v = ttnn.transformer.concatenate_heads(v)
            v = ttnn.reshape(v, (1, 1, batch, int(num_heads * hparams.v_head_dim)))
        else:
            v = ttnn.permute(v, (0, 2, 1, 3))
            v = ttnn.reshape(v, (1, batch, 1, int(num_heads * hparams.v_head_dim)))
            v = ttnn.permute(v, (0, 2, 1, 3))
        attn_out = attn_linear(v, w.w_o, device=device, cfg=cfg)
        ttnn.deallocate(v, force=False)

    _profile_add(profile, "attn_out_s", time.perf_counter() - t0 if profile is not None else 0.0)
    return attn_out
