# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Weight loading for Gemma4 attention with tensor parallelism support.

Uses HF-style weight format. Context-parallel global prefill folds the packed
640-channel cache permutations into Q/KV projection, norm, and output weights.

TP sharding (following gpt-oss pattern):
- QKV: column-parallel (shard output dim across TP devices)
- O_proj: row-parallel (shard input dim across TP devices)
- Norm weights: replicated across all devices
- Allreduce after O_proj recombines results
"""

import os
import pathlib
from dataclasses import dataclass
from typing import Union

import torch
from loguru import logger

import ttnn
from models.demos.gemma4.config import MeshConfig
from models.demos.gemma4.tt.dram_sharded import DramShardedLinear, can_dram_shard
from models.demos.gemma4.utils.general_utils import get_cache_file_name

from .global_kv_cache import GLOBAL_ROTARY_DIM, global_kv_indices

# DRAM-width-sharded QKV / O-proj decode matmuls (same size as the interleaved
# weight → no memory cost). On by default for tp>1; GEMMA4_ATTN_DRAM_SHARD=0
# falls back to plain interleaved matmuls.
_DRAM_SHARD_ATTN = os.environ.get("GEMMA4_ATTN_DRAM_SHARD", "1") != "0"


@dataclass(frozen=True)
class AttentionWeights:
    """Container for attention weight tensors — immutable after creation."""

    wqkv: Union[ttnn.Tensor, DramShardedLinear]  # Fused Q+K+V, column-parallel
    o_proj: Union[ttnn.Tensor, DramShardedLinear]  # Row-parallel sharded
    q_norm_weight: ttnn.Tensor  # Replicated across devices
    k_norm_weight: ttnn.Tensor  # Replicated across devices
    # Only the rotary quarter needs a separate K gamma in packed global prefill.
    k_norm_rotary_weight: ttnn.Tensor | None
    is_global: bool  # Controls K=V tying and partial RoPE
    kv_replicated: bool = False  # True when KV heads are replicated (not split) across TP devices
    # Fused Q+K only, no duplicate V columns -- global (K=V tied) layers only, prefill only.
    # None on sliding layers and whenever the tied projection is disabled. See load_attention_weights.
    wqk: ttnn.Tensor | None = None


def _cached_tensor_exists(cache_file_name) -> bool:
    """Whether ttnn.as_tensor would find a cache file for this name.

    as_tensor appends ``_dtype_<DTYPE>_layout_<LAYOUT>.tensorbin`` to the name it is given,
    so the caller's name is a prefix, not the path. Matched by glob rather than rebuilt here,
    which would duplicate ttnn's naming and drift from it.
    """
    if not cache_file_name:
        return False
    path = pathlib.Path(cache_file_name)
    return any(path.parent.glob(f"{path.name}_dtype_*.tensorbin"))


def load_attention_weights(
    mesh_device,
    config,
    state_dict,
    mesh_config: MeshConfig,
    weight_dtype=ttnn.bfloat16,
    tensor_cache_path=None,
) -> AttentionWeights:
    """
    Load and fuse attention weights with tensor parallelism.

    No Meta-format conversion needed — uses HF-style rotary_embedding.

    Global layers tie K and V to one projection (``attention_k_eq_v``), so the fused wqkv
    holds the same K columns twice and the projection matmul computes them twice: 3072 of
    5376x3072 output columns per device at TP=8, of which 512 are the duplicate. Those
    layers additionally get ``wqk`` -- the same weight without the V section -- which prefill
    uses together with the split op's tied mode. Decode keeps using wqkv, because its split
    (``nlp_create_qkv_heads_decode``) reads Q/K/V across input cores with mcast address
    state and has no tied mode; carrying both is 13.9 MB per device per global layer, which
    is the price of leaving the decode path untouched. Set GEMMA4_TIED_QKV=0 to skip
    building wqk and put prefill back on the wqkv path.
    """
    is_global = config.use_kv_tying
    tied_qkv = is_global and os.environ.get("GEMMA4_TIED_QKV", "1") == "1"
    q_size = config.num_attention_heads * config.head_dim
    kv_size = config.num_key_value_heads * config.head_dim
    tp = mesh_config.tp
    is_context_parallel = bool(mesh_config and mesh_config.prefill.sp > 1)

    # When KV heads < TP, each device gets the KV head(s) its Q heads map to via GQA.
    # E.g. 16 Q / 2 KV / 8 TP: devices 0-3 get KV head 0, devices 4-7 get KV head 1.
    kv_replicated = config.num_key_value_heads < tp

    # Compute o_proj padding for tile-aligned CCL
    hidden_size = config.hidden_size
    local_hidden = hidden_size // tp
    padded_local_hidden = ((local_hidden + 31) // 32) * 32
    o_proj_pad_size = padded_local_hidden - local_hidden

    if state_dict:
        q_w = state_dict["q_proj.weight"]  # [q_size, H]
        k_w = state_dict["k_proj.weight"]  # [kv_size, H]
        if is_global and is_context_parallel:
            rotary, nonrotary, value_order = global_kv_indices(config.head_dim, GLOBAL_ROTARY_DIM)
            query_order = torch.cat((rotary, nonrotary))
            q_w = q_w.reshape(config.num_attention_heads, config.head_dim, -1).index_select(1, query_order)
            q_w = q_w.reshape(q_size, -1)
            k_w = k_w.reshape(config.num_key_value_heads, config.head_dim, -1).index_select(1, value_order)
            k_w = k_w.reshape(kv_size, -1)

        if not is_global:
            v_w = state_dict["v_proj.weight"]  # [kv_size, H]
        else:
            v_w = k_w  # K=V tying: duplicate K as V

        if tp > 1:
            # Chunk Q/K/V per TP device, fuse per-device, then concatenate across devices
            # When kv_replicated, keep full K/V on each device instead of chunking
            num_q_heads = config.num_attention_heads
            num_kv_heads = config.num_key_value_heads
            head_dim = config.head_dim
            q_per_device = num_q_heads // tp

            qkv_list = []
            qk_list = []
            for i in range(tp):
                wq_chunk = torch.chunk(q_w, tp, dim=0)[i].transpose(-2, -1)
                if kv_replicated:
                    # GQA-aware KV assignment: each device gets the KV head its Q heads map to
                    kv_idx = (i * q_per_device) * num_kv_heads // num_q_heads
                    wk_chunk = k_w[kv_idx * head_dim : (kv_idx + 1) * head_dim].transpose(-2, -1)
                    wv_chunk = v_w[kv_idx * head_dim : (kv_idx + 1) * head_dim].transpose(-2, -1)
                else:
                    wk_chunk = torch.chunk(k_w, tp, dim=0)[i].transpose(-2, -1)
                    wv_chunk = torch.chunk(v_w, tp, dim=0)[i].transpose(-2, -1)
                qkv_list.append(torch.cat([wq_chunk, wk_chunk, wv_chunk], dim=-1))
                if tied_qkv:
                    qk_list.append(torch.cat([wq_chunk, wk_chunk], dim=-1))
            qkv = torch.cat(qkv_list, dim=-1).unsqueeze(0).unsqueeze(0)
            qk = torch.cat(qk_list, dim=-1).unsqueeze(0).unsqueeze(0) if tied_qkv else None
        else:
            # Single device: fuse Q+K+V directly
            qkv = (
                torch.cat(
                    [
                        q_w.transpose(-2, -1),
                        k_w.transpose(-2, -1),
                        v_w.transpose(-2, -1),
                    ],
                    dim=-1,
                )
                .unsqueeze(0)
                .unsqueeze(0)
            )
            qk = (
                torch.cat([q_w.transpose(-2, -1), k_w.transpose(-2, -1)], dim=-1).unsqueeze(0).unsqueeze(0)
                if tied_qkv
                else None
            )

        # Output projection
        o_w = state_dict["o_proj.weight"].transpose(-2, -1).unsqueeze(0).unsqueeze(0)
        if is_global and is_context_parallel:
            _, _, value_order = global_kv_indices(config.head_dim, GLOBAL_ROTARY_DIM)
            o_w = o_w.reshape(1, 1, config.num_attention_heads, config.head_dim, config.hidden_size)
            o_w = o_w.index_select(3, value_order)
            o_w = o_w.reshape(1, 1, q_size, config.hidden_size)
        if o_proj_pad_size > 0 and tp > 1:
            padded_hidden = padded_local_hidden * tp
            o_w = torch.nn.functional.pad(o_w, (0, padded_hidden - hidden_size), "constant", 0.0)

        # Per-head norm weights: [head_dim] -> [1, 1, head_dim/TILE_SIZE, TILE_SIZE]
        q_norm_flat = state_dict["q_norm.weight"].reshape(-1)
        k_norm_flat = state_dict["k_norm.weight"].reshape(-1)
        k_norm_w = k_norm_flat.reshape(1, 1, -1, ttnn.TILE_SIZE)
        if is_global:
            rotary, nonrotary, _ = global_kv_indices(config.head_dim, GLOBAL_ROTARY_DIM)
            # The 128-wide rotary op consumes NeoX halves, not adjacent pairs.
            rotary_neox = torch.sort(rotary).values
            k_norm_rotary_w = k_norm_flat.index_select(0, rotary_neox).reshape(1, 1, 1, -1)
            if is_context_parallel:
                query_order = torch.cat((rotary, nonrotary))
                packed_q_scale = torch.cat(
                    (torch.ones(GLOBAL_ROTARY_DIM, dtype=k_norm_flat.dtype), k_norm_flat.index_select(0, nonrotary))
                )
                q_norm_flat = q_norm_flat.index_select(0, query_order) * packed_q_scale
        else:
            k_norm_rotary_w = None
        q_norm_w = q_norm_flat.reshape(1, 1, -1, ttnn.TILE_SIZE)
    else:
        qkv = None
        qk = None
        o_w = None
        q_norm_w = None
        k_norm_w = None
        k_norm_rotary_w = None

    # Mesh mappers
    if tp > 1:
        col_mapper = mesh_config.column_parallel(mesh_device)
        row_mapper = mesh_config.row_parallel(mesh_device)
        replicate_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    else:
        col_mapper = None
        row_mapper = None
        replicate_mapper = None

    o_proj_cache_suffix = "_padded" if o_proj_pad_size > 0 and tp > 1 else ""
    if is_global and is_context_parallel:
        o_proj_cache_suffix += "_packed640"
    tp_suffix = f"_tp{tp}" if tp > 1 else ""
    # Tag the wqkv / o_proj cache filenames with their dtype so flipping
    # ``attention`` precision in precision_overrides.json doesn't reuse a
    # stale cached tensor at the previous dtype. q_norm / k_norm stay at
    # bfloat16 (no override) and don't need the suffix.
    from models.demos.gemma4.tt.precision import dtype_to_str

    dtype_suffix = f"_{dtype_to_str(weight_dtype)}"

    # Cache-only builds may predate the narrow Q+K cache; fall back to full QKV
    # until a full-weight build creates it.
    packed_cache_suffix = "_packed640" if is_global and is_context_parallel else ""
    qk_cache_name = get_cache_file_name(tensor_cache_path, f"wqk{packed_cache_suffix}{tp_suffix}{dtype_suffix}")
    if tied_qkv and qk is None and not _cached_tensor_exists(qk_cache_name):
        logger.warning(
            "No cached wqk for this global layer, so its QKV projection keeps the duplicate V "
            "columns. Rebuild the cache with GEMMA4_PREFILL_LOAD_FULL_WEIGHTS=1 once to write it."
        )
        tied_qkv = False

    num_local_heads = config.num_attention_heads // tp
    head_dim = config.head_dim
    q_per_dev = num_local_heads * head_dim
    kv_per_dev = head_dim if kv_replicated else (config.num_key_value_heads // tp) * head_dim
    qkv_n = q_per_dev + 2 * kv_per_dev
    oproj_k = q_per_dev
    oproj_n = hidden_size

    # Main's DRAM-width-sharded full-QKV remains the decode path.
    is_moe = bool(getattr(config, "enable_moe_block", False))
    # Context-parallel prefill keeps the pre-rebase interleaved representation.
    # Besides avoiding decode-only DRAM-sharding machinery, this preserves the
    # established tensor-cache names used by the long-context service.
    dram_shard = _DRAM_SHARD_ATTN and tp > 1 and not is_moe and not is_context_parallel
    qkv_cache = get_cache_file_name(tensor_cache_path, f"wqkv{packed_cache_suffix}{tp_suffix}{dtype_suffix}")
    oproj_cache = get_cache_file_name(tensor_cache_path, f"o_proj{o_proj_cache_suffix}{tp_suffix}{dtype_suffix}")
    qkv_cache_ws = (qkv_cache + ".ws") if qkv_cache else None
    oproj_cache_ws = (oproj_cache + ".ws") if oproj_cache else None

    if dram_shard and can_dram_shard(hidden_size, qkv_n, dtype=weight_dtype):
        wqkv = DramShardedLinear(
            qkv, mesh_device, col_mapper, k=hidden_size, n=qkv_n, dtype=weight_dtype, cache_file_name=qkv_cache_ws
        )
    else:
        wqkv = ttnn.as_tensor(
            qkv,
            device=mesh_device,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=col_mapper,
            cache_file_name=qkv_cache,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    # Prefill-only narrow Q+K projection. Keep it interleaved; unlike the full
    # QKV tensor it is not consumed by the DRAM-sharded decode path.
    wqk = (
        ttnn.as_tensor(
            qk,
            device=mesh_device,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=col_mapper,
            cache_file_name=qk_cache_name,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if tied_qkv
        else None
    )

    if dram_shard and o_proj_pad_size == 0 and can_dram_shard(oproj_k, oproj_n, dtype=weight_dtype):
        o_proj = DramShardedLinear(
            o_w, mesh_device, row_mapper, k=oproj_k, n=oproj_n, dtype=weight_dtype, cache_file_name=oproj_cache_ws
        )
    else:
        o_proj = ttnn.as_tensor(
            o_w,
            device=mesh_device,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=row_mapper,
            cache_file_name=oproj_cache,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    q_norm_weight = ttnn.as_tensor(
        q_norm_w,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=replicate_mapper,
        cache_file_name=get_cache_file_name(
            tensor_cache_path, f"q_norm.weight{'_packed640' if is_global and is_context_parallel else ''}{tp_suffix}"
        ),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    k_norm_weight = ttnn.as_tensor(
        k_norm_w,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=replicate_mapper,
        cache_file_name=get_cache_file_name(tensor_cache_path, f"k_norm.weight{tp_suffix}"),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    k_norm_rotary_weight = (
        ttnn.as_tensor(
            k_norm_rotary_w,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=replicate_mapper,
            cache_file_name=get_cache_file_name(tensor_cache_path, f"k_norm.rotary.weight{tp_suffix}"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if is_global and is_context_parallel
        else None
    )

    return AttentionWeights(
        wqkv=wqkv,
        o_proj=o_proj,
        q_norm_weight=q_norm_weight,
        k_norm_weight=k_norm_weight,
        k_norm_rotary_weight=k_norm_rotary_weight,
        is_global=is_global,
        kv_replicated=kv_replicated,
        wqk=wqk,
    )
