# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Weight loading for Gemma4 attention with tensor parallelism support.

Uses HF-style weight format (no Meta permutation needed since we use
ttnn.experimental.rotary_embedding instead of rotary_embedding_llama).

TP sharding (following gpt-oss pattern):
- QKV: column-parallel (shard output dim across TP devices)
- O_proj: row-parallel (shard input dim across TP devices)
- Norm weights: replicated across all devices
- Allreduce after O_proj recombines results
"""

import os
from dataclasses import dataclass, replace
from typing import Union

import torch

import ttnn
from models.demos.gemma4.config import MeshConfig
from models.demos.gemma4.tt.decode_matmul import DecodeLinear, flag, pipeline_row_for
from models.demos.gemma4.tt.dram_sharded import DramShardedLinear, can_dram_shard
from models.demos.gemma4.tt.sdpa_pair_split import (
    log_once,
    pair_heads,
    pair_o_proj_weight,
    pair_qkv_weight,
    pair_selector,
    pair_split_enabled,
    pair_split_reason,
)
from models.demos.gemma4.utils.general_utils import get_cache_file_name

# DRAM-width-sharded QKV / O-proj decode matmuls (same size as the interleaved
# weight → no memory cost). On by default for tp>1; GEMMA4_ATTN_DRAM_SHARD=0
# falls back to plain interleaved matmuls.
_DRAM_SHARD_ATTN = os.environ.get("GEMMA4_ATTN_DRAM_SHARD", "1") != "0"
# GEMMA4_ATTN_DECODE_1D=1: explicit 1D-multicast decode configs on the interleaved wqkv / o_proj matmuls (the
# MoE path), K streamed in 4-tile blocks. Measured on a P150 (26B-A4B TP=4, trace replay): sliding wqkv
# 53 -> 34 us, global wqkv 65 -> 47 us, o_proj 25 -> 18 us (sliding) / 46 -> 34 us (global), same HiFi2 numerics.
_ATTN_DECODE_BLOCK_W = int(os.environ.get("GEMMA4_ATTN_DECODE_BLOCK_W", "4"))  # env override for the K-block sweep


@dataclass(frozen=True)
class AttentionWeights:
    """Container for attention weight tensors — immutable after creation."""

    wqkv: Union[ttnn.Tensor, DramShardedLinear, DecodeLinear]  # Fused Q+K+V, column-parallel
    o_proj: Union[ttnn.Tensor, DramShardedLinear, DecodeLinear]  # Row-parallel sharded
    q_norm_weight: ttnn.Tensor  # Replicated across devices
    k_norm_weight: ttnn.Tensor  # Replicated across devices
    is_global: bool  # Controls K=V tying and partial RoPE
    kv_replicated: bool = False  # True when KV heads are replicated (not split) across TP devices
    # The q / k norm gammas as TILE [1, 1, 1, head_dim] rows (the form custom_ops/qkv_norm_rope reads), built once
    # here from the ROW_MAJOR weights above and owned by this container for the model's lifetime.
    q_norm_tile: ttnn.Tensor = None
    k_norm_tile: ttnn.Tensor = None
    # GEMMA4_SDPA_PAIR_SPLIT (tt/sdpa_pair_split.py): the decode view of a qualifying global layer -- this container
    # with the pair QKV / o_proj (the q heads of the chip's whole KV group) and ``pair_selector``, the per-chip
    # [1, tp, 1, 1] fp32 row picking ``lse_self - lse_partner`` out of the gathered LSEs. None: the 4-head path only.
    pair: "AttentionWeights" = None
    pair_selector: ttnn.Tensor = None


def load_attention_weights(
    mesh_device,
    config,
    state_dict,
    mesh_config: MeshConfig,
    weight_dtype=ttnn.bfloat16,
    tensor_cache_path=None,
    allow_pair_split=True,
) -> AttentionWeights:
    """
    Load and fuse attention weights with tensor parallelism.

    No Meta-format conversion needed — uses HF-style rotary_embedding.

    ``allow_pair_split``: False keeps a layer off ``GEMMA4_SDPA_PAIR_SPLIT`` whatever the knob (the MTP drafter, whose
    layers are KV-shared and never take that decode path, builds no pair copies).
    """
    is_global = config.use_kv_tying
    tp = mesh_config.tp
    fractured = bool(getattr(mesh_config, "weight_fracture", False))
    if fractured and tp != mesh_config.mesh_shape[0]:
        raise NotImplementedError("weight_fracture expects tp == mesh rows (heads over axis 0)")

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
            qkv = torch.cat(qkv_list, dim=-1).unsqueeze(0).unsqueeze(0)
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

        # Output projection
        o_w = state_dict["o_proj.weight"].transpose(-2, -1).unsqueeze(0).unsqueeze(0)
        if o_proj_pad_size > 0 and tp > 1:
            padded_hidden = padded_local_hidden * tp
            o_w = torch.nn.functional.pad(o_w, (0, padded_hidden - hidden_size), "constant", 0.0)

        # Per-head norm weights: [head_dim] -> [1, 1, head_dim/TILE_SIZE, TILE_SIZE]
        q_norm_w = state_dict["q_norm.weight"].reshape(1, 1, -1, ttnn.TILE_SIZE)
        k_norm_w = state_dict["k_norm.weight"].reshape(1, 1, -1, ttnn.TILE_SIZE)
    else:
        qkv = None
        o_w = None
        q_norm_w = None
        k_norm_w = None

    # Mesh mappers. Under weight_fracture attention still shards heads over tp_axis
    # and replicates across the other axis, so the plain-TP mappers apply.
    if tp > 1 or fractured:
        col_mapper = mesh_config.column_parallel(mesh_device)
        row_mapper = mesh_config.row_parallel(mesh_device)
        replicate_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    else:
        col_mapper = None
        row_mapper = None
        replicate_mapper = None

    o_proj_cache_suffix = "_padded" if o_proj_pad_size > 0 and tp > 1 else ""
    tp_suffix = f"_tp{tp}" if (tp > 1 or fractured) else ""
    # Tag the wqkv / o_proj cache filenames with their dtype so flipping
    # ``attention`` precision in precision_overrides.json doesn't reuse a
    # stale cached tensor at the previous dtype. q_norm / k_norm stay at
    # bfloat16 (no override) and don't need the suffix.
    from models.demos.gemma4.tt.precision import dtype_to_str

    dtype_suffix = f"_{dtype_to_str(weight_dtype)}"

    # Per-device fused-QKV output width and o_proj input width (concat_heads out).
    num_local_heads = config.num_attention_heads // tp
    head_dim = config.head_dim
    q_per_dev = num_local_heads * head_dim
    kv_per_dev = head_dim if kv_replicated else (config.num_key_value_heads // tp) * head_dim
    qkv_n = q_per_dev + 2 * kv_per_dev
    oproj_k = q_per_dev
    oproj_n = hidden_size

    # MoE (26B-A4B): sharded QKV/O-proj decode matmuls regress layer-decode PCC
    # on BH 1x4 (~0.93 vs 0.99). Dense models keep the opt.
    is_moe = bool(getattr(config, "enable_moe_block", False))
    # GEMMA4_DRAM_SHARD_TP1=1: the DRAM-sharded decode path on a single chip as well (the layer lab builds one
    # TP shard of the serving mesh on one chip, tests/unit/test_layer_lab.py).
    lab_tp1 = os.environ.get("GEMMA4_DRAM_SHARD_TP1", "0") == "1"
    dram_shard = _DRAM_SHARD_ATTN and (tp > 1 or lab_tp1) and not is_moe and not fractured
    decode_1d = flag("GEMMA4_ATTN_DECODE_1D")
    qkv_cache = get_cache_file_name(tensor_cache_path, f"wqkv{tp_suffix}{dtype_suffix}")
    oproj_cache = get_cache_file_name(tensor_cache_path, f"o_proj{o_proj_cache_suffix}{tp_suffix}{dtype_suffix}")
    qkv_cache_ws = (qkv_cache + ".ws") if qkv_cache else None
    oproj_cache_ws = (oproj_cache + ".ws") if oproj_cache else None

    if dram_shard and can_dram_shard(hidden_size, qkv_n, dtype=weight_dtype):
        wqkv = DramShardedLinear(
            qkv,
            mesh_device,
            col_mapper,
            k=hidden_size,
            n=qkv_n,
            dtype=weight_dtype,
            cache_file_name=qkv_cache_ws,
            pipeline_row=pipeline_row_for(mesh_device, hidden_size, "qkv", tp),
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
        if decode_1d:
            # Per-device K / N from the device tensor (the mesh tensor reports its local shape).
            wqkv = DecodeLinear(
                wqkv,
                mesh_device,
                k=int(wqkv.shape[-2]),
                n=int(wqkv.shape[-1]),
                enabled=True,
                max_block_w=_ATTN_DECODE_BLOCK_W,
            )

    if dram_shard and o_proj_pad_size == 0 and can_dram_shard(oproj_k, oproj_n, dtype=weight_dtype):
        o_proj = DramShardedLinear(
            o_w,
            mesh_device,
            row_mapper,
            k=oproj_k,
            n=oproj_n,
            dtype=weight_dtype,
            cache_file_name=oproj_cache_ws,
            pipeline_row=pipeline_row_for(mesh_device, hidden_size, "out_proj", tp),
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
        if decode_1d:
            o_proj = DecodeLinear(
                o_proj,
                mesh_device,
                k=int(o_proj.shape[-2]),
                n=int(o_proj.shape[-1]),
                enabled=True,
                max_block_w=_ATTN_DECODE_BLOCK_W,
            )
    q_norm_weight = ttnn.as_tensor(
        q_norm_w,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=replicate_mapper,
        cache_file_name=get_cache_file_name(tensor_cache_path, f"q_norm.weight{tp_suffix}"),
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

    def _norm_tile(w):
        return ttnn.to_layout(ttnn.reshape(w, (1, 1, 1, config.head_dim)), ttnn.TILE_LAYOUT)

    weights = AttentionWeights(
        wqkv=wqkv,
        o_proj=o_proj,
        q_norm_weight=q_norm_weight,
        k_norm_weight=k_norm_weight,
        is_global=is_global,
        kv_replicated=kv_replicated,
        q_norm_tile=_norm_tile(q_norm_weight),
        k_norm_tile=_norm_tile(k_norm_weight),
    )
    if allow_pair_split and pair_split_enabled():
        reason = pair_split_reason(
            is_sliding=bool(config.is_sliding),
            num_q_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            tp=tp,
            mesh_shape=getattr(mesh_device, "shape", None),
            tp_axis=getattr(mesh_config, "tp_axis", 1),
            kv_replicated=kv_replicated,
            fractured=fractured,
        )
        if reason is None:
            weights = replace(
                weights,
                pair=_load_pair_weights(
                    weights,
                    mesh_device,
                    config,
                    state_dict,
                    mesh_config,
                    weight_dtype,
                    tensor_cache_path,
                    dram_shard=dram_shard,
                    decode_1d=decode_1d,
                    col_mapper=col_mapper,
                    row_mapper=row_mapper,
                    o_proj_pad_size=o_proj_pad_size,
                    padded_local_hidden=padded_local_hidden,
                    o_proj_cache_suffix=o_proj_cache_suffix,
                    tp_suffix=tp_suffix,
                    dtype_suffix=dtype_suffix,
                ),
            )
        elif not config.is_sliding:
            log_once(f"the global layers keep the 4-head decode attention: {reason}")
    return weights


def _load_pair_weights(
    weights,
    mesh_device,
    config,
    state_dict,
    mesh_config,
    weight_dtype,
    tensor_cache_path,
    *,
    dram_shard,
    decode_1d,
    col_mapper,
    row_mapper,
    o_proj_pad_size,
    padded_local_hidden,
    o_proj_cache_suffix,
    tp_suffix,
    dtype_suffix,
) -> AttentionWeights:
    """The ``GEMMA4_SDPA_PAIR_SPLIT`` decode view of a qualifying global layer: ``weights`` with the pair QKV / o_proj
    (tt/sdpa_pair_split.py) and the partner selector, built like the 4-head pair above (DramShardedLinear where the
    shard fits, else interleaved, with the same column / row mappers) under their own ``*_pair_*`` cache names.

    Per device column i the fused QKV of the G = Hq / Hkv q heads of KV group i // 2 (natural head order) plus that
    group's K and V (V = K under K=V tying), and the o_proj rows of the same G heads: the TP/2 per-chip shapes (31B:
    5376 x 5120 and 4096 x 5376, its TP=4 shapes, so the DRAM-sharded pipeline rows are the TP/2 ones). They sit beside
    the 4-head weights, which prefill, the spec verify and the MTP drafter keep using: (5376 x 5120 + 4096 x 5376)
    bfloat8_b = ~52 MB per global layer per chip, ~0.5 GB per chip over the 31B's 10 global layers."""
    tp = mesh_config.tp
    hidden_size = config.hidden_size
    head_dim = config.head_dim
    layout_kw = dict(
        num_q_heads=config.num_attention_heads, num_kv_heads=config.num_key_value_heads, head_dim=head_dim, tp=tp
    )
    group = pair_heads(config.num_attention_heads, config.num_key_value_heads)
    qkv_n = (group + 2) * head_dim
    oproj_k = group * head_dim
    if state_dict:
        k_w = state_dict["k_proj.weight"]
        v_w = k_w if weights.is_global else state_dict["v_proj.weight"]
        qkv = pair_qkv_weight(state_dict["q_proj.weight"], k_w, v_w, **layout_kw).unsqueeze(0).unsqueeze(0)
        o_w = pair_o_proj_weight(state_dict["o_proj.weight"].transpose(-2, -1), **layout_kw)
        if o_proj_pad_size > 0:
            o_w = torch.nn.functional.pad(o_w, (0, padded_local_hidden * tp - hidden_size), "constant", 0.0)
        o_w = o_w.unsqueeze(0).unsqueeze(0)
    else:
        qkv = None
        o_w = None
    qkv_cache = get_cache_file_name(tensor_cache_path, f"wqkv_pair{tp_suffix}{dtype_suffix}")
    oproj_cache = get_cache_file_name(tensor_cache_path, f"o_proj_pair{o_proj_cache_suffix}{tp_suffix}{dtype_suffix}")

    if dram_shard and can_dram_shard(hidden_size, qkv_n, dtype=weight_dtype):
        wqkv = DramShardedLinear(
            qkv,
            mesh_device,
            col_mapper,
            k=hidden_size,
            n=qkv_n,
            dtype=weight_dtype,
            cache_file_name=(qkv_cache + ".ws") if qkv_cache else None,
            pipeline_row=pipeline_row_for(mesh_device, hidden_size, "qkv", tp // 2),
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
        if decode_1d:
            wqkv = DecodeLinear(
                wqkv,
                mesh_device,
                k=int(wqkv.shape[-2]),
                n=int(wqkv.shape[-1]),
                enabled=True,
                max_block_w=_ATTN_DECODE_BLOCK_W,
            )

    if dram_shard and o_proj_pad_size == 0 and can_dram_shard(oproj_k, hidden_size, dtype=weight_dtype):
        o_proj = DramShardedLinear(
            o_w,
            mesh_device,
            row_mapper,
            k=oproj_k,
            n=hidden_size,
            dtype=weight_dtype,
            cache_file_name=(oproj_cache + ".ws") if oproj_cache else None,
            pipeline_row=pipeline_row_for(mesh_device, hidden_size, "out_proj", tp // 2),
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
        if decode_1d:
            o_proj = DecodeLinear(
                o_proj,
                mesh_device,
                k=int(o_proj.shape[-2]),
                n=int(o_proj.shape[-1]),
                enabled=True,
                max_block_w=_ATTN_DECODE_BLOCK_W,
            )

    # Device column c holds e_c - e_{c^1}: built once here, before any trace capture, and owned by the weights.
    selector = ttnn.from_torch(
        pair_selector(tp),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=mesh_config.shard_mapper(mesh_device, tensor_dim=1),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    log_once(f"pair-split decode attention on the global layers: {group} q heads + 1 KV head per chip")
    return replace(weights, wqkv=wqkv, o_proj=o_proj, pair_selector=selector)
