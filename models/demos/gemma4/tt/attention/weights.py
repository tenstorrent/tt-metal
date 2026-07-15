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

from dataclasses import dataclass

import torch

import ttnn
from models.demos.gemma4.config import MeshConfig
from models.demos.gemma4.utils.general_utils import get_cache_file_name


@dataclass(frozen=True)
class AttentionWeights:
    """Container for attention weight tensors — immutable after creation."""

    wqkv: ttnn.Tensor  # Fused Q+K+V per TP device, column-parallel sharded
    o_proj: ttnn.Tensor  # Row-parallel sharded
    q_norm_weight: ttnn.Tensor  # Replicated across devices
    k_norm_weight: ttnn.Tensor  # Replicated across devices
    is_global: bool  # Controls K=V tying and partial RoPE
    kv_replicated: bool = False  # True when KV heads are replicated (not split) across TP devices
    # Phase 2a: optional persistent DRAM-width-sharded copies used only by decode
    # (peak-BW weight reads). None => interleaved decode. See tt/dram_sharded.py.
    wqkv_ds: object = None  # DramShardedMatmul | None
    o_proj_ds: object = None  # DramShardedMatmul | None


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
    """
    is_global = config.use_kv_tying
    q_size = config.num_attention_heads * config.head_dim
    kv_size = config.num_key_value_heads * config.head_dim
    tp = mesh_config.tp

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
    tp_suffix = f"_tp{tp}" if tp > 1 else ""
    # Tag the wqkv / o_proj cache filenames with their dtype so flipping
    # ``attention`` precision in precision_overrides.json doesn't reuse a
    # stale cached tensor at the previous dtype. q_norm / k_norm stay at
    # bfloat16 (no override) and don't need the suffix.
    from models.demos.gemma4.tt.precision import dtype_to_str

    dtype_suffix = f"_{dtype_to_str(weight_dtype)}"

    wqkv = ttnn.as_tensor(
        qkv,
        device=mesh_device,
        dtype=weight_dtype,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=col_mapper,
        cache_file_name=get_cache_file_name(tensor_cache_path, f"wqkv{tp_suffix}{dtype_suffix}"),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    o_proj = ttnn.as_tensor(
        o_w,
        device=mesh_device,
        dtype=weight_dtype,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=row_mapper,
        cache_file_name=get_cache_file_name(tensor_cache_path, f"o_proj{o_proj_cache_suffix}{tp_suffix}{dtype_suffix}"),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
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

    # Phase 2a: build DRAM-width-sharded decode copies of wqkv / o_proj (peak-BW
    # weight reads). Opt-in; interleaved weights above are untouched so prefill is
    # unchanged. Dims are per-device and depend on the layer type (sliding head_dim
    # vs global head_dim), so derive them from config rather than assuming square.
    from models.demos.gemma4.tt.dram_sharded import DramShardedMatmul, env_flag

    wqkv_ds = None
    o_proj_ds = None
    if env_flag("GEMMA4_DRAM_SHARDED_ATTN", "GEMMA4_DRAM_SHARDED"):
        head_dim = config.head_dim
        q_per_dev = (config.num_attention_heads // tp) * head_dim
        kv_per_dev = head_dim if kv_replicated else (config.num_key_value_heads // tp) * head_dim
        qkv_n = q_per_dev + 2 * kv_per_dev  # fused Q+K+V width on this device
        o_k = q_per_dev  # concat-heads width on this device = (heads/tp) * head_dim
        o_n = (padded_local_hidden * tp) if (o_proj_pad_size > 0 and tp > 1) else hidden_size
        wqkv_ds = DramShardedMatmul.try_build(mesh_device, wqkv, k=hidden_size, n=qkv_n, name=f"wqkv[{tp=}]")
        o_proj_ds = DramShardedMatmul.try_build(mesh_device, o_proj, k=o_k, n=o_n, name=f"o_proj[{tp=}]")

    return AttentionWeights(
        wqkv=wqkv,
        o_proj=o_proj,
        q_norm_weight=q_norm_weight,
        k_norm_weight=k_norm_weight,
        is_global=is_global,
        kv_replicated=kv_replicated,
        wqkv_ds=wqkv_ds,
        o_proj_ds=o_proj_ds,
    )
