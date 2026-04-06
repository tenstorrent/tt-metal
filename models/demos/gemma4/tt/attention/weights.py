# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
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
            qkv_list = []
            for i in range(tp):
                wq_chunk = torch.chunk(q_w, tp, dim=0)[i].transpose(-2, -1)
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

    wqkv = ttnn.as_tensor(
        qkv,
        device=mesh_device,
        dtype=weight_dtype,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=col_mapper,
        cache_file_name=get_cache_file_name(tensor_cache_path, "wqkv"),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    o_proj = ttnn.as_tensor(
        o_w,
        device=mesh_device,
        dtype=weight_dtype,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=row_mapper,
        cache_file_name=get_cache_file_name(tensor_cache_path, f"o_proj{o_proj_cache_suffix}"),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    q_norm_weight = ttnn.as_tensor(
        q_norm_w,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=replicate_mapper,
        cache_file_name=get_cache_file_name(tensor_cache_path, "q_norm.weight"),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    k_norm_weight = ttnn.as_tensor(
        k_norm_w,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=replicate_mapper,
        cache_file_name=get_cache_file_name(tensor_cache_path, "k_norm.weight"),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    return AttentionWeights(
        wqkv=wqkv,
        o_proj=o_proj,
        q_norm_weight=q_norm_weight,
        k_norm_weight=k_norm_weight,
        is_global=is_global,
    )
