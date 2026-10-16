# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Llama 3.1 8B attention weight loading: fused TP-sharded QKV weight (no bias), o_proj weight (no bias).
No attention sinks, no QK-norm.

IMPORTANT: Q and K weights are converted from HF convention to Meta convention for compatibility with
``ttnn.experimental.rotary_embedding_llama``. The Meta convention interleaves the rotary pairs:
    HF:   [d0, d1, ..., d_{dim/2-1}, d_{dim/2}, ..., d_{dim-1}]
    Meta: [d0, d_{dim/2}, d1, d_{dim/2+1}, ...] (interleaved pairs)
This permutation is applied to Q and K projection weights (but NOT V or O).
"""

from dataclasses import dataclass

import torch

import ttnn
from models.demos.llama_3_1_8b.tt.config import MeshConfig
from models.demos.llama_3_1_8b.utils.general_utils import get_cache_file_name
from models.demos.llama_3_1_8b.utils.substate import substate

from .config import AttentionConfig


@dataclass(frozen=True)
class AttentionWeights:
    """Container for attention weight tensors - immutable after creation.

    Llama 3.1 8B has no projection biases (attention_bias: false) and no attention sinks.
    """

    wqkv: ttnn.Tensor
    o_proj: ttnn.Tensor


def _convert_hf_to_meta_format(weight, head_dim):
    """Convert HF convention weights to Meta interleaved convention for RoPE compatibility.

    For each head, rearranges from:
        HF:   [d0, d1, ..., d_{dim/2-1}, d_{dim/2}, ..., d_{dim-1}]
    to:
        Meta: [d0, d_{dim/2}, d1, d_{dim/2+1}, ...] (interleaved pairs)

    Applied per-head to Q and K projection weights (first dim is num_heads * head_dim).
    """
    # weight shape: [num_heads * head_dim, hidden_size]
    num_heads = weight.shape[0] // head_dim
    # Reshape to [num_heads, head_dim, hidden_size]
    w = weight.reshape(num_heads, head_dim, -1)
    # Split head_dim into two halves: [num_heads, 2, head_dim//2, hidden_size]
    w = w.reshape(num_heads, 2, head_dim // 2, -1)
    # Interleave: [num_heads, head_dim//2, 2, hidden_size] -> [num_heads, head_dim, hidden_size]
    w = w.transpose(1, 2).contiguous().reshape(num_heads * head_dim, -1)
    return w


def load_attention_weights(
    mesh_device,
    config: AttentionConfig,
    state_dict,
    mesh_config: MeshConfig,
    weight_dtype=ttnn.bfloat8_b,
    tensor_cache_path=None,
) -> AttentionWeights:
    """
    Load and shard attention weights for Llama 3.1 8B.

    Args:
        mesh_device: TTNN mesh device
        config: Attention configuration
        state_dict: State dictionary containing weights (q_proj/k_proj/v_proj/o_proj weight only,
            no bias). None -> cache-only load.
        mesh_config: Mesh parallelization config
        weight_dtype: Data type for weights (default: bfloat8_b)
        tensor_cache_path: Optional path for weight caching

    Returns:
        AttentionWeights container with all loaded weights
    """
    hidden_size = config.hidden_size
    head_dim = config.head_dim

    if state_dict:
        # Extract projection weights from state dict
        q_proj_weight = substate(state_dict, "q_proj")["weight"]  # [num_heads * head_dim, hidden_size]
        k_proj_weight = substate(state_dict, "k_proj")["weight"]  # [num_kv_heads * head_dim, hidden_size]
        v_proj_weight = substate(state_dict, "v_proj")["weight"]  # [num_kv_heads * head_dim, hidden_size]

        # Convert Q and K from HF convention to Meta convention (interleaved pairs for RoPE)
        q_proj_weight = _convert_hf_to_meta_format(q_proj_weight, head_dim)
        k_proj_weight = _convert_hf_to_meta_format(k_proj_weight, head_dim)
        # V is NOT converted — it has no rotary embedding

        o_proj_weight = substate(state_dict, "o_proj")["weight"].transpose(-1, -2)

        # Create fused QKV weight.
        # Split Q, K, V across devices, then concatenate per device.
        # Q: 32 heads -> 8 per TP column; K/V: 8 heads -> 2 per TP column
        # Fused QKV per TP column: [4096, (8+2+2)*128] = [4096, 1536]
        qkv_list = []
        for i in range(mesh_config.tp):
            wq_selected = torch.chunk(q_proj_weight, mesh_config.tp, dim=0)[i]
            wk_selected = torch.chunk(k_proj_weight, mesh_config.tp, dim=0)[i]
            wv_selected = torch.chunk(v_proj_weight, mesh_config.tp, dim=0)[i]

            # Transpose for matmul: [hidden_size, local_dim]
            wq = wq_selected.transpose(-2, -1)
            wk = wk_selected.transpose(-2, -1)
            wv = wv_selected.transpose(-2, -1)

            # Concatenate Q, K, V: [hidden_size, local_q_dim + local_k_dim + local_v_dim]
            qkv = torch.cat([wq, wk, wv], dim=-1)
            qkv_list.append(qkv)

        # Concatenate across devices: [1, 1, hidden_size, total_qkv_dim]
        qkv_cat = torch.cat(qkv_list, dim=-1).unsqueeze(0).unsqueeze(0)

        # O proj: row-parallel [local_q_heads * head_dim, hidden_size] per TP column
        # For Llama with 8 Q heads per TP column: [8*128, 4096] = [1024, 4096]
        # No padding needed: local_hidden = 4096/4 = 1024, which is tile-aligned (1024/32 = 32)
    else:
        qkv_cat = None
        o_proj_weight = None

    # Clean mesh mapping using MeshConfig
    col_mesh_mapper = mesh_config.column_parallel(mesh_device)
    row_mesh_mapper = mesh_config.row_parallel(mesh_device)

    # Load QKV weight (column-parallel: heads sharded on the output/feature dim across TP)
    wqkv = ttnn.as_tensor(
        qkv_cat,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=weight_dtype,
        mesh_mapper=col_mesh_mapper,
        cache_file_name=get_cache_file_name(tensor_cache_path, "wqkv_meta"),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # o_proj is row-parallel (input/contraction dim sharded across TP)
    o_proj_tt = ttnn.as_tensor(
        o_proj_weight,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=weight_dtype,
        mesh_mapper=row_mesh_mapper,
        cache_file_name=get_cache_file_name(tensor_cache_path, "o_proj"),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    return AttentionWeights(
        wqkv=wqkv,
        o_proj=o_proj_tt,
    )
