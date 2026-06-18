# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import torch

import ttnn
from models.demos.minimax_m3.config import MeshConfig
from models.demos.minimax_m3.utils.general_utils import get_cache_file_name
from models.demos.minimax_m3.utils.substate import substate

from .config import AttentionConfig


@dataclass(frozen=True)
class AttentionWeights:
    """Container for attention weight tensors - immutable after creation.

    MiniMax-M2 has no q/k/v/o projection biases and no attention sinks, so only the
    two projection weights are stored, plus the QK-norm gains.
    """

    wqkv: ttnn.Tensor
    o_proj: ttnn.Tensor
    # QK-norm weights (full-width RMSNorm gains), sharded column-parallel to match
    # the Q/K projection sharding. None when use_qk_norm is False / no state_dict.
    q_norm: ttnn.Tensor | None = None
    k_norm: ttnn.Tensor | None = None


def load_attention_weights(
    mesh_device,
    config: AttentionConfig,
    state_dict,
    mesh_config: MeshConfig,
    weight_dtype=ttnn.bfloat8_b,
    bias_dtype=ttnn.bfloat16,
    tensor_cache_path=None,
) -> AttentionWeights:
    """
    Load and shard attention weights.

    Args:
        mesh_device: TTNN mesh device
        config: Attention configuration
        state_dict: State dictionary containing weights
        mesh_config: Mesh parallelization config
        weight_dtype: Data type for weights (default: bfloat8_b)
        bias_dtype: Data type for biases (default: bfloat16)
        tensor_cache_path: Optional path for weight caching

    Returns:
        AttentionWeights container with all loaded weights
    """

    # Compute o_proj padding size based on config/mesh (independent of state_dict)
    hidden_size = config.hidden_size
    local_hidden = hidden_size // mesh_config.tp
    padded_local_hidden = ((local_hidden + 31) // 32) * 32  # Round up to tile boundary
    o_proj_pad_size = padded_local_hidden - local_hidden
    o_proj_cache_suffix = f"_padded" if o_proj_pad_size > 0 and mesh_config.tp > 1 else ""

    if state_dict:
        # Extract projection weights from state dict
        q_proj_weight = substate(state_dict, "q_proj")["weight"]  # [num_heads * head_dim, hidden_size]
        k_proj_weight = substate(state_dict, "k_proj")["weight"]  # [num_kv_heads * head_dim, hidden_size]
        v_proj_weight = substate(state_dict, "v_proj")["weight"]  # [num_kv_heads * head_dim, hidden_size]

        o_proj = substate(state_dict, "o_proj")["weight"].transpose(-1, -2)

        # QK-norm gains (MiniMax-M2). Full-width RMSNorm over the whole Q (head_dim *
        # num_heads) and K (head_dim * num_kv_heads) projection output. Reshaped to
        # [1, 1, 1, width] so the column-parallel mapper shards them across TP the
        # same way as the Q/K projection columns.
        q_norm_w = substate(state_dict, "q_norm").get("weight") if "q_norm.weight" in state_dict else None
        k_norm_w = substate(state_dict, "k_norm").get("weight") if "k_norm.weight" in state_dict else None
        if q_norm_w is not None:
            q_norm_w = q_norm_w.reshape(1, 1, 1, -1)
        if k_norm_w is not None:
            k_norm_w = k_norm_w.reshape(1, 1, 1, -1)

        # Create fused QKV weight
        # Split Q, K, V across devices, then concatenate per device
        qkv_list = []
        for i in range(mesh_config.tp):
            # Chunk weights across tensor parallel dimension
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

        # Concatenate across devices: [hidden_size, total_qkv_dim]
        qkv_cat = torch.cat(qkv_list, dim=-1).unsqueeze(0).unsqueeze(0)  # [1, 1, hidden_size, total_qkv_dim]

        # MiniMax-M2 has no q/k/v/o projection biases and no attention sinks.

        # Pad o_proj output dimension for tile alignment in CCL operations.
        # Without padding, local_hidden = hidden_size / TP may not be tile-aligned,
        # causing CCL to do expensive Untilize->Pad->Tilize cycles internally.
        if o_proj_pad_size > 0 and mesh_config.tp > 1:
            # Pad the output dimension of o_proj weight: [input_dim, hidden_size] -> [input_dim, padded_hidden]
            # Each TP device's output goes from local_hidden to padded_local_hidden
            padded_hidden = padded_local_hidden * mesh_config.tp
            o_proj = torch.nn.functional.pad(o_proj, (0, padded_hidden - hidden_size), "constant", value=0.0)

    else:
        # If state_dict is not provided, create empty tensors for weights
        qkv_cat = None
        o_proj = None
        q_norm_w = None
        k_norm_w = None

    # Clean mesh mapping using MeshConfig
    col_mesh_mapper = mesh_config.column_parallel(mesh_device)
    row_mesh_mapper = mesh_config.row_parallel(mesh_device)

    # Load QKV weight
    wqkv = ttnn.as_tensor(
        qkv_cat,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=weight_dtype,
        mesh_mapper=col_mesh_mapper,
        cache_file_name=get_cache_file_name(tensor_cache_path, "wqkv"),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    o_proj_tt = ttnn.as_tensor(
        o_proj,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=weight_dtype,
        mesh_mapper=row_mesh_mapper,
        cache_file_name=get_cache_file_name(tensor_cache_path, f"o_proj{o_proj_cache_suffix}"),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # QK-norm gains, sharded column-parallel (same axis as Q/K projection columns)
    # so each TP device holds the gains for its local Q/K slice. Kept in bfloat16.
    q_norm_tt = (
        ttnn.as_tensor(
            q_norm_w,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=col_mesh_mapper,
            cache_file_name=get_cache_file_name(tensor_cache_path, "q_norm"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if q_norm_w is not None
        else None
    )
    k_norm_tt = (
        ttnn.as_tensor(
            k_norm_w,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=col_mesh_mapper,
            cache_file_name=get_cache_file_name(tensor_cache_path, "k_norm"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if k_norm_w is not None
        else None
    )

    return AttentionWeights(
        wqkv=wqkv,
        o_proj=o_proj_tt,
        q_norm=q_norm_tt,
        k_norm=k_norm_tt,
    )
