# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import torch

import ttnn
from models.demos.gpt_oss.config import MeshConfig
from models.demos.gpt_oss.utils.general_utils import get_cache_file_name
from models.demos.gpt_oss.utils.substate import substate

from .config import AttentionConfig


@dataclass(frozen=True)
class AttentionWeights:
    """Container for attention weight tensors - immutable after creation"""

    wqkv: ttnn.Tensor
    wqkv_bias: ttnn.Tensor
    o_proj: ttnn.Tensor
    o_proj_bias: ttnn.Tensor
    decode_sinks: ttnn.Tensor
    sinks: ttnn.Tensor


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
    # Extract projection weights from state dict
    q_proj_weight = substate(state_dict, "q_proj")["weight"]  # [num_heads * head_dim, hidden_size]
    k_proj_weight = substate(state_dict, "k_proj")["weight"]  # [num_kv_heads * head_dim, hidden_size]
    v_proj_weight = substate(state_dict, "v_proj")["weight"]  # [num_kv_heads * head_dim, hidden_size]

    # Calculate padded hidden size for row-sharding (must be divisible by num_rows * TILE_SIZE)
    hidden_size = config.hidden_size
    num_rows = mesh_config.mesh_shape[0]
    shard_chunk_size = num_rows * ttnn.TILE_SIZE  # e.g., 4 * 32 = 128
    if hidden_size % shard_chunk_size != 0:
        padded_hidden_size = ((hidden_size + shard_chunk_size - 1) // shard_chunk_size) * shard_chunk_size
    else:
        padded_hidden_size = hidden_size

    # Load and pad o_proj weights for 2D sharding
    o_proj = substate(state_dict, "o_proj")["weight"].transpose(-1, -2)  # [hidden_size, hidden_size]
    o_proj_bias = substate(state_dict, "o_proj")["bias"]  # [hidden_size]

    # Pad BOTH dimensions of o_proj to padded_hidden_size
    if padded_hidden_size > hidden_size:
        padding_size = padded_hidden_size - hidden_size
        assert padding_size % num_rows == 0, "Expert down_proj weight padding size must be divisible by number of rows"
        local_padding_size = padding_size // num_rows  # (padded_hidden_size - hidden_size) / num_rows

        assert hidden_size % num_rows == 0, "Hidden size must be divisible by number of rows"
        local_hidden_size = hidden_size // num_rows
        # Pad both dimensions: [hidden_size, hidden_size] -> [padded_hidden_size, padded_hidden_size]
        o_proj_old = torch.nn.functional.pad(o_proj, (0, padding_size, 0, 0), value=0.0)
        o_proj_sliced = [o_proj[:, i*local_hidden_size:(i+1)*local_hidden_size] for i in range(num_rows)]
        o_proj = torch.cat([torch.nn.functional.pad(o, (0, local_padding_size, 0, 0), value=0.0) for o in o_proj_sliced], dim=-1)
        # Pad bias: [hidden_size] -> [padded_hidden_size]
        o_proj_bias_old = torch.nn.functional.pad(o_proj_bias, (0, padding_size), value=0.0)
        o_proj_bias_sliced = [o_proj_bias[i*local_hidden_size:(i+1)*local_hidden_size] for i in range(num_rows)]
        o_proj_bias = torch.cat([torch.nn.functional.pad(d, (0, local_padding_size), value=0.0) for d in o_proj_bias_sliced], dim=-1)

    print("o_proj_bias", o_proj.shape, o_proj_bias.shape)
    # Create fused QKV weight for 2D sharding
    # For 2D sharding: don't pre-chunk, let mesh_mapper handle distribution
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
    qkv_cat = torch.cat(qkv_list, dim=-1)  # [hidden_size, total_qkv_dim]

    # Pad input dimension (hidden_size) to padded_hidden_size for row-sharding
    if padded_hidden_size > hidden_size:
        padding_size = padded_hidden_size - hidden_size
        qkv_cat_old = torch.nn.functional.pad(qkv_cat, (0, 0, 0, padding_size), value=0.0)
        assert padding_size % num_rows == 0, "QKV weight padding size must be divisible by number of rows"
        local_padding_size = padding_size // num_rows  # (padded_hidden_size - hidden_size) / num_rows

        assert hidden_size % num_rows == 0, "Hidden size must be divisible by number of rows"
        local_hidden_size = hidden_size // num_rows
        qkv_cat_sliced = [qkv_cat[i*local_hidden_size:(i+1)*local_hidden_size, :] for i in range(num_rows)]
        qkv_cat = torch.cat([torch.nn.functional.pad(d, (0, 0, 0, local_padding_size), value=0.0) for d in qkv_cat_sliced], dim=0)
        # Pad first dimension: [hidden_size, total_qkv_dim] -> [padded_hidden_size, total_qkv_dim]

    # Add batch dimensions: [padded_hidden_size, total_qkv_dim] -> [1, 1, padded_hidden_size, total_qkv_dim]
    qkv_cat = qkv_cat.unsqueeze(0).unsqueeze(0)
    print("qkv_cat", qkv_cat.shape)

    # 2D sharding mappers
    # QKV: input dim (hidden) across ROWS, output dim (qkv) across COLUMNS
    qkv_2d_mapper = mesh_config.attention_2d_qkv(mesh_device)
    # WO: input dim (hidden) across COLUMNS, output dim (hidden) across ROWS
    wo_2d_mapper = mesh_config.attention_2d_wo(mesh_device)

    # For biases and other 1D-sharded tensors
    col_mesh_mapper = mesh_config.column_parallel(mesh_device)
    row_mesh_mapper = mesh_config.row_parallel(mesh_device)

    # Load QKV weight with 2D sharding
    wqkv = ttnn.as_tensor(
        qkv_cat,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=weight_dtype,
        mesh_mapper=qkv_2d_mapper,
        cache_file_name=get_cache_file_name(tensor_cache_path, "wqkv_2d"),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Handle biases - column-sharded for output dimension
    q_proj_bias = substate(state_dict, "q_proj")["bias"]
    k_proj_bias = substate(state_dict, "k_proj")["bias"]
    v_proj_bias = substate(state_dict, "v_proj")["bias"]

    qkv_bias_list = []
    for i in range(mesh_config.tp):
        q_bias_selected = torch.chunk(q_proj_bias, mesh_config.tp, dim=0)[i]
        k_bias_selected = torch.chunk(k_proj_bias, mesh_config.tp, dim=0)[i]
        v_bias_selected = torch.chunk(v_proj_bias, mesh_config.tp, dim=0)[i]
        qkv_bias = torch.cat([q_bias_selected, k_bias_selected, v_bias_selected], dim=-1)
        qkv_bias_list.append(qkv_bias)

    qkv_bias_cat = torch.cat(qkv_bias_list, dim=-1)  # [total_qkv_dim]

    wqkv_bias = ttnn.as_tensor(
        qkv_bias_cat,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=bias_dtype,
        mesh_mapper=col_mesh_mapper,  # Column-sharded on output dimension
        cache_file_name=get_cache_file_name(tensor_cache_path, "wqkv_bias"),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Attention sinks (GPT-OSS specific feature)
    sinks = state_dict["sinks"].reshape(1, config.num_heads, 1, 1)
    decode_sinks = torch.nn.functional.pad(
        sinks.view(-1, 1), (0, ttnn.TILE_SIZE - sinks.shape[-1]), "constant", value=0.0
    )
    decode_sinks /= config.scaling

    # Output projection with 2D sharding
    num_rows = mesh_config.mesh_shape[0]

    # Handle bias for 2D sharding
    # After WO matmul + allreduce_cols, output is: row-sharded, column-replicated
    # Bias should match: row-sharded (already padded_hidden_size), column-replicated
    #
    # For 2D mesh: shard dim=-1 (padded_hidden_size) across mesh axis 0 (rows)
    # For 1D mesh: column-parallel sharding (TP dimension)
    if num_rows > 1:
        # 2D mesh: shard along rows
        # dims=(-1, None) means: shard tensor dim -1 on mesh axis 0 (rows), replicate on mesh axis 1 (cols)
        o_bias_mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_config.mesh_shape, dims=(-1, None))
    else:
        # 1D mesh: pad bias by TP factor and use column-parallel sharding
        if mesh_config.tp > 1:
            o_proj_bias = torch.cat([o_proj_bias] + [torch.zeros_like(o_proj_bias)] * (mesh_config.tp - 1), dim=-1)
        o_bias_mapper = col_mesh_mapper

    o_proj_tt = ttnn.as_tensor(
        o_proj.unsqueeze(0).unsqueeze(0),  # [1, 1, hidden_size, hidden_size]
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=weight_dtype,
        mesh_mapper=wo_2d_mapper,  # 2D sharding: input across cols, output across rows
        cache_file_name=get_cache_file_name(tensor_cache_path, "o_proj_2d"),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    o_proj_bias_tt = ttnn.as_tensor(
        o_proj_bias,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=bias_dtype,
        mesh_mapper=o_bias_mapper,  # Choose based on mesh shape
        cache_file_name=get_cache_file_name(tensor_cache_path, "o_proj_bias"),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    decode_sinks_tt = ttnn.as_tensor(
        decode_sinks,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=bias_dtype,
        mesh_mapper=mesh_config.row_parallel(mesh_device),
        cache_file_name=get_cache_file_name(tensor_cache_path, "decode_sinks"),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    sinks_tt = ttnn.as_tensor(
        sinks,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=bias_dtype,
        mesh_mapper=mesh_config.sequence_parallel(mesh_device),
        cache_file_name=get_cache_file_name(tensor_cache_path, "sinks"),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    return AttentionWeights(
        wqkv=wqkv,
        wqkv_bias=wqkv_bias,
        o_proj=o_proj_tt,
        o_proj_bias=o_proj_bias_tt,
        decode_sinks=decode_sinks_tt,
        sinks=sinks_tt,
    )
