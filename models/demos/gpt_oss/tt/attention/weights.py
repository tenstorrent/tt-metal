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

    o_proj = substate(state_dict, "o_proj")["weight"].transpose(-1, -2)
    o_proj_bias = substate(state_dict, "o_proj")["bias"]

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

    # Handle biases - create fused QKV bias
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
        mesh_mapper=col_mesh_mapper,
        cache_file_name=get_cache_file_name(tensor_cache_path, "wqkv_bias"),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Attention sinks (GPT-OSS specific feature)
    #
    # IMPORTANT: TT SDPA kernels apply `scale` inside the exp path for BOTH QK and sinks.
    # HF GPT-OSS behavior is: QK logits are scaled, sinks are NOT additionally scaled.
    # To match HF, we provide sinks in "pre-divided" form: sink_input = sink / scale,
    # so that the kernel's internal multiplication by `scale` yields the original sink values.
    sinks = state_dict["sinks"].reshape(1, config.num_heads, 1, 1)
    sinks_for_sdpa = sinks / config.scaling
    decode_sinks = torch.nn.functional.pad(
        sinks.view(-1, 1), (0, ttnn.TILE_SIZE - sinks.shape[-1]), "constant", value=0.0
    )
    decode_sinks /= config.scaling

    # Output projection
    # Pad o_proj output dimension for tile alignment in CCL operations.
    # Without padding, local_hidden = hidden_size / TP may not be tile-aligned (e.g., 2880/8 = 360),
    # causing CCL to do expensive Untilize->Pad->Tilize cycles internally.
    hidden_size = config.hidden_size
    local_hidden = hidden_size // mesh_config.tp
    padded_local_hidden = ((local_hidden + 31) // 32) * 32  # Round up to tile boundary
    o_proj_pad_size = padded_local_hidden - local_hidden

    if o_proj_pad_size > 0 and mesh_config.tp > 1:
        # Pad the output dimension of o_proj weight: [input_dim, hidden_size] -> [input_dim, padded_hidden]
        # Each TP device's output goes from local_hidden to padded_local_hidden
        padded_hidden = padded_local_hidden * mesh_config.tp
        o_proj = torch.nn.functional.pad(o_proj, (0, padded_hidden - hidden_size), "constant", value=0.0)
        # Pad bias similarly
        o_proj_bias = torch.nn.functional.pad(o_proj_bias, (0, padded_hidden - hidden_size), "constant", value=0.0)

    if mesh_config.tp > 1:
        o_proj_bias = torch.cat([o_proj_bias] + [torch.zeros_like(o_proj_bias)] * (mesh_config.tp - 1), dim=-1)

    # Use unique cache key when padding is applied
    o_proj_cache_suffix = f"_padded{padded_local_hidden}" if o_proj_pad_size > 0 and mesh_config.tp > 1 else ""
    o_proj_tt = ttnn.as_tensor(
        o_proj,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=weight_dtype,
        mesh_mapper=row_mesh_mapper,
        cache_file_name=get_cache_file_name(tensor_cache_path, f"o_proj{o_proj_cache_suffix}"),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    o_proj_bias_tt = ttnn.as_tensor(
        o_proj_bias,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=bias_dtype,
        mesh_mapper=col_mesh_mapper,
        cache_file_name=get_cache_file_name(tensor_cache_path, f"o_proj_bias{o_proj_cache_suffix}"),
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
        sinks_for_sdpa,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=bias_dtype,
        mesh_mapper=mesh_config.sequence_parallel(mesh_device),
        # Bump cache key since values are now pre-divided by `config.scaling`.
        cache_file_name=get_cache_file_name(tensor_cache_path, "sinks_div_scale"),
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
