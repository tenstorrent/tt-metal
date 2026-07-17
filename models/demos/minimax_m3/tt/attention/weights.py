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

    MiniMax-M3 has no q/k/v/o projection biases and no attention sinks, so only the
    two projection weights are stored, plus the QK-norm gains.
    """

    wqkv: ttnn.Tensor
    o_proj: ttnn.Tensor
    # QK-norm weights (full-width RMSNorm gains), sharded column-parallel to match
    # the Q/K projection sharding. None when use_qk_norm is False / no state_dict.
    q_norm: ttnn.Tensor | None = None
    k_norm: ttnn.Tensor | None = None
    # MSA index branch (sparse layers 3-59 only; None for dense layers 0-2). index_q_proj is
    # column-parallel (4 index heads -> 1 per TP col); index_k_proj + both index norms are
    # REPLICATED (index_k is the single shared head, present on every TP col). See tt/attention/msa.py.
    index_q_proj: ttnn.Tensor | None = None
    index_k_proj: ttnn.Tensor | None = None
    index_q_norm: ttnn.Tensor | None = None
    index_k_norm: ttnn.Tensor | None = None


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

        # QK-norm gains (MiniMax-M3): per-head RMSNorm over head_dim, so the gain is a
        # single [head_dim] vector shared across all heads (replicated, NOT TP-sharded).
        # Gemma (1 + w) is folded in here (fp32, pre-bf16 cast) when use_gemma_norm, exactly
        # like tt/rms_norm.py; reshaped to (1, 1, head_dim/32, 32) ROW_MAJOR to match the
        # weight layout ttnn.rms_norm expects (see RMSNorm class / apply_qk_norm_per_head).
        q_norm_w = substate(state_dict, "q_norm").get("weight") if "q_norm.weight" in state_dict else None
        k_norm_w = substate(state_dict, "k_norm").get("weight") if "k_norm.weight" in state_dict else None

        def _prep_qk_norm_gain(w):
            if w is None:
                return None
            if config.use_gemma_norm:
                w = w.float() + 1.0
            return w.reshape(1, 1, -1, ttnn.TILE_SIZE)

        q_norm_w = _prep_qk_norm_gain(q_norm_w)
        k_norm_w = _prep_qk_norm_gain(k_norm_w)

        # MSA index branch (only present on sparse layers): index_q_proj [n_index*idx_dim, hidden],
        # index_k_proj [idx_dim, hidden] (1 shared head). Transpose for matmul; index_q is sharded
        # col-parallel (dim -1), index_k replicated. index_*_norm are per-head gains like q/k_norm.
        has_index = "index_q_proj.weight" in state_dict
        if has_index:
            index_q_proj_w = substate(state_dict, "index_q_proj")["weight"].transpose(-1, -2)
            index_q_proj_w = index_q_proj_w.unsqueeze(0).unsqueeze(0)  # [1,1,hidden, n_index*idx_dim]
            index_k_proj_w = substate(state_dict, "index_k_proj")["weight"].transpose(-1, -2)
            index_k_proj_w = index_k_proj_w.unsqueeze(0).unsqueeze(0)  # [1,1,hidden, idx_dim]
            index_q_norm_w = _prep_qk_norm_gain(substate(state_dict, "index_q_norm")["weight"])
            index_k_norm_w = _prep_qk_norm_gain(substate(state_dict, "index_k_norm")["weight"])
        else:
            index_q_proj_w = index_k_proj_w = index_q_norm_w = index_k_norm_w = None

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

        # MiniMax-M3 has no q/k/v/o projection biases and no attention sinks.

        # Pad o_proj output dimension for tile alignment in CCL operations.
        # Without padding, local_hidden = hidden_size / TP may not be tile-aligned,
        # causing CCL to do expensive Untilize->Pad->Tilize cycles internally.
        if o_proj_pad_size > 0 and mesh_config.tp > 1:
            # Pad the output dimension of o_proj weight: [input_dim, hidden_size] -> [input_dim, padded_hidden]
            # Each TP device's output goes from local_hidden to padded_local_hidden
            padded_hidden = padded_local_hidden * mesh_config.tp
            o_proj = torch.nn.functional.pad(o_proj, (0, padded_hidden - hidden_size), "constant", value=0.0)

    else:
        # Cache-only loading (empty state_dict): pass None for every torch weight so ttnn.as_tensor
        # loads each tilized tensor from disk. Which OPTIONAL weights to build (q/k-norm, MSA index
        # branch) can't be read from an absent state_dict, so decide from the attention config below.
        qkv_cat = None
        o_proj = None
        q_norm_w = None
        k_norm_w = None
        index_q_proj_w = index_k_proj_w = index_q_norm_w = index_k_norm_w = None

    # Whether to build the optional weights: from source presence when we have a state_dict, else
    # (cache-only) from the model config — use_qk_norm gates q/k-norm, is_sparse gates the MSA index
    # branch (only sparse layers cached those). torch weights stay None in cache-only mode -> cache load.
    build_qk_norm = (q_norm_w is not None) if state_dict else bool(config.use_qk_norm)
    build_index = (index_q_proj_w is not None) if state_dict else bool(config.is_sparse)

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

    # QK-norm gains (M3 per-head): a single [head_dim] gain replicated across all devices
    # (head_dim is not TP-sharded, so every device norms its local heads with the same gain).
    # ROW_MAJOR layout to match ttnn.rms_norm's weight expectation. Kept in bfloat16.
    replicate_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    q_norm_tt = (
        ttnn.as_tensor(
            q_norm_w,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=replicate_mapper,
            cache_file_name=get_cache_file_name(tensor_cache_path, "q_norm"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if build_qk_norm
        else None
    )
    k_norm_tt = (
        ttnn.as_tensor(
            k_norm_w,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=replicate_mapper,
            cache_file_name=get_cache_file_name(tensor_cache_path, "k_norm"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if build_qk_norm
        else None
    )

    # MSA index branch (sparse layers only): index_q_proj column-parallel (4 heads -> 1/col),
    # index_k_proj + norms replicated (index_k is the single shared head). Kept bf16.
    def _as_index(w, mapper, layout, name, dtype=ttnn.bfloat16):
        # build_index drives construction; w is None in cache-only mode -> loads from the cache.
        if not build_index:
            return None
        return ttnn.as_tensor(
            w,
            device=mesh_device,
            layout=layout,
            dtype=dtype,
            mesh_mapper=mapper,
            cache_file_name=get_cache_file_name(tensor_cache_path, name),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    index_q_proj_tt = _as_index(index_q_proj_w, col_mesh_mapper, ttnn.TILE_LAYOUT, "index_q_proj", weight_dtype)
    index_k_proj_tt = _as_index(index_k_proj_w, replicate_mapper, ttnn.TILE_LAYOUT, "index_k_proj", weight_dtype)
    index_q_norm_tt = _as_index(index_q_norm_w, replicate_mapper, ttnn.ROW_MAJOR_LAYOUT, "index_q_norm")
    index_k_norm_tt = _as_index(index_k_norm_w, replicate_mapper, ttnn.ROW_MAJOR_LAYOUT, "index_k_norm")

    return AttentionWeights(
        wqkv=wqkv,
        o_proj=o_proj_tt,
        q_norm=q_norm_tt,
        k_norm=k_norm_tt,
        index_q_proj=index_q_proj_tt,
        index_k_proj=index_k_proj_tt,
        index_q_norm=index_q_norm_tt,
        index_k_norm=index_k_norm_tt,
    )
