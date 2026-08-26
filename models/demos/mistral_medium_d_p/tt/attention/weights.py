# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Mistral-Medium-3.5 attention weight loading: fused TP-sharded QKV, row-parallel o_proj.

Simpler than the gpt-oss loader in three ways, all verified against ``MistralAttention``:
**no projection biases**, **no attention sinks**, **no QK-norm** — so the container holds exactly
two tensors.

Sharding on the (8,4) Blackhole Galaxy (TP=4 on the cols):

    wqkv    [1, 1, 12288, 14336]  column-parallel -> 3584 per chip = (24 Q + 2 K + 2 V) * 128
    o_proj  [12288, 12288]        row-parallel    -> K = 3072 per chip = 24 Q heads * 128

Both are tile-aligned (3584 = 112 tiles, 3072 = 96 tiles), so unlike gpt-oss (2880/8 = 360) the CCL
tile-alignment padding below is a no-op for the production shape. It is kept generic so a different
TP factor still works.
"""

from dataclasses import dataclass

import torch

import ttnn
from models.demos.mistral_medium_d_p.config import MeshConfig
from models.demos.mistral_medium_d_p.utils.general_utils import get_cache_file_name
from models.demos.mistral_medium_d_p.utils.substate import substate

from .config import AttentionConfig


@dataclass(frozen=True)
class AttentionWeights:
    """Container for attention weight tensors - immutable after creation.

    Mistral has bias-free projections, no sinks and no QK-norm, so this is just the fused QKV and
    the output projection.
    """

    wqkv: ttnn.Tensor
    o_proj: ttnn.Tensor


def load_attention_weights(
    mesh_device,
    config: AttentionConfig,
    state_dict,
    mesh_config: MeshConfig,
    weight_dtype=ttnn.bfloat8_b,
    tensor_cache_path=None,
) -> AttentionWeights:
    """Load and shard attention weights.

    Args:
        mesh_device: TTNN mesh device
        config: Attention configuration
        state_dict: ``q_proj.weight`` / ``k_proj.weight`` / ``v_proj.weight`` / ``o_proj.weight``.
            The q/k projections are expected ALREADY swizzled to Meta interleaved RoPE order by
            ``convert_hf_qkv_to_meta_format`` (see model_config.ModelArgs.load_state_dict).
            Empty dict -> cache-only load.
        mesh_config: Mesh parallelization config
        weight_dtype: Data type for weights (default: bfloat8_b)
        tensor_cache_path: Optional path for weight caching

    Returns:
        AttentionWeights container
    """
    hidden_size = config.hidden_size
    local_hidden = hidden_size // mesh_config.tp
    padded_local_hidden = ((local_hidden + 31) // 32) * 32
    o_proj_pad_size = padded_local_hidden - local_hidden
    o_proj_cache_suffix = "_padded" if o_proj_pad_size > 0 and mesh_config.tp > 1 else ""

    if state_dict:
        for k in ("q_proj", "k_proj", "v_proj", "o_proj"):
            sub = substate(state_dict, k)
            assert "weight" in sub, f"attention state_dict is missing {k}.weight"
            assert "bias" not in sub, (
                f"{k} carries a bias, but Mistral/Ministral3 projections are bias-free "
                "(MistralAttention builds every nn.Linear with bias=False). Refusing to silently "
                "drop it — check the checkpoint."
            )

        q_proj_weight = substate(state_dict, "q_proj")["weight"]  # [num_heads * head_dim, hidden]
        k_proj_weight = substate(state_dict, "k_proj")["weight"]  # [num_kv_heads * head_dim, hidden]
        v_proj_weight = substate(state_dict, "v_proj")["weight"]
        o_proj = substate(state_dict, "o_proj")["weight"].transpose(-1, -2)  # -> [n_q*hd, hidden]

        # Fused QKV: split Q, K and V across TP first, then concatenate per device, so device i holds
        # [Q_i | K_i | V_i] contiguously — the layout nlp_create_qkv_heads expects.
        qkv_list = []
        for i in range(mesh_config.tp):
            wq = torch.chunk(q_proj_weight, mesh_config.tp, dim=0)[i].transpose(-2, -1)
            wk = torch.chunk(k_proj_weight, mesh_config.tp, dim=0)[i].transpose(-2, -1)
            wv = torch.chunk(v_proj_weight, mesh_config.tp, dim=0)[i].transpose(-2, -1)
            qkv_list.append(torch.cat([wq, wk, wv], dim=-1))
        qkv_cat = torch.cat(qkv_list, dim=-1).unsqueeze(0).unsqueeze(0)

        if o_proj_pad_size > 0 and mesh_config.tp > 1:
            padded_hidden = padded_local_hidden * mesh_config.tp
            o_proj = torch.nn.functional.pad(o_proj, (0, padded_hidden - hidden_size), "constant", value=0.0)
    else:
        # Cache-only loading (empty state_dict): ttnn.as_tensor loads each tilized tensor from disk.
        qkv_cat = None
        o_proj = None

    wqkv = ttnn.as_tensor(
        qkv_cat,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=weight_dtype,
        mesh_mapper=mesh_config.column_parallel(mesh_device),
        cache_file_name=get_cache_file_name(tensor_cache_path, "wqkv"),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # o_proj is row-parallel: the contraction dim (n_q*head_dim) is sharded across TP, so each chip
    # produces a partial sum over the full hidden dim and the caller reduces.
    o_proj_tt = ttnn.as_tensor(
        o_proj,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=weight_dtype,
        mesh_mapper=mesh_config.row_parallel(mesh_device),
        cache_file_name=get_cache_file_name(tensor_cache_path, f"o_proj{o_proj_cache_suffix}"),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    return AttentionWeights(wqkv=wqkv, o_proj=o_proj_tt)
