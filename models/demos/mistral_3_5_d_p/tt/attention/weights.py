# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Mistral-Medium-3.5 attention weight loading: fused TP-sharded QKV weight and a tile-aligned o_proj.

Adapted from ``gpt_oss_d_p/tt/attention/weights.py``. What the donor carries and Mistral does not:

  * **no q/k/v/o biases** — ``Ministral3Attention`` builds all four projections with ``bias=False``
    (``attention_bias`` is absent from the config), so the fused ``wqkv_bias`` and the row-parallel
    ``o_proj_bias`` (with its replicate-to-first-shard trick) are both gone;
  * **no attention sinks** — no ``sinks`` tensor to load, scale-divide, or head-shard;
  * **no QK-norm**, **no MSA index branch**.

What transfers unchanged is the part this entry exists for: the per-device Q|K|V concat that makes
one column-parallel fused matmul out of three projections, o_proj as row-parallel with its
tile-alignment padding, and the cache-only (empty ``state_dict``) load path.

The q/k projections are expected ALREADY swizzled to Meta format (``convert_hf_qkv_to_meta_format``,
reached through ``tt/model_config.py``), because the on-device rope is the Meta-interleaved
``rotary_embedding_indexed``.

At the spec's TP=8: local Q = 96/8 = 12 heads (1536), local K = local V = 1 head (128), so the fused
per-device width is 1792 — tile-aligned, as is ``hidden_size/tp = 1536``, so the donor's o_proj pad
is a no-op here. It is kept because it costs nothing and a future TP would need it.
"""

from dataclasses import dataclass

import torch

import ttnn
from models.demos.mistral_3_5_d_p.tt.config import MeshConfig
from models.demos.mistral_3_5_d_p.utils.general_utils import get_cache_file_name
from models.demos.mistral_3_5_d_p.utils.substate import substate

from .config import AttentionConfig


@dataclass(frozen=True)
class AttentionWeights:
    """Container for attention weight tensors — immutable after creation.

    Two tensors only: Mistral's attention carries no bias and no sink, so the donor's five-field
    container collapses to the fused column-parallel QKV and the row-parallel output projection.
    """

    wqkv: ttnn.Tensor
    o_proj: ttnn.Tensor


def load_attention_weights(
    mesh_device,
    config: AttentionConfig,
    state_dict,
    mesh_config: MeshConfig,
    weight_dtype=None,
    tensor_cache_path=None,
) -> AttentionWeights:
    """
    Load and shard attention weights.

    Args:
        mesh_device: TTNN mesh device
        config: Attention configuration
        state_dict: State dict holding ``{q,k,v,o}_proj.weight`` (q/k already Meta-swizzled).
            An empty dict means cache-only loading.
        mesh_config: Mesh parallelization config
        weight_dtype: Data type for weights; defaults to the spec's ``dataformats.weights.attention``
        tensor_cache_path: Optional path for weight caching

    Returns:
        AttentionWeights container with the fused QKV and the output projection
    """
    from models.demos.mistral_3_5_d_p.spec import SPEC

    weight_dtype = SPEC.attention_weight_dtype if weight_dtype is None else weight_dtype

    # o_proj padding (config/mesh-derived). local_hidden = hidden_size/TP may not be tile-aligned,
    # which would force CCL Untilize->Pad->Tilize; pad to a tile boundary to avoid it. At the spec's
    # 12288/8 = 1536 this is already aligned, so pad_size is 0 and nothing is added.
    hidden_size = config.hidden_size
    local_hidden = hidden_size // mesh_config.tp
    padded_local_hidden = ((local_hidden + 31) // 32) * 32
    o_proj_pad_size = padded_local_hidden - local_hidden
    o_proj_cache_suffix = "_padded" if o_proj_pad_size > 0 and mesh_config.tp > 1 else ""

    if state_dict:
        q_proj_weight = substate(state_dict, "q_proj")["weight"]  # [num_heads * head_dim, hidden_size]
        k_proj_weight = substate(state_dict, "k_proj")["weight"]  # [num_kv_heads * head_dim, hidden_size]
        v_proj_weight = substate(state_dict, "v_proj")["weight"]  # [num_kv_heads * head_dim, hidden_size]
        o_proj = substate(state_dict, "o_proj")["weight"].transpose(-1, -2)

        # Fused QKV weight: chunk Q, K and V across the TP devices, then concatenate PER DEVICE so
        # one column-parallel matmul produces that device's q|k|v slab in the order
        # nlp_create_qkv_heads expects.
        qkv_list = []
        for i in range(mesh_config.tp):
            wq = torch.chunk(q_proj_weight, mesh_config.tp, dim=0)[i].transpose(-2, -1)
            wk = torch.chunk(k_proj_weight, mesh_config.tp, dim=0)[i].transpose(-2, -1)
            wv = torch.chunk(v_proj_weight, mesh_config.tp, dim=0)[i].transpose(-2, -1)
            qkv_list.append(torch.cat([wq, wk, wv], dim=-1))
        # Concatenate across devices: [1, 1, hidden_size, total_qkv_dim]
        qkv_cat = torch.cat(qkv_list, dim=-1).unsqueeze(0).unsqueeze(0)

        if o_proj_pad_size > 0 and mesh_config.tp > 1:
            padded_hidden = padded_local_hidden * mesh_config.tp
            o_proj = torch.nn.functional.pad(o_proj, (0, padded_hidden - hidden_size), "constant", value=0.0)
    else:
        # Cache-only loading (empty state_dict): pass None for every torch weight so ttnn.as_tensor
        # loads each tilized tensor from disk.
        qkv_cat = None
        o_proj = None

    col_mesh_mapper = mesh_config.column_parallel(mesh_device)
    row_mesh_mapper = mesh_config.row_parallel(mesh_device)

    # Fused QKV is column-parallel (heads sharded on the output/feature dim across TP).
    wqkv = ttnn.as_tensor(
        qkv_cat,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=weight_dtype,
        mesh_mapper=col_mesh_mapper,
        cache_file_name=get_cache_file_name(tensor_cache_path, "wqkv"),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # o_proj is row-parallel (input/contraction dim sharded across TP), closed by a TP all-reduce.
    o_proj_tt = ttnn.as_tensor(
        o_proj,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=weight_dtype,
        mesh_mapper=row_mesh_mapper,
        cache_file_name=get_cache_file_name(tensor_cache_path, f"o_proj{o_proj_cache_suffix}"),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    return AttentionWeights(wqkv=wqkv, o_proj=o_proj_tt)
