# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Llama 3.1 8B attention weight loading: fused TP-sharded QKV weight + row-parallel o_proj.

Copied from ``gpt_oss_d_p/tt/attention/weights.py``. What transfers is the per-TP-shard Q|K|V
interleave — the fused QKV weight is built by chunking q/k/v across TP *first* and concatenating
each device's slice, so that after column-parallel sharding a chip's fused block is exactly its own
[Q_local | K_local | V_local] and ``nlp_create_qkv_heads`` splits it correctly. Building the fused
weight the other way round (concat then shard) puts another chip's K in this chip's Q and fails as
degraded PCC, not as an error.

Llama-specific deletions from the donor, all following ``attention_bias: false`` and the spec's
empty ``attention.features``:

  * **no biases** — q/k/v/o are unbiased, so there is no fused QKV bias and no o_proj bias;
  * **no attention sinks** — nothing to pre-divide by ``config.scaling``;
  * **no o_proj tile padding** — gpt-oss needs it because 2880/8 = 360 is not tile-aligned;
    4096/4 = 1024 is, so the padding branch is dropped rather than left dormant.

q/k projection rows are expected already in **Meta format** (``reverse_permute``d) for the on-device
interleaved RoPE — done once at checkpoint load in ``tt/model_config.py``, not here.
"""

from dataclasses import dataclass

import torch

import ttnn
from models.demos.llama3_1_8b_d_p.tt.config import MeshConfig
from models.demos.llama3_1_8b_d_p.utils.general_utils import get_cache_file_name
from models.demos.llama3_1_8b_d_p.utils.substate import substate

from .config import AttentionConfig


@dataclass(frozen=True)
class AttentionWeights:
    """Container for attention weight tensors — immutable after creation.

    Only two tensors: Llama carries no projection biases and no attention sinks.
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
        state_dict: ``self_attn.*`` substate with q_proj/k_proj/v_proj/o_proj ``weight``. Empty dict
            => cache-only load.
        mesh_config: Mesh parallelization config
        weight_dtype: on-device weight dtype. Defaults to bfloat8_b, the accuracy-mode value
            tt_transformers uses for WQKV/WO; the spec's ``numerics.attn_weights`` says bfp4 and the
            two sources disagree (spec known_risks). bfp8 is the conservative default for bring-up;
            pass bfloat4_b to measure the spec value.
        tensor_cache_path: Optional path for weight caching
    """
    if state_dict:
        q_proj_weight = substate(state_dict, "q_proj")["weight"]  # [num_heads * head_dim, hidden]
        k_proj_weight = substate(state_dict, "k_proj")["weight"]  # [num_kv_heads * head_dim, hidden]
        v_proj_weight = substate(state_dict, "v_proj")["weight"]  # [num_kv_heads * head_dim, hidden]
        o_proj = substate(state_dict, "o_proj")["weight"].transpose(-1, -2)

        for name, w, rows in (
            ("q_proj", q_proj_weight, config.num_heads * config.head_dim),
            ("k_proj", k_proj_weight, config.num_kv_heads * config.head_dim),
            ("v_proj", v_proj_weight, config.num_kv_heads * config.head_dim),
        ):
            assert w.shape == (
                rows,
                config.hidden_size,
            ), f"{name} has shape {tuple(w.shape)}, expected {(rows, config.hidden_size)}"

        # Fused QKV: chunk each projection across TP FIRST, then concat per device, so each chip's
        # fused block is its own [Q_local | K_local | V_local].
        qkv_list = []
        for i in range(mesh_config.tp):
            wq = torch.chunk(q_proj_weight, mesh_config.tp, dim=0)[i].transpose(-2, -1)
            wk = torch.chunk(k_proj_weight, mesh_config.tp, dim=0)[i].transpose(-2, -1)
            wv = torch.chunk(v_proj_weight, mesh_config.tp, dim=0)[i].transpose(-2, -1)
            qkv_list.append(torch.cat([wq, wk, wv], dim=-1))
        qkv_cat = torch.cat(qkv_list, dim=-1).unsqueeze(0).unsqueeze(0)  # [1, 1, hidden, total_qkv]
    else:
        # Cache-only loading: pass None so ttnn.as_tensor loads each tilized tensor from disk.
        qkv_cat = None
        o_proj = None

    col_mesh_mapper = mesh_config.column_parallel(mesh_device)
    row_mesh_mapper = mesh_config.row_parallel(mesh_device)

    # QKV weight is column-parallel: heads shard on the output/feature dim across TP.
    wqkv = ttnn.as_tensor(
        qkv_cat,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=weight_dtype,
        mesh_mapper=col_mesh_mapper,
        cache_file_name=get_cache_file_name(tensor_cache_path, "wqkv"),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # o_proj is row-parallel: the input/contraction dim shards across TP. hidden/tp = 1024 is
    # tile-aligned for Llama, so (unlike gpt-oss) there is no alignment padding to add or strip.
    assert (config.hidden_size // mesh_config.tp) % ttnn.TILE_SIZE == 0, (
        f"hidden_size/tp ({config.hidden_size}/{mesh_config.tp}) is not tile-aligned; this package "
        f"dropped the donor's o_proj padding branch because Llama never needs it"
    )
    o_proj_tt = ttnn.as_tensor(
        o_proj,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=weight_dtype,
        mesh_mapper=row_mesh_mapper,
        cache_file_name=get_cache_file_name(tensor_cache_path, "o_proj"),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    return AttentionWeights(wqkv=wqkv, o_proj=o_proj_tt)
