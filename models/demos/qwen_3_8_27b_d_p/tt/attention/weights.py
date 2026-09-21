# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Attention weight loading, including the fused QKV+gate projection.

**Why the fusion needs a permutation.** Qwen3.5's ``q_proj`` is twice the usual width: per head it
emits ``[q(head_dim) | gate(head_dim)]``, interleaved head by head. ``nlp_create_qkv_heads`` wants
``[all q | all k | all v]`` contiguous, and a column-parallel shard hands TP rank ``r`` the ``r``-th
contiguous slice of the fused output. Both constraints are satisfied by one load-time row
permutation: lay the fused weight out **rank-block-major**, so rank ``r``'s slice is exactly

    [ its q heads | its k heads | its v heads | its gate heads ]

and an ordinary ``column_parallel`` mapper delivers it. Nothing at runtime has to know.

Taking ``q_proj.weight.shape[0] // head_dim`` as the head count — the natural reading for any other
GQA model — silently halves it here, which is why the head split is written out rather than derived.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

import ttnn

from ...config import MeshConfig
from ...reference.config import Qwen35TextConfig
from ...utils.general_utils import get_cache_file_name
from ...utils.substate import substate


def build_fused_qkvg_weight(
    q_proj: torch.Tensor,
    k_proj: torch.Tensor,
    v_proj: torch.Tensor,
    *,
    cfg: Qwen35TextConfig,
    tp: int,
) -> torch.Tensor:
    """-> ``[hidden, tp * (n_q_local + 2*n_kv_local + n_q_local) * head_dim]``, rank-block-major.

    Inputs are the HF ``[out, in]`` weights. The output is already transposed for ``ttnn.linear``.
    """
    hd = cfg.head_dim
    n_q, n_kv = cfg.num_attention_heads, cfg.num_key_value_heads
    assert n_q % tp == 0 and n_kv % tp == 0, f"tp={tp} must divide {n_q} q heads and {n_kv} kv heads"
    q_local, kv_local = n_q // tp, n_kv // tp
    hidden = q_proj.shape[1]

    # q_proj rows are grouped per head as [q(hd) | gate(hd)] — split that apart first.
    qg = q_proj.reshape(n_q, 2, hd, hidden)
    q_rows, gate_rows = qg[:, 0], qg[:, 1]
    k_rows = k_proj.reshape(n_kv, hd, hidden)
    v_rows = v_proj.reshape(n_kv, hd, hidden)

    blocks = []
    for r in range(tp):
        blocks.extend(
            [
                q_rows[r * q_local : (r + 1) * q_local].reshape(-1, hidden),
                k_rows[r * kv_local : (r + 1) * kv_local].reshape(-1, hidden),
                v_rows[r * kv_local : (r + 1) * kv_local].reshape(-1, hidden),
                gate_rows[r * q_local : (r + 1) * q_local].reshape(-1, hidden),
            ]
        )
    return torch.cat(blocks, dim=0).transpose(0, 1).contiguous()


@dataclass
class AttentionWeights:
    wqkvg: ttnn.Tensor  # fused, rank-block-major [hidden, per_rank_width * tp]
    o_proj: ttnn.Tensor  # row-parallel [n_q*head_dim, hidden]
    q_norm: ttnn.Tensor  # [head_dim] gain, replicated (Gemma fold applied at load)
    k_norm: ttnn.Tensor


def load_attention_weights(
    mesh_device,
    cfg: Qwen35TextConfig,
    state_dict: dict[str, torch.Tensor],
    *,
    mesh_config: MeshConfig,
    weight_dtype=ttnn.bfloat8_b,
    tensor_cache_path: Optional[str] = None,
) -> AttentionWeights:
    tp = mesh_config.tp

    fused = None
    o_w = None
    if state_dict:
        fused = (
            build_fused_qkvg_weight(
                substate(state_dict, "q_proj")["weight"],
                substate(state_dict, "k_proj")["weight"],
                substate(state_dict, "v_proj")["weight"],
                cfg=cfg,
                tp=tp,
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )
        # o_proj is row-parallel on its INPUT dim; the contiguous split matches the q-head split
        # above (rank r owns heads [r*q_local, (r+1)*q_local)), so no permutation is needed.
        o_w = substate(state_dict, "o_proj")["weight"].transpose(0, 1).unsqueeze(0).unsqueeze(0)

    wqkvg = ttnn.as_tensor(
        fused,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=weight_dtype,
        mesh_mapper=mesh_config.column_parallel(mesh_device),
        cache_file_name=get_cache_file_name(tensor_cache_path, "wqkvg"),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    o_proj = ttnn.as_tensor(
        o_w,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=weight_dtype,
        mesh_mapper=mesh_config.row_parallel(mesh_device),
        cache_file_name=get_cache_file_name(tensor_cache_path, "o_proj"),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    def _norm_gain(name: str) -> ttnn.Tensor:
        torch_w = None
        if state_dict:
            # The Gemma (1 + w) fold, in fp32 before the bf16 cast — same as tt/rms_norm.py.
            torch_w = (substate(state_dict, name)["weight"].float() + 1.0).reshape(1, 1, -1, ttnn.TILE_SIZE)
        return ttnn.as_tensor(
            torch_w,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            cache_file_name=get_cache_file_name(tensor_cache_path, name),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_config.replicate(mesh_device),
        )

    return AttentionWeights(wqkvg=wqkvg, o_proj=o_proj, q_norm=_norm_gain("q_norm"), k_norm=_norm_gain("k_norm"))
