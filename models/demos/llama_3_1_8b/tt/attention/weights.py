# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Attention weight loading and TP sharding.

Two things happen here that nothing downstream repeats:

1. **QKV fusion.** q/k/v are concatenated into one ``[1, 1, hidden, qkv_total]`` weight laid out
   *per TP device* — device ``i``'s q slice, then its k slice, then its v slice — so that after a
   column-parallel shard each device's single matmul output is exactly the ``[q | k | v]`` block
   that ``nlp_create_qkv_heads`` expects. Concatenating q, k and v globally first and sharding the
   result would interleave the wrong pieces onto each device.
2. **Meta RoPE permutation.** q_proj and k_proj rows are interleaved into the Meta order that
   ``rotary_embedding_llama`` implements (``utils/rope_layout.py``). v_proj and o_proj are NOT
   permuted — V never rotates, and o_proj consumes attention output, not a rotated vector.

Llama-3.1-8B has ``attention_bias: false``, so there are no q/k/v/o biases to load; the loader
asserts that rather than silently dropping a bias a different checkpoint might carry.

o_proj output-dim padding: ``hidden_size`` 4096 / tp 4 = 1024, already tile-aligned, so the padding
branch M3 needs (its 6144/4 = 1536 is aligned too, but the code is general) is a no-op here. It is
kept as an assert instead of dead code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

import ttnn

from ...utils.general import cache_name, substate
from ...utils.rope_layout import meta_permute_proj
from .config import AttentionConfig


@dataclass(frozen=True)
class AttentionWeights:
    """``wqkv`` is column-parallel (shards q/k/v heads across TP); ``o_proj`` is row-parallel."""

    wqkv: ttnn.Tensor
    o_proj: ttnn.Tensor


def load_attention_weights(
    mesh_device,
    config: AttentionConfig,
    mesh_config,
    state_dict: Optional[dict] = None,
    weight_dtype=ttnn.bfloat8_b,
    tensor_cache_path: Optional[str] = None,
) -> AttentionWeights:
    tp = mesh_config.tp
    assert config.num_heads % tp == 0, f"{config.num_heads} q heads do not split over tp={tp}"
    assert config.num_kv_heads % tp == 0, f"{config.num_kv_heads} kv heads do not split over tp={tp}"
    local_hidden = config.hidden_size // tp
    assert local_hidden % ttnn.TILE_SIZE == 0, (
        f"hidden_size {config.hidden_size} / tp {tp} = {local_hidden} is not tile-aligned; o_proj "
        f"would need output-dim padding, which this loader does not implement"
    )

    qkv_cat = None
    o_proj = None
    if state_dict:
        for banned in ("q_proj.bias", "k_proj.bias", "v_proj.bias", "o_proj.bias"):
            assert banned not in state_dict, f"unexpected {banned}: this loader assumes attention_bias=false"

        d = config.head_dim
        q_w = meta_permute_proj(substate(state_dict, "q_proj")["weight"], d)  # [n_q*d, hidden]
        k_w = meta_permute_proj(substate(state_dict, "k_proj")["weight"], d)  # [n_kv*d, hidden]
        v_w = substate(state_dict, "v_proj")["weight"]  # [n_kv*d, hidden] — no rope, no permute
        o_proj = substate(state_dict, "o_proj")["weight"].transpose(-1, -2)  # [n_q*d, hidden]

        per_device = []
        for i in range(tp):
            wq = torch.chunk(q_w, tp, dim=0)[i].transpose(-2, -1)  # [hidden, n_q_local*d]
            wk = torch.chunk(k_w, tp, dim=0)[i].transpose(-2, -1)
            wv = torch.chunk(v_w, tp, dim=0)[i].transpose(-2, -1)
            per_device.append(torch.cat([wq, wk, wv], dim=-1))
        qkv_cat = torch.cat(per_device, dim=-1).unsqueeze(0).unsqueeze(0)

    wqkv = ttnn.as_tensor(
        qkv_cat,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=weight_dtype,
        mesh_mapper=mesh_config.column_parallel(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cache_file_name=cache_name(tensor_cache_path, "wqkv"),
    )
    o_proj_tt = ttnn.as_tensor(
        o_proj.unsqueeze(0).unsqueeze(0) if o_proj is not None else None,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=weight_dtype,
        mesh_mapper=mesh_config.row_parallel(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cache_file_name=cache_name(tensor_cache_path, "o_proj"),
    )
    return AttentionWeights(wqkv=wqkv, o_proj=o_proj_tt)
