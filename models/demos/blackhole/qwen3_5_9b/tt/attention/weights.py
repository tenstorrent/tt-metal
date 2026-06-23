# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Weight loading + tensor-parallel sharding for the Qwen3.5 full-attention block.

Loads one attention layer's projections from the raw HF submodule state dict and
shards them across the mesh (column-parallel Q/K/V/gate, row-parallel out_proj),
returning the immutable Qwen35AttentionWeights bundle attention.py consumes. At
tp=1 each "shard" is the full weight, so the same loader serves single-device runs.
"""
import os
from dataclasses import dataclass

import torch

import ttnn
from models.demos.blackhole.qwen3_5_9b.tt import tp_common as tpc
from models.demos.blackhole.qwen3_5_9b.utils.general_utils import get_cache_file_name


@dataclass(frozen=True)
class Qwen35AttentionWeights:
    # Q/K/V are fused into one column-parallel weight so the on-device head split
    # can use ttnn.experimental.nlp_create_qkv_heads (and ..._decode) instead of a
    # linear-per-projection + manual reshape/transpose. The gate (Qwen3.5 gated
    # attention) is a 4th projection with n_heads heads that the 3-way QKV head op
    # cannot split, so it stays separate.
    wqkv: ttnn.Tensor  # fused q_proj | k_proj | v_proj, per-device [Q|K|V], column-parallel
    wg: ttnn.Tensor  # gate_proj, column-parallel
    wo: ttnn.Tensor  # o_proj, row-parallel
    w_q_norm: ttnn.Tensor  # Replicated across devices
    w_k_norm: ttnn.Tensor  # Replicated across devices


def load_attention_weights(mesh_device, state_dict, args, tensor_cache_path=None) -> Qwen35AttentionWeights:
    """Load + shard one attention layer's weights into a Qwen35AttentionWeights bundle.

    state_dict is the raw HF ``self_attn`` submodule dict (q_proj/k_proj/v_proj/o_proj/
    q_norm/k_norm). The query and gate projections, fused in the checkpoint, are split
    here; Q/K/V are then re-fused in the per-device [Q|K|V] order nlp_create_qkv_heads
    expects and column-parallel sharded, while o_proj is row-parallel sharded and the
    q/k norms are replicated. tensor_cache_path, if given, caches the converted shards.
    """
    if tensor_cache_path is not None:
        os.makedirs(tensor_cache_path, exist_ok=True)

    def split_q_and_gate(w):
        """Split the fused query+gate projection into separate wq and wg weights.

        The HF checkpoint ships query and gate interleaved per head as
        ``[2 * num_heads * head_dim, hidden]``; this de-interleaves them into

            wq: [num_heads * head_dim, hidden]
            wg: [num_heads * head_dim, hidden]
        """
        NH, HD = args.n_heads, args.head_dim
        w_q_and_gate = w.reshape(NH, 2 * HD, -1)
        wq = w_q_and_gate[:, :HD, :].reshape(NH * HD, -1)
        wg = w_q_and_gate[:, HD:, :].reshape(NH * HD, -1)
        return wq, wg

    wq, wg = split_q_and_gate(state_dict["q_proj.weight"])

    # Fuse Q+K+V along the out dim in the per-device [Q|K|V] order nlp_create_qkv_heads
    # consumes. prepare_attn_qkv reduces to cat([wq, wk, wv]) at tp=1 (the 9B / rob.py
    # single-device case) and interleaves per device for TP>1. shard_w(dim=-1) then
    # column-parallel-shards it, handing each device its own [q|k|v] heads.
    wqkv = tpc.prepare_attn_qkv(
        wq,
        state_dict["k_proj.weight"],
        state_dict["v_proj.weight"],
        args.n_heads,
        args.n_kv_heads,
        args.head_dim,
        args.num_devices,
    )

    return Qwen35AttentionWeights(
        wqkv=tpc.shard_w(
            wqkv,
            mesh_device,
            dim=-1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_path=get_cache_file_name(tensor_cache_path, "wqkv"),
            dtype=ttnn.bfloat8_b,
        ),
        wg=tpc.shard_w(
            wg,
            mesh_device,
            dim=-1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_path=get_cache_file_name(tensor_cache_path, "wg"),
            dtype=ttnn.bfloat8_b,
        ),
        # Row-parallel: shard input dim → reduce-scatter after
        wo=tpc.shard_w(
            state_dict["o_proj.weight"],
            mesh_device,
            dim=0,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_path=get_cache_file_name(tensor_cache_path, "wo"),
            dtype=ttnn.bfloat8_b,
        ),
        w_q_norm=tpc.replicate(state_dict["q_norm.weight"].to(torch.float32), mesh_device, None),
        w_k_norm=tpc.replicate(state_dict["k_norm.weight"].to(torch.float32), mesh_device, None),
    )
