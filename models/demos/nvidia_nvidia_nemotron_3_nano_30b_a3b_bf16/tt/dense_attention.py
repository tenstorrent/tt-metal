# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""DenseAttention block — TP=4 column/row-parallel on QB 4-chip Blackhole.

GQA (32 Q / 2 KV heads, head_dim=128).  No RoPE (HF has it
commented out with a TODO for this model variant).

TP strategy:
  wq [4096, 2688]: column-parallel → [1024, 2688]/device
  wk [256,  2688]: replicated (2 KV heads < TP=4, cannot shard)
  wv [256,  2688]: replicated
  SDPA: on host (q gathered, k/v from first replica)
  wo [2688, 4096]: row-parallel → [2688, 1024]/device
  all_reduce after wo projection
"""

import torch
import torch.nn.functional as F

import ttnn
from ttnn import MeshDevice

from .tp import _col, _host_rep, _host_sharded, _rep, _row, _shard_act, all_reduce

NUM_HEADS = 32
NUM_KV_HEADS = 2
HEAD_DIM = 128
HIDDEN_SIZE = 2688
NORM_EPS = 1e-5


def dense_attention_forward(
    mesh_device: MeshDevice,
    hidden_states: torch.Tensor,  # [B, S, 2688] bf16 CPU
    norm_weight: torch.Tensor,  # [2688] bf16 CPU
    wq: torch.Tensor,  # [4096, 2688] bf16 CPU
    wk: torch.Tensor,  # [256,  2688] bf16 CPU
    wv: torch.Tensor,  # [256,  2688] bf16 CPU
    wo: torch.Tensor,  # [2688, 4096] bf16 CPU
    norm_eps: float = NORM_EPS,
) -> torch.Tensor:
    """DenseAttention with TP=4 column/row-parallel projections and residual.

    Returns [B, S, 2688] bfloat16 (CPU).
    """
    residual = hidden_states
    B, S, _ = hidden_states.shape

    # 1. Pre-norm (replicated on all 4 devices) → normed_tt [B, S, 2688]
    h_tt = _rep(hidden_states, mesh_device)
    w_tt = _rep(norm_weight.unsqueeze(0), mesh_device)
    normed_tt = ttnn.rms_norm(h_tt, epsilon=norm_eps, weight=w_tt)

    # 2. Q projection: column-parallel → [B, S, 1024]/device
    wq_tt = _col(wq, mesh_device)  # [1024, 2688]/device
    q_tt = ttnn.linear(normed_tt, wq_tt, transpose_b=True)  # [B, S, 1024]/device

    # 3. K projection: replicated (2 KV heads < TP=4) → [B, S, 256] on all devices
    wk_tt = _rep(wk, mesh_device)
    k_tt = ttnn.linear(normed_tt, wk_tt, transpose_b=True)  # [B, S, 256] replicated

    # 4. V projection: replicated → [B, S, 256] on all devices
    wv_tt = _rep(wv, mesh_device)
    v_tt = ttnn.linear(normed_tt, wv_tt, transpose_b=True)  # [B, S, 256] replicated

    # 5. Bring Q to host by concatenating shards along dim=2 → [B, S, 4096]
    q_host = _host_sharded(q_tt, mesh_device, concat_dim=2)  # [B, S, 4096]
    # 6-7. Bring K, V to host; take first B rows from replicated copies
    k_host = _host_rep(k_tt, mesh_device, B)  # [B, S, 256]
    v_host = _host_rep(v_tt, mesh_device, B)  # [B, S, 256]

    # 8. GQA SDPA on host
    q = q_host.view(B, S, NUM_HEADS, HEAD_DIM).transpose(1, 2)  # [B, 32, S, 128]
    k = k_host.view(B, S, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)  # [B,  2, S, 128]
    v = v_host.view(B, S, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)  # [B,  2, S, 128]

    n_rep = NUM_HEADS // NUM_KV_HEADS
    k = k.unsqueeze(2).expand(B, NUM_KV_HEADS, n_rep, S, HEAD_DIM).reshape(B, NUM_HEADS, S, HEAD_DIM)
    v = v.unsqueeze(2).expand(B, NUM_KV_HEADS, n_rep, S, HEAD_DIM).reshape(B, NUM_HEADS, S, HEAD_DIM)

    is_causal = S > 1
    attn_out = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=is_causal).to(
        torch.bfloat16
    )  # [B, 32, S, 128]
    attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, NUM_HEADS * HEAD_DIM)  # [B, S, 4096]

    # 9. Shard attn_out back to devices along dim=2 → [B, S, 1024]/device
    attn_shard_tt = _shard_act(attn_out, mesh_device, dim=2)  # [B, S, 1024]/device

    # 10-11. O projection: row-parallel → partial [B, S, 2688]/device
    wo_tt = _row(wo, mesh_device)  # [2688, 1024]/device
    out_tt = ttnn.linear(attn_shard_tt, wo_tt, transpose_b=True)  # [B, S, 2688] partial/device

    # 12. All-reduce to sum partials → full [B, S, 2688] on all devices
    result_tt = all_reduce(out_tt)

    # 13. Bring result to host
    out = _host_rep(result_tt, mesh_device, B)  # [B, S, 2688]

    # 14. Residual
    return (residual + out).bfloat16()
