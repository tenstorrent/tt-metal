# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""DenseAttention block — TP=4 column/row-parallel on QB 4-chip Blackhole.

GQA (32 Q / 2 KV heads, head_dim=128).  No RoPE (HF has it
commented out with a TODO for this model variant).

TP strategy:
  wq [4096, 2688]: column-parallel → [1024, 2688]/device → 8 Q heads/device
  wk [256,  2688]: replicated (2 KV heads < TP=4)
  wv [256,  2688]: replicated
  SDPA: ttnn.transformer.scaled_dot_product_attention (prefill path, S≥1)
        q [B,8,S,128], k [B,2,S,128], v [B,2,S,128] per device — GQA handled.
        For production decode (with KV cache) use scaled_dot_product_attention_decode.
  wo [2688, 4096]: row-parallel → [2688, 1024]/device
  all_reduce after wo; residual add on device.
"""

import torch

import ttnn
from ttnn import MeshDevice

from .tp import _col, _rep, _row, all_reduce

NUM_HEADS = 32
NUM_KV_HEADS = 2
HEAD_DIM = 128
TP = 4
HIDDEN_SIZE = 2688
NORM_EPS = 1e-5


def dense_attention_forward(
    mesh_device: MeshDevice,
    hidden_states: ttnn.Tensor,  # [B, S, 2688] bf16 on device
    norm_weight: torch.Tensor,  # [2688] bf16 CPU
    wq: torch.Tensor,  # [4096, 2688] bf16 CPU
    wk: torch.Tensor,  # [256,  2688] bf16 CPU
    wv: torch.Tensor,  # [256,  2688] bf16 CPU
    wo: torch.Tensor,  # [2688, 4096] bf16 CPU
    norm_eps: float = NORM_EPS,
) -> ttnn.Tensor:
    """Returns [B, S, 2688] bfloat16 on device (replicated) with residual applied."""
    residual = hidden_states
    B = hidden_states.shape[0]
    S = hidden_states.shape[1]

    # 1. Pre-norm (weight replicated, input already on device)
    w_tt = _rep(norm_weight.unsqueeze(0), mesh_device)
    normed_tt = ttnn.rms_norm(hidden_states, epsilon=norm_eps, weight=w_tt)

    # 2. Q: column-parallel → [B, S, 1024]/device (8 heads/device)
    wq_tt = _col(wq, mesh_device)
    q_tt = ttnn.linear(normed_tt, wq_tt, transpose_b=True)  # [B, S, 1024]/device

    # 3. K: replicated (2 KV heads < TP=4)
    wk_tt = _rep(wk, mesh_device)
    k_tt = ttnn.linear(normed_tt, wk_tt, transpose_b=True)  # [B, S, 256]/device

    # 4. V: replicated
    wv_tt = _rep(wv, mesh_device)
    v_tt = ttnn.linear(normed_tt, wv_tt, transpose_b=True)  # [B, S, 256]/device

    # 5. Reshape to [B, heads, S, head_dim] for SDPA
    q_4d = ttnn.reshape(q_tt, [B, S, NUM_HEADS // TP, HEAD_DIM])  # [B, S, 8, 128]
    q_4d = ttnn.permute(q_4d, [0, 2, 1, 3])  # [B, 8, S, 128]

    k_4d = ttnn.reshape(k_tt, [B, S, NUM_KV_HEADS, HEAD_DIM])  # [B, S, 2, 128]
    k_4d = ttnn.permute(k_4d, [0, 2, 1, 3])  # [B, 2, S, 128]

    v_4d = ttnn.reshape(v_tt, [B, S, NUM_KV_HEADS, HEAD_DIM])  # [B, S, 2, 128]
    v_4d = ttnn.permute(v_4d, [0, 2, 1, 3])  # [B, 2, S, 128]

    # 6. Prefill SDPA — handles GQA (8 Q heads, 2 KV heads) automatically.
    #    For production decode with KV cache use scaled_dot_product_attention_decode.
    is_causal = S > 1
    attn_out = ttnn.transformer.scaled_dot_product_attention(
        q_4d, k_4d, v_4d, is_causal=is_causal
    )  # [B, 8, S, 128]/device

    # 7. Reshape back to [B, S, 1024] for row-parallel O projection
    attn_out = ttnn.permute(attn_out, [0, 2, 1, 3])  # [B, S, 8, 128]
    attn_out = ttnn.reshape(attn_out, [B, S, (NUM_HEADS // TP) * HEAD_DIM])  # [B, S, 1024]

    # 8. O projection: row-parallel → partial [B, S, 2688]/device
    wo_tt = _row(wo, mesh_device)
    out_tt = ttnn.linear(attn_out, wo_tt, transpose_b=True)  # [B, S, 2688] partial/device

    # 9. All-reduce → full [B, S, 2688] + residual
    result_tt = all_reduce(out_tt)
    return ttnn.add(residual, result_tt)
