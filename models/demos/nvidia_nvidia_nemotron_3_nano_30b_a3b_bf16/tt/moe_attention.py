# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""MoEAttention — TP=4 column/row-parallel on QB 4-chip Blackhole.

Unlike DenseAttention, this function:
  - Takes PRE-NORMED hidden_states (no pre-norm applied here)
  - Applies RoPE to Q and K
  - Returns the O_proj output only (no pre-norm, no residual)

TP strategy:
  wq [4096, 2688]: column-parallel → [1024, 2688]/device
  wk [256,  2688]: replicated (2 KV heads < TP=4, cannot shard)
  wv [256,  2688]: replicated
  Q col-sharded → host gather on dim=2 → [B, S, 4096]
  K, V replicated → host slice [:B] → [B, S, 256]
  RoPE + GQA SDPA on host
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


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


def _rope_cos_sin(position_ids, head_dim, rope_theta, partial_rotary_factor, attention_scaling, dtype):
    rot_dim = int(head_dim * partial_rotary_factor)
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, rot_dim, 2, dtype=torch.float32) / rot_dim))
    B, S = position_ids.shape
    inv_freq_exp = inv_freq[None, :, None].expand(B, -1, 1).float()
    pos_exp = position_ids[:, None, :].float()
    freqs = (inv_freq_exp @ pos_exp).transpose(1, 2)  # [B, S, rot_dim/2]
    emb = torch.cat([freqs, freqs], dim=-1)  # [B, S, rot_dim]
    return (emb.cos() * attention_scaling).to(dtype), (emb.sin() * attention_scaling).to(dtype)


def moe_attention_forward(
    mesh_device: MeshDevice,
    hidden_states: torch.Tensor,  # [B, S, 2688] bf16 CPU — PRE-NORMED
    wq: torch.Tensor,  # [4096, 2688]
    wk: torch.Tensor,  # [256,  2688]
    wv: torch.Tensor,  # [256,  2688]
    wo: torch.Tensor,  # [2688, 4096]
    position_ids: torch.Tensor,  # [B, S] int64
    num_heads: int = NUM_HEADS,
    num_kv_heads: int = NUM_KV_HEADS,
    head_dim: int = HEAD_DIM,
    rope_theta: float = 10000.0,
    partial_rotary_factor: float = 1.0,
    attention_scaling: float = 1.0,
) -> torch.Tensor:
    """Attention core for MoE transformer layers — with RoPE, no pre-norm, no residual.

    Returns [B, S, 2688] bfloat16 (CPU).
    """
    B, S, _ = hidden_states.shape
    num_groups = num_heads // num_kv_heads

    # 1. Load input onto all devices (replicated)
    h_tt = _rep(hidden_states, mesh_device)

    # 2. Q: column-parallel → [B, S, 1024]/device
    q_tt = ttnn.linear(h_tt, _col(wq, mesh_device), transpose_b=True)

    # 3. K, V: replicated (2 KV heads < TP=4)
    k_tt = ttnn.linear(h_tt, _rep(wk, mesh_device), transpose_b=True)
    v_tt = ttnn.linear(h_tt, _rep(wv, mesh_device), transpose_b=True)

    # 4. Gather Q shards along dim=2 → [B, S, 4096]; slice K, V replicas
    q_host = _host_sharded(q_tt, mesh_device, concat_dim=2)  # [B, S, 4096]
    k_host = _host_rep(k_tt, mesh_device, B)  # [B, S, 256]
    v_host = _host_rep(v_tt, mesh_device, B)  # [B, S, 256]

    # 5. Reshape to [B, nH, S, D]
    q = q_host.view(B, S, num_heads, head_dim).transpose(1, 2).float()
    k = k_host.view(B, S, num_kv_heads, head_dim).transpose(1, 2).float()
    v = v_host.view(B, S, num_kv_heads, head_dim).transpose(1, 2).float()

    # 6. RoPE
    cos, sin = _rope_cos_sin(
        position_ids, head_dim, rope_theta, partial_rotary_factor, attention_scaling, dtype=torch.float32
    )
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    rot_dim = cos.shape[-1]
    q = torch.cat([(q[..., :rot_dim] * cos) + (_rotate_half(q[..., :rot_dim]) * sin), q[..., rot_dim:]], dim=-1)
    k = torch.cat([(k[..., :rot_dim] * cos) + (_rotate_half(k[..., :rot_dim]) * sin), k[..., rot_dim:]], dim=-1)

    # 7. GQA expand KV
    k = k.unsqueeze(2).expand(-1, -1, num_groups, -1, -1).reshape(B, num_heads, S, head_dim)
    v = v.unsqueeze(2).expand(-1, -1, num_groups, -1, -1).reshape(B, num_heads, S, head_dim)

    # 8. SDPA causal on host
    is_causal = S > 1
    attn_out = (
        F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, scale=head_dim**-0.5)
        .transpose(1, 2)
        .contiguous()
        .view(B, S, num_heads * head_dim)
        .bfloat16()
    )  # [B, S, 4096]

    # 9. O projection: row-parallel → partial [B, S, 2688]/device → all_reduce
    attn_shard_tt = _shard_act(attn_out, mesh_device, dim=2)
    out_tt = ttnn.linear(attn_shard_tt, _row(wo, mesh_device), transpose_b=True)
    result_tt = all_reduce(out_tt)

    return _host_rep(result_tt, mesh_device, B)
