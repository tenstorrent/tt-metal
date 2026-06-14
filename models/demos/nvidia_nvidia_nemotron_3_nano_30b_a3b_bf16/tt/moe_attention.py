# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""MoEAttention — attention core used in MoE transformer layers of NemotronH-30B.

Unlike DenseAttention, this function:
  - Takes PRE-NORMED hidden_states (no pre-norm applied here)
  - Applies RoPE to Q and K
  - Returns the O_proj output only (no pre-norm, no residual)
"""

import torch
import torch.nn.functional as F

import ttnn
from ttnn import MeshDevice

NUM_HEADS = 32
NUM_KV_HEADS = 2
HEAD_DIM = 128

_R = ttnn.ReplicateTensorToMesh
_C = ttnn.ConcatMeshToTensor


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

    def to_dev(t):
        return ttnn.from_torch(
            t.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=_R(mesh_device)
        )

    def to_host(t):
        return ttnn.to_torch(t, mesh_composer=_C(mesh_device, dim=0)).bfloat16()

    # QKV projections on device
    h_tt = to_dev(hidden_states)
    q_tt = ttnn.linear(h_tt, to_dev(wq), transpose_b=True)  # [B, S, 4096]
    k_tt = ttnn.linear(h_tt, to_dev(wk), transpose_b=True)  # [B, S, 256]
    v_tt = ttnn.linear(h_tt, to_dev(wv), transpose_b=True)  # [B, S, 256]

    q = to_host(q_tt).view(B, S, num_heads, head_dim).transpose(1, 2).float()  # [B,nH,S,D]
    k = to_host(k_tt).view(B, S, num_kv_heads, head_dim).transpose(1, 2).float()  # [B,nKV,S,D]
    v = to_host(v_tt).view(B, S, num_kv_heads, head_dim).transpose(1, 2).float()  # [B,nKV,S,D]

    # RoPE
    cos, sin = _rope_cos_sin(
        position_ids, head_dim, rope_theta, partial_rotary_factor, attention_scaling, dtype=torch.float32
    )
    cos = cos.unsqueeze(1)  # [B, 1, S, rot_dim]
    sin = sin.unsqueeze(1)
    rot_dim = cos.shape[-1]
    q = torch.cat([(q[..., :rot_dim] * cos) + (_rotate_half(q[..., :rot_dim]) * sin), q[..., rot_dim:]], dim=-1)
    k = torch.cat([(k[..., :rot_dim] * cos) + (_rotate_half(k[..., :rot_dim]) * sin), k[..., rot_dim:]], dim=-1)

    # GQA expand KV
    k = k.unsqueeze(2).expand(-1, -1, num_groups, -1, -1).reshape(B, num_heads, S, head_dim)
    v = v.unsqueeze(2).expand(-1, -1, num_groups, -1, -1).reshape(B, num_heads, S, head_dim)

    # SDPA causal on host
    attn_out = (
        F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=True,
            scale=head_dim**-0.5,
        )
        .transpose(1, 2)
        .contiguous()
        .view(B, S, num_heads * head_dim)
        .bfloat16()
    )

    # O projection on device
    out_tt = ttnn.linear(to_dev(attn_out), to_dev(wo), transpose_b=True)
    return to_host(out_tt)
