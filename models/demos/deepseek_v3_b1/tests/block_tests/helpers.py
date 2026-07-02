# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for attention/MoE reference implementations and tests."""

import random
import torch


def seed_all(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)


def assert_close(a, b, **kw):
    diff = (a - b).abs()
    flat_a = a.flatten()[:5]
    flat_b = b.flatten()[:5]
    print(f"  max_diff={diff.max().item():.2e}  mean_diff={diff.mean().item():.2e}")
    print(f"  expected[:5]={flat_b.tolist()}")
    print(f"  got[:5]     ={flat_a.tolist()}")
    torch.testing.assert_close(a, b, **kw)


def _rotate_half(x):
    """Rotate half the hidden dims: [x1, x2] → [-x2, x1]."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _rotate_half_interleaved(x):
    """Interleaved rotation: pairs (even, odd) → (-odd, even)."""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rotary_pos_emb(q, k, cos, sin):
    """
    Standard RoPE (Llama/Qwen/Mixtral pattern).

    q, k: [b, heads, seq, head_dim]
    cos, sin: [b, seq, head_dim]  (from RotaryEmbedding)

    Applies rotation to ALL head_dim dimensions.
    """
    cos = cos.unsqueeze(1)  # [b, 1, seq, head_dim]
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def apply_rotary_pos_emb_interleaved(q, k, cos, sin):
    """
    Interleaved RoPE (DeepSeek V3 pattern with rope_interleave=True).

    Rearranges q/k dims from interleaved to paired layout before applying
    standard rotate_half, matching HF's apply_rotary_pos_emb_interleave.
    """
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    # Rearrange: [a, b, c, d] → [a, c, b, d] (deinterleave pairs)
    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
    b, h, s, d = k.shape
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def apply_rotary_pos_emb_glm4(q, k, cos, sin):
    """
    GLM-4 RoPE: interleaved rotation with partial rotary support.

    cos/sin may be smaller than head_dim (partial_rotary_factor).
    Unrotated dims are passed through unchanged.
    """
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    # GLM4 uses interleaved cos/sin — take first half and repeat_interleave
    cos = cos[..., : cos.shape[-1] // 2].repeat_interleave(2, dim=-1)
    sin = sin[..., : sin.shape[-1] // 2].repeat_interleave(2, dim=-1)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = (q_rot * cos) + (_rotate_half_interleaved(q_rot) * sin)
    k_embed = (k_rot * cos) + (_rotate_half_interleaved(k_rot) * sin)
    return torch.cat([q_embed, q_pass], dim=-1), torch.cat([k_embed, k_pass], dim=-1)


def repeat_kv(x, n_rep):
    """Repeat KV heads to match Q heads. x: [b, kv_heads, seq, hd]"""
    if n_rep == 1:
        return x
    b, kv_heads, seq, hd = x.shape
    x = x[:, :, None, :, :].expand(b, kv_heads, n_rep, seq, hd)
    return x.reshape(b, kv_heads * n_rep, seq, hd)


def rms_norm(x, weight, eps=1e-6):
    """RMSNorm matching HuggingFace implementation."""
    input_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return (weight * x).to(input_dtype)


def make_profiles(size="tiny"):
    """Return toy profiles that preserve structural constraints per family."""
    if size == "tiny":
        return {
            "deepseek_v3": dict(
                hidden_size=64, num_q_heads=8, num_kv_heads=2, head_dim=8,
                top_k=2, num_experts=4, moe_intermediate=128,
                kv_latent_dim=32, qk_rope_head_dim=4, max_context=256_000,
                use_mla=True, use_moe=True,
                ffn_intermediate=128,
            ),
            "kimi_k25": dict(
                hidden_size=64, num_q_heads=8, num_kv_heads=2, head_dim=8,
                top_k=2, num_experts=4, moe_intermediate=128,
                kv_latent_dim=32, qk_rope_head_dim=4, max_context=256_000,
                use_mla=True, use_moe=True,
                ffn_intermediate=128,
            ),
            "glm4_355b": dict(
                hidden_size=64, num_q_heads=8, num_kv_heads=4, head_dim=8,
                top_k=1, num_experts=1, moe_intermediate=128,
                kv_latent_dim=None, max_context=200_000,
                use_mla=False, use_moe=False,
                ffn_intermediate=128,
            ),
            "gpt_oss_120b": dict(
                hidden_size=64, num_q_heads=8, num_kv_heads=2, head_dim=8,
                top_k=2, num_experts=4, moe_intermediate=128,
                kv_latent_dim=None, max_context=128_000,
                use_mla=False, use_moe=True,
                ffn_intermediate=128,
            ),
        }
    raise ValueError(f"Unknown size: {size}")
