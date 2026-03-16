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
                kv_latent_dim=32, max_context=256_000,
                use_mla=True, use_moe=True,
                ffn_intermediate=128,
            ),
            "kimi_k25": dict(
                hidden_size=64, num_q_heads=8, num_kv_heads=2, head_dim=8,
                top_k=2, num_experts=4, moe_intermediate=128,
                kv_latent_dim=32, max_context=256_000,
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
