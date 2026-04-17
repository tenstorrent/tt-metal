"""tt-nn Fast3R encoder attention (with CroCo 2D RoPE).

Approach: permute Q and K rows of the QKV projection + build permuted cos/sin
tables so that the standard Llama-style `ttnn.experimental.rotary_embedding`
(rotate_half on full head_dim) reproduces the 2D RoPE used in CroCo/Fast3R.

See `reference/rope.py` for the torch implementation this mirrors.
"""
from __future__ import annotations

from typing import Dict

import torch
import ttnn
from safetensors import safe_open

from models.experimental.fast3r.reference.model import Fast3RConfig
from models.experimental.fast3r.reference.rope import build_rope2d_cos_sin
from models.experimental.fast3r.tt.attention import TtAttention
from models.experimental.fast3r.tt.block import TtLayerNorm, TtNormMlp
from models.experimental.fast3r.tt.mlp import to_device_bias, to_device_weight


def _head_dim_perm(head_dim: int) -> torch.Tensor:
    """Permutation that maps CroCo head_dim layout [y_front, y_back, x_front, x_back]
    to the Llama-friendly layout [y_front, x_front, y_back, x_back].
    """
    assert head_dim % 4 == 0
    q = head_dim // 4
    return torch.tensor(
        list(range(0, q))
        + list(range(2 * q, 3 * q))
        + list(range(q, 2 * q))
        + list(range(3 * q, 4 * q))
    )


def _permute_qk_in_qkv(qkv_w: torch.Tensor, qkv_b: torch.Tensor, num_heads: int, embed_dim: int):
    """Permute the head_dim positions within Q and K output rows of the fused QKV projection.
    V rows are left untouched.

    qkv_w shape (3*C, C); rows partitioned as [Q(C), K(C), V(C)].
    """
    head_dim = embed_dim // num_heads
    perm = _head_dim_perm(head_dim)
    full_perm = torch.arange(3 * embed_dim)
    for chunk_start in (0, embed_dim):  # Q, then K
        for h in range(num_heads):
            base = chunk_start + h * head_dim
            full_perm[base : base + head_dim] = base + perm
    return qkv_w[full_perm], qkv_b[full_perm]


def _build_permuted_rope_cache(device, cfg: Fast3RConfig):
    head_dim = cfg.embed_dim // cfg.num_heads
    grid = cfg.img_size // cfg.patch_size
    cos, sin = build_rope2d_cos_sin(grid, grid, head_dim, base=cfg.rope_base)
    perm = _head_dim_perm(head_dim)
    cos_p = cos[:, perm].unsqueeze(0).unsqueeze(0)  # (1, 1, N, Dh)
    sin_p = sin[:, perm].unsqueeze(0).unsqueeze(0)
    return (
        ttnn.from_torch(cos_p, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
        ttnn.from_torch(sin_p, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
    )


class TtEncoderAttention:
    """Self-attention with on-device 2D RoPE applied to Q and K."""

    def __init__(
        self,
        device,
        cfg: Fast3RConfig,
        qkv_w: torch.Tensor,
        qkv_b: torch.Tensor,
        proj_w: torch.Tensor,
        proj_b: torch.Tensor,
        cos_cache: ttnn.Tensor,
        sin_cache: ttnn.Tensor,
    ):
        self.num_heads = cfg.num_heads
        qkv_w_p, qkv_b_p = _permute_qk_in_qkv(qkv_w, qkv_b, cfg.num_heads, cfg.embed_dim)
        self.qkv_w = to_device_weight(device, qkv_w_p)
        self.qkv_b = to_device_bias(device, qkv_b_p)
        self.proj_w = to_device_weight(device, proj_w)
        self.proj_b = to_device_bias(device, proj_b)
        self.cos = cos_cache
        self.sin = sin_cache

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        qkv = ttnn.linear(x, self.qkv_w, bias=self.qkv_b)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv, num_heads=self.num_heads, transpose_k_heads=False
        )
        ttnn.deallocate(qkv)
        q_rot = ttnn.experimental.rotary_embedding(q, self.cos, self.sin)
        k_rot = ttnn.experimental.rotary_embedding(k, self.cos, self.sin)
        ttnn.deallocate(q); ttnn.deallocate(k)
        attn = ttnn.transformer.scaled_dot_product_attention(q_rot, k_rot, v, is_causal=False)
        ttnn.deallocate(q_rot); ttnn.deallocate(k_rot); ttnn.deallocate(v)
        out = ttnn.experimental.nlp_concat_heads(attn)
        ttnn.deallocate(attn)
        return ttnn.linear(out, self.proj_w, bias=self.proj_b)


class TtEncoderBlock:
    def __init__(self, device, cfg: Fast3RConfig, sd: Dict[str, torch.Tensor], cos_cache, sin_cache):
        self.norm1 = TtLayerNorm(device, sd["norm1.weight"], sd["norm1.bias"])
        self.attn = TtEncoderAttention(
            device, cfg,
            sd["attn.qkv.weight"], sd["attn.qkv.bias"],
            sd["attn.proj.weight"], sd["attn.proj.bias"],
            cos_cache, sin_cache,
        )
        self.norm_mlp = TtNormMlp(
            device,
            sd["norm2.weight"], sd["norm2.bias"],
            sd["mlp.fc1.weight"], sd["mlp.fc1.bias"],
            sd["mlp.fc2.weight"], sd["mlp.fc2.bias"],
        )

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x = ttnn.add(x, self.attn(self.norm1(x)))
        x = ttnn.add(x, self.norm_mlp(x))
        return x
