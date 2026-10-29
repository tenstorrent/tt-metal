# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Self-attention temporal transformer used inside the AudioX CLIP conditioner.

Mirrors ``audiox/models/temptransformer.py:SA_Transformer`` for the inference
path: pre-norm self-attention + GELU MLP, ``depth=4`` blocks, ``heads=16``,
``dim_head=64``, ``mlp_dim = dim*4``. No rotary, no cross-attn — this is a
plain ViT-style block stack run over per-frame CLIP token features.

Param naming is flatter than upstream (no nested ``Sequential``/``Residual``
wrappers) so the surrounding code reads cleanly; the pretrained loader
remaps upstream keys onto these names."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SAAttention(nn.Module):
    """Standard scaled-dot-product self-attention. Fused QKV projection,
    optional output projection (skipped when ``heads*dim_head == dim`` and
    ``heads == 1`` — never the case in AudioX, but kept for parity)."""

    def __init__(self, dim: int, heads: int, dim_head: int):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim) if (heads != 1 or dim_head != dim) else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D]
        b, n, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (t.reshape(b, n, self.heads, -1).transpose(1, 2) for t in qkv)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)


class SAFeedForward(nn.Module):
    """Two-layer GELU MLP. Hidden dim is ``mlp_dim`` (= 4*dim in upstream)."""

    def __init__(self, dim: int, mlp_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(dim, mlp_dim)
        self.linear2 = nn.Linear(mlp_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(F.gelu(self.linear1(x)))


class SATransformerBlock(nn.Module):
    """Pre-norm self-attn + pre-norm FF, both with residual."""

    def __init__(self, dim: int, heads: int, dim_head: int, mlp_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SAAttention(dim, heads=heads, dim_head=dim_head)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = SAFeedForward(dim, mlp_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class SATransformer(nn.Module):
    """Stack of ``depth`` blocks plus a final LayerNorm."""

    def __init__(self, dim: int, depth: int, heads: int, dim_head: int, mlp_dim: int):
        super().__init__()
        self.blocks = nn.ModuleList(
            [SATransformerBlock(dim, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.norm(x)
