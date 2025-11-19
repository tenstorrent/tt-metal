# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
PyTorch reference implementation of Resampler for PCC validation.

Simplified from reference_pytorch/minicpm_official/resampler.py for testing.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


def get_2d_sincos_pos_embed(embed_dim: int, image_size: Tuple[int, int]) -> np.ndarray:
    """Generate 2D sinusoidal positional embeddings."""
    if isinstance(image_size, int):
        grid_h_size, grid_w_size = image_size, image_size
    else:
        grid_h_size, grid_w_size = image_size[0], image_size[1]

    grid_h = np.arange(grid_h_size, dtype=np.float32)
    grid_w = np.arange(grid_w_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    """Generate 2D positional embeddings from grid."""
    assert embed_dim % 2 == 0

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])

    emb = np.concatenate([emb_h, emb_w], axis=-1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """Generate 1D sinusoidal positional embeddings."""
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega

    out = np.einsum("hw,d->hwd", pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)

    emb = np.concatenate([emb_sin, emb_cos], axis=-1)
    return emb


class PyTorchResampler(nn.Module):
    """
    Simplified PyTorch Resampler for PCC validation.

    This is a minimal implementation focusing on the core resampling logic
    for testing against TTNN implementation.
    """

    def __init__(
        self,
        num_queries: int = 64,
        embed_dim: int = 3584,
        num_heads: int = 28,
        kv_dim: int = None,
        max_size: Tuple[int, int] = (70, 70),
    ):
        super().__init__()
        self.num_queries = num_queries
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.kv_dim = kv_dim if kv_dim is not None else embed_dim
        self.max_size = max_size

        # Learnable queries
        self.query = nn.Parameter(torch.zeros(num_queries, embed_dim))

        # Optional KV projection
        if kv_dim is not None and kv_dim != embed_dim:
            self.kv_proj = nn.Linear(kv_dim, embed_dim, bias=False)
        else:
            self.kv_proj = nn.Identity()

        # Attention layers
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            batch_first=True,
            bias=True,
        )

        # Layer norms
        self.ln_q = nn.LayerNorm(embed_dim)
        self.ln_kv = nn.LayerNorm(embed_dim)
        self.ln_post = nn.LayerNorm(embed_dim)

        # Final projection
        self.proj = nn.Parameter((embed_dim**-0.5) * torch.randn(embed_dim, embed_dim))

        # Positional embeddings
        pos_embed = torch.from_numpy(get_2d_sincos_pos_embed(embed_dim, max_size)).float()
        self.register_buffer("pos_embed", pos_embed, persistent=False)

    def forward(self, x: torch.Tensor, tgt_sizes: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of resampler.

        Args:
            x: Input features [batch_size, seq_len, kv_dim]
            tgt_sizes: Target sizes [batch_size, 2] (height, width)

        Returns:
            torch.Tensor: Resampled features [batch_size, num_queries, embed_dim]
        """
        batch_size = x.shape[0]

        # Apply KV projection
        x = self.kv_proj(x)

        # Apply LayerNorm to KV
        x = self.ln_kv(x)

        # Prepare positional embeddings
        pos_embeds = []
        max_patch_len = 0
        for i in range(batch_size):
            tgt_h, tgt_w = int(tgt_sizes[i, 0]), int(tgt_sizes[i, 1])
            patch_len = tgt_h * tgt_w
            max_patch_len = max(max_patch_len, patch_len)

            pos_embed_i = self.pos_embed[:tgt_h, :tgt_w, :].reshape(patch_len, -1)
            pos_embeds.append(pos_embed_i)

        # Pad positional embeddings
        pos_embeds_padded = []
        for pos_embed in pos_embeds:
            if pos_embed.shape[0] < max_patch_len:
                pad_size = max_patch_len - pos_embed.shape[0]
                padding = torch.zeros(pad_size, pos_embed.shape[1], dtype=pos_embed.dtype, device=pos_embed.device)
                pos_embed = torch.cat([pos_embed, padding], dim=0)
            pos_embeds_padded.append(pos_embed)

        pos_embeds_tensor = torch.stack(pos_embeds_padded, dim=0)

        # Add positional embeddings to keys
        x_with_pos = x + pos_embeds_tensor

        # Prepare queries
        queries_normalized = self.ln_q(self.query)
        queries_batched = queries_normalized.unsqueeze(0).repeat(batch_size, 1, 1)

        # Cross-attention
        attention_output, _ = self.attn(
            query=queries_batched,
            key=x_with_pos,
            value=x,
            need_weights=False,
        )

        # Post-attention processing
        attention_output = self.ln_post(attention_output)
        output = attention_output @ self.proj

        return output
