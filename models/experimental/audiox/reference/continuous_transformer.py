# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch
import torch.nn as nn

from models.experimental.audiox.reference.rotary import RotaryEmbedding
from models.experimental.audiox.reference.transformer_block import TransformerBlock


class ContinuousTransformer(nn.Module):
    """Stack of AudioX DiT transformer blocks with optional input/output
    projections and shared rotary embeddings. Mirrors the prepend-conditioning
    path from audiox/models/transformer.py:ContinuousTransformer (no adaLN,
    no causal, no sinusoidal/abs pos, no return_info, no masking)."""

    def __init__(
        self,
        dim: int,
        depth: int,
        dim_in: Optional[int] = None,
        dim_out: Optional[int] = None,
        dim_heads: int = 64,
        cross_attend: bool = False,
        cond_token_dim: Optional[int] = None,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.project_in = nn.Linear(dim_in, dim, bias=False) if dim_in is not None else nn.Identity()
        self.project_out = nn.Linear(dim, dim_out, bias=False) if dim_out is not None else nn.Identity()
        self.rotary_pos_emb = RotaryEmbedding(max(dim_heads // 2, 32))
        self.layers = nn.ModuleList(
            [
                TransformerBlock(dim, dim_heads=dim_heads, cross_attend=cross_attend, dim_context=cond_token_dim)
                for _ in range(depth)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        prepend_embeds: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.project_in(x)
        if prepend_embeds is not None:
            assert prepend_embeds.shape[-1] == x.shape[-1], "prepend dim must match sequence dim"
            x = torch.cat((prepend_embeds, x), dim=-2)

        rotary_pos_emb = self.rotary_pos_emb.forward_from_seq_len(x.shape[1])

        for layer in self.layers:
            x = layer(x, context=context, rotary_pos_emb=rotary_pos_emb)

        return self.project_out(x)
