# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Token embedding lookup -- the model's entry point.

Turns token ids into ebedding vectors that flow through all 64 layers.
"""

import ttnn
from models.common.lightweightmodule import LightweightModule


class TtQwen36Embedding(LightweightModule):
    """
    Token embedding table.

    Dimensions:
        V = 248320    vocab size (already padded to nicely divisible size in the checkpoint)
        D = 5120      hidden size

    Weight:
        embed_tokens  [248320, 5120]   ~2.5 GB in bf16 -- the single largest
                                       tensor in the model, alongside lm_head

    Shapes:
        token_ids  [B, T]        uint32, ROW_MAJOR layout
        output     [B, T, D]     bfloat16, TILE layout

    Note: embeddings are NOT tied to the lm_head in this model
    (`tie_word_embeddings: false`), so the two big tables are separate weights.

    To implement:
        1. load the table as ROW_MAJOR (ttnn.embedding wants it that way)
        2. ttnn.embedding(token_ids, weight, layout=ttnn.TILE_LAYOUT)
    """

    def __init__(self, device):
        self.device = device

    def forward(self, token_ids: ttnn.Tensor) -> ttnn.Tensor:
        """[B, T] uint32 -> [B, T, D] bfloat16."""
        return token_ids
