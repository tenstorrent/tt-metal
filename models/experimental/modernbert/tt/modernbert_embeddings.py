# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""TTNN ModernBERT embeddings: token lookup followed by LayerNorm.

There are no positional embeddings; ModernBERT carries position through RoPE
inside attention, so this block is lookup + normalise only.

The LayerNorm is weight-only, matching norm_bias=False.
"""

import ttnn


class TtnnModernBertEmbeddings:
    def __init__(self, parameters, config):
        self.tok_embeddings = parameters["tok_embeddings"]
        self.norm = parameters["norm"]
        self.eps = config.norm_eps

    def __call__(self, input_ids):
        """input_ids: ttnn uint32 tensor, ROW_MAJOR_LAYOUT, shape (B, S).

        Returns (B, S, hidden_size) in TILE_LAYOUT.
        """
        hidden = ttnn.embedding(input_ids, self.tok_embeddings, layout=ttnn.TILE_LAYOUT)
        # bias is omitted entirely: every ModernBERT LayerNorm has norm_bias=False
        normed = ttnn.layer_norm(hidden, weight=self.norm, epsilon=self.eps)
        ttnn.deallocate(hidden)
        return normed
