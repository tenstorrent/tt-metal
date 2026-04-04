# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Token embedding for Qwen3-Coder-Next."""

import torch
import torch.nn as nn

from models.demos.qwen3_coder_next.tt.model_config import Qwen3CoderNextConfig


class Embedding(nn.Module):
    """Token embedding layer.

    Converts token IDs to dense vectors of hidden_size dimensions.
    Vocab size is 151,936 for Qwen3-Coder-Next.
    """

    def __init__(self, config: Qwen3CoderNextConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Convert token IDs to embeddings.

        Args:
            input_ids: (batch, seq_len) integer token IDs.

        Returns:
            Embeddings (batch, seq_len, hidden_size).
        """
        return self.embed_tokens(input_ids)
