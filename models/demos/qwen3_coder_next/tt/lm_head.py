# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Language model head for Qwen3-Coder-Next."""

import torch
import torch.nn as nn

from models.demos.qwen3_coder_next.tt.model_config import Qwen3CoderNextConfig


class LMHead(nn.Module):
    """Output projection from hidden_size to vocab_size.

    Projects the final hidden states to logits over the vocabulary.
    """

    def __init__(self, config: Qwen3CoderNextConfig):
        super().__init__()
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project hidden states to vocabulary logits.

        Args:
            hidden_states: (batch, seq_len, hidden_size).

        Returns:
            Logits (batch, seq_len, vocab_size).
        """
        return self.lm_head(hidden_states)
