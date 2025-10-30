# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Feed-forward network for SpeechT5 model.
Adapted from transformers.models.speecht5.modeling_speecht5
"""

import torch
import torch.nn as nn


class SpeechT5FeedForward(nn.Module):
    """
    Feed-forward network module for SpeechT5.
    Uses two linear layers with GELU activation and dropout.
    """

    def __init__(
        self,
        hidden_size: int,
        ffn_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.intermediate_dropout = nn.Dropout(dropout)
        self.intermediate_dense = nn.Linear(hidden_size, ffn_dim)
        self.intermediate_act_fn = nn.GELU()
        self.output_dense = nn.Linear(ffn_dim, hidden_size)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size]

        Returns:
            output: [batch, seq_len, hidden_size]
        """
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.intermediate_dropout(hidden_states)

        hidden_states = self.output_dense(hidden_states)
        hidden_states = self.output_dropout(hidden_states)

        return hidden_states
