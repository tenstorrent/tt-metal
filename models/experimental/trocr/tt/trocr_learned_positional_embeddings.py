# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn

import ttnn


class TtTrOCRLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        config=None,
        base_address="",
        state_dict=None,
        device=None,
    ):
        # TrOCR is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        self.offset = 2
        self.device = device
        self.config = config
        self.base_address = base_address
        self.weights = state_dict[f"{self.base_address}.weight"]
        super().__init__(num_embeddings + self.offset, embedding_dim, _weight=self.weights)

    def forward(self, input_ids: ttnn.Tensor, past_key_values_length: int = 0):
        """`input_ids' shape is expected to be [bsz x seqlen]."""
        bsz, seq_len = input_ids.get_legacy_shape()[2:]
        positions = torch.arange(
            past_key_values_length,
            past_key_values_length + seq_len,
            dtype=torch.long,
            device=self.weight.device,
        ).expand(bsz, -1)
        return super().forward(positions + self.offset)
