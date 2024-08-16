# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional
import torch
import torch.nn as nn

from models.utility_functions import (
    torch_to_tt_tensor_rm,
)

import ttnn


class TtDistilBert_Embeddings(nn.Module):
    def __init__(self, config, state_dict=None, base_address="", device=None):
        super().__init__()
        self.config = config
        self.device = device

        self.word_embedding_weight = state_dict[f"{base_address}.word_embeddings.weight"]

        self.word_embeddings = nn.Embedding(
            num_embeddings=self.config.vocab_size,
            embedding_dim=self.config.dim,
            padding_idx=self.config.pad_token_id,
            _weight=self.word_embedding_weight,
        )

        self.position_embedding_weight = state_dict[f"{base_address}.position_embeddings.weight"]
        self.position_embeddings = nn.Embedding(
            num_embeddings=self.config.max_position_embeddings,
            embedding_dim=self.config.dim,
            _weight=self.position_embedding_weight,
        )

        self.gamma = torch_to_tt_tensor_rm(state_dict[f"{base_address}.LayerNorm.weight"], self.device)
        self.beta = torch_to_tt_tensor_rm(state_dict[f"{base_address}.LayerNorm.bias"], self.device)

        self.LayerNorm = ttnn.layer_norm

        self.register_buffer(
            "position_ids",
            torch.arange(self.config.max_position_embeddings).expand((1, -1)),
            persistent=False,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        input_embeds: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        """
        Torch tensor is passed as input for embedding to address low pcc
        """
        if input_ids is not None:
            input_embeds = self.word_embeddings(input_ids)
            input_embeds = torch_to_tt_tensor_rm(input_embeds, self.device, put_on_device=True)

        seq_length = input_embeds.get_legacy_shape()[-2]

        if hasattr(self, "position_ids"):
            position_ids = self.position_ids[:, :seq_length]
        else:
            position_ids = torch.arange(seq_length, dtype=torch.long)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        position_embeddings = self.position_embeddings(position_ids)
        position_embeddings = torch_to_tt_tensor_rm(position_embeddings, self.device, put_on_device=True)

        embeddings = ttnn.add(input_embeds, position_embeddings)
        embeddings = self.LayerNorm(embeddings, epsilon=1e-12, weight=self.gamma, bias=self.beta)

        return embeddings
