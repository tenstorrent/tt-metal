# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
import torch
import ttnn


class TtClipEmbeddings(nn.Module):
    def __init__(
        self,
        device,
        state_dict,
        module_path,
        model_config,
    ):
        super().__init__()
        embedding_dtype = ttnn.bfloat16

        self.tt_position_embedding_weights = ttnn.from_torch(
            state_dict[f"{module_path}.position_embedding.weight"],
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=embedding_dtype,
            device=device,
        )

        self.tt_token_embedding_weights = ttnn.from_torch(
            state_dict[f"{module_path}.token_embedding.weight"],
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=embedding_dtype,
            device=device,
        )

        self.max_position_embeddings = state_dict[f"{module_path}.position_embedding.weight"].shape[0]
        self.vocab_size = state_dict[f"{module_path}.token_embedding.weight"].shape[0]
        self.device = device

    def forward(self, input_ids):
        seq_length = input_ids.shape[-1]

        # truncate seq if >max_position_embeddings
        if seq_length > self.max_position_embeddings:
            input_ids = input_ids[:, : self.max_position_embeddings]
            seq_length = self.max_position_embeddings

        position_ids = torch.arange(seq_length).expand((1, -1))
        position_ids = ttnn.from_torch(position_ids, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=self.device)

        input_embeddings = ttnn.embedding(input_ids, self.tt_token_embedding_weights, layout=ttnn.TILE_LAYOUT)
        position_embeddings = ttnn.embedding(position_ids, self.tt_position_embedding_weights, layout=ttnn.TILE_LAYOUT)

        return input_embeddings + position_embeddings
