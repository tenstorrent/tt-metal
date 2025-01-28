# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from torch import nn
from typing import Optional


class ttnn_CLIPTextEmbeddings:
    def __init__(self, config, parameters):
        embed_dim = config.hidden_size

        self.token_embedding = ttnn.embedding
        self.position_embedding = ttnn.embedding

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_ids = parameters["position_ids"]

    def __call__(
        self,
        input_ids=None,
        position_ids=None,
        inputs_embeds=None,
        parameters=None,
    ) -> ttnn.Tensor:
        seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]

        if position_ids is None:
            position_ids = self.position_ids[:, 0:seq_length]

        input_ids = ttnn.to_layout(input_ids, layout=ttnn.ROW_MAJOR_LAYOUT)
        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(
                input_ids,
                parameters.token_embedding.weight,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )

        position_embeddings = self.position_embedding(
            position_ids,
            parameters.position_embedding.weight,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        embeddings = inputs_embeds + position_embeddings

        return embeddings
