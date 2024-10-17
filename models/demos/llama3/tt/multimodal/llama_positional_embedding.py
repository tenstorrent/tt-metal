# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional
import torch
import torch.nn as nn
import itertools

import ttnn
from models.utility_functions import (
    nearest_32,
)
from models.common.lightweightmodule import LightweightModule

from ttnn import ShardTensorToMesh, ConcatMeshToTensor, ReplicateTensorToMesh

TILE_SIZE = 32


class TtLlamaPositionalEmbedding(LightweightModule):
    """Positional Embedding layer.
    Adds positional embeddings corresponding to each patch
    and tile location in the input tensor.
    """

    def __init__(
        self,
        mesh_device,
        state_dict,
        state_dict_prefix,
        weight_cache_path,
        dtype,
        configuration,
    ):
        super().__init__()

        self.mesh_device = mesh_device

        positional_embedding = state_dict[f"{state_dict_prefix}positional_embedding"]
        gated_positional_embedding = state_dict[f"{state_dict_prefix}gated_positional_embedding"]
        gated_positional_embedding_gate = state_dict[f"{state_dict_prefix}gated_positional_embedding_gate"]

        positional_embedding = positional_embedding.reshape(1, *positional_embedding.shape)  # Add batch dimensions
        self.positional_embedding = ttnn.as_tensor(
            positional_embedding,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        padded_gated_embeddings, self.ar_mapping = self.generate_padded_gated_embeddings(
            gated_positional_embedding, gated_positional_embedding_gate
        )
        self.padded_gated_positional_embedding = ttnn.as_tensor(
            padded_gated_embeddings,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        # Add batch and ntok dimensions
        gated_positional_embedding_gate = gated_positional_embedding_gate.unsqueeze(0).unsqueeze(0)
        self.gated_positional_embedding_gate = ttnn.as_tensor(
            (
                1 - gated_positional_embedding_gate.tanh()
            ),  # NOTE: The reference code has does the 1 - gate.tanh() at inference time
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def generate_padded_gated_embeddings(self, gated_embedding, gate):
        num_tiles = gated_embedding.shape[0]

        # Generate all possible aspect ratios (H * W must be less than or equal to num_tiles)
        ratios = list(itertools.product(range(1, num_tiles + 1), repeat=2))
        ratios = [x for x in ratios if x[0] * x[1] <= num_tiles]

        # Used to select the correct embedding for a given aspect ratio
        ar_mapping = {value: index for index, value in enumerate(ratios)}

        padded_embeds = []
        for h, w in ratios:
            out_pos_embed = torch.zeros(
                1, num_tiles, *gated_embedding.shape[2:]
            )  # 0th dim holds all possible embeddings for different aspect ratios
            out_pos_embed[0, : w * h] = gated_embedding[:h, :w].reshape(1, w * h, *gated_embedding.shape[2:])

            padded_embeds.append(out_pos_embed * gate.tanh())

        return torch.cat(padded_embeds, dim=0), ar_mapping

    def forward(self, x: ttnn.Tensor, ar: torch.Tensor):
        bsz, num_chunks, num_tokens, dim = x.shape

        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.reshape(x, [bsz * num_chunks, num_tokens, dim])
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

        pos_embed_ = self.positional_embedding * self.gated_positional_embedding_gate

        # Broadcast in batch dim # NOTE: TTNN broadcast add gives PCC issues
        pos_embed_ = ttnn.concat([pos_embed_] * x.shape[0], dim=0)
        x = x + pos_embed_

        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.reshape(x, [bsz, num_chunks, num_tokens, dim])
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

        # Get the correct embeddings for the given aspect ratios
        gated_pos_embed_ = []
        for [h, w] in ar:
            if isinstance(h, torch.Tensor):
                h, w = h.item(), w.item()
            idx = self.ar_mapping[(h, w)]
            gated_pos_embed_.append(
                self.padded_gated_positional_embedding[idx : idx + 1],  # Select the correct embedding
            )
        gated_pos_embed_ = ttnn.concat(gated_pos_embed_, dim=0)  # Concat batch

        x = x + gated_pos_embed_

        return x
