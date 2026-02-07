# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import csv
import itertools
import os

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule

_pos_emb_collected = set()
if os.path.exists("llama_positional_embedding_1d_performance.csv"):
    with open("llama_positional_embedding_1d_performance.csv", "r") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if row:
                _pos_emb_collected.add(",".join(row))

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
        self._model_name = configuration.model_name if hasattr(configuration, "model_name") else "unknown"

        positional_embedding = state_dict[f"{state_dict_prefix}positional_embedding"]
        gated_positional_embedding = state_dict[f"{state_dict_prefix}gated_positional_embedding"]
        gated_positional_embedding_gate = state_dict[f"{state_dict_prefix}gated_positional_embedding_gate"]
        positional_embedding = positional_embedding.unsqueeze(0)  # Add batch dimensions
        pos_embed_device = ttnn.as_tensor(
            positional_embedding,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        # as_tensor returns (padded_shape, dim) which is incorrect, this reshape updates the padded size to the correct size
        self.positional_embedding = ttnn.reshape(pos_embed_device, positional_embedding.shape, pos_embed_device.shape)

        padded_gated_embeddings, self.ar_mapping = self.generate_padded_gated_embeddings(
            gated_positional_embedding, gated_positional_embedding_gate
        )
        padded_gated_embed = ttnn.as_tensor(
            padded_gated_embeddings,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        # as_tensor returns (padded_shape, dim) which is incorrect, this reshape updates the padded size to the correct size
        self.padded_gated_positional_embedding = ttnn.reshape(
            padded_gated_embed, padded_gated_embeddings.shape, padded_gated_embed.shape
        )

        # Add batch and ntok dimensions
        gated_positional_embedding_gate = gated_positional_embedding_gate.unsqueeze(0).unsqueeze(0)
        self.gated_positional_embedding_gate = ttnn.as_tensor(
            (
                1 - gated_positional_embedding_gate.tanh()
            ),  # NOTE: The reference code has does the 1 - gate.tanh() at inference time
            dtype=ttnn.bfloat16,
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
        _file_exists = os.path.exists("llama_positional_embedding_1d_performance.csv")
        with open("llama_positional_embedding_1d_performance.csv", "a") as _f:
            if not _file_exists:
                _f.write(
                    "x_dtype,x_shape_0,x_shape_1,x_shape_2,x_shape_3,"
                    "pos_emb_shape_0,pos_emb_shape_1,pos_emb_shape_2,"
                    "padded_gated_emb_shape_0,padded_gated_emb_shape_1,padded_gated_emb_shape_2,padded_gated_emb_shape_3,"
                    "device_shape_x,device_shape_y,ar_shape,ar_dtype,aspect_ratios,model_name\n"
                )
            _dev_shape = list(self.mesh_device.shape) if hasattr(self.mesh_device, "shape") else [1, 1]
            _ar_shape = "x".join(str(d) for d in ar.shape)
            _ar_str = ";".join(
                f"{h.item() if hasattr(h, 'item') else h}x{w.item() if hasattr(w, 'item') else w}" for h, w in ar
            )
            _entry = (
                f"{x.dtype},{x.shape[0]},{x.shape[1]},{x.shape[2]},{x.shape[3]},"
                f"{self.positional_embedding.shape[0]},{self.positional_embedding.shape[1]},{self.positional_embedding.shape[2]},"
                f"{self.padded_gated_positional_embedding.shape[0]},{self.padded_gated_positional_embedding.shape[1]},{self.padded_gated_positional_embedding.shape[2]},{self.padded_gated_positional_embedding.shape[3]},"
                f"{_dev_shape[0]},{_dev_shape[1]},{_ar_shape},{ar.dtype},{_ar_str},{self._model_name}"
            )
            if _entry not in _pos_emb_collected:
                _pos_emb_collected.add(_entry)
                _f.write(f"{_entry}\n")

        bsz, num_chunks, num_tokens, dim = x.shape
        x = ttnn.reshape(x, [bsz * num_chunks, num_tokens, dim])

        pos_embed_ = self.positional_embedding * self.gated_positional_embedding_gate

        # Broadcast in batch dim # NOTE: TTNN broadcast add gives PCC issues
        pos_embed_ = ttnn.concat([pos_embed_] * x.shape[0], dim=0)
        x = x + pos_embed_

        x = ttnn.reshape(x, [bsz, num_chunks, num_tokens, dim])

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
