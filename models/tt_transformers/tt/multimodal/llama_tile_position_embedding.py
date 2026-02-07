# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import csv
import itertools
import os

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.utility_functions import nearest_32

_tile_pos_collected = set()
if os.path.exists("llama_tile_position_embedding_1d_performance.csv"):
    with open("llama_tile_position_embedding_1d_performance.csv", "r") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if row:
                _tile_pos_collected.add(",".join(row))


class TtLlamaTilePositionEmbedding(LightweightModule):
    """Tile Position Embedding layer.
    Arguments:
        num_tiles: Input channels.
        width: Width/Dim for the model.
        gate: Use gating mechanism.
        embedding: Position embedding tensor.
        embedding_config: TTNN configuration for embedding tensor.
    Input: (bsz, num_chunks, ntok, dim)
    Output: (bsz, num_chunks, ntok, dim)
    """

    def __init__(
        self,
        mesh_device,
        state_dict,
        state_dict_prefix,
        dtype,
        num_tiles: int,
        width: int,
        gated=False,
        model_name: str = "unknown",
    ):
        super().__init__()

        self.mesh_device = mesh_device
        self._model_name = model_name

        self.num_tiles = num_tiles
        self.width = width
        self.gated = gated

        embedding = state_dict[f"{state_dict_prefix}embedding"]

        padded_embeddings, self.ar_mapping = self.generate_padded_embeddings(embedding, num_tiles, width)
        self.padded_embeddings = ttnn.as_tensor(
            padded_embeddings,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        if self.gated:
            gate = state_dict[f"{state_dict_prefix}gate"]
            self.gate = ttnn.as_tensor(
                gate,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )

    def generate_padded_embeddings(self, embedding: torch.Tensor, num_tiles, width):
        TILE_SIZE = 32
        embedding = torch.cat(
            [
                embedding,
            ]
            * TILE_SIZE,
            dim=2,
        )  # Pad to tile size

        # Generate all possible aspect ratios (H * W must be less than or equal to num_tiles)
        ratios = list(itertools.product(range(1, num_tiles + 1), repeat=2))
        ratios = [x for x in ratios if x[0] * x[1] <= num_tiles]

        # Used to select the correct embedding for a given aspect ratio
        ar_mapping = {value: index for index, value in enumerate(ratios)}

        padded_embeds = []
        for h, w in ratios:
            out_pos_embed = torch.zeros(
                1, num_tiles, TILE_SIZE, width
            )  # 0th dim holds all possible embeddings for different aspect ratios
            out_pos_embed[0, : w * h] = embedding[:h, :w].reshape(1, w * h, TILE_SIZE, width)
            padded_embeds.append(out_pos_embed)

        return torch.cat(padded_embeds, dim=0), ar_mapping

    def forward(self, x: ttnn.Tensor, ar: torch.Tensor, num_tiles: int = None):
        _file_exists = os.path.exists("llama_tile_position_embedding_1d_performance.csv")
        with open("llama_tile_position_embedding_1d_performance.csv", "a") as _f:
            if not _file_exists:
                _f.write(
                    "x_dtype,x_shape_0,x_shape_1,x_shape_2,x_shape_3,"
                    "padded_emb_shape_0,padded_emb_shape_1,padded_emb_shape_2,padded_emb_shape_3,"
                    "num_tiles_cfg,width,gated,device_shape_x,device_shape_y,"
                    "num_tiles,ar_shape,ar_dtype,aspect_ratios,model_name\n"
                )
            _dev_shape = list(self.mesh_device.shape) if hasattr(self.mesh_device, "shape") else [1, 1]
            _num_tiles_arg = num_tiles if num_tiles is not None else self.num_tiles
            _ar_shape = "x".join(str(d) for d in ar.shape)
            _ar_str = ";".join(
                f"{h.item() if hasattr(h, 'item') else h}x{w.item() if hasattr(w, 'item') else w}" for h, w in ar
            )
            _entry = (
                f"{x.dtype},{x.shape[0]},{x.shape[1]},{x.shape[2]},{x.shape[3]},"
                f"{self.padded_embeddings.shape[0]},{self.padded_embeddings.shape[1]},{self.padded_embeddings.shape[2]},{self.padded_embeddings.shape[3]},"
                f"{self.num_tiles},{self.width},{self.gated},{_dev_shape[0]},{_dev_shape[1]},"
                f"{_num_tiles_arg},{_ar_shape},{ar.dtype},{_ar_str},{self._model_name}"
            )
            if _entry not in _tile_pos_collected:
                _tile_pos_collected.add(_entry)
                _f.write(f"{_entry}\n")

        if num_tiles is None:
            num_tiles = self.num_tiles
        elif num_tiles > self.num_tiles:
            assert False, "_dynamic_resize is currently not supported for TtLllamaTilePositionEmbedding"

        # Get the correct embeddings for the given aspect ratios
        out_pos_embed = []
        for [h, w] in ar:
            if isinstance(h, torch.Tensor):
                h, w = h.item(), w.item()
            idx = self.ar_mapping[(h, w)]
            out_pos_embed.append(
                self.padded_embeddings[idx : idx + 1],  # Select the correct embedding
            )
        out_pos_embed = ttnn.concat(out_pos_embed, dim=0)  # Concat batch

        # Apply gating mechanism
        if self.gated:
            out_pos_embed = out_pos_embed * ttnn.tanh(self.gate)

        # Broadcast along ntok dimension
        out_pos_embed = ttnn.concat(
            [
                out_pos_embed,
            ]
            * (nearest_32(x.shape[2]) // 32),
            dim=2,
        )

        # Concatenate with input tensor
        x = x + out_pos_embed[:, : x.shape[1], : x.shape[2], :]

        return x
