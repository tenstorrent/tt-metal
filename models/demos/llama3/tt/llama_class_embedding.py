# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn

import ttnn
from models.common.lightweightmodule import LightweightModule

from ttnn import ReplicateTensorToMesh


class TtLlamaClassEmbedding(LightweightModule):
    """Class Embedding layer."""

    def __init__(
        self,
        mesh_device,
        class_embedding: torch.Tensor,
        dtype,
    ):
        super().__init__()

        self.mesh_device = mesh_device

        # Add batch and ntoks dimensions
        class_embedding = class_embedding.reshape(1, 1, *class_embedding.shape)

        self.class_embedding = ttnn.as_tensor(
            class_embedding,
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def forward(self, x: ttnn.Tensor):
        bsz = x.shape[0]

        # Broadcast class embedding to match input batch size
        class_embedding = ttnn.concat([self.class_embedding] * bsz, dim=0)  # Broadcast batch size

        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.concat([class_embedding, x], dim=1)
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

        return x
