"""
This is the VisionEmbedding implementation for the Janus-Pro-7B
This implementation combines patch_conv followed by Embeddings as a submodule.
"""

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0


import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.experimental.janus_pro.tt.janus_pro_conv2d_patch import TtJanusProConv2dPatch


class TtJanusProVisionEmbeddings(LightweightModule):
    def __init__(
        self,
        mesh_device,
        state_dict,
        state_dict_prefix,
        dtype,
        image_size,
        patch_size,
        num_channels,
        hidden_dim,
        bias=True,
    ):
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.num_channels = num_channels
        self.mesh_device = mesh_device

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_ids = ttnn.arange(0, self.num_positions, 1, dtype=ttnn.uint32, device=self.mesh_device)
        self.position_ids = ttnn.reshape(self.position_ids, (1, -1))

        self.patch_embed = TtJanusProConv2dPatch(
            mesh_device=mesh_device,
            state_dict=state_dict,
            state_dict_prefix=f"{state_dict_prefix}patch_embedding.",
            dtype=dtype,
            in_channels=num_channels,
            out_channels=hidden_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=bias,
        )

        # Positional embedding
        positional_embedding = state_dict[f"{state_dict_prefix}position_embedding.positional_embedding"]

        self.pos_emb_weights = ttnn.as_tensor(
            positional_embedding,
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def forward(self, pixel_values: torch.Tensor) -> ttnn.Tensor:
        """
        Args:
            pixel_values: torch.Tensor of shape (B, C, H, W)
        Returns:
            embeddings: ttnn.Tensor of shape (B, num_patches, hidden_dim)
        """
        # Conv2d patch returns [1, B, num_patches, hidden_dim]; the leading 1 comes from the
        # 4D linear weight inside TtJanusProConv2dPatch, so the batch axis is at index 1.
        patch_embeddings = self.patch_embed(pixel_values)
        batch_size = patch_embeddings.shape[1]
        patch_embeddings = ttnn.reshape(patch_embeddings, (batch_size, -1, self.hidden_dim))
        positional_embeddings = ttnn.embedding(self.position_ids, self.pos_emb_weights, layout=ttnn.TILE_LAYOUT)
        embeddings = ttnn.add(patch_embeddings, positional_embeddings)
        return embeddings
