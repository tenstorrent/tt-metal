"""
Patch and position embeddings for the Janus-Pro-7B vision tower: the patch projection plus the
positional embedding added to it.

HF reference: `vision_model.embeddings` (`ModelArgs.reference_vision_embedding`).
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
        hidden_dim,
        configuration=None,
        bias=True,
    ):
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.mesh_device = mesh_device

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_ids = ttnn.arange(0, self.num_positions, 1, dtype=ttnn.uint32, device=self.mesh_device)
        self.position_ids = ttnn.reshape(self.position_ids, (1, -1))

        # The projection writes `ln_1`'s block shard directly, so neither the position add below
        # nor the first block has to reshard. Interleaved instead when the shape has no 2D config,
        # or when a caller builds this module standalone and passes no configuration.
        program_config = configuration and configuration.vision_patch_embed_program_config(1, self.num_patches)
        self.out_memory_config = (
            configuration.vision_norm_shard_configs(self.num_patches, hidden_dim)[0]
            if program_config is not None
            else None
        )

        self.patch_embed = TtJanusProConv2dPatch(
            mesh_device=mesh_device,
            state_dict=state_dict,
            state_dict_prefix=f"{state_dict_prefix}patch_embedding.",
            dtype=dtype,
            out_channels=hidden_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=bias,
            program_config=program_config,
            out_memory_config=self.out_memory_config,
        )

        # Positional embedding
        positional_embedding = state_dict[f"{state_dict_prefix}position_embedding.positional_embedding"]

        self.pos_emb_weights = ttnn.as_tensor(
            positional_embedding,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        # position_ids is a fixed arange, so the embedding lookup is constant — precompute it once.
        self.positional_embeddings = ttnn.embedding(self.position_ids, self.pos_emb_weights, layout=ttnn.TILE_LAYOUT)
        if self.out_memory_config is not None:
            # Resharded once here rather than per forward: position ids are a fixed arange, so
            # this tensor is a constant and the add wants it in the projection's layout.
            self.positional_embeddings = ttnn.to_memory_config(self.positional_embeddings, self.out_memory_config)

    def _add_position(self, patch_embeddings: ttnn.Tensor) -> ttnn.Tensor:
        # patch_embeddings: [1, B, num_patches, hidden_dim]
        batch_size = patch_embeddings.shape[1]
        patch_embeddings = ttnn.reshape(patch_embeddings, (batch_size, -1, self.hidden_dim))
        return ttnn.add(patch_embeddings, self.positional_embeddings, memory_config=self.out_memory_config)

    def forward(self, pixel_values: torch.Tensor) -> ttnn.Tensor:
        """
        Args:
            pixel_values: torch.Tensor of shape (B, C, H, W)
        Returns:
            embeddings: ttnn.Tensor of shape (B, num_patches, hidden_dim)
        """
        return self._add_position(self.patch_embed(pixel_values))

    def forward_device(self, patches: ttnn.Tensor) -> ttnn.Tensor:
        """Same as :meth:`forward` from patches already on device, so the caller can keep the
        host im2col and the transfer outside a trace region."""
        return self._add_position(self.patch_embed.forward_device(patches))
