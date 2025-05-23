# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch


# adapted from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/models/embeddings.py
class PatchEmbed(torch.nn.Module):
    def __init__(
        self,
        *,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
        pos_embed_max_size: int,
    ) -> None:
        super().__init__()

        self.pos_embed_max_size = pos_embed_max_size
        self.patch_size = patch_size

        self.proj = torch.nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=(patch_size, patch_size),
            stride=patch_size,
        )

        self.register_buffer("pos_embed", torch.zeros((1, pos_embed_max_size**2, embed_dim)))

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        height, width = latent.shape[-2:]

        latent = self.proj(latent)
        latent = latent.flatten(2).transpose(1, 2)  # BCHW -> BNC

        pos_embed = self._cropped_pos_embed(height, width)

        return (latent + pos_embed).to(latent.dtype)

    def _cropped_pos_embed(self, height: int, width: int) -> torch.Tensor:
        height = height // self.patch_size
        width = width // self.patch_size

        top = (self.pos_embed_max_size - height) // 2
        left = (self.pos_embed_max_size - width) // 2

        spatial_pos_embed = self.pos_embed.reshape(1, self.pos_embed_max_size, self.pos_embed_max_size, -1)
        spatial_pos_embed = spatial_pos_embed[:, top : top + height, left : left + width, :]
        return spatial_pos_embed.reshape(1, -1, spatial_pos_embed.shape[-1])
