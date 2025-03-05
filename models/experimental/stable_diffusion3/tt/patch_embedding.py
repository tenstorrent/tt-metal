# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import ttnn

from .conv2d import TtConv2d, TtConv2dParameters
from .substate import substate

from .utils import from_torch

if TYPE_CHECKING:
    import torch


@dataclass
class TtPatchEmbedParameters:
    proj: TtConv2dParameters
    pos_embed: ttnn.Tensor

    @classmethod
    def from_torch(
        cls,
        state: dict[str, torch.Tensor],
        *,
        device: ttnn.Device,
        out_channels: int,
    ) -> TtPatchEmbedParameters:
        return cls(
            proj=TtConv2dParameters.from_torch(
                substate(state, "proj"), dtype=ttnn.bfloat16, out_channels=out_channels, device=device
            ),
            pos_embed=from_torch(
                state["pos_embed"], layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_device=device, shard_dim=None
            ),
        )

    @property
    def pos_embed_max_size(self) -> int:
        return math.isqrt(self.pos_embed.shape[1])

    @property
    def patch_size(self) -> int:
        return list(self.proj.weight.shape)[-2]


class TtPatchEmbed:
    def __init__(self, parameters: TtPatchEmbedParameters, mesh_device) -> None:
        super().__init__()

        self._pos_embed_max_size = parameters.pos_embed_max_size
        self._proj = TtConv2d(parameters.proj, device=mesh_device)
        self._pos_embed = parameters.pos_embed
        self._out_height = parameters.patch_size
        self._out_width = parameters.patch_size

    def __call__(self, latent: ttnn.Tensor) -> ttnn.Tensor:
        batch_size = latent.shape[0]
        c = latent.shape[3]

        latent = self._proj(latent)
        latent = ttnn.squeeze(latent, 0)
        pos_embed = self._cropped_pos_embed(self._out_height, self._out_width)
        return latent + pos_embed

    def _cropped_pos_embed(self, height: int, width: int) -> ttnn.Tensor:
        top = (self._pos_embed_max_size - height) // 2
        left = (self._pos_embed_max_size - width) // 2

        spatial_pos_embed = self._pos_embed.reshape([1, self._pos_embed_max_size, self._pos_embed_max_size, -1])
        spatial_pos_embed = spatial_pos_embed[:, top : top + height, left : left + width, :]
        return spatial_pos_embed.reshape([1, -1, spatial_pos_embed.shape[-1]])
