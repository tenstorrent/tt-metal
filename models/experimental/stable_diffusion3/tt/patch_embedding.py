# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import ttnn

from .conv2d import TtConv2d, TtConv2dParameters
from .substate import substate

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
    ) -> TtPatchEmbedParameters:
        return cls(
            proj=TtConv2dParameters.from_torch(substate(state, "proj"), dtype=ttnn.bfloat16),
            pos_embed=ttnn.from_torch(state["pos_embed"], dtype=ttnn.bfloat16, device=device),
        )

    @property
    def pos_embed_max_size(self) -> int:
        return math.isqrt(self.pos_embed.shape[1])

    @property
    def patch_size(self) -> int:
        return list(self.proj.weight.shape)[-1]


class TtPatchEmbed:
    def __init__(self, parameters: TtPatchEmbedParameters) -> None:
        super().__init__()

        self._pos_embed_max_size = parameters.pos_embed_max_size
        self._proj = TtConv2d(parameters.proj, stride=(parameters.patch_size, parameters.patch_size))
        self._pos_embed = parameters.pos_embed

    def __call__(self, latent: ttnn.Tensor) -> ttnn.Tensor:
        latent, [batch_size, out_height, out_width, c] = self._proj.call_without_reshape(
            latent,
            conv_config=ttnn.Conv2dConfig(
                shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED  # https://github.com/tenstorrent/tt-metal/issues/17787
            ),
        )

        assert list(latent.shape) == list(latent.padded_shape)
        assert (out_height * out_width) % 32 == 0
        latent = latent.reshape([batch_size, out_height * out_width, c])

        pos_embed = self._cropped_pos_embed(out_height, out_width)
        pos_embed = ttnn.to_layout(pos_embed, ttnn.TILE_LAYOUT)

        return latent + pos_embed

    def _cropped_pos_embed(self, height: int, width: int) -> ttnn.Tensor:
        top = (self._pos_embed_max_size - height) // 2
        left = (self._pos_embed_max_size - width) // 2

        spatial_pos_embed = self._pos_embed.reshape([1, self._pos_embed_max_size, self._pos_embed_max_size, -1])
        spatial_pos_embed = spatial_pos_embed[:, top : top + height, left : left + width, :]
        return spatial_pos_embed.reshape([1, -1, spatial_pos_embed.shape[-1]])
