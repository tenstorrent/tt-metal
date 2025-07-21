# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import torch
import ttnn

from .patch_embedding_conv2d import TtPatchEmbeddingConv2d, TtPatchEmbeddingConv2dParameters
from .substate import substate
from .utils import from_torch_fast


@dataclass
class TtPatchEmbedParameters:
    proj: TtPatchEmbeddingConv2dParameters
    pos_embed: ttnn.Tensor

    @classmethod
    def from_torch(
        cls,
        state: dict[str, torch.Tensor],
        *,
        device: ttnn.Device,
        hidden_dim_padding: int,
        out_channels: int,
    ) -> TtPatchEmbedParameters:
        pos_embed_param = state["pos_embed"]
        if os.environ["MESH_DEVICE"] == "T3K":
            pos_embed_param = torch.nn.functional.pad(
                pos_embed_param, pad=(0, hidden_dim_padding), mode="constant", value=0
            )

        return cls(
            proj=TtPatchEmbeddingConv2dParameters.from_torch(
                substate(state, "proj"),
                dtype=ttnn.bfloat16,
                hidden_dim_padding=hidden_dim_padding,
                out_channels=out_channels,
                device=device,
            ),
            pos_embed=from_torch_fast(
                pos_embed_param, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device, shard_dim=-1
            ),
        )

    @property
    def pos_embed_max_size(self) -> int:
        return math.isqrt(self.pos_embed.shape[1])


class TtPatchEmbed:
    def __init__(self, parameters: TtPatchEmbedParameters, mesh_device) -> None:
        super().__init__()

        self._pos_embed_max_size = parameters.pos_embed_max_size
        self._proj = TtPatchEmbeddingConv2d(parameters.proj, device=mesh_device)
        self._pos_embed = parameters.pos_embed
        self._patch_size = 2

    def __call__(self, latent: ttnn.Tensor) -> ttnn.Tensor:
        batch_size_, in_height, in_width, c_ = latent.shape
        out_height = in_height // self._patch_size
        out_width = in_width // self._patch_size

        latent = self._proj(latent)
        latent = ttnn.reshape(latent, (batch_size_, out_height * out_width, -1))
        pos_embed = self._cropped_pos_embed(out_height, out_width)
        return latent + pos_embed

    @property
    def patch_size(self) -> int:
        return self._patch_size

    def _cropped_pos_embed(self, height: int, width: int) -> ttnn.Tensor:
        top = (self._pos_embed_max_size - height) // 2
        left = (self._pos_embed_max_size - width) // 2

        spatial_pos_embed = self._pos_embed.reshape([1, self._pos_embed_max_size, self._pos_embed_max_size, -1])
        spatial_pos_embed = spatial_pos_embed[:, top : top + height, left : left + width, :]
        return spatial_pos_embed.reshape([1, -1, spatial_pos_embed.shape[-1]])
