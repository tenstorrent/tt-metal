# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import torch
import ttnn

from .fun_conv2d import sd_conv2d, TtConv2dParameters
from .substate import substate
from .utils import from_torch_fast
from .parallel_config import DiTParallelConfig


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
        parallel_config: DiTParallelConfig,
        hidden_dim_padding: int,
        out_channels: int,
    ) -> TtPatchEmbedParameters:
        pos_embed_param = state["pos_embed"]
        if os.environ["MESH_DEVICE"] == "T3K":
            pos_embed_param = torch.nn.functional.pad(
                pos_embed_param, pad=(0, hidden_dim_padding), mode="constant", value=0
            )

        return cls(
            proj=TtConv2dParameters.from_torch(
                substate(state, "proj"),
                dtype=ttnn.bfloat16,
                hidden_dim_padding=hidden_dim_padding,
                out_channels=out_channels,
                device=device,
                parallel_config=parallel_config,
            ),
            pos_embed=from_torch_fast(
                pos_embed_param, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device, shard_dim=-1
            ),
        )

    @property
    def pos_embed_max_size(self) -> int:
        return math.isqrt(self.pos_embed.shape[1])

    @property
    def patch_size(self) -> int:
        return self.proj.kernel_size[0]


def sd_patch_embed(
    latent: ttnn.Tensor, parameters: TtPatchEmbedParameters, parallel_config: DiTParallelConfig
) -> ttnn.Tensor:
    def _cropped_pos_embed(height: int, width: int) -> ttnn.Tensor:
        top = (parameters.pos_embed_max_size - height) // 2
        left = (parameters.pos_embed_max_size - width) // 2

        spatial_pos_embed = parameters.pos_embed.reshape(
            [1, parameters.pos_embed_max_size, parameters.pos_embed_max_size, -1]
        )
        spatial_pos_embed = spatial_pos_embed[:, top : top + height, left : left + width, :]
        return spatial_pos_embed.reshape([1, -1, spatial_pos_embed.shape[-1]])

    batch_size_, in_height, in_width, c_ = latent.shape
    out_height = in_height // parameters.patch_size
    out_width = in_width // parameters.patch_size

    latent = sd_conv2d(latent, parameters.proj, parallel_config=parallel_config)
    latent = ttnn.reshape(latent, (batch_size_, out_height * out_width, -1))
    pos_embed = _cropped_pos_embed(out_height, out_width)
    return latent + pos_embed
