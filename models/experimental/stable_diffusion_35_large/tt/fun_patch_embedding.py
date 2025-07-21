# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import ttnn

from .fun_conv2d import sd_conv2d, TtConv2dParameters
from .substate import substate
from .utils import from_torch_fast_2d
from .parallel_config import DiTParallelConfig, StableDiffusionParallelManager


@dataclass
class TtPatchEmbedParameters:
    proj: TtConv2dParameters
    pos_embed: ttnn.Tensor

    @classmethod
    def from_torch(
        cls,
        state: dict[str, torch.Tensor],
        *,
        dtype: ttnn.DataType | None = None,
        device: ttnn.Device,
        parallel_config: DiTParallelConfig,
        hidden_dim_padding: int,
        out_channels: int,
        height: int,
        width: int,
    ) -> TtPatchEmbedParameters:
        pos_embed_param = state["pos_embed"]
        if hidden_dim_padding > 0:
            pos_embed_param = torch.nn.functional.pad(
                pos_embed_param, pad=(0, hidden_dim_padding), mode="constant", value=0
            )

        def _cropped_pos_embed(height: int, width: int) -> ttnn.Tensor:
            pos_embed_max_size = math.isqrt(pos_embed_param.shape[1])
            top = (pos_embed_max_size - height) // 2
            left = (pos_embed_max_size - width) // 2

            spatial_pos_embed = pos_embed_param.reshape([1, pos_embed_max_size, pos_embed_max_size, -1])
            spatial_pos_embed = spatial_pos_embed[:, top : top + height, left : left + width, :]
            return spatial_pos_embed.reshape([1, -1, spatial_pos_embed.shape[-1]])

        sd_conv = TtConv2dParameters.from_torch(
            substate(state, "proj"),
            dtype=dtype,
            hidden_dim_padding=hidden_dim_padding,
            out_channels=out_channels,
            device=device,
            parallel_config=parallel_config,
        )

        out_height = height // sd_conv.kernel_size[0]
        out_width = width // sd_conv.kernel_size[0]
        cropped_pos_embed_param = _cropped_pos_embed(out_height, out_width)
        dims = [None, None]
        dims[parallel_config.sequence_parallel.mesh_axis] = -2
        dims[parallel_config.tensor_parallel.mesh_axis] = -1
        return cls(
            proj=sd_conv,
            pos_embed=from_torch_fast_2d(
                cropped_pos_embed_param,
                mesh_device=device,
                mesh_shape=parallel_config.cfg_parallel.mesh_shape,
                dims=dims,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
            ),
        )

    @property
    def pos_embed_max_size(self) -> int:
        return math.isqrt(self.pos_embed.shape[1])

    @property
    def patch_size(self) -> int:
        return self.proj.kernel_size[0]


def sd_patch_embed(
    latent: ttnn.Tensor, parameters: TtPatchEmbedParameters, parallel_manager: StableDiffusionParallelManager
) -> ttnn.Tensor:
    batch_size_, in_height, in_width, c_ = latent.shape
    out_height = in_height // parameters.patch_size
    out_width = in_width // parameters.patch_size
    latent = sd_conv2d(latent, parameters.proj, parallel_manager=parallel_manager)
    latent = ttnn.reshape(latent, (batch_size_, out_height * out_width, -1))
    return latent + parameters.pos_embed
