# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import ttnn

# from .conv2d import TtConv2d, TtConv2dParameters
from .fun_conv2d import vae_conv2d, TtConv2dParameters
from .fun_group_norm import vae_group_norm, TtGroupNormParameters
from .fun_linear import vae_linear, TtLinearParameters
from ..parallel_config import VAEParallelConfig

if TYPE_CHECKING:
    import torch

## Parameters


@dataclass
class TtResnetBlock2DParameters:
    norm1: TtGroupNormParameters
    norm2: TtGroupNormParameters
    conv1: TtConv2dParameters
    conv2: TtConv2dParameters
    conv_shortcut: TtConv2dParameters | None

    @classmethod
    def from_torch(
        cls,
        resnet_block: torch.nn.Module,
        *,
        dtype: ttnn.DataType | None = None,
        parallel_config: VAEParallelConfig,
    ) -> TtResnetBlock2DParameters:
        return cls(
            norm1=TtGroupNormParameters.from_torch(resnet_block.norm1, parallel_config=parallel_config),
            norm2=TtGroupNormParameters.from_torch(resnet_block.norm2, parallel_config=parallel_config),
            conv1=TtConv2dParameters.from_torch(resnet_block.conv1, dtype=dtype, parallel_config=parallel_config),
            conv2=TtConv2dParameters.from_torch(resnet_block.conv2, dtype=dtype, parallel_config=parallel_config),
            conv_shortcut=TtLinearParameters.from_torch(
                resnet_block.conv_shortcut, dtype=dtype, parallel_config=parallel_config, is_conv=True
            )
            if resnet_block.conv_shortcut
            else None,
        )


def resnet_block(x_in: ttnn.Tensor, parameters: TtResnetBlock2DParameters) -> ttnn.Tensor:
    residual = x_in
    x = vae_group_norm(x_in, parameters.norm1)
    x = ttnn.silu(x)
    x = vae_conv2d(x, parameters.conv1)
    x = vae_group_norm(x, parameters.norm2)
    x = ttnn.silu(x)
    x = vae_conv2d(x, parameters.conv2)
    if parameters.conv_shortcut is not None:
        residual = vae_linear(residual, parameters.conv_shortcut)
    x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
    residual = ttnn.to_layout(residual, ttnn.TILE_LAYOUT)
    return x + residual
