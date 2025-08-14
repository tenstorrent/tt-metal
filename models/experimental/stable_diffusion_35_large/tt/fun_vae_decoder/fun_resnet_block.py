# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import ttnn

# from .conv2d import TtConv2d, TtConv2dParameters
from .fun_conv2d import vae_conv2d, TtConv2dParameters
from .fun_group_norm import vae_group_norm, TtGroupNormParameters
from ..parallel_config import VAEParallelConfig

if TYPE_CHECKING:
    import torch

## Parameters


# Assumptions: If input is sharded, output will be sharded. Otherwise, If input is not sharded, output will be replicated across mesh.
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
        # conv_mesh_sharded_output: bool = False,
        gn_allow_sharded_compute: bool = True,
        mesh_sharded_input: bool = True,
    ) -> TtResnetBlock2DParameters:
        if resnet_block.conv_shortcut is not None:
            conv_shortcut = TtConv2dParameters.from_torch(
                resnet_block.conv_shortcut,
                dtype=dtype,
                parallel_config=parallel_config,
                mesh_sharded_input=mesh_sharded_input,
                mesh_sharded_output=mesh_sharded_input,
            )
        else:
            conv_shortcut = None

        return cls(
            norm1=TtGroupNormParameters.from_torch(
                resnet_block.norm1,
                parallel_config=parallel_config,
                allow_sharded_compute=gn_allow_sharded_compute,
                mesh_sharded_input=mesh_sharded_input,
            ),
            norm2=TtGroupNormParameters.from_torch(
                resnet_block.norm2,
                parallel_config=parallel_config,
                allow_sharded_compute=gn_allow_sharded_compute,
                mesh_sharded_input=mesh_sharded_input,
            ),
            conv1=TtConv2dParameters.from_torch(
                resnet_block.conv1, dtype=dtype, parallel_config=parallel_config, mesh_sharded_output=mesh_sharded_input
            ),
            conv2=TtConv2dParameters.from_torch(
                resnet_block.conv2, dtype=dtype, parallel_config=parallel_config, mesh_sharded_output=mesh_sharded_input
            ),
            conv_shortcut=conv_shortcut,
        )


# NOTE: CCL Calls with persistent buffer affect the programming expectations
def resnet_block(x_in: ttnn.Tensor, parameters: TtResnetBlock2DParameters) -> ttnn.Tensor:
    residual = ttnn.clone(x_in)
    x = vae_group_norm(x_in, parameters.norm1)
    x = ttnn.silu(x)
    x = vae_conv2d(x, parameters.conv1)
    x = vae_group_norm(x, parameters.norm2)
    x = ttnn.silu(x)
    x = vae_conv2d(x, parameters.conv2)
    if parameters.conv_shortcut is not None:
        residual = vae_conv2d(residual, parameters.conv_shortcut)
    x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
    residual = ttnn.to_layout(residual, ttnn.TILE_LAYOUT)
    return x + residual
