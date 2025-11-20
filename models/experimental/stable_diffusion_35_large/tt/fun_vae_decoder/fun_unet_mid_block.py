# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import ttnn

from .fun_resnet_block import resnet_block, TtResnetBlock2DParameters
from .fun_attention import vae_attention, TtAttentionParameters
from ..parallel_config import VAEParallelConfig

if TYPE_CHECKING:
    import torch

## Parameters


@dataclass
class TtUNetMidBlock2DParameters:
    attentions: list[TtAttentionParameters]
    resnets: list[TtResnetBlock2DParameters]

    @classmethod
    def from_torch(
        cls,
        unet_mid_block: torch.nn.Module,
        *,
        dtype: ttnn.DataType | None = None,
        parallel_config: VAEParallelConfig,
        gn_allow_sharded_compute: bool = True,  # TODO: Move into resnet specific parameters
        mesh_sharded_input: bool = True,  # TODO: Move into resnet specific parameters
    ) -> TtUNetMidBlock2DParameters:
        return cls(
            attentions=[
                TtAttentionParameters.from_torch(
                    attention, dtype=dtype, parallel_config=parallel_config, mesh_sharded_input=mesh_sharded_input
                )
                for attention in unet_mid_block.attentions
            ],
            resnets=[
                TtResnetBlock2DParameters.from_torch(
                    resnet_block,
                    dtype=dtype,
                    parallel_config=parallel_config,
                    gn_allow_sharded_compute=gn_allow_sharded_compute,
                    mesh_sharded_input=mesh_sharded_input,
                )
                for resnet_block in unet_mid_block.resnets
            ],
        )


def unet_mid_block(x: ttnn.Tensor, parameters: TtUNetMidBlock2DParameters) -> ttnn.Tensor:
    x = resnet_block(x, parameters.resnets[0])
    x = vae_attention(x, parameters.attentions[0])
    x = resnet_block(x, parameters.resnets[1])

    return x
