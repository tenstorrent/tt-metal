# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from loguru import logger
from typing import TYPE_CHECKING

import ttnn

# from .conv2d import TtConv2d, TtConv2dParameters
from .fun_resnet_block import resnet_block, TtResnetBlock2DParameters
from .fun_upsample2d import vae_upsample2d, TtUpsample2DParameters
from ..parallel_config import VAEParallelConfig

if TYPE_CHECKING:
    import torch

## Parameters


@dataclass
class TtUpDecoderBlock2DParameters:
    resnets: list[TtResnetBlock2DParameters]
    upsamplers: list[TtUpsample2DParameters]

    @classmethod
    def from_torch(
        cls,
        updecoder_block: torch.nn.Module,
        *,
        dtype: ttnn.DataType | None = None,
        parallel_config: VAEParallelConfig,
    ) -> TtUpDecoderBlock2DParameters:
        return cls(
            resnets=[
                TtResnetBlock2DParameters.from_torch(resnet_block, dtype=dtype, parallel_config=parallel_config)
                for resnet_block in (updecoder_block.resnets or [])
            ],
            upsamplers=[
                TtUpsample2DParameters.from_torch(torch_upsample, dtype=dtype, parallel_config=parallel_config)
                for torch_upsample in (updecoder_block.upsamplers or [])
            ],
        )


def updecoder_block(x: ttnn.Tensor, parameters: TtUpDecoderBlock2DParameters) -> ttnn.Tensor:
    for idx, resnet_params in enumerate(parameters.resnets):
        logger.info(f"resnet: {idx} <-> {x.shape}")
        x = resnet_block(x, resnet_params)

    for idx, upsample_params in enumerate(parameters.upsamplers):
        logger.info(f"upsample: {idx} <-> {x.shape}")
        x = vae_upsample2d(x, upsample_params)

    return x
