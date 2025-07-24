# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import ttnn

# from .conv2d import TtConv2d, TtConv2dParameters
from .fun_conv2d import vae_conv2d, TtConv2dParameters
from .fun_unet_mid_block import unet_mid_block, TtUNetMidBlock2DParameters
from .fun_updecoder_block import updecoder_block, TtUpDecoderBlock2DParameters
from .fun_group_norm import vae_group_norm, TtGroupNormParameters
from ..parallel_config import VAEParallelConfig

if TYPE_CHECKING:
    import torch

## Parameters


@dataclass
class TtVaeDecoderParameters:
    conv_in: TtConv2dParameters
    mid_block: TtUNetMidBlock2DParameters
    up_blocks: list[TtUpDecoderBlock2DParameters]
    conv_norm_out: TtGroupNormParameters
    conv_out: TtConv2dParameters

    @classmethod
    def from_torch(
        cls,
        torch_vae_decoder: torch.nn.Module,
        *,
        dtype: ttnn.DataType | None = None,
        parallel_config: VAEParallelConfig,
    ) -> TtVaeDecoderParameters:
        return cls(
            conv_in=TtConv2dParameters.from_torch(
                torch_vae_decoder.conv_in, dtype=dtype, parallel_config=parallel_config
            ),
            mid_block=TtUNetMidBlock2DParameters.from_torch(
                torch_vae_decoder.mid_block, dtype=dtype, parallel_config=parallel_config
            ),
            up_blocks=[
                TtUpDecoderBlock2DParameters.from_torch(up_block, dtype=dtype, parallel_config=parallel_config)
                for up_block in (torch_vae_decoder.up_blocks or [])
            ],
            conv_norm_out=TtGroupNormParameters.from_torch(
                torch_vae_decoder.conv_norm_out, parallel_config=parallel_config
            ),
            conv_out=TtConv2dParameters.from_torch(
                torch_vae_decoder.conv_out, dtype=dtype, parallel_config=parallel_config
            ),
        )


# TODO: Verify upscale_dtype from reference code
def sd_vae_decode(x: ttnn.Tensor, parameters: TtVaeDecoderParameters) -> ttnn.Tensor:
    x = vae_conv2d(x, parameters.conv_in)
    x = unet_mid_block(x, parameters.mid_block)

    for up_block_params in parameters.up_blocks[0:4]:
        x = updecoder_block(x, up_block_params)

    x = vae_group_norm(x, parameters.conv_norm_out)
    x = ttnn.silu(x)
    x = vae_conv2d(x, parameters.conv_out)

    return x
