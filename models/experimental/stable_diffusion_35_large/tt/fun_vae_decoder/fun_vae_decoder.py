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
    parallel_config: VAEParallelConfig

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
                torch_vae_decoder.conv_in, dtype=dtype, parallel_config=parallel_config, mesh_sharded_output=True
            ),
            mid_block=TtUNetMidBlock2DParameters.from_torch(
                torch_vae_decoder.mid_block,
                dtype=dtype,
                parallel_config=parallel_config,
                gn_allow_sharded_compute=True,
                mesh_sharded_input=True,
            ),
            up_blocks=[
                TtUpDecoderBlock2DParameters.from_torch(
                    up_block,
                    dtype=dtype,
                    parallel_config=parallel_config,
                    gn_allow_sharded_compute=True,  # (False if idx == len(torch_vae_decoder.up_blocks) - 1 else True),
                    mesh_sharded_input=True,
                )  # No sharded compute for group norm for the last up decoder layer
                for idx, up_block in enumerate(torch_vae_decoder.up_blocks or [])
            ],
            conv_norm_out=TtGroupNormParameters.from_torch(
                torch_vae_decoder.conv_norm_out,
                parallel_config=parallel_config,
                allow_sharded_compute=True,  # No sharded compute for group norm for the last group norm because of hanging issue
                mesh_sharded_input=True,
            ),
            conv_out=TtConv2dParameters.from_torch(
                torch_vae_decoder.conv_out,
                dtype=dtype,
                parallel_config=parallel_config,  # This shouldn't use TP compute. If using TP, set mesh_sharded_output=False
            ),
            parallel_config=parallel_config,
        )


# TODO: Verify upscale_dtype from reference code
def sd_vae_decode(x: ttnn.Tensor, parameters: TtVaeDecoderParameters) -> ttnn.Tensor:
    x = vae_conv2d(x, parameters.conv_in)
    # ttnn.ReadDeviceProfiler(parameters.parallel_config.device)
    x = unet_mid_block(x, parameters.mid_block)
    # ttnn.ReadDeviceProfiler(parameters.parallel_config.device)

    for up_block_params in parameters.up_blocks:
        x = updecoder_block(x, up_block_params)
        # ttnn.ReadDeviceProfiler(parameters.parallel_config.device)

    x = vae_group_norm(x, parameters.conv_norm_out)
    # ttnn.ReadDeviceProfiler(parameters.parallel_config.device)
    x = ttnn.silu(x)
    # ttnn.ReadDeviceProfiler(parameters.parallel_config.device)
    x = vae_conv2d(x, parameters.conv_out)

    return x
