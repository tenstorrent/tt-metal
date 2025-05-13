# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from diffusers import AutoencoderKL

import ttnn
from models.demos.wormhole.stable_diffusion.tt.vae.ttnn_vae_configs import (
    MIDBLOCK_RESNET_CONV_CHANNEL_SPLIT_FACTORS,
    MIDBLOCK_RESNET_NORM_NUM_BLOCKS,
    UPBLOCK_RESNET_CONV_CHANNEL_SPLIT_FACTORS,
    UPBLOCK_RESNET_NORM_NUM_BLOCKS,
)
from models.demos.wormhole.stable_diffusion.tt.vae.ttnn_vae_resnet import ResnetBlock
from models.utility_functions import skip_for_blackhole
from tests.ttnn.utils_for_testing import assert_with_pcc


@skip_for_blackhole("Blackhole PCC bad until GN issues fixed (#20760)")
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "input_channels, input_height, input_width, out_channels, norm_num_blocks, conv_channel_split_factors, block, block_id, resnet_block_id",
    [
        # fmt: off
        (512, 64, 64, 512, MIDBLOCK_RESNET_NORM_NUM_BLOCKS[0], MIDBLOCK_RESNET_CONV_CHANNEL_SPLIT_FACTORS[0], "mid", None, 0),
        (512, 64, 64, 512, MIDBLOCK_RESNET_NORM_NUM_BLOCKS[1], MIDBLOCK_RESNET_CONV_CHANNEL_SPLIT_FACTORS[1], "mid", None, 1),
        (512, 64, 64, 512, UPBLOCK_RESNET_NORM_NUM_BLOCKS[0][0], UPBLOCK_RESNET_CONV_CHANNEL_SPLIT_FACTORS[0][0], "up", 0, 0),
        (512, 64, 64, 512, UPBLOCK_RESNET_NORM_NUM_BLOCKS[0][1], UPBLOCK_RESNET_CONV_CHANNEL_SPLIT_FACTORS[0][1], "up", 0, 1),
        (512, 64, 64, 512, UPBLOCK_RESNET_NORM_NUM_BLOCKS[0][2], UPBLOCK_RESNET_CONV_CHANNEL_SPLIT_FACTORS[0][2], "up", 0, 2),
        (512, 128, 128, 512, UPBLOCK_RESNET_NORM_NUM_BLOCKS[1][0], UPBLOCK_RESNET_CONV_CHANNEL_SPLIT_FACTORS[1][0], "up", 1, 0),
        (512, 128, 128, 512, UPBLOCK_RESNET_NORM_NUM_BLOCKS[1][1], UPBLOCK_RESNET_CONV_CHANNEL_SPLIT_FACTORS[1][1], "up", 1, 1),
        (512, 128, 128, 512, UPBLOCK_RESNET_NORM_NUM_BLOCKS[1][2], UPBLOCK_RESNET_CONV_CHANNEL_SPLIT_FACTORS[1][2], "up", 1, 2),
        (512, 256, 256, 256, UPBLOCK_RESNET_NORM_NUM_BLOCKS[2][0], UPBLOCK_RESNET_CONV_CHANNEL_SPLIT_FACTORS[2][0], "up", 2, 0),
        (256, 256, 256, 256, UPBLOCK_RESNET_NORM_NUM_BLOCKS[2][1], UPBLOCK_RESNET_CONV_CHANNEL_SPLIT_FACTORS[2][1], "up", 2, 1),
        (256, 256, 256, 256, UPBLOCK_RESNET_NORM_NUM_BLOCKS[2][2], UPBLOCK_RESNET_CONV_CHANNEL_SPLIT_FACTORS[2][2], "up", 2, 2),
        (256, 512, 512, 128, UPBLOCK_RESNET_NORM_NUM_BLOCKS[3][0], UPBLOCK_RESNET_CONV_CHANNEL_SPLIT_FACTORS[3][0], "up", 3, 0),
        (128, 512, 512, 128, UPBLOCK_RESNET_NORM_NUM_BLOCKS[3][1], UPBLOCK_RESNET_CONV_CHANNEL_SPLIT_FACTORS[3][1], "up", 3, 1),
        (128, 512, 512, 128, UPBLOCK_RESNET_NORM_NUM_BLOCKS[3][2], UPBLOCK_RESNET_CONV_CHANNEL_SPLIT_FACTORS[3][2], "up", 3, 2),
        # fmt: on
    ],
)
def test_vae_resnet(
    device,
    input_channels,
    input_height,
    input_width,
    out_channels,
    norm_num_blocks,
    conv_channel_split_factors,
    block,
    block_id,
    resnet_block_id,
    use_program_cache,
):
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")

    if block == "mid":
        torch_resnet = vae.decoder.mid_block.resnets[resnet_block_id]
    else:
        torch_resnet = vae.decoder.up_blocks[block_id].resnets[resnet_block_id]

    # Run pytorch model
    torch_input = torch.randn([1, input_channels, input_height, input_width])
    torch_output = torch_resnet(torch_input, temb=None)

    # Initialize ttnn model
    ttnn_model = ResnetBlock(
        torch_resnet,
        device,
        input_channels,
        input_height,
        input_width,
        out_channels,
        norm_num_blocks[0],
        norm_num_blocks[1],
        conv_channel_split_factors[0],
        conv_channel_split_factors[1],
    )

    # Prepare ttnn input
    ttnn_input = ttnn.from_torch(
        torch_input.permute([0, 2, 3, 1]), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b
    )

    # Run ttnn model twice to test program cache and weights reuse
    ttnn_output = ttnn_model(ttnn_input)
    ttnn_output.deallocate(True)
    ttnn_input = ttnn.from_torch(
        torch_input.permute([0, 2, 3, 1]), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b
    )
    ttnn_output = ttnn_model(ttnn_input)

    ttnn_output = ttnn.reshape(ttnn_output, [1, input_height, input_width, out_channels])
    ttnn_output = ttnn.permute(ttnn_output, [0, 3, 1, 2])
    ttnn_output = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_output, ttnn_output, 0.99)
