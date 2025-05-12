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
    UPBLOCK_UPSAMPLE_CONV_CHANNEL_SPLIT_FACTORS,
)
from models.demos.wormhole.stable_diffusion.tt.vae.ttnn_vae_decoder import VaeDecoder
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 8 * 8192}], indirect=True)
@pytest.mark.parametrize(
    """input_channels, input_height, input_width, out_channels, output_height, output_width,
    midblock_in_channels, midblock_resnet_norm_blocks, midblock_conv_channel_split_factors,
    upblock_out_channels, upblock_out_dimensions, upblock_resnet_norm_blocks,
    upblock_resnet_conv_in_channel_split_factors, upblock_upsample_conv_channel_split_factors""",
    [
        (
            4,  # input_channels
            64,  # input_height
            64,  # input_width
            3,  # out_channels
            512,  # output_height
            512,  # output_width
            512,  # midblock_in_channels
            MIDBLOCK_RESNET_NORM_NUM_BLOCKS,
            MIDBLOCK_RESNET_CONV_CHANNEL_SPLIT_FACTORS,
            [512, 512, 256, 128],  # upblock_out_channels
            [128, 256, 512, 512],  # upblock_out_dimensions
            UPBLOCK_RESNET_NORM_NUM_BLOCKS,
            UPBLOCK_RESNET_CONV_CHANNEL_SPLIT_FACTORS,
            UPBLOCK_UPSAMPLE_CONV_CHANNEL_SPLIT_FACTORS,
        ),
    ],
)
def test_decoder(
    device,
    input_channels,
    input_height,
    input_width,
    out_channels,
    output_height,
    output_width,
    midblock_in_channels,
    midblock_resnet_norm_blocks,
    midblock_conv_channel_split_factors,
    upblock_out_channels,
    upblock_out_dimensions,
    upblock_resnet_norm_blocks,
    upblock_resnet_conv_in_channel_split_factors,
    upblock_upsample_conv_channel_split_factors,
    use_program_cache,
):
    torch.manual_seed(0)

    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
    torch_decoder = vae.decoder

    # Run pytorch model
    torch_input = torch.randn([1, input_channels, input_height, input_width])
    torch_output = torch_decoder(torch_input)

    # Initialize ttnn model
    ttnn_model = VaeDecoder(
        torch_decoder,
        device,
        input_channels,
        input_height,
        input_width,
        out_channels,
        output_height,
        output_width,
        midblock_in_channels,
        midblock_resnet_norm_blocks,
        midblock_conv_channel_split_factors,
        upblock_out_channels,
        upblock_out_dimensions,
        upblock_resnet_norm_blocks,
        upblock_resnet_conv_in_channel_split_factors,
        upblock_upsample_conv_channel_split_factors,
    )

    # Prepare ttnn input
    ttnn_input = ttnn.from_torch(
        torch_input.permute([0, 2, 3, 1]), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
    )

    # Run ttnn model twice to test program cache and weights reuse
    ttnn_output = ttnn_model(ttnn_input)
    ttnn_input = ttnn.from_torch(
        torch_input.permute([0, 2, 3, 1]), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
    )
    ttnn_output = ttnn_model(ttnn_input)

    ttnn_output = ttnn.reshape(ttnn_output, [1, output_height, output_width, out_channels])
    ttnn_output = ttnn.permute(ttnn_output, [0, 3, 1, 2])
    ttnn_output = ttnn.to_torch(ttnn_output)

    # TODO: Improve PCC (issue #21131)
    assert_with_pcc(torch_output, ttnn_output, 0.959)
