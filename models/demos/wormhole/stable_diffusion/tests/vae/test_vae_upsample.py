# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from diffusers import AutoencoderKL

import ttnn
from models.demos.wormhole.stable_diffusion.tt.vae.ttnn_vae_configs import UPBLOCK_UPSAMPLE_CONV_CHANNEL_SPLIT_FACTORS
from models.demos.wormhole.stable_diffusion.tt.vae.ttnn_vae_upsample import UpsampleBlock
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "input_channels, input_height, input_width, out_channels, output_height, output_width, conv_channel_split_factor, block_id",
    [
        (512, 64, 64, 512, 128, 128, UPBLOCK_UPSAMPLE_CONV_CHANNEL_SPLIT_FACTORS[0], 0),
        (512, 128, 128, 512, 256, 256, UPBLOCK_UPSAMPLE_CONV_CHANNEL_SPLIT_FACTORS[1], 1),
        (256, 256, 256, 256, 512, 512, UPBLOCK_UPSAMPLE_CONV_CHANNEL_SPLIT_FACTORS[2], 2),
    ],
)
def test_upsample(
    device,
    input_channels,
    input_height,
    input_width,
    out_channels,
    output_height,
    output_width,
    conv_channel_split_factor,
    block_id,
    use_program_cache,
):
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
    torch_upsample = vae.decoder.up_blocks[block_id].upsamplers[0]

    # Run pytorch model
    torch_input = torch.randn([1, input_channels, input_height, input_width])
    torch_output = torch_upsample(torch_input)

    # Initialize ttnn model
    ttnn_model = UpsampleBlock(
        torch_upsample,
        device,
        input_channels,
        input_height,
        input_width,
        out_channels,
        output_height,
        output_width,
        conv_channel_split_factor[0],
        conv_channel_split_factor[1],
    )

    # Prepare ttnn input
    ttnn_input = ttnn.from_torch(
        torch_input.permute([0, 2, 3, 1]),
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Run ttnn model twice to test program cache and weights reuse
    ttnn_output = ttnn_model(ttnn_input)
    ttnn_output.deallocate(True)
    ttnn_output = ttnn_model(ttnn_input)

    ttnn_output = ttnn.reshape(
        ttnn_output,
        [1, output_height, output_width, out_channels],
    )
    ttnn_output = ttnn.permute(ttnn_output, [0, 3, 1, 2])
    ttnn_output = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_output, ttnn_output, 0.99)
