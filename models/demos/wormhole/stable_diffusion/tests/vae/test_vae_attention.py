# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from diffusers import AutoencoderKL

import ttnn
from models.demos.wormhole.stable_diffusion.tt.vae.ttnn_vae_attention import Attention
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "input_shape",
    [
        ([1, 512, 64, 64]),
    ],
)
def test_vae_attention(
    device,
    input_shape,
    use_program_cache,
):
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
    torch_attention = vae.decoder.mid_block.attentions[0]

    batch_size, in_channels, height, width = input_shape

    # Run pytorch model
    torch_input = torch.randn(input_shape)
    torch_output = torch_attention(torch_input)

    # Initialize ttnn model
    ttnn_model = Attention(torch_attention, device, in_channels)

    # Prepare ttnn input
    ttnn_input = ttnn.from_torch(
        torch_input.permute([0, 2, 3, 1]), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b
    )

    # Run ttnn model twice to test program cache
    ttnn_output = ttnn_model(ttnn_input)
    ttnn_output = ttnn_model(ttnn_input)

    ttnn_output = ttnn.permute(ttnn_output, [0, 1, 3, 2])
    ttnn_output = ttnn_output.reshape([batch_size, in_channels, height, width])
    ttnn_output = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_output, ttnn_output, 0.99)
