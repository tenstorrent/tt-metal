# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from diffusers import AutoencoderKL

import ttnn
from models.demos.wormhole.stable_diffusion.tt.vae.ttnn_vae_attention import Attention
from tests.ttnn.utils_for_testing import assert_with_pcc


# num heads 1
@pytest.mark.parametrize(
    "input_shape",
    [
        ([1, 512, 64, 64]),
    ],
)
def test_upsample(
    device,
    input_shape,
    use_program_cache,
):
    torch.manual_seed(0)
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
    torch_attention = vae.decoder.mid_block.attentions[0]

    batch_size, in_channels, height, width = input_shape

    # Run pytorch model
    torch_input = torch.randn(input_shape)
    torch_output = torch_attention(torch_input)

    # Initialize ttnn model
    ttnn_model = Attention(torch_attention, device, in_channels)

    ttnn_input = torch_input.reshape([1, 1, input_shape[1], input_shape[2] * input_shape[3]])
    ttnn_input = ttnn_input.permute(0, 1, 3, 2)

    # Prepare ttnn input
    ttnn_input = ttnn.from_torch(
        ttnn_input,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Run ttnn model twice to test program cache and weights reuse
    ttnn_output = ttnn_model(ttnn_input)
    ttnn_output = ttnn.permute(ttnn_output, [0, 1, 3, 2])
    ttnn_output = ttnn_output.reshape([batch_size, in_channels, height, width])

    ttnn_output = ttnn.to_torch(ttnn_output)
    assert_with_pcc(torch_output, ttnn_output, 0.99)
