# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.demos.wormhole.stable_diffusion.tt.vae.ttnn_vae_configs import (
    get_default_compute_config,
    get_default_conv_config,
)
from models.demos.wormhole.stable_diffusion.tt.vae.ttnn_vae_utils import (
    prepare_split_conv_weights_bias,
    split_conv_and_run,
)
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "in_channels, input_height, input_width, out_channels, output_height, output_width, conv_in_channel_split_factor, conv_out_channel_split_factor",
    [
        (512, 128, 128, 512, 128, 128, 1, 1),
        (512, 256, 256, 512, 256, 256, 2, 2),
        (256, 512, 512, 256, 512, 512, 4, 2),
    ],
)
def test_split_conv(
    device,
    in_channels,
    input_height,
    input_width,
    out_channels,
    output_height,
    output_width,
    conv_in_channel_split_factor,
    conv_out_channel_split_factor,
    use_program_cache,
):
    torch_input = torch.randn([1, in_channels, input_height, input_width])
    torch_weights = torch.randn([out_channels, in_channels, 3, 3])
    torch_biases = torch.randn([out_channels])

    torch_output = torch.nn.functional.conv2d(
        torch_input, torch_weights, bias=torch_biases, stride=(1, 1), padding=(1, 1), groups=1
    )

    conv_weights, conv_bias = prepare_split_conv_weights_bias(
        in_channels,
        out_channels,
        conv_in_channel_split_factor,
        conv_out_channel_split_factor,
        torch_weights,
        torch_biases.unsqueeze(0).unsqueeze(0).unsqueeze(0),
    )

    ttnn_input = ttnn.from_torch(
        torch_input.permute([0, 2, 3, 1]),
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = split_conv_and_run(
        ttnn_input,
        conv_weights,
        conv_bias,
        device,
        in_channels,
        input_height,
        input_width,
        out_channels,
        conv_in_channel_split_factor,
        conv_out_channel_split_factor,
        get_default_compute_config(device),
        get_default_conv_config(),
    )

    ttnn_output = ttnn.to_memory_config(ttnn_output, ttnn.DRAM_MEMORY_CONFIG)
    ttnn_output = ttnn.reshape(ttnn_output, [1, output_height, output_width, out_channels])
    ttnn_output = ttnn.permute(ttnn_output, [0, 3, 1, 2])
    ttnn_output = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_output, ttnn_output, 0.99)
