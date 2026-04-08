# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.demos.unet_3d.torch_impl.decoder import DecoderTch
from models.demos.unet_3d.ttnn_impl.decoder import Decoder
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "batch_size, in_channels, out_channels, num_groups, kernel_size, depth, height, width",
    [
        (1, 32, 32, 4, 3, 32, 32, 16),
        (1, 64, 32, 4, 3, 16, 32, 4),
        (1, 128, 32, 16, 3, 8, 16, 16),
        (1, 256, 32, 8, 3, 4, 8, 8),
        (1, 32, 32, 8, 3, 32, 8, 16),
        (1, 64, 32, 8, 3, 16, 4, 32),
        (1, 256, 32, 8, 3, 4, 8, 8),
        (1, 32, 32, 8, 3, 32, 8, 8),
        (1, 64, 32, 8, 3, 4, 8, 32),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=True)
def test_decoder(
    device,
    batch_size,
    in_channels,
    out_channels,
    num_groups,
    kernel_size,
    depth,
    height,
    width,
    model_location_generator,
):
    torch_input = torch.randn(batch_size, in_channels, depth, height, width).to(dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(
        torch_input.permute(0, 2, 3, 4, 1), dtype=ttnn.bfloat16, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    torch_skip_connection = torch.randn(batch_size, out_channels, depth * 2, height * 2, width * 2).to(
        dtype=torch.bfloat16
    )
    ttnn_skip_connection = ttnn.from_torch(
        torch_skip_connection.permute(0, 2, 3, 4, 1),
        dtype=ttnn.bfloat16,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    reference_model = DecoderTch(
        in_channels=in_channels + out_channels,
        out_channels=out_channels,
        num_groups=num_groups,
        kernel_size=kernel_size,
    ).to(dtype=torch.bfloat16)

    ttnn_model = Decoder(
        device=device,
        in_channels=in_channels + out_channels,
        out_channels=out_channels,
        num_groups=num_groups,
        kernel_size=kernel_size,
    )
    params_dict = reference_model.state_dict()
    ttnn_model.load_state_dict(device, params_dict)

    torch_output = reference_model(torch_input, torch_skip_connection).permute(0, 2, 3, 4, 1)
    ttnn_output = ttnn_model(ttnn_input, ttnn_skip_connection, device)

    ttnn_output = ttnn.to_layout(ttnn_output, ttnn.ROW_MAJOR_LAYOUT)
    ttnn_output = ttnn.to_torch(ttnn_output.cpu())

    assert_with_pcc(torch_output, ttnn_output, pcc=0.99)
