# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.demos.unet_3d.ttnn_impl.conv3d import Conv3D
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "batch_size, in_channels, out_channels, depth, height, width, kernel_size",
    [
        (1, 32, 32, 32, 64, 64, 3),
        (1, 64, 64, 16, 32, 32, 3),
        (1, 128, 64, 8, 16, 16, 3),
        (1, 256, 128, 4, 8, 8, 3),
        (1, 32, 32, 32, 64, 64, 1),
        (1, 64, 32, 16, 32, 32, 5),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=True)
def test_conv3d(
    device, batch_size, in_channels, out_channels, depth, height, width, kernel_size, model_location_generator
):
    torch_input = torch.randn(batch_size, in_channels, depth, height, width).to(dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(
        torch_input.permute(0, 2, 3, 4, 1), dtype=ttnn.bfloat16, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    reference_model = torch.nn.Conv3d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=1,
        padding="same",
        dilation=1,
        bias=True,
    ).to(dtype=torch.bfloat16)

    ttnn_model = Conv3D(device=device, in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
    params_dict = reference_model.state_dict()
    ttnn_model.load_state_dict(device, params_dict)

    torch_output = reference_model(torch_input).permute(0, 2, 3, 4, 1)

    ttnn_output = ttnn_model(ttnn_input)
    ttnn_output = ttnn.to_layout(ttnn_output, ttnn.ROW_MAJOR_LAYOUT)
    ttnn_output = ttnn.to_torch(ttnn_output.cpu())

    assert_with_pcc(torch_output, ttnn_output, pcc=0.99)
