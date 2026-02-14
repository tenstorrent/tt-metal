# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.demos.unet_3d.torch_impl.encoder import EncoderTch
from models.demos.unet_3d.ttnn_impl.encoder import Encoder
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "is_bottleneck, batch_size, in_channels, hid_channels, out_channels, num_groups, kernel_size, depth, height, width",
    [
        (False, 1, 32, 64, 32, 4, 3, 32, 64, 64),
        (False, 1, 64, 32, 32, 4, 3, 16, 32, 32),
        (False, 1, 128, 64, 32, 16, 3, 8, 16, 16),
        (False, 1, 256, 128, 32, 8, 3, 4, 8, 8),
        (False, 1, 32, 64, 32, 8, 3, 32, 64, 64),
        (False, 1, 64, 128, 32, 8, 3, 16, 32, 32),
        (True, 1, 256, 128, 32, 8, 3, 4, 8, 8),
        (True, 1, 32, 64, 32, 8, 3, 32, 64, 64),
        (True, 1, 64, 128, 32, 8, 3, 16, 32, 32),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=True)
def test_encoder(
    device,
    is_bottleneck,
    batch_size,
    in_channels,
    hid_channels,
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

    reference_model = EncoderTch(
        is_bottleneck=is_bottleneck,
        in_channels=in_channels,
        hid_channels=hid_channels,
        out_channels=out_channels,
        num_groups=num_groups,
        kernel_size=kernel_size,
    ).to(dtype=torch.bfloat16)

    ttnn_model = Encoder(
        device=device,
        is_bottleneck=is_bottleneck,
        in_channels=in_channels,
        hid_channels=hid_channels,
        out_channels=out_channels,
        num_groups=num_groups,
        kernel_size=kernel_size,
    )
    params_dict = reference_model.state_dict()
    ttnn_model.load_state_dict(device, params_dict)

    torch_output = reference_model(torch_input)
    ttnn_output = ttnn_model(ttnn_input, device)

    if is_bottleneck:
        torch_output = torch_output.permute(0, 2, 3, 4, 1)
    else:
        torch_output = torch_output[0].permute(0, 2, 3, 4, 1)
        ttnn_output = ttnn_output[0]
    ttnn_output = ttnn.to_layout(ttnn_output, ttnn.ROW_MAJOR_LAYOUT)
    ttnn_output = ttnn.to_torch(ttnn_output.cpu())

    assert_with_pcc(torch_output, ttnn_output, pcc=0.99)
