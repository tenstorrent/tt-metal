# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.demos.unet_3d.torch_impl.model import UNet3DTch
from models.demos.unet_3d.ttnn_impl.model import UNet3D
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "batch_size, in_channels, out_channels, base_channels, num_groups, depth, height, width",
    [
        (1, 1, 1, 16, 8, 32, 32, 64),
        (1, 32, 32, 32, 8, 16, 64, 16),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=True)
def test_model(
    device,
    batch_size,
    in_channels,
    out_channels,
    base_channels,
    num_groups,
    depth,
    height,
    width,
    model_location_generator,
):
    torch_input = torch.randn(batch_size, in_channels, depth, height, width).to(dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, device=device, memory_config=ttnn.L1_MEMORY_CONFIG)

    reference_model = UNet3DTch(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=base_channels,
        num_groups=num_groups,
    ).to(dtype=torch.bfloat16)

    reference_model.eval()

    ttnn_model = UNet3D(
        device=device,
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=base_channels,
        num_groups=num_groups,
    )
    params_dict = reference_model.state_dict()
    ttnn_model.load_state_dict(params_dict)

    torch_output = reference_model(torch_input)
    ttnn_output = ttnn_model(ttnn_input)

    ttnn_output = ttnn.to_layout(ttnn_output, ttnn.ROW_MAJOR_LAYOUT)
    ttnn_output = ttnn.to_torch(ttnn_output.cpu())

    assert_with_pcc(torch_output, ttnn_output, pcc=0.97)
