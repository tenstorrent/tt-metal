# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.demos.unet_3d.torch_impl.model import UNet3DTch
from models.demos.unet_3d.ttnn_impl.model import UNet3D


@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=True)
def test_encoder(
    device,
):
    batch_size, in_channels, out_channels, base_channels, num_groups, depth, height, width = 1, 1, 1, 16, 8, 32, 32, 64
    torch_input = torch.randn(batch_size, in_channels, depth, height, width).to(dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, device=device, memory_config=ttnn.L1_MEMORY_CONFIG)
    reference_model = UNet3DTch(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=base_channels,
        num_groups=num_groups,
    ).to(dtype=torch.bfloat16)
    params_dict = reference_model.state_dict()
    ttnn_model = UNet3D(
        device=device,
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=base_channels,
        num_groups=num_groups,
    )
    ttnn_model.load_state_dict(params_dict)

    _ttnn_output = ttnn_model(ttnn_input)
