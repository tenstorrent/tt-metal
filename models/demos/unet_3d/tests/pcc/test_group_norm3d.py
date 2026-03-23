# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.demos.unet_3d.ttnn_impl.group_norm3d import GroupNorm3D
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "batch_size, num_channels, num_groups, depth, height, width",
    [
        (1, 32, 4, 32, 64, 64),
        (1, 64, 4, 16, 32, 32),
        (1, 128, 16, 8, 16, 16),
        (1, 256, 8, 4, 8, 8),
        (1, 32, 8, 32, 64, 64),
        (1, 64, 8, 16, 32, 32),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=True)
def test_groupnorm3d(device, batch_size, num_channels, num_groups, depth, height, width, model_location_generator):
    torch_input = torch.randn(batch_size, num_channels, depth, height, width).to(dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(
        torch_input.permute(0, 2, 3, 4, 1), dtype=ttnn.bfloat16, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    reference_model = torch.nn.GroupNorm(
        num_groups=num_groups,
        num_channels=num_channels,
        eps=1e-05,
        affine=True,
    ).to(dtype=torch.bfloat16)

    ttnn_model = GroupNorm3D(device=device, num_channels=num_channels, num_groups=num_groups)
    params_dict = reference_model.state_dict()
    ttnn_model.load_state_dict(device, params_dict)

    torch_output = reference_model(torch_input).permute(0, 2, 3, 4, 1)

    ttnn_output = ttnn_model(ttnn_input, device)
    ttnn_output = ttnn.to_layout(ttnn_output, ttnn.ROW_MAJOR_LAYOUT)
    ttnn_output = ttnn.to_torch(ttnn_output.cpu())

    assert_with_pcc(torch_output, ttnn_output, pcc=0.99)
