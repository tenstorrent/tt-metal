# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.demos.unet_3d.ttnn_impl.upsample3d import upsample3d
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "batch_size, num_channels, kernel_size, depth, height, width",
    [
        (1, 32, 2, 32, 32, 64),
        (1, 64, 8, 4, 8, 32),
        (1, 128, 4, 8, 16, 16),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=True)
def test_upsample3d(device, batch_size, num_channels, kernel_size, depth, height, width, model_location_generator):
    torch_input = torch.randn(batch_size, num_channels, depth, height, width).to(dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(
        torch_input.permute(0, 2, 3, 4, 1), dtype=ttnn.bfloat16, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    reference_model = torch.nn.Upsample(
        scale_factor=kernel_size,
        mode="nearest",
    ).to(dtype=torch.bfloat16)

    reference_model.eval()

    ttnn_output = upsample3d(
        ttnn_input,
        scale_factor=kernel_size,
    )
    torch_output = reference_model(torch_input).permute(0, 2, 3, 4, 1)

    ttnn_output = ttnn.to_layout(ttnn_output, ttnn.ROW_MAJOR_LAYOUT)
    ttnn_output = ttnn.to_torch(ttnn_output.cpu())

    assert_with_pcc(torch_output, ttnn_output, pcc=0.99)
