# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F
from loguru import logger

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=True)
@pytest.mark.parametrize(
    "input_shape, grid_shape",
    [
        ((1, 256, 12, 40), (1, 7, 25281, 2)),
        ((1, 256, 24, 80), (1, 7, 25281, 2)),
        ((1, 256, 48, 160), (1, 7, 25281, 2)),
        ((16, 32, 100, 100), (16, 10000, 4, 2)),
        ((48, 32, 12, 20), (48, 3567, 8, 2)),
        ((8, 32, 100, 100), (8, 300, 4, 2)),
        ((8, 32, 100, 100), (8, 2000, 4, 2)),
        ((16, 32, 50, 50), (16, 10000, 1, 2)),
        ((48, 32, 80, 45), (48, 4832, 1, 2)),
        ((48, 32, 40, 23), (48, 4832, 1, 2)),
        ((48, 32, 20, 12), (48, 4832, 1, 2)),
        ((48, 32, 10, 6), (48, 4832, 1, 2)),
        ((8, 32, 50, 50), (8, 3604, 1, 2)),
    ],
)
def test_grid_sample_near_uniform_grid(device, input_shape, grid_shape):
    torch.manual_seed(0)

    _, grid_h, grid_w, _ = grid_shape

    torch_input_nchw = torch.randn(input_shape, dtype=torch.float32)
    torch_input_nhwc = torch_input_nchw.permute(0, 2, 3, 1).to(torch.bfloat16)

    torch_grid = torch.zeros(grid_shape, dtype=torch.bfloat16)

    # Generate varied grid coordinates to test interpolation
    for h in range(grid_h):
        for w in range(grid_w):
            if grid_w > 1:
                x_coord = 2.0 * (w + 0.5) / grid_w - 1.0
                # Add some variation to test interpolation between pixels
                x_coord += 0.1 * torch.randn(1).item()
                x_coord = max(-1.0, min(1.0, x_coord))  # Clamp to valid range
            else:
                x_coord = 0.0

            if grid_h > 1:
                y_coord = 2.0 * (h + 0.5) / grid_h - 1.0
                # Add some variation to test interpolation between pixels
                y_coord += 0.1 * torch.randn(1).item()
                y_coord = max(-1.0, min(1.0, y_coord))  # Clamp to valid range
            else:
                y_coord = 0.0

            torch_grid[:, h, w, 0] = x_coord  # x coordinate
            torch_grid[:, h, w, 1] = y_coord  # y coordinate

    torch_grid_float = torch_grid.to(torch.float32)
    torch_output_nchw = F.grid_sample(
        torch_input_nchw, torch_grid_float, mode="bilinear", padding_mode="zeros", align_corners=False
    )

    torch_output_nhwc = torch_output_nchw.permute(0, 2, 3, 1).to(torch.bfloat16)

    ttnn_input = ttnn.from_torch(torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_grid = ttnn.from_torch(torch_grid, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    ttnn_output = ttnn.grid_sample(ttnn_input, ttnn_grid)
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    pcc_passed, pcc_message = assert_with_pcc(torch_output_nhwc, ttnn_output_torch, pcc=0.99)
    logger.info(pcc_message)
