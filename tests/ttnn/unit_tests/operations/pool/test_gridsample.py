# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "input_shape",
    [
        [1, 1, 1, 3],
        [1, 1, 32, 32],
    ],
)
@pytest.mark.parametrize("align_corners", [True])
@pytest.mark.parametrize(
    "grid",
    [
        torch.tensor(
            [
                [0.25, 0.75],
                [0.75, 0.25],
                [0.5, 1.5],
                [1.5, 0.5],
                [0.0, 0.0],
                [1.0, 1.0],
            ],
            dtype=torch.float32,
        ).view(1, 6, 1, 2)
    ],
)
def test_gridsample(device, input_shape, align_corners, grid):
    torch.manual_seed(0)

    input_tensor = torch.rand((input_shape), dtype=torch.float32)
    torch_output = torch.nn.functional.grid_sample(input_tensor, grid, mode="bilinear", align_corners=align_corners)

    input_tensor = ttnn.from_torch(input_tensor, device=device)
    grid_tensor = ttnn.from_torch(grid, device=device)

    output_tensor = ttnn.gridsample(input_tensor, grid_tensor, mode="bilinear", align_corners=align_corners)
    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output, output_tensor, 0.99999)
