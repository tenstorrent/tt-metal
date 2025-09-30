# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import pytest
from models.common.utility_functions import run_for_wormhole_b0
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@run_for_wormhole_b0()
def test_case_multiply(device, reset_seeds):
    nx = 400
    torch_this_coors = torch.load(
        "models/experimental/pointpillars/test_case/this_coors_input.pt"
    )  # torch.Size([6522, 4])

    # Torch implementation
    # Multiply third column of this_coors with 400 and add fourth column of this_coors to the resultant
    torch_indices = torch_this_coors[:, 2] * nx + torch_this_coors[:, 3]
    # print("torch_indices: ", torch_indices.shape) # torch.Size([6522])

    # TTNN implementation
    ttnn_this_coors = ttnn.from_torch(
        torch_this_coors, device=device, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT
    )  # Shape([6522, 4])

    # Unsqueeze the input to 4d for ttnn slicing
    ttnn_this_coors = ttnn.unsqueeze(ttnn_this_coors, dim=0)
    ttnn_this_coors = ttnn.unsqueeze(ttnn_this_coors, dim=0)  # Shape([1, 1, 6522, 4])

    # Get the sliced tensors of third and fourth column of this_coors
    col3 = ttnn.slice(
        ttnn_this_coors,
        starts=[0, 0, 0, 2],
        ends=[1, 1, 6522, 3],
        steps=[1, 1, 1, 1],
    )  # Shape([1, 1, 6522, 1])
    col4 = ttnn.slice(
        ttnn_this_coors,
        starts=[0, 0, 0, 3],
        ends=[1, 1, 6522, 4],
        steps=[1, 1, 1, 1],
    )  # Shape([1, 1, 6522, 1])

    # Squeeze the sliced tensors to 3d
    col3 = ttnn.squeeze(col3, dim=3)  # Shape([1, 1, 6522])
    col4 = ttnn.squeeze(col4, dim=3)  # Shape([1, 1, 6522])

    # Creating tensor of 400 to multiply
    nx = ttnn.from_torch(torch.tensor([nx]), dtype=ttnn.uint32, device=device, layout=ttnn.TILE_LAYOUT)
    nx = ttnn.unsqueeze(nx, dim=0)
    nx = ttnn.unsqueeze(nx, dim=0)  # Shape([1, 1, 1])

    # Multiplying with nx=400
    indices_mul = ttnn.multiply(col3, nx)  # Shape([1, 1, 6522])

    # Adding with fourth column of input tensor
    indices_add = ttnn.add(indices_mul, col4)  # Shape([1, 1, 6522])

    indices_add = ttnn.squeeze(indices_add, dim=0)
    indices_add = ttnn.squeeze(indices_add, dim=0)  # Shape([6522])
    ttnn_indices_add = ttnn.to_torch(indices_add)

    # Compare the pcc of torch and ttnn implementation
    assert_with_pcc(torch_indices, ttnn_indices_add, 0.99)  # PCC: 0.04236401504801242
