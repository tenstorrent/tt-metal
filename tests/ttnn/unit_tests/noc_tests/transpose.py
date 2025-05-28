# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

compute_grid = ttnn.CoreGrid(x=13, y=10)  # actual 13, 10
num_tiles_per_dim = 10
tensor_size = (32 * num_tiles_per_dim * compute_grid.y, 32 * num_tiles_per_dim * compute_grid.x)
# x,y=multiple of (compute_grid.y*compute_grid.x*32) <= sqrt(max elements per core * num cores) so that its shardable in any orientation and maxes out L1
# but numbers above should be good enough


def test_transpose(device):
    # Create input tensor
    torch_input = torch.rand(tensor_size, dtype=torch.bfloat16)

    # print(torch_input)

    # TT operations
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    tt_output = ttnn.permute(tt_input, (1, 0))

    # Convert back to PyTorch for comparison
    tt_result = ttnn.to_torch(tt_output)

    # print(tt_result)

    # PyTorch reference operations
    torch_ref = torch_input.view(tensor_size)
    torch_ref = torch_ref.transpose(0, 1)

    # Compare results
    assert_with_pcc(torch_ref, tt_result, 0.9999)
