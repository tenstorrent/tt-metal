# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

compute_grid = ttnn.CoreCoord(14, 10)
num_tiles_per_dim = 8
tensor_size = (32 * num_tiles_per_dim * compute_grid.y, 32 * num_tiles_per_dim * compute_grid.x)


def test_concat(device):
    # Create input tensor
    torch_input1 = torch.rand(tensor_size, dtype=torch.bfloat16)
    torch_input2 = torch.rand(tensor_size, dtype=torch.bfloat16)

    # TT operations
    tt_input1 = ttnn.from_torch(
        torch_input1,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    tt_input2 = ttnn.from_torch(
        torch_input2,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    tt_output = ttnn.concat([tt_input1, tt_input2], dim=1)

    ## Convert back to PyTorch for comparison
    # tt_result = ttnn.to_torch(tt_output)
    #
    ## print((tt_input).memory_config())
    #
    ## PyTorch reference operations
    # torch_ref = torch_input.view(tensor_size)
    # torch_ref = torch.concat([torch_input, torch_input], dim=1)
    #
    ## Compare results
    # assert_with_pcc(torch_ref, torch_ref, 0.9999)
