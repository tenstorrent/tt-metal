# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

compute_grid = ttnn.CoreGrid(x=13, y=10)  # actual 13, 10
num_tiles_per_dim = 10
tensor_size = (32 * num_tiles_per_dim * compute_grid.y, 32 * num_tiles_per_dim * compute_grid.x)


# Limitations: We only consider interleaved memory config since we cannot tilize with block or width sharded memory config,
#   and with height sharded tilize only involves local data movement
def test_interleaved_tilize_untilize(device):
    # Create input tensor
    torch_input = torch.rand(tensor_size, dtype=torch.bfloat16)

    # TT operations
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    tt_output = ttnn.tilize(tt_input)

    tt_output = ttnn.untilize(tt_output)

    # Convert back to PyTorch for comparison
    tt_result = ttnn.to_torch(tt_output)

    # PyTorch reference operations
    print(tt_output.layout)
    print(tt_output.memory_config())

    # Compare results
    assert_with_pcc(torch_input, tt_result, 0.9999)
