# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


def test_transpose_with_reshape(device):
    # Create input tensor
    torch_input = torch.rand((1, 1, 2048, 512), dtype=torch.bfloat16)

    # TT operations
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    tt_input = tt_input.reshape(1, 2048, 4, 128)
    tt_output = ttnn.transpose(tt_input, 1, 2)

    # Convert back to PyTorch for comparison
    tt_result = ttnn.to_torch(tt_output)

    # PyTorch reference operations
    torch_ref = torch_input.view(1, 2048, 4, 128)
    torch_ref = torch_ref.transpose(1, 2)

    # Compare results
    assert_with_pcc(torch_ref, tt_result, 0.9999)
