# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_equal


@pytest.mark.parametrize("height", [256])
@pytest.mark.parametrize("width", [512])
def test_bh_dram_read(device, height, width):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((height, width), dtype=torch.bfloat16)

    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Read-only op: returns nothing, must not mutate the input.
    result = ttnn.bh_dram_read(input_tensor)
    assert result is None

    output_tensor = ttnn.to_torch(input_tensor)
    assert_equal(torch_input_tensor, output_tensor)
