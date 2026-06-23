# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_equal


# Awkward tile counts: 1 tile, fewer tiles than DRAM banks (banks get 0 pages),
# a prime non-multiple of the 16 KB packet (8-tile) read, and a large multiple.
@pytest.mark.parametrize(
    "height, width",
    [
        (32, 32),  # 1 tile
        (32, 32 * 5),  # 5 tiles  (< 8 banks)
        (32, 32 * 13),  # 13 tiles (prime, forces over-read tail)
        (32 * 7, 32 * 11),  # 77 tiles
        (1024, 1024),  # 1024 tiles
    ],
)
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
