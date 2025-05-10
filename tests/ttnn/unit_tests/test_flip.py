# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "input_tensor, dim",
    [
        (torch.arange(12).reshape([4, 3]), [0]),
        (torch.arange(12).reshape([4, 3]), [0]),
        (torch.arange(6).reshape([2, 3]), [0]),
        (torch.arange(6).reshape([2, 3]), [0]),
        (torch.arange(16).reshape([4, 4]), [0]),
        (torch.arange(16).reshape([4, 4]), [0]),
        (torch.arange(25).reshape([5, 5]), [0]),
        (torch.arange(25).reshape([5, 5]), [0]),
        ([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]], [0, 2]),
        ([[[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]]], [0, 2, 3]),
        (torch.arange(24).reshape([2, 4, 3]), [0, 1]),
        (torch.arange(24).reshape([2, 4, 3]), [1, 0]),
        (torch.arange(30).reshape([3, 5, 2]), [0, 1]),
        (torch.arange(30).reshape([3, 5, 2]), [0, 1]),
        (torch.arange(48).reshape([2, 3, 4, 2]), [0, 1]),
        (torch.arange(48).reshape([2, 3, 4, 2]), [1, 2]),
        (torch.arange(48).reshape([2, 3, 4, 2]), [2]),
        (torch.arange(48).reshape([2, 3, 4, 2]), [0, 2]),
        (torch.arange(48).reshape([2, 3, 4, 2]), [2, 0]),
        (torch.arange(100).reshape([10, 10]), [0]),
        (torch.arange(120).reshape([5, 4, 6]), [0, 1]),
        (torch.arange(120).reshape([5, 4, 6]), [1]),
        (torch.arange(120).reshape([5, 4, 6]), [0]),
        (torch.arange(120).reshape([5, 4, 6]), [0, 1]),
        (torch.arange(8).reshape([2, 2, 2]), [0, 1]),
        (torch.arange(8).reshape([2, 2, 2]), [1]),
        (torch.arange(8).reshape([2, 2, 2]), [0]),
        (torch.arange(8).reshape([2, 2, 2]), [0, 1]),
        (
            [
                [
                    [[[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]]],
                    [[[[[17, 18], [19, 20]], [[21, 22], [23, 24]]], [[[25, 26], [27, 28]], [[29, 30], [31, 32]]]]],
                ]
            ],
            [0, 4],
        ),
    ],
)
def test_ttnn_flip(device, input_tensor, dim):
    tensor = torch.tensor(input_tensor, dtype=torch.bfloat16)
    ttnn_tensor = ttnn.from_torch(tensor)

    ttnn_tensor = ttnn.to_layout(ttnn_tensor, ttnn.ROW_MAJOR_LAYOUT)

    ttnn_tensor = ttnn.to_device(ttnn_tensor, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    torch_flipped = torch.flip(tensor, dim)
    ttnn_flipped = ttnn.flip(ttnn_tensor, dim)

    ttnn_flipped_torch = ttnn.to_torch(ttnn_flipped)

    assert_with_pcc(torch_flipped, ttnn_flipped_torch)
    assert (
        torch_flipped.shape == ttnn_flipped_torch.shape
    ), f"Shape mismatch: {torch_flipped.shape} vs {ttnn_flipped_torch.shape}"
