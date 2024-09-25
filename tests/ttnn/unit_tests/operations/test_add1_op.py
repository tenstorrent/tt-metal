# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc, assert_equal


@pytest.mark.parametrize("w", [[1, 4, 32, 32], [1, 4, 8, 32]])
def test_add_one_tile(device, w):
    torch_input_tensor = (
        torch.ones(
            (w),
        )
        * 10
    )
    torch_output_tensor = torch_input_tensor + 1
    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.add1(input_tensor)

    output_tensor = ttnn.to_torch(output_tensor)
    torch.set_printoptions(linewidth=200, threshold=10000, precision=15, sci_mode=False, edgeitems=17)
    assert_equal(torch_output_tensor, output_tensor)


@pytest.mark.parametrize("w", [[1, 4, 32, 32], [1, 4, 8, 32], [1, 2, 24, 24]])
def test_add_one_rm(device, w):
    torch_input_tensor = (
        torch.ones(
            (w),
        )
        * 23
    )
    torch_output_tensor = torch_input_tensor + 1
    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, device=device)

    output_tensor = ttnn.add1(input_tensor)
    output_tensor = ttnn.to_torch(output_tensor)
    assert_equal(torch_output_tensor, output_tensor)
