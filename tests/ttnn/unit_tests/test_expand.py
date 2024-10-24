# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest


@pytest.mark.parametrize(
    "input_shape",
    [
        [1, 32],
        [8, 1],
        [32, 1],
    ],
    ids=["1d", "2d", "large_2d"],
)
@pytest.mark.parametrize(
    "output_shape",
    [
        [-1, 32],
        [32, -1, 32],
        [32, 32, -1, 32],
        [32, 32, 32, -1, 32],
        [4, 4, 4096, -1, 32],
    ],
    ids=["2d", "3d", "4d", "5d", "random_large_5d"],
)
@pytest.mark.parametrize(
    "tensor_layout",
    [
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.TILE_LAYOUT,
    ],
)
def test_expand(input_shape, output_shape, tensor_layout, device):
    torch.manual_seed(2024)
    torch_input_tensor = torch.rand(input_shape, dtype=torch.float32)
    torch_output_tensor = torch_input_tensor.expand(output_shape)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=tensor_layout, device=device)
    output_tensor = ttnn.expand(input_tensor, output_shape)

    output_tensor = ttnn.to_torch(output_tensor)
    assert torch.equal(torch_output_tensor, output_tensor)


@pytest.mark.parametrize(
    "tensor_layout",
    [
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.TILE_LAYOUT,
    ],
)
def test_expand_callback(tensor_layout, device, use_program_cache):
    num_program_cache_entries_list = []
    for i in range(2):
        test_expand([32, 1], [32, 32, 32], tensor_layout, device)
        num_program_cache_entries_list.append(device.num_program_cache_entries())

    assert num_program_cache_entries_list[0] > 0
    assert num_program_cache_entries_list[0] == num_program_cache_entries_list[1]
