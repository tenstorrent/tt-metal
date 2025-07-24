# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time


def random_torch_tensor(dtype, shape):
    if dtype == ttnn.uint16:
        return torch.randint(0, 100, shape).to(torch.int16)
    if dtype == ttnn.int32:
        return torch.randint(-(2**31), 2**31, shape, dtype=torch.int32)
    return torch.rand(shape).bfloat16().float()


@pytest.mark.parametrize(
    "input_shape, size",
    [
        # [(4, 1), (4, 2)],
        # [(1, 32), (32, -1)],
        # [(1, 32), (64, 32)],
        # [(8, 1), (8, 8)],
        [(1, 2, 3, 1), (3, -1, -1, 2)],
    ],
)
@pytest.mark.parametrize(
    "tensor_layout",
    [
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.TILE_LAYOUT,
    ],
)
@pytest.mark.parametrize("dtype", [ttnn.uint32])
def test_expand(input_shape, size, tensor_layout, dtype, device):
    torch.manual_seed(2024)
    # torch_input_tensor = torch.rand(input_shape, dtype=torch.float32)
    torch_input_tensor = random_torch_tensor(dtype, input_shape)
    torch_output_tensor = torch_input_tensor.expand(size)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=tensor_layout, device=device, dtype=dtype)
    output_tensor = ttnn.expand(input_tensor, size)

    output_tensor = ttnn.to_torch(output_tensor)
    result = check_with_pcc(torch_output_tensor, output_tensor, 0.999)
    print(torch_output_tensor)
    print(output_tensor)
    assert result[0], f"Expand operation failed with PCC check {result[1]}"
    # assert torch.allclose(torch_output_tensor, output_tensor, atol=1e-1, rtol=1e-2)


@pytest.mark.parametrize(
    "tensor_layout",
    [
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.TILE_LAYOUT,
    ],
)
def test_expand_callback(tensor_layout, device):
    num_program_cache_entries_list = []
    for i in range(2):
        test_expand([32, 1], [32, 32], tensor_layout, device)
        num_program_cache_entries_list.append(device.num_program_cache_entries())

    assert num_program_cache_entries_list[0] > 0
    assert num_program_cache_entries_list[0] == num_program_cache_entries_list[1]
