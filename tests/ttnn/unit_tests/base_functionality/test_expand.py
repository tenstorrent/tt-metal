# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest


def random_torch_tensor(dtype, shape):
    if dtype == ttnn.uint16:
        return torch.randint(0, 100, shape).to(torch.int16)
    if dtype == ttnn.int32:
        return torch.randint(-(2**31), 2**31, shape, dtype=torch.int32)
    if dtype == ttnn.uint32:
        return torch.randint(0, 2**31, shape, dtype=torch.int32)
    return torch.rand(shape).bfloat16().float()


@pytest.mark.parametrize(
    "input_shape, output_shape",
    [
        [(4, 1), (4, 2)],
        [(1, 32), (32, -1)],
        [(1, 32), (64, 32)],
        [(8, 1), (8, 8)],
        [(8, 1), (-1, 32)],
    ],
)
@pytest.mark.parametrize(
    "tensor_layout",
    [
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.TILE_LAYOUT,
    ],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.int32])
def test_expand(input_shape, output_shape, tensor_layout, dtype, device):
    torch.manual_seed(2024)
    torch_input_tensor = random_torch_tensor(dtype, input_shape)
    torch_output_tensor = torch_input_tensor.expand(output_shape)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=tensor_layout, device=device)
    with device.cache_entries_counter.measure():
        output_tensor = ttnn.expand(input_tensor, output_shape)

    output_tensor = ttnn.to_torch(output_tensor)
    assert torch.allclose(torch_output_tensor, output_tensor, atol=1e-1, rtol=1e-2)


@pytest.mark.parametrize(
    "tensor_layout",
    [
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.TILE_LAYOUT,
    ],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.int32])
def test_expand_callback(tensor_layout, dtype, device):
    test_expand([32, 1], [32, 32], tensor_layout, dtype, device)
    num_cache_entries = device.cache_entries_counter.total
    assert num_cache_entries > 0

    test_expand([32, 1], [32, 32], tensor_layout, dtype, device)
    assert device.cache_entries_counter.total == num_cache_entries
