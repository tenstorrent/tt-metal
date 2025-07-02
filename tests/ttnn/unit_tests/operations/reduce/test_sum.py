# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import torch_random


@pytest.mark.parametrize("batch_size", [1, 16])
@pytest.mark.parametrize("h", [32, 64, 41, 37])
@pytest.mark.parametrize("w", [32, 64, 31, 63])
@pytest.mark.parametrize("dim", [-1, -2, (2, 1)])
@pytest.mark.parametrize("keepdim", [True, False])
def test_sum(device, batch_size, h, w, dim, keepdim):
    torch.manual_seed(0)

    torch_input_tensor = torch_random((batch_size, h, w), -100, 100, dtype=torch.bfloat16)
    torch_output_tensor = torch.sum(torch_input_tensor, dim=dim, keepdim=keepdim)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.sum(input_tensor, dim=dim, keepdim=keepdim)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)

    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor)


@pytest.mark.parametrize("batch_size", [1, 16])
@pytest.mark.parametrize("h", [32, 64, 41, 37])
@pytest.mark.parametrize("w", [32, 64, 31, 63])
def test_sum_global(device, batch_size, h, w):
    torch.manual_seed(0)

    torch_input_tensor = torch_random((batch_size, h, w), -100, 100, dtype=torch.bfloat16)
    torch_output_tensor = torch.sum(torch_input_tensor)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.sum(input_tensor)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)

    output_tensor = ttnn.to_torch(output_tensor)
    output_tensor = output_tensor

    assert_with_pcc(torch_output_tensor, output_tensor)


@pytest.mark.parametrize("n", [1, 9])
@pytest.mark.parametrize("c", [1, 9])
@pytest.mark.parametrize("h", [9, 37])
@pytest.mark.parametrize("w", [9, 63])
@pytest.mark.parametrize("dim", [None, 0, 1, 2, 3])
def test_sum_4d(device, n, c, h, w, dim):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((n, c, h, w), dtype=torch.float32)
    torch_output_tensor = torch.sum(torch_input_tensor, dim=dim)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.sum(input_tensor, dim=dim)
    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)


@pytest.mark.parametrize(
    "shapes",
    [
        ([2, 1, 512, 2048], [1, 1, 256, 256], 2, 4),
        ([4, 4, 128, 128], [2, 2, 64, 64], 2, 4),
        ([4, 4, 128, 128], [2, 2, 64, 64], 0, 0),
    ],
)
@pytest.mark.parametrize("keepdim", [True])
def test_sum_nd_shard(device, shapes, keepdim):
    dim = -2
    input_shape, shard_shape, end_x, end_y = shapes
    torch_input_tensor = torch.rand(input_shape)
    torch_output_tensor = torch.sum(torch_input_tensor, dim, keepdim)

    memory_config = ttnn.MemoryConfig(
        buffer_type=ttnn.BufferType.L1,
        nd_shard_spec=ttnn.NdShardSpec(
            shard_shape,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(end_x, end_y))}),
        ),
    )
    input_tensor = ttnn.from_torch(
        torch_input_tensor, dtype=ttnn.float32, device=device, layout=ttnn.TILE_LAYOUT, memory_config=memory_config
    )
    op_output_tensor = ttnn.sum(input_tensor, dim=dim, keepdim=keepdim)
    output_tensor = ttnn.to_torch(op_output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.999)
