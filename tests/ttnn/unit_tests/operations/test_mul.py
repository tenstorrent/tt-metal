# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from torch.nn import functional as F


# fmt: off
@pytest.mark.parametrize("scalar", [3.0])
# fmt: on
def test_multiply_not_4D(device, scalar):
    torch_input_tensor_a = torch.arange(32).to(dtype=torch.bfloat16)
    torch_input_tensor_b = torch.arange(32).to(dtype=torch.bfloat16)

    torch_output_tensor = torch_input_tensor_a * torch_input_tensor_b * scalar

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)

    output = input_tensor_a * input_tensor_b * scalar
    output = ttnn.to_torch(output, torch_rank=1)

    assert_with_pcc(torch_output_tensor, output, 0.9999)


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [64])
def test_mul_4D(device, h, w):
    torch_input_tensor_a = torch.rand((5, 64, h, w), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand((5, 64, h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.mul(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)
    output = ttnn.mul(input_tensor_a, input_tensor_b)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output_tensor, output, 0.9999)


# fmt: off
@pytest.mark.parametrize("scalar", [3.0, 0.125])
# fmt: on
def test_multiply_with_scalar(device, scalar):
    torch_input_tensor_a = torch.arange(1024).reshape(32, 32).to(dtype=torch.bfloat16)
    torch_output_tensor = scalar * torch_input_tensor_a

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    output = scalar * input_tensor_a
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output_tensor, output, 0.9999)


@pytest.mark.parametrize("output_memory_config", [ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
@pytest.mark.parametrize("input_shard_orientation", [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR])
@pytest.mark.parametrize("scalar", [3.0, 0.125])
def test_multiply_with_scalar_sharded(device, scalar, input_shard_orientation, output_memory_config):
    torch.manual_seed(0)
    torch_input_tensor_a = torch.rand(1024 * 32, dtype=torch.bfloat16).reshape(32, 32, 32)
    torch_output_tensor = scalar * torch_input_tensor_a

    shard_config = ttnn.create_sharded_memory_config(
        shape=(32, 32),
        core_grid=ttnn.CoreGrid(y=4, x=8),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=input_shard_orientation,
        use_height_and_width_as_shard_shape=True,
    )
    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, memory_config=shard_config, device=device
    )
    output = ttnn.mul(input_tensor_a, scalar, memory_config=output_memory_config)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output_tensor, output, 0.9999)


@pytest.mark.skip(reason="Unable to multiply scalar to tensor with int")
# fmt: off
@pytest.mark.parametrize("input_a,scalar", [
        ([13, 16, 42, 42], 0.125)
    ])
# fmt: on
def test_multiply_int32_with_scalar(device, input_a, scalar):
    torch_input_tensor_a = torch.as_tensor(input_a, dtype=torch.int32)
    torch_output_tensor = scalar * torch_input_tensor_a
    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    output = scalar * input_tensor_a
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output_tensor, output, 0.9999)
