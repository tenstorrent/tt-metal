# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_equal, assert_with_ulp

pytestmark = pytest.mark.use_module_device


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

    assert_with_ulp(expected_result=torch_output_tensor, actual_result=output, ulp_threshold=0)


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

    assert_equal(torch_output_tensor, output)


def test_binary_mul_bf16_scalar(device):
    torch_dtype = torch.bfloat16
    ttnn_dtype = ttnn.bfloat16

    x_torch = torch.tensor(
        [
            [
                10,
                10.0625,
                10.125,
                100,
                1000,
            ]
        ],
        dtype=torch_dtype,
    ).repeat(32, 8)
    y_torch = 0.1

    z_torch_mul = torch.mul(x_torch, y_torch)

    x_tt = ttnn.from_torch(x_torch, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = y_torch
    z_tt_mul = ttnn.mul(x_tt, y_tt)

    tt_out_mul = ttnn.to_torch(z_tt_mul)

    assert_with_ulp(expected_result=z_torch_mul, actual_result=tt_out_mul, ulp_threshold=0)
