# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc, assert_with_ulp

pytestmark = pytest.mark.use_module_device


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

    assert_with_ulp(torch_output_tensor, output, 0)


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

    assert_with_ulp(torch_output_tensor, output, 0)


# fmt: off
@pytest.mark.parametrize("scalar", [3.0, 0.125])
# fmt: on
def test_multiply_with_scalar(device, scalar):
    torch_input_tensor_a = torch.arange(1024).reshape(32, 32).to(dtype=torch.bfloat16)
    torch_output_tensor = scalar * torch_input_tensor_a

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    output = scalar * input_tensor_a
    output = ttnn.to_torch(output)

    assert_with_ulp(torch_output_tensor, output, 0)


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

    assert_with_ulp(torch_output_tensor, output, 0)


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


#  #14840: use DRAM config
@pytest.mark.parametrize("output_memory_config", [ttnn.DRAM_MEMORY_CONFIG])
@pytest.mark.parametrize("scalar", [0.125])
@pytest.mark.parametrize("batch_size", [6, 7, 8])
def test_multiply_float32_with_scalar_sharded(device, scalar, batch_size, output_memory_config):
    torch.manual_seed(0)
    torch_input_tensor_a = torch.rand((batch_size, 16, 384, 384), dtype=torch.float32)
    torch_output_tensor = scalar * torch_input_tensor_a

    # GS has smaller L1 than WH
    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG, device=device
    )
    output = ttnn.mul(input_tensor_a, scalar, memory_config=output_memory_config)
    output = ttnn.to_torch(output)

    assert_with_ulp(torch_output_tensor, output, 0)


def test_binary_mul_div_bf16(device):
    torch_dtype = torch.bfloat16
    ttnn_dtype = ttnn.bfloat16

    x_torch = torch.tensor([[508, 17]], dtype=torch_dtype)
    y_torch = torch.tensor([[748, 17]], dtype=torch_dtype)

    z_torch_mul = torch.mul(x_torch, y_torch)
    z_torch_div = torch.div(x_torch, y_torch)

    x_tt = ttnn.from_torch(x_torch, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt_mul = ttnn.mul(x_tt, y_tt, use_legacy=False)
    z_tt_div = ttnn.div(x_tt, y_tt, use_legacy=False)

    tt_out_mul = ttnn.to_torch(z_tt_mul)
    tt_out_div = ttnn.to_torch(z_tt_div)

    torch.set_printoptions(linewidth=200, threshold=10000, precision=15, sci_mode=False, edgeitems=17)
    print("z_tt_mul", z_tt_mul)
    print("tt_out_mul", tt_out_mul)
    print("z_torch_mul", z_torch_mul)
    # print("z_tt_div", z_tt_div)
    # print("tt_out_div", tt_out_div)
    # print("z_torch_div", z_torch_div)

    assert_with_ulp(z_torch_mul, tt_out_mul, 0)
    assert_with_ulp(z_torch_div, tt_out_div, 0)


def test_binary_mul_div_bf16_scalar(device):
    torch_dtype = torch.bfloat16
    ttnn_dtype = ttnn.bfloat16

    x_torch = torch.tensor([[508]], dtype=torch_dtype)
    y_torch = 748

    z_torch_mul = torch.mul(x_torch, y_torch)
    z_torch_div = torch.div(x_torch, y_torch)

    x_tt = ttnn.from_torch(x_torch, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = y_torch
    z_tt_mul = ttnn.mul(x_tt, y_tt, use_legacy=False)
    z_tt_div = ttnn.div(x_tt, y_tt, use_legacy=False)

    tt_out_mul = ttnn.to_torch(z_tt_mul)
    tt_out_div = ttnn.to_torch(z_tt_div)

    assert_with_ulp(z_torch_mul, tt_out_mul, 0)
    assert_with_ulp(z_torch_div, tt_out_div, 0)
