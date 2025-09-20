# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

import pytest
from models.utility_functions import torch_random
from functools import partial
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import is_wormhole_b0


@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.remainder,
    ],
)
def test_remainder_fp32(device, ttnn_function):
    torch.manual_seed(213919)
    x_torch = torch.rand([2, 3, 64, 64], dtype=torch.float32)
    y_torch = torch.rand([2, 3, 64, 64], dtype=torch.float32)
    golden_fn = ttnn.get_golden_function(ttnn_function)
    z_torch = golden_fn(x_torch, y_torch, device=device)
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt_div = ttnn.remainder(x_tt, y_tt)
    tt_out = ttnn.to_torch(z_tt_div)

    status = ttnn.pearson_correlation_coefficient(z_torch, tt_out) >= 0.999
    assert status


@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.div_no_nan,
    ],
)
def test_div_no_nan_fp32(device, ttnn_function):
    x_torch = torch.tensor(
        [
            [
                15,
                0,
                2,
                0,
            ]
        ],
        dtype=torch.float32,
    )
    y_torch = torch.tensor(
        [
            [
                10,
                0,
                0,
                2,
            ]
        ],
        dtype=torch.float32,
    )
    # direct div sfpu result before where: tt out in torch TorchTensor([[1.5000,    nan,    inf, 0.0000]])
    # predicate b==0 tt out in torch TorchTensor([[0., 1., 1., 0.]])
    # t1 false tt out in torch TorchTensor([[1.5000,    nan,    nan, 0.0000]]) 0 * nan = nan, 0 * inf = nan ?
    # t2 true  tt out in torch TorchTensor([[0., 0., 0., 0.]])
    # t1 + t2 tt out in torch TorchTensor([[1.5000,    nan,    nan, 0.0000]]) 0 + nan = nan
    # final div_result = tt out in torch TorchTensor([[1.5000,    nan,    nan, 0.0000]])

    golden_fn = ttnn.get_golden_function(ttnn_function)
    z_torch = golden_fn(x_torch, y_torch)
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt = ttnn.from_torch(z_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt_div = ttnn.div_no_nan(x_tt, y_tt)
    tt_out = ttnn.to_torch(z_tt_div)

    status = ttnn.pearson_correlation_coefficient(z_torch, tt_out) >= 0.999
    assert status


@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.remainder,
    ],
)
def test_remainder_forge(device, ttnn_function):
    torch.manual_seed(213919)
    input1 = torch.randn(2, 32, 32)
    input2 = torch.randn(2, 32, 32)

    golden_fn = ttnn.get_golden_function(ttnn_function)
    torch_output = golden_fn(input1, input2, device=device)

    input1 = ttnn.from_torch(input1, dtype=ttnn.float32)
    input2 = ttnn.from_torch(input2, dtype=ttnn.float32)

    input1 = ttnn.to_device(input1, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    input2 = ttnn.to_device(input2, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    input1 = ttnn.to_layout(input1, ttnn.TILE_LAYOUT)
    input2 = ttnn.to_layout(input2, ttnn.TILE_LAYOUT)

    output = ttnn.remainder(input1, input2)

    output = ttnn.to_torch(output)

    status = ttnn.pearson_correlation_coefficient(torch_output, output) >= 0.999
    assert status


def generate_torch_tensor(shape, low, high, step=0.0025, dtype=torch.float32):
    num_elements = torch.prod(torch.tensor(shape))
    values = torch.arange(low, high + step, step, dtype=dtype)

    if values.numel() < num_elements:
        values = values.repeat((num_elements // values.numel()) + 1)
    values = values[:num_elements]
    return values.reshape(shape)


@pytest.mark.parametrize(
    "input_shapes",
    [[64, 640], [2, 32, 320], [2, 1, 1024, 1024], [1, 1, 32, 32], [1, 3, 320, 384], [1, 2, 32, 64, 64]],
)
def test_binary_fmod_bf16(
    device,
    input_shapes,
):
    torch_input_tensor_a = generate_torch_tensor(input_shapes, -100, 100, step=0.22, dtype=torch.bfloat16)
    torch_input_tensor_b = generate_torch_tensor(input_shapes, -80, 120, step=0.11, dtype=torch.bfloat16)
    torch_output_tensor = torch.fmod(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn.fmod(input_tensor_a, input_tensor_b)
    output = ttnn.to_torch(output)

    pcc = ttnn.pearson_correlation_coefficient(torch_output_tensor, output)
    assert pcc >= 0.99


# This test was added for #17362
# If input is a multiple of the scalar, the result should be 0, but both Torch and TT output either 0 or the scalar value itself depending on the operands.
# This inconsistency is persistent due to some fp precision loss in both Torch and TT.
# Eg: torch.remainder of (3, 1.5) = 0.0 and of (3, 0.003) = 0.003
# Eg: ttnn.remainder of (4, 0.004) = 0.004 and of (3, 0.003) = 0.0
@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([6, 5, 320, 320])),
        (torch.Size([3, 123, 115])),
        (torch.Size([69, 178])),
        (torch.Size([1024])),
        (torch.Size([])),
    ),
)
@pytest.mark.parametrize("scalar", [-0.0029, -0.002, -0.0005, 0.0, 0.0007, 0.001, 0.0025])
def test_unary_fmod(input_shapes, scalar, device):
    torch.manual_seed(0)
    if len(input_shapes) == 0:
        torch_input_tensor = torch.tensor(5.0, dtype=torch.bfloat16)
    else:
        torch_input_tensor = gen_func_with_cast_tt(
            partial(torch_random, low=-100, high=100, dtype=torch.bfloat16), ttnn.bfloat16
        )(input_shapes)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    golden_function = ttnn.get_golden_function(ttnn.fmod)
    torch_output_tensor = golden_function(torch_input_tensor, scalar, device=device)

    output_tensor = ttnn.fmod(input_tensor_a, scalar)
    output_tensor = ttnn.to_torch(output_tensor)

    if scalar == 0.0:
        assert ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor) >= 0.999
    else:
        assert torch.allclose(output_tensor, torch_output_tensor, atol=0.001, rtol=0)


@pytest.mark.parametrize(
    "input_shapes",
    ((torch.Size([1, 2, 32, 256])),),
)
@pytest.mark.parametrize(
    "ttnn_fn, atol",
    [
        (ttnn.remainder, 1e-10),
    ],
)
def test_width_sharded(device, input_shapes, ttnn_fn, atol):
    torch.manual_seed(0)
    high = 1000
    low = 600
    in_data1 = torch.rand((input_shapes), dtype=torch.bfloat16) * (high - low) + low
    high = 900
    low = 200
    in_data2 = torch.rand((input_shapes), dtype=torch.bfloat16) * (high - low) + low

    shard_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange((0, 0), (0, 7)),
        }
    )
    shard_spec = ttnn.ShardSpec(shard_grid, [64, 32], ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardMode.PHYSICAL)
    input_mem_config = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.WIDTH_SHARDED, ttnn.types.BufferType.L1, shard_spec
    )

    input_tensor1 = ttnn.from_torch(
        in_data1,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mem_config,
    )

    input_tensor2 = ttnn.from_torch(
        in_data2,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mem_config,
    )
    output_tensor = ttnn_fn(
        input_tensor1,
        input_tensor2,
        memory_config=input_mem_config,
    )

    output_tensor = ttnn.to_torch(output_tensor)
    golden_function = ttnn.get_golden_function(ttnn_fn)
    golden_tensor = golden_function(in_data1, in_data2, device=device)

    assert torch.allclose(golden_tensor, output_tensor, atol=atol)


@pytest.mark.parametrize(
    "input_shapes, shard_type, shard_shape",
    [
        (torch.Size([1, 4, 32, 960]), ttnn.types.TensorMemoryLayout.WIDTH_SHARDED, [128, 32]),
        (torch.Size([1, 2, 480, 32]), ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, [32, 32]),
        (torch.Size([1, 1, 512, 512]), ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, [64, 512]),
        (torch.Size([1, 1, 512, 512]), ttnn.types.TensorMemoryLayout.BLOCK_SHARDED, [64, 64]),
    ],
)
@pytest.mark.parametrize(
    "input_range",
    [
        {"high": 100, "low": -100},
        {"high": 3e38, "low": 1e-38},
        {"high": -1e38, "low": -3e-38},
    ],
)
def test_typecast_sharded(device, input_shapes, shard_type, shard_shape, input_range):
    if is_wormhole_b0 and device.get_num_devices() > 1:
        pytest.skip("Testing for 8 X 8 grid")

    high = input_range["high"]
    low = input_range["low"]
    in_data1 = torch.rand((input_shapes), dtype=torch.bfloat16) * (high - low) + low
    shard_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange((0, 0), (7, 7)),
        }
    )
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardMode.PHYSICAL)
    input_mem_config = ttnn.MemoryConfig(shard_type, ttnn.types.BufferType.L1, shard_spec)

    input_tensor1 = ttnn.from_torch(
        in_data1,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mem_config,
    )

    output_tensor = ttnn.typecast(
        input_tensor1,
        ttnn.float32,
        memory_config=input_mem_config,
    )

    output_tensor = ttnn.to_torch(output_tensor)
    golden_tensor = in_data1.to(torch.float32)

    assert torch.allclose(golden_tensor, output_tensor, atol=1e-6)
