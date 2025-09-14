# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc, assert_allclose


@pytest.mark.parametrize(
    "torch_dtype, ttnn_dtype, atol", [(torch.bfloat16, ttnn.bfloat16, 0.008), (torch.float32, ttnn.float32, 0.002)]
)
def test_tanh_range(device, torch_dtype, ttnn_dtype, atol):
    torch_input_tensor_a = torch.tensor(
        [
            [
                [
                    [
                        -1.8125,
                        -2.828125,
                        -3.125,
                        -3.234375,
                        -2.765625,
                        -1.890625,
                        -3.359375,
                        -2.0625,
                        -3.015625,
                        -2.203125,
                        -2.015625,
                        -2.9375,
                        -1.3046875,
                        -1.359375,
                        -1.3984375,
                        -1.2265625,
                        -2,
                        -3,
                        -1.5,
                        -2.5,
                        -3.5,
                        -3.75,
                        -3.359375,
                        -1.8828125,
                        -3.255,
                        -0.9,
                        -0.1,
                        0.25,
                        0.75,
                        -0.8359375,
                        -0.5,
                        0.9,
                    ]
                ]
            ]
        ],
        dtype=torch_dtype,
    )
    torch_output_tensor = torch.tanh(torch_input_tensor_a)
    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn_dtype,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output_tensor = ttnn.tanh(input_tensor_a, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    output_tensor = ttnn.to_torch(output_tensor)

    assert_allclose(output_tensor, torch_output_tensor, rtol=1e-05, atol=atol)
    pcc, pcc_msg = assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)
    # pcc_msg 0.9999663646890817, fast_and_approximate_mode=True pcc 0.9978378297942829
    # pcc_msg 0.9999583453515977 - fpu arithmetic, pcc_msg 0.9999669593009368 sfpu arithmetic
    # fp32 pcc_msg 0.9999829606828651 (fast_and_approximate_mode=False) , 0.9977552960423647 (fast_and_approximate_mode=True)
    # Single-tile tanh: accurate = 7886ns, approx = 1789ns (~77% faster)
    assert pcc


@pytest.mark.parametrize(
    "torch_dtype, ttnn_dtype, atol", [(torch.bfloat16, ttnn.bfloat16, 0.008), (torch.float32, ttnn.float32, 0.002)]
)
@pytest.mark.parametrize(
    "high, low",
    [
        (1, -1),
        (100, -100),
        (4, -4),
    ],
)
def test_tanh_inplace(device, high, low, torch_dtype, ttnn_dtype, atol):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.rand([1, 9, 8192], dtype=torch_dtype) * (high - low) + low
    torch_output_tensor = torch.tanh(torch_input_tensor_a)
    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn_dtype,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.tanh(input_tensor_a, memory_config=ttnn.DRAM_MEMORY_CONFIG, output_tensor=input_tensor_a)
    output_tensor = ttnn.to_torch(input_tensor_a)

    assert_allclose(output_tensor, torch_output_tensor, rtol=1e-05, atol=atol)
    pcc, pcc_msg = assert_with_pcc(torch_output_tensor, output_tensor, 0.999)
    assert pcc


@pytest.mark.parametrize(
    "torch_dtype, ttnn_dtype, atol", [(torch.bfloat16, ttnn.bfloat16, 0.008), (torch.float32, ttnn.float32, 0.002)]
)
@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 3, 320, 384])),
        (torch.Size([2, 4, 320, 1024])),
        (torch.Size([1, 9, 8192])),
    ),
)
@pytest.mark.parametrize(
    "high, low",
    [
        (1, -1),  # pcc_msg 0.99998
        (100, -100),
        (10000, -10000),
        (4, -4),  # pcc_msg 0.9999948671754642
    ],
)
def test_tanh_accuracy(device, input_shapes, high, low, torch_dtype, ttnn_dtype, atol):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((input_shapes), dtype=torch_dtype) * (high - low) + low
    golden_function = ttnn.get_golden_function(ttnn.tanh)
    torch_output_tensor = golden_function(torch_input_tensor)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    output = ttnn.tanh(input_tensor)
    output_tensor = ttnn.to_torch(output)

    assert_allclose(output_tensor, torch_output_tensor, rtol=1e-05, atol=atol)
    pcc, pcc_msg = assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)
    # pcc_msg 0.9999 or above
    assert pcc


@pytest.mark.parametrize("torch_dtype, ttnn_dtype, atol", [(torch.bfloat16, ttnn.bfloat16, 0.008)])
@pytest.mark.parametrize(
    "input_shapes",
    ((torch.Size([1, 1, 89600, 32])),),
)
@pytest.mark.parametrize(
    "high, low",
    [
        (1, -1),
        (100, -100),
        (10000, -10000),
        (4, -4),
    ],
)
def test_tanh_height_sharded(device, input_shapes, high, low, torch_dtype, ttnn_dtype, atol):
    torch.manual_seed(0)

    in_data = torch.rand((input_shapes), dtype=torch_dtype) * (high - low) + low
    shard_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(7, 6),
            ),
        }
    )
    n_cores = 56
    N, C, H, W = in_data.shape
    shard_spec = ttnn.ShardSpec(shard_grid, [N * C * H // n_cores, W], ttnn.ShardOrientation.ROW_MAJOR)
    input_mem_config = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, shard_spec
    )
    input_tensor1 = ttnn.from_torch(
        in_data,
        dtype=ttnn_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mem_config,
    )
    output_tensor = ttnn.tanh(input_tensor1)
    output_tensor = ttnn.to_torch(output_tensor)
    golden_function = ttnn.get_golden_function(ttnn.tanh)
    golden_tensor = golden_function(in_data)

    assert_allclose(output_tensor, golden_tensor, rtol=1e-05, atol=atol)
    pcc, pcc_msg = assert_with_pcc(golden_tensor, output_tensor, 0.999)
    assert pcc


def return_mem_config(mem_config_string):
    if mem_config_string == "l1_height_sharded_rm":
        return ttnn.create_sharded_memory_config(
            shape=(512 // 8, 512),
            core_grid=ttnn.CoreGrid(y=2, x=4),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
    elif mem_config_string == "l1_height_sharded_cm":
        return ttnn.create_sharded_memory_config(
            shape=(512, 512 // 8),
            core_grid=ttnn.CoreGrid(y=2, x=4),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.COL_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
    elif mem_config_string == "l1_width_sharded_rm":
        return ttnn.create_sharded_memory_config(
            shape=(512, 512 // 8),
            core_grid=ttnn.CoreGrid(y=2, x=4),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
    elif mem_config_string == "l1_width_sharded_cm":
        return ttnn.create_sharded_memory_config(
            shape=(512 // 8, 512),
            core_grid=ttnn.CoreGrid(y=2, x=4),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.COL_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
    elif mem_config_string == "l1_block_sharded_rm":
        return ttnn.create_sharded_memory_config(
            shape=(512 // 2, 512 // 4),
            core_grid=ttnn.CoreGrid(y=2, x=4),
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
    elif mem_config_string == "l1_block_sharded_cm":
        return ttnn.create_sharded_memory_config(
            shape=(512 // 2, 512 // 4),
            core_grid=ttnn.CoreGrid(y=2, x=4),
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.COL_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
    raise ("Input mem_config_string is not valid!")


@pytest.mark.parametrize(
    "torch_dtype, ttnn_dtype, atol", [(torch.bfloat16, ttnn.bfloat16, 0.008), (torch.float32, ttnn.float32, 0.002)]
)
@pytest.mark.parametrize(
    "high, low",
    [
        (1, -1),
        (100, -100),
        (10000, -10000),
        (4, -4),
    ],
)
@pytest.mark.parametrize(
    "input_mem_config",
    [
        "l1_height_sharded_rm",
        "l1_height_sharded_cm",
        "l1_width_sharded_rm",
        "l1_width_sharded_cm",
        "l1_block_sharded_rm",
        "l1_block_sharded_cm",
    ],
)
def test_tanh_sharded(device, high, low, input_mem_config, torch_dtype, ttnn_dtype, atol):
    torch.manual_seed(0)

    in_data = torch.rand([1, 1, 512, 512], dtype=torch_dtype) * (high - low) + low

    input_tensor1 = ttnn.from_torch(
        in_data,
        dtype=ttnn_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=return_mem_config(input_mem_config),
    )
    output_tensor = ttnn.tanh(input_tensor1)
    output_tensor = ttnn.to_torch(output_tensor)
    golden_function = ttnn.get_golden_function(ttnn.tanh)
    golden_tensor = golden_function(in_data)

    assert_allclose(output_tensor, golden_tensor, rtol=1e-05, atol=atol)
    pcc, pcc_msg = assert_with_pcc(golden_tensor, output_tensor, 0.999)
    assert pcc
