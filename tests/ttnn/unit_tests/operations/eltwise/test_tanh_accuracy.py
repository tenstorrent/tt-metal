# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


def test_tanh_range(device):
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
        dtype=torch.bfloat16,
    )
    torch_output_tensor = torch.tanh(torch_input_tensor_a)
    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output_tensor = ttnn.tanh(input_tensor_a, memory_config=ttnn.DRAM_MEMORY_CONFIG, accuracy=True)

    output_tensor = ttnn.to_torch(output_tensor)

    pcc, pcc_msg = assert_with_pcc(torch_output_tensor, output_tensor, 0.999)
    # pcc_msg 0.9999663646890817, accuracy=False pcc 0.9978378297942829
    # pcc_msg 0.9999583453515977 - fpu arithmetic, pcc_msg 0.9999669593009368 sfpu arithmetic
    assert pcc


@pytest.mark.parametrize(
    "high, low",
    [
        (1, -1),
        (100, -100),
        (4, -4),
    ],
)
def test_tanh_inplace(device, high, low):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.rand([1, 9, 8192], dtype=torch.bfloat16) * (high - low) + low
    torch_output_tensor = torch.tanh(torch_input_tensor_a)
    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.tanh(input_tensor_a, accuracy=True, memory_config=ttnn.DRAM_MEMORY_CONFIG, output_tensor=input_tensor_a)
    output_tensor = ttnn.to_torch(input_tensor_a)

    pcc, pcc_msg = assert_with_pcc(torch_output_tensor, output_tensor, 0.999)
    assert pcc


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
        (1, -1),  # pcc_msg 0.99989
        (100, -100),
        (10000, -10000),
        (4, -4),  # pcc_msg 0.999985108313941
    ],
)
def test_tanh_accuracy(device, input_shapes, high, low):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((input_shapes), dtype=torch.bfloat16) * (high - low) + low
    golden_function = ttnn.get_golden_function(ttnn.tanh)
    torch_output_tensor = golden_function(torch_input_tensor)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    output = ttnn.tanh(input_tensor, accuracy=True)
    output_tensor = ttnn.to_torch(output)
    pcc, pcc_msg = assert_with_pcc(torch_output_tensor, output_tensor, 0.999)
    # pcc_msg 0.9999 or above
    assert pcc


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
def test_tanh_height_sharded(device, input_shapes, high, low):
    torch.manual_seed(0)

    in_data = torch.rand((input_shapes), dtype=torch.bfloat16) * (high - low) + low
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
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mem_config,
    )
    output_tensor = ttnn.tanh(input_tensor1, accuracy=True)
    output_tensor = ttnn.to_torch(output_tensor)
    golden_function = ttnn.get_golden_function(ttnn.tanh)
    golden_tensor = golden_function(in_data)

    pcc, pcc_msg = assert_with_pcc(golden_tensor, output_tensor, 0.999)
    # (-4,4) pcc_msg 0.9992087814894364, accuracy: pcc_msg 0.9999851389543495
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
def test_tanh_sharded(device, high, low, input_mem_config):
    torch.manual_seed(0)

    in_data = torch.rand([1, 1, 512, 512], dtype=torch.bfloat16) * (high - low) + low

    input_tensor1 = ttnn.from_torch(
        in_data,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=return_mem_config(input_mem_config),
    )
    output_tensor = ttnn.tanh(input_tensor1, accuracy=True)
    output_tensor = ttnn.to_torch(output_tensor)
    golden_function = ttnn.get_golden_function(ttnn.tanh)
    golden_tensor = golden_function(in_data)

    pcc, pcc_msg = assert_with_pcc(golden_tensor, output_tensor, 0.999)
    assert pcc
