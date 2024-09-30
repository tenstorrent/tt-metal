# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


def test_repeat(device):
    torch_input_tensor = torch.randn((1, 2, 4, 4), dtype=torch.bfloat16)
    repeat_shape = (1, 2, 1, 1)

    torch_result = torch_input_tensor.repeat(repeat_shape)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn.repeat(input_tensor, ttnn.Shape(repeat_shape))
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_result, output, 0.9999)


@pytest.mark.parametrize(
    "input_shape, input_sharded_memory_config_args",
    [
        (
            (1, 2, 128, 128),
            dict(core_grid=ttnn.CoreGrid(y=4, x=1), strategy=ttnn.ShardStrategy.HEIGHT),
        ),
        (
            (1, 2, 128, 128),
            dict(core_grid=ttnn.CoreGrid(y=4, x=1), strategy=ttnn.ShardStrategy.WIDTH),
        ),
    ],
)
@pytest.mark.parametrize(
    "repeat_shape",
    [
        (1, 2, 1, 1),
        (1, 4, 1, 1),
        (1, 1, 4, 1),
        (4, 1, 1, 1),
        (1, 1, 1, 4),
    ],
)
def test_repeat_sharded(input_shape, input_sharded_memory_config_args, repeat_shape, device):
    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)

    input_sharded_memory_config = ttnn.create_sharded_memory_config(input_shape, **input_sharded_memory_config_args)

    torch_result = torch_input_tensor.repeat(repeat_shape)

    input_tensor = ttnn.from_torch(
        torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device, memory_config=input_sharded_memory_config
    )

    output = ttnn.repeat(input_tensor, ttnn.Shape(repeat_shape))
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_result, output, 0.9999)
