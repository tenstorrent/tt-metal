# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import random
import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc, check_with_pcc

Y, X = (8, 8)


def run_tests(
    input_shape,
    dtype,
    dlayout,
    sharding_strategy,
    shard_orientation,
    tensor_hw_as_shard_shape,
    torch_op,
    ttnn_op,
    device,
):
    random.seed(0)
    data_seed = random.randint(0, 20000000)
    torch.manual_seed(data_seed)

    torch_input_tensor_a = torch.Tensor(size=input_shape).uniform_(-100, 100).to(torch.bfloat16)
    torch_output_tensor = torch_op(torch_input_tensor_a)

    sharded_config = ttnn.create_sharded_memory_config(
        shape=input_shape,
        core_grid=ttnn.CoreGrid(y=Y, x=X),
        strategy=sharding_strategy,
        orientation=shard_orientation,
        use_height_and_width_as_shard_shape=tensor_hw_as_shard_shape,
    )

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=dtype,
        layout=dlayout,
        device=device,
        memory_config=sharded_config,
    )

    output_tensor = ttnn_op(input_tensor_a, memory_config=sharded_config)
    output_tensor = ttnn.to_torch(output_tensor)

    [passed, message] = check_with_pcc(torch_output_tensor, output_tensor, 0.999)
    assert passed, f"PCC={message}"


test_sweep_args = [
    (
        (16, 1, 256, 1024),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.ShardStrategy.BLOCK,
        ttnn.ShardOrientation.COL_MAJOR,
        False,
    ),
    (
        (2, 32, 256, 256),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.ShardStrategy.BLOCK,
        ttnn.ShardOrientation.COL_MAJOR,
        False,
    ),
    (
        (2, 5, 256, 1280),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.ShardStrategy.BLOCK,
        ttnn.ShardOrientation.COL_MAJOR,
        False,
    ),
    (
        (16, 1, 256, 1024),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.ShardStrategy.BLOCK,
        ttnn.ShardOrientation.ROW_MAJOR,
        False,
    ),
    (
        (2, 32, 256, 256),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.ShardStrategy.BLOCK,
        ttnn.ShardOrientation.ROW_MAJOR,
        False,
    ),
    (
        (2, 5, 256, 1280),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.ShardStrategy.BLOCK,
        ttnn.ShardOrientation.ROW_MAJOR,
        False,
    ),
    (
        (7936, 256),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.ShardStrategy.BLOCK,
        ttnn.ShardOrientation.ROW_MAJOR,
        False,
    ),
    (
        (5, 1, 32, 10240),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.ShardStrategy.WIDTH,
        ttnn.ShardOrientation.COL_MAJOR,
        False,
    ),
    (
        (2, 1, 352, 2048),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.ShardStrategy.WIDTH,
        ttnn.ShardOrientation.ROW_MAJOR,
        False,
    ),
    (
        (5, 1, 32, 10240),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.ShardStrategy.HEIGHT,
        ttnn.ShardOrientation.COL_MAJOR,
        False,
    ),
    (
        (2, 1, 352, 2048),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.ShardStrategy.HEIGHT,
        ttnn.ShardOrientation.ROW_MAJOR,
        False,
    ),
    (
        (5, 1, 160, 32),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.ShardStrategy.HEIGHT,
        ttnn.ShardOrientation.COL_MAJOR,
        True,
    ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, sharding_strategy, shard_orientation, hw_as_shard_shape",
    (test_sweep_args),
)
def test_eltwise_lgamma(input_shape, dtype, dlayout, sharding_strategy, shard_orientation, hw_as_shard_shape, device):
    run_tests(
        input_shape,
        dtype,
        dlayout,
        sharding_strategy,
        shard_orientation,
        hw_as_shard_shape,
        torch.lgamma,
        ttnn.lgamma,
        device,
    )


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, sharding_strategy, shard_orientation, hw_as_shard_shape",
    (test_sweep_args),
)
def test_eltwise_multigammaln(
    input_shape, dtype, dlayout, sharding_strategy, shard_orientation, hw_as_shard_shape, device
):
    run_tests(
        input_shape,
        dtype,
        dlayout,
        sharding_strategy,
        shard_orientation,
        hw_as_shard_shape,
        ttnn.get_golden_function(ttnn.multigammaln),
        ttnn.multigammaln,
        device,
    )
