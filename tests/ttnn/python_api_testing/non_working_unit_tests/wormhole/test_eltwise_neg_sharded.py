# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import random
import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc, check_with_pcc


def run_tests(
    input_shape,
    dtype,
    dlayout,
    sharding_strategy,
    shard_orientation,
    tensor_hw_as_shard_shape,
    X,
    Y,
    torch_op,
    ttnn_op,
    device,
):
    random.seed(0)
    data_seed = random.randint(0, 20000000)
    torch.manual_seed(data_seed)

    torch_input_tensor_a = torch.Tensor(size=input_shape).uniform_(-9, 9).to(torch.bfloat16)
    torch_output_tensor = torch_op(torch_input_tensor_a)
    print(X)
    print(Y)
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
        (1, 2, 112, 448),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.ShardStrategy.BLOCK,
        ttnn.ShardOrientation.COL_MAJOR,
        False,
        7,
        7,
    ),
    ((416, 32), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.ShardStrategy.BLOCK, ttnn.ShardOrientation.COL_MAJOR, True, 3, 3),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, sharding_strategy, shard_orientation, hw_as_shard_shape, X, Y",
    (test_sweep_args),
)
def test_eltwise_neg(
    input_shape, dtype, dlayout, sharding_strategy, shard_orientation, hw_as_shard_shape, X, Y, device
):
    run_tests(
        input_shape,
        dtype,
        dlayout,
        sharding_strategy,
        shard_orientation,
        hw_as_shard_shape,
        X,
        Y,
        ttnn.get_golden_function(ttnn.neg),
        ttnn.neg,
        device,
    )
