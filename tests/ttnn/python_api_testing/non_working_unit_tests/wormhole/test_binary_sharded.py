# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
from functools import partial
import random
import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc, check_with_pcc
from models.utility_functions import torch_random
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt


def run_binary_tests(
    input_shape,
    dtype,
    dlayout,
    Y,
    X,
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

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), dtype[0]
    )(input_shape)
    torch_input_tensor_b = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), dtype[1]
    )(input_shape)

    torch_input_tensor_a.requires_grad = True
    torch_input_tensor_b.requires_grad = True

    torch_output_tensor = torch_op(torch_input_tensor_a, torch_input_tensor_b)

    sharded_config = ttnn.create_sharded_memory_config(
        shape=input_shape,
        core_grid=ttnn.CoreGrid(y=Y, x=X),
        strategy=sharding_strategy,
        orientation=shard_orientation,
        use_height_and_width_as_shard_shape=tensor_hw_as_shard_shape,
    )

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=dtype[0],
        layout=dlayout,
        device=device,
        memory_config=sharded_config,
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=dtype[1],
        layout=dlayout,
        device=device,
        memory_config=sharded_config,
    )

    output_tensor = ttnn_op(input_tensor_a, input_tensor_b, memory_config=sharded_config)
    output_tensor = ttnn.to_torch(output_tensor).to(torch.float32)

    passed, output_string = check_with_pcc(torch_output_tensor, output_tensor, 0.999)

    assert passed, f"{output_string}\n{torch_output_tensor}\n{output_tensor}"


test_sweep_args = [
    (
        (256, 2, 5, 1536),
        [ttnn.bfloat16, ttnn.bfloat16],
        ttnn.TILE_LAYOUT,
        8,
        8,
        ttnn.ShardStrategy.BLOCK,
        ttnn.ShardOrientation.COL_MAJOR,
        False,
    ),
    (
        (1, 2, 128, 256),
        [ttnn.float32, ttnn.float32],
        ttnn.TILE_LAYOUT,
        8,
        8,
        ttnn.ShardStrategy.BLOCK,
        ttnn.ShardOrientation.ROW_MAJOR,
        False,
    ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, Y, X, sharding_strategy, shard_orientation, hw_as_shard_shape",
    (test_sweep_args),
)
def test_binary_fmod_sharded(
    input_shape, dtype, dlayout, Y, X, sharding_strategy, shard_orientation, hw_as_shard_shape, device
):
    run_binary_tests(
        input_shape,
        dtype,
        dlayout,
        Y,
        X,
        sharding_strategy,
        shard_orientation,
        hw_as_shard_shape,
        ttnn.get_golden_function(ttnn.fmod),
        ttnn.fmod,
        device,
    )
