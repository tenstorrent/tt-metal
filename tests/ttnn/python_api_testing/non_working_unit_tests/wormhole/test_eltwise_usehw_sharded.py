# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import random
import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc, check_with_pcc
from tests.ttnn.python_api_testing.sweep_tests import ttnn_ops
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_rand_inf

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
    gen_infs,
    device,
):
    random.seed(0)
    data_seed = random.randint(0, 20000000)
    torch.manual_seed(data_seed)

    if gen_infs:
        torch_input_tensor_a = gen_rand_inf(input_shape, low=-100, high=100)
    else:
        torch_input_tensor_a = torch.Tensor(size=input_shape).uniform_(-50, 50).to(torch.bfloat16)

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
        (1, 25, 160, 32),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.ShardStrategy.BLOCK,
        ttnn.ShardOrientation.COL_MAJOR,
        True,
    ),
    (
        (1, 25, 160, 32),
        ttnn.bfloat8_b,
        ttnn.TILE_LAYOUT,
        ttnn.ShardStrategy.BLOCK,
        ttnn.ShardOrientation.COL_MAJOR,
        True,
    ),
    (
        (1, 2, 1248, 32),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.ShardStrategy.BLOCK,
        ttnn.ShardOrientation.COL_MAJOR,
        True,
    ),
    (
        (1, 2, 1248, 32),
        ttnn.bfloat8_b,
        ttnn.TILE_LAYOUT,
        ttnn.ShardStrategy.BLOCK,
        ttnn.ShardOrientation.COL_MAJOR,
        True,
    ),
    (
        (1, 2, 1472, 32),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.ShardStrategy.BLOCK,
        ttnn.ShardOrientation.COL_MAJOR,
        True,
    ),
    (
        (1, 2, 1472, 32),
        ttnn.bfloat8_b,
        ttnn.TILE_LAYOUT,
        ttnn.ShardStrategy.BLOCK,
        ttnn.ShardOrientation.COL_MAJOR,
        True,
    ),
    (
        (2, 1, 224, 128),
        ttnn.bfloat8_b,
        ttnn.TILE_LAYOUT,
        ttnn.ShardStrategy.BLOCK,
        ttnn.ShardOrientation.COL_MAJOR,
        True,
    ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, sharding_strategy, shard_orientation, hw_as_shard_shape",
    (test_sweep_args),
)
def test_eltwise_isfinite(input_shape, dtype, dlayout, sharding_strategy, shard_orientation, hw_as_shard_shape, device):
    run_tests(
        input_shape,
        dtype,
        dlayout,
        sharding_strategy,
        shard_orientation,
        hw_as_shard_shape,
        torch.isfinite,
        ttnn.isfinite,
        True,
        device,
    )


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, sharding_strategy, shard_orientation, hw_as_shard_shape",
    (test_sweep_args),
)
def test_eltwise_isinf(input_shape, dtype, dlayout, sharding_strategy, shard_orientation, hw_as_shard_shape, device):
    run_tests(
        input_shape,
        dtype,
        dlayout,
        sharding_strategy,
        shard_orientation,
        hw_as_shard_shape,
        torch.isinf,
        ttnn.isinf,
        True,
        device,
    )


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, sharding_strategy, shard_orientation, hw_as_shard_shape",
    (test_sweep_args),
)
def test_eltwise_isnan(input_shape, dtype, dlayout, sharding_strategy, shard_orientation, hw_as_shard_shape, device):
    run_tests(
        input_shape,
        dtype,
        dlayout,
        sharding_strategy,
        shard_orientation,
        hw_as_shard_shape,
        torch.isnan,
        ttnn.isnan,
        True,
        device,
    )


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, sharding_strategy, shard_orientation, hw_as_shard_shape",
    (test_sweep_args),
)
def test_eltwise_isneginf(input_shape, dtype, dlayout, sharding_strategy, shard_orientation, hw_as_shard_shape, device):
    run_tests(
        input_shape,
        dtype,
        dlayout,
        sharding_strategy,
        shard_orientation,
        hw_as_shard_shape,
        torch.isneginf,
        ttnn.isneginf,
        True,
        device,
    )


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, sharding_strategy, shard_orientation, hw_as_shard_shape",
    (test_sweep_args),
)
def test_eltwise_exp(input_shape, dtype, dlayout, sharding_strategy, shard_orientation, hw_as_shard_shape, device):
    run_tests(
        input_shape,
        dtype,
        dlayout,
        sharding_strategy,
        shard_orientation,
        hw_as_shard_shape,
        torch.exp,
        ttnn.exp,
        False,
        device,
    )


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, sharding_strategy, shard_orientation, hw_as_shard_shape",
    (test_sweep_args),
)
def test_eltwise_sin(input_shape, dtype, dlayout, sharding_strategy, shard_orientation, hw_as_shard_shape, device):
    run_tests(
        input_shape,
        dtype,
        dlayout,
        sharding_strategy,
        shard_orientation,
        hw_as_shard_shape,
        torch.sin,
        ttnn.sin,
        False,
        device,
    )


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, sharding_strategy, shard_orientation, hw_as_shard_shape",
    (test_sweep_args),
)
def test_eltwise_cos(input_shape, dtype, dlayout, sharding_strategy, shard_orientation, hw_as_shard_shape, device):
    run_tests(
        input_shape,
        dtype,
        dlayout,
        sharding_strategy,
        shard_orientation,
        hw_as_shard_shape,
        torch.cos,
        ttnn.cos,
        False,
        device,
    )


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, sharding_strategy, shard_orientation, hw_as_shard_shape",
    (test_sweep_args),
)
def test_eltwise_abs(input_shape, dtype, dlayout, sharding_strategy, shard_orientation, hw_as_shard_shape, device):
    run_tests(
        input_shape,
        dtype,
        dlayout,
        sharding_strategy,
        shard_orientation,
        hw_as_shard_shape,
        torch.abs,
        ttnn.abs,
        False,
        device,
    )


def nop(x, memory_config=None):
    return x


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, sharding_strategy, shard_orientation, hw_as_shard_shape",
    (test_sweep_args),
)
def test_eltwise_nop(input_shape, dtype, dlayout, sharding_strategy, shard_orientation, hw_as_shard_shape, device):
    run_tests(
        input_shape,
        dtype,
        dlayout,
        sharding_strategy,
        shard_orientation,
        hw_as_shard_shape,
        nop,
        nop,
        False,
        device,
    )
