# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

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
        torch_input_tensor_a = torch.Tensor(size=input_shape).uniform_(-100, 100).to(torch.bfloat16)

    torch_output_tensor = torch.isfinite(torch_input_tensor_a)

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

    output_tensor = ttnn.isfinite(input_tensor_a, memory_config=sharded_config)
    output_tensor = ttnn.to_torch(output_tensor)

    [passed, message] = check_with_pcc(torch_output_tensor, output_tensor, 0.999)
    assert passed, f"PCC={message}"


# X 8 Y 8 input_shape [256, 2, 5, 1536] DataType.BFLOAT16 Layout.TILE ShardStrategy.BLOCK ShardOrientation.COL_MAJOR tensor_hw_as_shard_shape False
# (False, '0.0011681649683373467')
# X 8 Y 8 input_shape [256, 2, 5, 1536] DataType.BFLOAT8_B Layout.TILE ShardStrategy.BLOCK ShardOrientation.COL_MAJOR tensor_hw_as_shard_shape False
# (False, '0.0001750748520755012')
# X 8 Y 8 input_shape [1, 256, 2, 2304] DataType.BFLOAT16 Layout.TILE ShardStrategy.BLOCK ShardOrientation.COL_MAJOR tensor_hw_as_shard_shape False
# (False, '3.9930462497966295e-05')
# X 8 Y 8 input_shape [1, 256, 2, 2304] DataType.BFLOAT8_B Layout.TILE ShardStrategy.BLOCK ShardOrientation.COL_MAJOR tensor_hw_as_shard_shape False
# (False, '-0.000952861622351697')
# X 8 Y 8 input_shape [32, 4, 8, 768] DataType.BFLOAT16 Layout.TILE ShardStrategy.BLOCK ShardOrientation.COL_MAJOR tensor_hw_as_shard_shape False
# (False, '0.00028035922183673165')
# X 8 Y 8 input_shape [32, 4, 8, 768] DataType.BFLOAT8_B Layout.TILE ShardStrategy.BLOCK ShardOrientation.COL_MAJOR tensor_hw_as_shard_shape False
# (False, '0.0016899023358967573')

test_sweep_args = [
    (
        (256, 2, 5, 1536),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.ShardStrategy.BLOCK,
        ttnn.ShardOrientation.COL_MAJOR,
        False,
    ),
    (
        (256, 2, 5, 1536),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.ShardStrategy.BLOCK,
        ttnn.ShardOrientation.ROW_MAJOR,
        False,
    ),
    (
        (256, 2, 5, 1536),
        ttnn.bfloat8_b,
        ttnn.TILE_LAYOUT,
        ttnn.ShardStrategy.BLOCK,
        ttnn.ShardOrientation.COL_MAJOR,
        False,
    ),
    (
        (1, 256, 2, 2304),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.ShardStrategy.BLOCK,
        ttnn.ShardOrientation.COL_MAJOR,
        False,
    ),
    (
        (1, 256, 2, 2304),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.ShardStrategy.BLOCK,
        ttnn.ShardOrientation.ROW_MAJOR,
        False,
    ),
    (
        (1, 256, 2, 2304),
        ttnn.bfloat8_b,
        ttnn.TILE_LAYOUT,
        ttnn.ShardStrategy.BLOCK,
        ttnn.ShardOrientation.COL_MAJOR,
        False,
    ),
    (
        (32, 4, 8, 768),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.ShardStrategy.BLOCK,
        ttnn.ShardOrientation.COL_MAJOR,
        False,
    ),
    (
        (32, 4, 8, 768),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.ShardStrategy.BLOCK,
        ttnn.ShardOrientation.ROW_MAJOR,
        False,
    ),
    (
        (32, 4, 8, 768),
        ttnn.bfloat8_b,
        ttnn.TILE_LAYOUT,
        ttnn.ShardStrategy.BLOCK,
        ttnn.ShardOrientation.COL_MAJOR,
        False,
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
        torch.isposinf,
        ttnn.isposinf,
        True,
        device,
    )
