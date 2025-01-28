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


def run_binary_bw_tests(
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

    torch_grad_tensor = torch.Tensor(size=input_shape).uniform_(-50, 50).to(torch.bfloat16)
    torch_input_tensor_a = torch.Tensor(size=input_shape).uniform_(-50, 50).to(torch.bfloat16)
    torch_input_tensor_b = torch.Tensor(size=input_shape).uniform_(-50, 50).to(torch.bfloat16)

    torch_input_tensor_a.requires_grad = True
    torch_input_tensor_b.requires_grad = True

    torch_output_tensors = torch_op(torch_grad_tensor, torch_input_tensor_a, torch_input_tensor_b)

    sharded_config = ttnn.create_sharded_memory_config(
        shape=input_shape,
        core_grid=ttnn.CoreGrid(y=Y, x=X),
        strategy=sharding_strategy,
        orientation=shard_orientation,
        use_height_and_width_as_shard_shape=tensor_hw_as_shard_shape,
    )

    grad_tensor = ttnn.from_torch(
        torch_grad_tensor,
        dtype=dtype[0],
        layout=dlayout,
        device=device,
        memory_config=sharded_config,
    )
    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=dtype[1],
        layout=dlayout,
        device=device,
        memory_config=sharded_config,
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=dtype[2],
        layout=dlayout,
        device=device,
        memory_config=sharded_config,
    )

    output_tensors = ttnn_op(grad_tensor, input_tensor_a, input_tensor_b, memory_config=sharded_config)
    output_tensors = ttnn.to_torch(output_tensor)

    passed = []
    output_string = ""
    for i in range(len(torch_output_tensors)):
        output_tensor = ttnn.to_torch(output_tensors[i])
        passed_, output_string_ = check_with_pcc(torch_output_tensors[i], output_tensor, 0.999)
        passed.append(passed_)
        output_string += output_string_ + ", "

    if all(passed):
        passed = True
    else:
        passed = False

    output_string = output_string[:-2]
    e2e_perf = stop_measuring_time(start_time)

    assert passed, f"{output_string}"


test_sweep_args = [
    (
        (256, 2, 5, 1536),
        [ttnn.bfloat16, ttnn.bfloat16, ttnn.bfloat16],
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
def test_binary_backward_add(
    input_shape, dtype, dlayout, sharding_strategy, shard_orientation, hw_as_shard_shape, device
):
    run_binary_bw_tests(
        input_shape,
        dtype,
        dlayout,
        sharding_strategy,
        shard_orientation,
        hw_as_shard_shape,
        ttnn.get_golden_function(ttnn.add_bw),
        ttnn.add_bw,
        device,
    )
