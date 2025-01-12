# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.unit_tests.operations.eltwise.backward.utility_funcs import (
    data_gen_with_range_batch_norm,
    compare_results_batch_norm,
)
from itertools import product


@pytest.mark.parametrize(
    "input_shapes",
    [
        *(torch.Size([n, c, 32, 32]) for n, c in product([1, 2, 3, 4], [1, 2, 3])),
        torch.Size([4, 4, 32, 32]),
        *(torch.Size([n, c, 23, 23]) for n, c in product([1, 2, 3, 4], [1, 2, 3])),
        torch.Size([4, 4, 23, 23]),
        *(torch.Size([n, c, 64, 120]) for n, c in product([1, 2], [1, 2, 3])),
        torch.Size([3, 1, 64, 120]),
        torch.Size([3, 2, 64, 120]),
    ],
)
@pytest.mark.parametrize("training", [False])
@pytest.mark.parametrize("weight", [True, False])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("eps", [1.0, 0.0, 2.34, 1e-05])
def test_batch_norm(input_shapes, training, weight, bias, eps, device):
    in_data, input_tensor = data_gen_with_range_batch_norm(input_shapes, 5, 10, device, is_input=True)
    mean_data, mean_tensor = (
        data_gen_with_range_batch_norm(input_shapes, 4, 10, device) if (not training) else (None, None)
    )
    var_data, var_tensor = (
        data_gen_with_range_batch_norm(input_shapes, 4, 20, device) if (not training) else (None, None)
    )
    weight_data, weight_tensor = data_gen_with_range_batch_norm(input_shapes, 4, 10, device) if weight else (None, None)
    bias_data, bias_tensor = data_gen_with_range_batch_norm(input_shapes, 4, 10, device) if bias else (None, None)

    tt_output_tensor_on_device = ttnn.batch_norm(
        input_tensor,
        running_mean=mean_tensor,
        running_var=var_tensor,
        training=training,
        eps=eps,
        weight=weight_tensor,
        bias=bias_tensor,
    )
    tt_output = ttnn.to_torch(tt_output_tensor_on_device)
    # ttnn.set_printoptions(profile="full")
    # print("TT result : ", tt_output, tt_output.shape)
    # torch.set_printoptions(precision=5, sci_mode=False)
    torch_result = torch.nn.functional.batch_norm(
        input=in_data,
        running_mean=mean_data,
        running_var=var_data,
        weight=weight_data,
        bias=bias_data,
        training=training,
        eps=eps,
    )
    # print("Torch result : ",torch_result)
    comp_pass = compare_results_batch_norm([tt_output], [torch_result])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    [
        torch.Size([3, 2, 32, 32]),
    ],
)
@pytest.mark.parametrize("mem_layout", [ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.TensorMemoryLayout.HEIGHT_SHARDED])
def test_batch_norm_program_cache_and_default(input_shapes, mem_layout, device):
    N, H, W, C = input_shapes
    in_data, input_tensor = data_gen_with_range_batch_norm(input_shapes, 5, 10, device, is_input=True)
    mean_data, mean_tensor = data_gen_with_range_batch_norm(input_shapes, 4, 10, device)
    var_data, var_tensor = data_gen_with_range_batch_norm(input_shapes, 4, 20, device)

    grid_size = ttnn.CoreGrid(y=1, x=8)
    grid_coord = ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1)
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
    shard_shape = N * H * W // grid_size.x, C // grid_size.y
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.COL_MAJOR)
    sharded_mem_config = ttnn.MemoryConfig(mem_layout, ttnn.types.BufferType.L1, shard_spec)

    if mem_layout is not ttnn.TensorMemoryLayout.INTERLEAVED:
        pytest.xfail("Input tensors to batch norm must be interleaved")

    tt_output_tensor_on_device = ttnn.batch_norm(
        input_tensor, running_mean=mean_tensor, running_var=var_tensor, memory_config=sharded_mem_config
    )
    tt_output = ttnn.to_torch(tt_output_tensor_on_device)
    torch_result = torch.nn.functional.batch_norm(input=in_data, running_mean=mean_data, running_var=var_data)
    comp_pass = compare_results_batch_norm([tt_output], [torch_result])
    assert comp_pass
