# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

pytestmark = pytest.mark.use_module_device

import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_numeric_metrics
from models.common.utility_functions import torch_random

TEST_PADDING_VALUE = -42


@pytest.mark.parametrize(
    "shapes",
    [
        ([2, 1, 512, 2048], [1, 1, 256, 256], 2, 4),
        ([4, 4, 128, 128], [2, 2, 64, 64], 2, 4),
        ([4, 4, 128, 128], [2, 2, 64, 64], 0, 0),
    ],
)
@pytest.mark.parametrize("keepdim", [True])
def test_sum_nd_shard(device, shapes, keepdim):
    torch.manual_seed(0)
    dim = -2
    input_shape, shard_shape, end_x, end_y = shapes
    torch_input_tensor = torch.rand(input_shape)
    torch_output_tensor = torch.sum(torch_input_tensor, dim, keepdim)

    memory_config = ttnn.MemoryConfig(
        buffer_type=ttnn.BufferType.L1,
        nd_shard_spec=ttnn.NdShardSpec(
            shard_shape,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(end_x, end_y))}),
        ),
    )
    input_tensor = ttnn.from_torch(
        torch_input_tensor, dtype=ttnn.float32, device=device, layout=ttnn.TILE_LAYOUT, memory_config=memory_config
    )
    input_tensor = ttnn.fill_implicit_tile_padding(input_tensor, TEST_PADDING_VALUE)
    op_output_tensor = ttnn.sum(input_tensor, dim=dim, keepdim=keepdim)
    output_tensor = ttnn.to_torch(op_output_tensor)
    # test for equivalance
    assert_numeric_metrics(
        torch_output_tensor,
        output_tensor,
        pcc_threshold=0.999,
        rtol=0.001,
        atol=0.194,
        frobenius_threshold=0.001,
    )


@pytest.mark.parametrize(
    "sub_core_grids",
    (
        # single core
        ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 0))]),
        # multiple disjoint cores
        ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 6)),
                ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 6)),
            ]
        ),
    ),
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
@pytest.mark.parametrize("shape", [(4, 32, 63), (4, 32, 63, 63), (16, 41, 63, 63)])
def test_sum_subcores(device, sub_core_grids, dtype, shape):
    torch.manual_seed(0)

    # Prepare Torch input/output
    torch_input_tensor = torch_random(shape, -100, 100, dtype=torch.bfloat16)
    torch_output_tensor = torch.sum(torch_input_tensor)

    # Prepare TTNN input/output
    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device, dtype=dtype)
    input_tensor = ttnn.fill_implicit_tile_padding(input_tensor, TEST_PADDING_VALUE)
    output_tensor = ttnn.sum(input_tensor, sub_core_grids=sub_core_grids)

    # Compare
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    if dtype == ttnn.bfloat16:
        pcc_threshold = 0.999
        rtol = 1e-06
        atol = 4100
        frobenius_threshold = 0.008
    else:
        pcc_threshold = 0.999
        rtol = 0.015
        atol = 4200
        frobenius_threshold = 0.015
    # test for equivalance
    assert_numeric_metrics(
        torch_output_tensor,
        output_tensor,
        pcc_threshold=pcc_threshold,
        rtol=rtol,
        atol=atol,
        frobenius_threshold=frobenius_threshold,
    )
