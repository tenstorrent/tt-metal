# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from models.utility_functions import skip_for_grayskull, get_devices_for_t3000
from tests.ttnn.unit_tests.operations.test_reduce_scatter_post_commit import run_reduce_scatter_sharded_test


@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "num_devices, num_links",
    [
        # (4, 1),
        (8, 1),
    ],
)
@pytest.mark.parametrize("dim", [3])
@pytest.mark.parametrize(
    "tensor_mem_layout",
    [
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        # ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        # ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    ],
)
@pytest.mark.parametrize("tensor_layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("orientation", [ttnn.ShardOrientation.ROW_MAJOR])
@pytest.mark.parametrize(
    "input_dtype",
    [
        # ttnn.bfloat16,
        ttnn.bfloat8_b,
    ],
)
@pytest.mark.parametrize(
    "per_chip_output_shape,output_shard_shape,shard_grid",
    (
        # LLama
        (
            (1, 1, 32, 1024),
            (32, 32),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
        ),
    ),
    ids=["llama70b_t3k_decode"],
)
@pytest.mark.parametrize(
    "n_buffer",
    (
        # LLama
        1,
        2,
        3,
        4,
        6,
        8,
    ),
)
@pytest.mark.parametrize(
    "n_worker",
    (
        # LLama
        2,
        4,
        8,
        10,
        12,
    ),
)
@pytest.mark.parametrize("num_iters", [1000])
@pytest.mark.parametrize("math_op", [ttnn.ReduceType.Sum])
@pytest.mark.parametrize("enable_async", [True])
@pytest.mark.parametrize("device_params", [{"trace_region_size": 17068032}], indirect=True)
def test_width_sharded_reduce_scatter_post_commit(
    t3k_mesh_device,
    num_devices,
    per_chip_output_shape,
    output_shard_shape,
    dim,
    num_links,
    math_op,
    shard_grid,
    orientation,
    input_dtype,
    tensor_layout,
    tensor_mem_layout,
    use_program_cache,
    function_level_defaults,
    enable_async,
    num_iters,
    n_worker,
    n_buffer,
):
    logger.info(f"Running for n_worker={n_worker}, n_buffer={n_buffer}:")
    run_reduce_scatter_sharded_test(
        t3k_mesh_device,
        num_devices,
        per_chip_output_shape,
        output_shard_shape,
        dim,
        num_links,
        math_op,
        shard_grid,
        orientation,
        input_dtype,
        tensor_layout,
        tensor_mem_layout,
        use_program_cache=use_program_cache,
        function_level_defaults=function_level_defaults,
        enable_async=enable_async,
        num_iters=num_iters,
        n_worker=n_worker,
        n_buffer=n_buffer,
        trace_mode=True,
    )
