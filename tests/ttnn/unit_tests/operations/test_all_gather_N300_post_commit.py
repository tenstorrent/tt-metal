# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from models.utility_functions import skip_for_grayskull
from tests.ttnn.unit_tests.operations.test_all_gather import (
    run_all_gather_on_n300_impl,
    run_all_gather_sharded_n300,
)


# Enumerate the post-commit cases explicitly
@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_devices, num_links, output_shape, dim, layout",
    [
        (2, 1, [1, 1, 64, 16384], 3, ttnn.TILE_LAYOUT),
        (2, 1, [8, 5, 32, 768], 3, ttnn.TILE_LAYOUT),
        (2, 1, [1, 1, 32, 736], 3, ttnn.TILE_LAYOUT),
        (2, 1, [1, 1, 32, 704], 3, ttnn.TILE_LAYOUT),
        (2, 1, [1, 1, 64, 704], 3, ttnn.TILE_LAYOUT),
        (2, 1, [1, 1, 32, 736], 3, ttnn.ROW_MAJOR_LAYOUT),
        (2, 1, [1, 1, 32, 704], 3, ttnn.ROW_MAJOR_LAYOUT),
        (2, 1, [1, 1, 64, 704], 3, ttnn.ROW_MAJOR_LAYOUT),
        (2, 1, [4, 1, 256, 32], 0, ttnn.ROW_MAJOR_LAYOUT),
        (2, 1, [8, 1, 256, 32], 0, ttnn.ROW_MAJOR_LAYOUT),
        (2, 1, [1, 1, 32, 8192], 3, ttnn.ROW_MAJOR_LAYOUT),
        (2, 1, [8, 5, 13, 512], 3, ttnn.ROW_MAJOR_LAYOUT),
        (2, 1, [8, 5, 13, 768], 3, ttnn.ROW_MAJOR_LAYOUT),
        (2, 1, [8, 8, 256, 384], 1, ttnn.ROW_MAJOR_LAYOUT),
    ],
)
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,
    ],
)
@pytest.mark.parametrize(
    "mem_config",
    [
        ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM),
    ],
)
@pytest.mark.parametrize("num_iters", [1])
@pytest.mark.parametrize("enable_async", [True, False])
def test_all_gather_on_n300_post_commit(
    n300_mesh_device,
    num_devices,
    output_shape,
    dim,
    num_links,
    input_dtype,
    layout,
    mem_config,
    num_iters,
    use_program_cache,
    function_level_defaults,
    enable_async,
):
    run_all_gather_on_n300_impl(
        n300_mesh_device,
        num_devices,
        output_shape,
        dim,
        num_links,
        input_dtype,
        layout,
        mem_config,
        use_program_cache,
        function_level_defaults,
        all_gather_operation=ttnn.all_gather,
        num_iters=num_iters,
        enable_async=enable_async,
    )


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize("num_devices", [2])
@pytest.mark.parametrize("dim", [3])
@pytest.mark.parametrize("tensor_layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,
    ],
)
@pytest.mark.parametrize(
    "tensor_mem_layout",
    [
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    ],
)
@pytest.mark.parametrize("orientation", [ttnn.ShardOrientation.ROW_MAJOR])
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize(
    "input_shape, input_shard_shape,shard_grid",
    (
        (
            (1, 1, 128, 8192),
            (128, 256),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
        ),
        (
            (1, 1, 32, 1792),
            (32, 32),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))}),
        ),
    ),
)
@pytest.mark.parametrize("enable_async", [True])
def test_all_gather_sharded_n300_post_commit(
    n300_mesh_device,
    num_devices,
    input_shape,
    input_shard_shape,
    shard_grid,
    dim,
    num_links,
    orientation,
    input_dtype,
    tensor_layout,
    tensor_mem_layout,
    # num_cores,
    use_program_cache,
    function_level_defaults,
    enable_async,
):
    run_all_gather_sharded_n300(
        n300_mesh_device,
        num_devices,
        input_shape,
        input_shard_shape,
        shard_grid,
        dim,
        num_links,
        orientation,
        input_dtype,
        tensor_layout,
        tensor_mem_layout,
        # num_cores,
        use_program_cache,
        function_level_defaults,
        all_gather_operation=ttnn.all_gather,
        enable_async=enable_async,
    )
