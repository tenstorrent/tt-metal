# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
from models.utility_functions import skip_for_grayskull
from tests.ttnn.unit_tests.operations.test_reduce_scatter_post_commit import (
    run_reduce_scatter_test,
    run_reduce_scatter_sharded_test,
)


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "num_devices, num_links",
    [
        (2, 1),
    ],
)
@pytest.mark.parametrize(
    "per_chip_output_shape, scatter_dim, layout",
    [
        ([1, 1, 32, 4096], 3, ttnn.TILE_LAYOUT),
        ([1, 1, 32, 2048], 3, ttnn.TILE_LAYOUT),
        ([1, 1, 32, 1024], 3, ttnn.TILE_LAYOUT),
        ([1, 1, 768, 4096], 3, ttnn.TILE_LAYOUT),
        ([1, 1, 768, 2048], 3, ttnn.TILE_LAYOUT),
        ([1, 1, 768, 1024], 3, ttnn.TILE_LAYOUT),
    ],
)
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,
        ttnn.bfloat8_b,
    ],
)
@pytest.mark.parametrize(
    "mem_config",
    [
        ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM),
        ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1),
    ],
)
@pytest.mark.parametrize("math_op", [ttnn.ReduceType.Sum])
@pytest.mark.parametrize("enable_async", [True])
def test_ring_reduce_scatter_n300_post_commit(
    n300_mesh_device,
    num_devices,
    per_chip_output_shape,
    scatter_dim,
    num_links,
    math_op,
    input_dtype,
    layout,
    mem_config,
    use_program_cache,
    function_level_defaults,
    enable_async,
    num_iters=5,
):
    run_reduce_scatter_test(
        n300_mesh_device,
        num_devices,
        per_chip_output_shape,
        scatter_dim,
        num_links,
        math_op,
        input_dtype,
        layout,
        mem_config,
        use_program_cache,
        function_level_defaults,
        num_iters=num_iters,
        enable_async=enable_async,
    )


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "num_devices, num_links",
    [
        (2, 1),
    ],
)
@pytest.mark.parametrize("dim", [3])
@pytest.mark.parametrize("tensor_layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("orientation", [ttnn.ShardOrientation.ROW_MAJOR])
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,
    ],
)
@pytest.mark.parametrize(
    "per_chip_output_shape,output_shard_shape,shard_grid,tensor_mem_layout",
    (
        (
            (1, 1, 32, 1792),
            (32, 32),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))}),
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ),
        (
            (1, 1, 1792, 32),
            (32, 32),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))}),
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ),
        (
            (1, 1, 224, 256),
            (32, 32),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))}),
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ),
    ),
)
@pytest.mark.parametrize("math_op", [ttnn.ReduceType.Sum])
@pytest.mark.parametrize("enable_async", [True])
def test_width_sharded_reduce_scatter_N300_post_commit(
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
    num_iters=5,
):
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
    )
