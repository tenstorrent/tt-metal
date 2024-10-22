# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from tests.ttnn.unit_tests.operations.test_all_gather_TG_post_commit import (
    run_line_all_gather_on_TG_with_mesh_tensor_along_rows,
)
from models.utility_functions import skip_for_grayskull

from ttnn import ShardTensor2dMesh, ConcatMesh2dToTensor


# Enumerate the post-commit cases explicitly
@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_devices, num_links",
    [(4, 3)],
)
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,
        ttnn.bfloat8_b,
    ],
)
@pytest.mark.parametrize("shard_grid_orientation", [ttnn.ShardOrientation.ROW_MAJOR])
@pytest.mark.parametrize(
    "tensor_mem_layout,per_chip_output_shape, dim, input_shard_shape,shard_grid,layout",
    (
        # LLama
        (
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            (1, 1, 32, 1024 * 4),
            3,
            (32, 32),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
            ttnn.TILE_LAYOUT,
        ),
        (
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            (4, 1, 32, 1280),
            0,
            (32, 32),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 4))}),
            ttnn.TILE_LAYOUT,
        ),
    ),
)
@pytest.mark.parametrize("replication_factor", [8])
@pytest.mark.parametrize("enable_async", [True])
@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4_grid")], indirect=True)
def test_line_all_gather_sharded_on_TG_rows_post_commit(
    mesh_device,
    num_devices,
    per_chip_output_shape,
    input_shard_shape,
    shard_grid,
    shard_grid_orientation,
    tensor_mem_layout,
    dim,
    num_links,
    input_dtype,
    layout,
    use_program_cache,
    function_level_defaults,
    enable_async,
    replication_factor,
    num_iters=1,
):
    input_shard_spec = ttnn.ShardSpec(
        shard_grid,
        input_shard_shape,
        shard_grid_orientation,
        False,
    )
    run_line_all_gather_on_TG_with_mesh_tensor_along_rows(
        mesh_device,
        num_devices,
        per_chip_output_shape,
        tensor_mem_layout,
        dim,
        num_links,
        input_dtype,
        layout,
        ttnn.BufferType.L1,
        use_program_cache,
        function_level_defaults,
        enable_async=enable_async,
        input_shard_spec=input_shard_spec,
        num_iters=num_iters,
        num_all_gather_instances=replication_factor,
        cluster_axis=1,
    )


# Enumerate the post-commit cases explicitly
@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_devices, num_links",
    [(8, 4), (8, 3), (8, 2)],
)
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,
        ttnn.bfloat8_b,
    ],
)
@pytest.mark.parametrize("shard_grid_orientation", [ttnn.ShardOrientation.ROW_MAJOR])
@pytest.mark.parametrize(
    "tensor_mem_layout, input_shape, dim, input_shard_shape,shard_grid,layout",
    (
        (
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            (8, 1, 32, 2048),
            0,
            (32, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
            ttnn.TILE_LAYOUT,
        ),
        (
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            (1, 8, 32, 2048),
            1,
            (32, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
            ttnn.TILE_LAYOUT,
        ),
        (
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            (1, 1, 256, 2048),
            2,
            (32, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
            ttnn.TILE_LAYOUT,
        ),
        (
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            (1, 1, 32, 16384),
            3,
            (32, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
            ttnn.TILE_LAYOUT,
        ),
        (
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (8, 1, 2048, 32),
            0,
            (64, 32),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
            ttnn.TILE_LAYOUT,
        ),
        (
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (1, 8, 2048, 32),
            1,
            (64, 32),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
            ttnn.TILE_LAYOUT,
        ),
        (
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (1, 1, 16384, 32),
            2,
            (64, 32),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
            ttnn.TILE_LAYOUT,
        ),
        (
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (1, 1, 2048, 256),
            3,
            (64, 32),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
            ttnn.TILE_LAYOUT,
        ),
    ),
)
@pytest.mark.parametrize("replication_factor", [4])
@pytest.mark.parametrize("enable_async", [True])
@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4_grid")], indirect=True)
def test_line_all_gather_sharded_on_TG_cols_post_commit(
    mesh_device,
    num_devices,
    input_shape,
    input_shard_shape,
    shard_grid,
    shard_grid_orientation,
    tensor_mem_layout,
    dim,
    num_links,
    input_dtype,
    layout,
    use_program_cache,
    function_level_defaults,
    enable_async,
    replication_factor,
    num_iters=1,
):
    input_shard_spec = ttnn.ShardSpec(
        shard_grid,
        input_shard_shape,
        shard_grid_orientation,
        False,
    )

    run_line_all_gather_on_TG_with_mesh_tensor_along_rows(
        mesh_device,
        num_devices,
        input_shape,
        tensor_mem_layout,
        dim,
        num_links,
        input_dtype,
        layout,
        ttnn.BufferType.L1,
        use_program_cache,
        function_level_defaults,
        enable_async=enable_async,
        num_iters=num_iters,
        input_shard_spec=input_shard_spec,
        num_all_gather_instances=replication_factor,
        cluster_axis=0,
    )


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_devices, num_links, per_chip_output_shape, dim, layout",
    [
        (8, 4, [1, 8, 32, 1280], 1, ttnn.TILE_LAYOUT),
        (8, 4, [8, 1, 32, 1280], 0, ttnn.TILE_LAYOUT),
        (8, 4, [1, 8, 32, 2048], 1, ttnn.TILE_LAYOUT),
        (8, 4, [1, 8, 32, 2304], 1, ttnn.TILE_LAYOUT),
        (8, 4, [1, 8, 32, 4096], 1, ttnn.TILE_LAYOUT),
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
    "buffer_type",
    [
        ttnn.BufferType.DRAM,
        ttnn.BufferType.L1,
    ],
)
@pytest.mark.parametrize("enable_async", [True])
@pytest.mark.parametrize("replication_factor", [4])  # 1, 4])
@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4_grid")], indirect=True)
def test_line_all_gather_on_TG_cols_nightly(
    mesh_device,
    num_devices,
    per_chip_output_shape,
    dim,
    num_links,
    input_dtype,
    layout,
    buffer_type,
    use_program_cache,
    function_level_defaults,
    enable_async,
    replication_factor,
    num_iters=1,
):
    run_line_all_gather_on_TG_with_mesh_tensor_along_rows(
        mesh_device,
        num_devices,
        per_chip_output_shape,
        ttnn.TensorMemoryLayout.INTERLEAVED,
        dim,
        num_links,
        input_dtype,
        layout,
        buffer_type,
        use_program_cache,
        function_level_defaults,
        enable_async=enable_async,
        num_iters=num_iters,
        num_all_gather_instances=replication_factor,
        cluster_axis=0,
    )
