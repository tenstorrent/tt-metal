# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from models.utility_functions import skip_for_grayskull
from tests.ttnn.unit_tests.operations.ccl.test_ccl_common import (
    create_and_load_sub_device_manager_with_fabric_interface,
    teardown_fabric_interface,
    create_global_semaphore_with_same_address,
)

from tests.ttnn.unit_tests.operations.ccl.test_all_gather_TG_post_commit import (
    run_line_all_gather_on_TG_with_mesh_tensor_along_rows,
)
from tests.ttnn.unit_tests.operations.ccl.test_reduce_scatter_TG_nightly import (
    run_line_reduce_scatter_on_TG_with_mesh_tensor_along_rows,
)
from tests.ttnn.unit_tests.operations.ccl.test_all_reduce_async import (
    run_all_reduce_with_mesh_tensor_along_row,
)
from models.perf.benchmarking_utils import BenchmarkProfiler


PREFETCHER_NOC1_RING = [
    (6, 6),
    (6, 7),
    (6, 9),
    (6, 0),
    (6, 1),
    (6, 2),
    (6, 4),
    (6, 5),
    (5, 5),
    (5, 6),
    (5, 7),
    (5, 9),
    (5, 0),
    (5, 1),
    (5, 2),
    (5, 4),
    (1, 4),
    (1, 5),
    (1, 9),
    (1, 0),
    (2, 0),
    (2, 4),
    (2, 5),
    (2, 9),
]


def get_core_range_set(output_core_grid):
    if isinstance(output_core_grid, ttnn.CoreGrid):
        output_core_range_set = ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(output_core_grid.x - 1, output_core_grid.y - 1)),
            ]
        )
    else:
        output_core_range_set = ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(
                    ttnn.CoreCoord(x, y),
                    ttnn.CoreCoord(x, y),
                )
                for x, y in output_core_grid
            ]
        )
    return output_core_range_set


# Enumerate the post-commit cases explicitly
@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_devices, num_links",
    [
        (4, 3),
    ],
)
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat8_b,
    ],
)
@pytest.mark.parametrize(
    "num_iters",
    [
        5000,
    ],
)
@pytest.mark.parametrize("shard_grid_orientation", [ttnn.ShardOrientation.ROW_MAJOR])
@pytest.mark.parametrize(
    "tensor_mem_layout, output_shape, dim, input_shard_shape,input_shard_grid,output_shard_shape, output_shard_grid, layout, perf_target_us",
    (
        (  # AllGather after SDPA
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (1, 32, 32, 128),
            1,
            (32, 128),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))}),
            (32, 128),
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9)),
                    ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(5, 1)),
                }
            ),
            ttnn.TILE_LAYOUT,
            32,
        ),
        (  # AllGather after Binary Mult+Silu
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            (1, 1, 32, 3840),
            3,
            (32, 32),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 4))}),
            (32, 160),
            get_core_range_set(PREFETCHER_NOC1_RING),
            ttnn.TILE_LAYOUT,
            25,
        ),
    ),
)
@pytest.mark.parametrize("replication_factor", [8])
@pytest.mark.parametrize("enable_async", [True])
@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4_grid")], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 17068032}], indirect=True)
def test_line_all_gather_sharded_on_TG_rows_llama(
    mesh_device,
    num_devices,
    output_shape,
    input_shard_shape,
    input_shard_grid,
    output_shard_shape,
    output_shard_grid,
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
    num_iters,
    perf_target_us,
):
    if len(mesh_device.get_devices()) != 32:
        pytest.skip("Not TG!")
    input_shard_spec = ttnn.ShardSpec(
        input_shard_grid,
        input_shard_shape,
        shard_grid_orientation,
    )

    if output_shard_grid is not None and output_shard_shape is not None:
        output_shard_spec = ttnn.ShardSpec(
            output_shard_grid,
            output_shard_shape,
            shard_grid_orientation,
        )
    else:
        output_shard_spec = None

    profiler = BenchmarkProfiler()

    run_line_all_gather_on_TG_with_mesh_tensor_along_rows(
        mesh_device,
        num_devices,
        output_shape,
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
        output_shard_spec=output_shard_spec,
        num_all_gather_instances=replication_factor,
        cluster_axis=1,
        profiler=profiler,
        trace_mode=True,
        use_all_gather_async=True,
        enable_persistent_fabric=True,
        create_persistent_fabric=True,
        teardown_persistent_fabric=True,
    )

    latency_us = profiler.get_duration("all-gather-async-trace") / num_iters * 1e6
    if perf_target_us is not None:
        assert (
            latency_us < perf_target_us
        ), f"Measured latency {latency_us} us is greater than target {perf_target_us} us"


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_devices, num_links",
    [
        (4, 2),
    ],
)
@pytest.mark.parametrize(
    "tensor_mem_layout, per_chip_input_shape, dim, input_shard_shape,shard_grid,layout",
    (
        (  # ReduceScatter After FF1/3  (~100 us)
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            (1, 1, 32, 3840),
            3,
            (32, 160),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 2))}),
            ttnn.TILE_LAYOUT,
        ),
    ),
)
@pytest.mark.parametrize("shard_grid_orientation", [ttnn.ShardOrientation.ROW_MAJOR])
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,
        # ttnn.bfloat8_b,
    ],
)
@pytest.mark.parametrize(
    "buffer_type",
    [
        ttnn.BufferType.L1,
    ],
)
@pytest.mark.parametrize("enable_async", [True])
@pytest.mark.parametrize("replication_factor", [8])
@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4_grid")], indirect=True)
@pytest.mark.parametrize("math_op", [ttnn.ReduceType.Sum])
def test_line_reduce_scatter_sharded_on_TG_rows_llama(
    mesh_device,
    num_devices,
    per_chip_input_shape,
    tensor_mem_layout,
    input_shard_shape,
    shard_grid,
    shard_grid_orientation,
    dim,
    num_links,
    math_op,
    input_dtype,
    layout,
    buffer_type,
    use_program_cache,
    function_level_defaults,
    enable_async,
    replication_factor,
    num_iters=10,
):
    if len(mesh_device.get_devices()) != 32:
        pytest.skip("Not TG!")
    input_shard_spec = ttnn.ShardSpec(
        shard_grid,
        input_shard_shape,
        shard_grid_orientation,
    )

    logger.warning("sharding not used due to issue #16699")

    run_line_reduce_scatter_on_TG_with_mesh_tensor_along_rows(
        mesh_device,
        num_devices,
        per_chip_input_shape,
        ttnn.TensorMemoryLayout.INTERLEAVED,  # tensor_mem_layout,
        dim,
        num_links,
        math_op,
        input_dtype,
        layout,
        buffer_type,
        use_program_cache,
        function_level_defaults,
        enable_async=enable_async,
        # input_shard_spec=input_shard_spec,
        num_iters=num_iters,
        num_reduce_scatter_instances=replication_factor,
        cluster_axis=1,
        use_reduce_scatter_async=True,
        enable_persistent_fabric=True,
        create_persistent_fabric=True,
        teardown_persistent_fabric=True,
    )


# Enumerate the post-commit cases explicitly
@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_devices, num_links, per_chip_output_shape, layout",
    [
        (4, 1, [1, 1, 32, 1280], ttnn.TILE_LAYOUT),  # AllReduce after QKV (~110 us)
    ],
)
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,
    ],
)
@pytest.mark.parametrize(
    "buffer_type",
    [
        ttnn.BufferType.L1,
    ],
)
@pytest.mark.parametrize("replication_factor", [8])  # 1, 8])
@pytest.mark.parametrize("enable_async", [True])
@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4_grid")], indirect=True)
@pytest.mark.parametrize("math_op", [ttnn.ReduceType.Sum])
def test_line_all_reduce_on_TG_rows_llama(
    mesh_device,
    num_devices,
    per_chip_output_shape,
    num_links,
    math_op,
    input_dtype,
    layout,
    buffer_type,
    use_program_cache,
    function_level_defaults,
    enable_async,
    replication_factor,
    num_iters=10,
):
    if len(mesh_device.get_devices()) != 32:
        pytest.skip("Not TG!")

    logger.warning("sharding not used due to issue #16699")

    run_all_reduce_with_mesh_tensor_along_row(
        mesh_device,
        num_devices,
        per_chip_output_shape,
        num_links,
        math_op,
        input_dtype,
        layout,
        buffer_type,
        use_program_cache,
        function_level_defaults,
        enable_async=enable_async,
        num_iters=num_iters,
        num_all_reduce_instances=replication_factor,
        cluster_axis=1,
        enable_persistent_fabric=True,
        create_persistent_fabric=True,
        teardown_persistent_fabric=True,
    )


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_devices, num_links, per_chip_output_shape, layout",
    [
        (8, 1, [1, 1, 32, 2048], ttnn.TILE_LAYOUT),  # AllReduce after DO and AllReduce after FF2 (~240 us)
        # multi-links fail https://github.com/tenstorrent/tt-metal/issues/16699
    ],
)
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,
    ],
)
@pytest.mark.parametrize(
    "buffer_type",
    [
        ttnn.BufferType.L1,
    ],
)
@pytest.mark.parametrize("enable_async", [True])
@pytest.mark.parametrize("replication_factor", [4])
@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4_grid")], indirect=True)
@pytest.mark.parametrize("math_op", [ttnn.ReduceType.Sum])
def test_line_all_reduce_on_TG_cols_llama(
    mesh_device,
    num_devices,
    per_chip_output_shape,
    num_links,
    math_op,
    input_dtype,
    layout,
    buffer_type,
    use_program_cache,
    function_level_defaults,
    enable_async,
    replication_factor,
    num_iters=10,
):
    if len(mesh_device.get_devices()) != 32:
        pytest.skip("Not TG!")

    logger.warning("sharding not used due to issue #16699")

    run_all_reduce_with_mesh_tensor_along_row(
        mesh_device,
        num_devices,
        per_chip_output_shape,
        num_links,
        math_op,
        input_dtype,
        layout,
        buffer_type,
        use_program_cache,
        function_level_defaults,
        enable_async=enable_async,
        num_iters=num_iters,
        num_all_reduce_instances=replication_factor,
        cluster_axis=0,
        enable_persistent_fabric=True,
        create_persistent_fabric=True,
        teardown_persistent_fabric=True,
    )
