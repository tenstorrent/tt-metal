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
from tests.ttnn.unit_tests.operations.ccl.test_new_all_reduce import (
    run_all_reduce_impl,
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


CORE_RANGE_SET_1x1 = ttnn.CoreRangeSet(
    {
        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0)),
    }
)


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
    "num_iters, warmup_iters",
    [
        (2500, 100),
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
            14.5,
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
            15.5,
        ),
        (  # AllGather for layernorm
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            (1, 1, 32, 128),
            3,
            (32, 32),
            CORE_RANGE_SET_1x1,
            (32, 128),
            CORE_RANGE_SET_1x1,
            ttnn.TILE_LAYOUT,
            9.5,
        ),
    ),
)
@pytest.mark.parametrize("replication_factor", [8])
@pytest.mark.parametrize("enable_async", [True])
@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4_grid")], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 17068032}], indirect=True)
def test_all_gather_tg_llama(
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
    warmup_iters,
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
        warmup_iters=warmup_iters,
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

    time_taken = profiler.get_duration("all-gather-async-trace") - profiler.get_duration(
        "all-gather-async-trace-warmup"
    )
    effective_iter = num_iters - warmup_iters
    latency_us = time_taken / effective_iter * 1e6
    logger.info(f"Time taken: {time_taken} s")
    logger.info(f"Time per iter: {latency_us} us")
    if perf_target_us is not None:
        assert (
            latency_us < perf_target_us
        ), f"Measured latency {latency_us} us is greater than target {perf_target_us} us"


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "output_shape, cluster_axis, num_links, input_num_cores, output_num_cores, perf_target_us",
    [
        ([1, 1, 32, 2048], 0, 4, 24, 16, 35.5),  # FF2/DO all reduce
        ([1, 1, 32, 1280], 1, 3, 24, 40, 34.5),  # QKV all reduce
        ([1, 1, 32, 3584], 1, 3, 24, 24, 40),  # FF1 all reduce
    ],
)
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat8_b,
    ],
)
@pytest.mark.parametrize(
    "num_iters, warmup_iters",
    [
        (2500, 100),
    ],
)
@pytest.mark.parametrize("enable_async", [True])
@pytest.mark.parametrize("trace_mode", [True])
@pytest.mark.parametrize(
    "device_params",
    [{"trace_region_size": 23887872}],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        (8, 4),
    ],
    indirect=True,
)
def test_all_reduce_tg_llama(
    mesh_device,
    output_shape,
    cluster_axis,
    input_dtype,
    num_links,
    input_num_cores,
    output_num_cores,
    num_iters,
    warmup_iters,
    perf_target_us,
    enable_async,
    trace_mode,
    use_program_cache,
    function_level_defaults,
):
    profiler = BenchmarkProfiler()

    run_all_reduce_impl(
        mesh_device,
        output_shape,
        cluster_axis,
        input_dtype,
        num_links,
        input_num_cores,
        output_num_cores,
        num_iters=num_iters,
        warmup_iters=warmup_iters,
        enable_async=enable_async,
        trace_mode=trace_mode,
        validate_all=False,
        profiler=profiler,
    )

    time_taken = profiler.get_duration("all-reduce-async-trace") - profiler.get_duration(
        "all-reduce-async-trace-warmup"
    )
    effective_iter = num_iters - warmup_iters
    latency_us = time_taken / effective_iter * 1e6
    logger.info(f"Time taken: {time_taken} s")
    logger.info(f"Time per iter: {latency_us} us")
    if perf_target_us is not None:
        assert (
            latency_us < perf_target_us
        ), f"Measured latency {latency_us} us is greater than target {perf_target_us} us"
