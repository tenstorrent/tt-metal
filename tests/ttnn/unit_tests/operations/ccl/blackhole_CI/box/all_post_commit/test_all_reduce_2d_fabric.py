# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger
import ttnn

from models.common.utility_functions import skip_for_n_or_less_dev
from tests.ttnn.unit_tests.operations.ccl.blackhole_CI.box.nightly.test_all_gather_nightly import validate_test

from models.demos.llama3_70b_galaxy.tt.model_config import (
    PREFETCHER_NOC1_GRID,
)
from models.perf.benchmarking_utils import BenchmarkProfiler


SUB_DEVICE_CRS = ttnn.CoreRangeSet(
    [
        ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9)),
        ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 9)),
    ]
)

QKV_CRS = ttnn.num_cores_to_corerangeset_in_subcoregrids(ttnn.CoreCoord(1, 0), 10, SUB_DEVICE_CRS, row_wise=True)

RING_CRS = ttnn.CoreRangeSet(
    [
        ttnn.CoreRange(
            ttnn.CoreCoord(x, y),
            ttnn.CoreCoord(x, y),
        )
        for x, y in PREFETCHER_NOC1_GRID
    ]
)

# Import the actual run_all_reduce_impl from the parent test file
from tests.ttnn.unit_tests.operations.ccl.blackhole_CI.box.all_post_commit.test_new_all_reduce import (
    run_all_reduce_impl,
)


@skip_for_n_or_less_dev(1)
@pytest.mark.parametrize(
    "output_shape, cluster_axis, num_links, input_num_cores, input_core_range_set, output_num_cores, output_core_range_set",
    [
        ([1, 1, 32, 1280], 0, 2, 24, RING_CRS, 10, QKV_CRS),  # QKV all reduce
    ],
)
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,
    ],
)
@pytest.mark.parametrize(
    "num_iters, warmup_iters",
    [
        (100, 10),
    ],
)
@pytest.mark.parametrize("trace_mode", [True])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "trace_region_size": 23887872,
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
        }
    ],
    indirect=True,
)
def test_all_reduce_2d_fabric(
    bh_2d_mesh_device,
    output_shape,
    cluster_axis,
    input_dtype,
    num_links,
    input_num_cores,
    input_core_range_set,
    output_num_cores,
    output_core_range_set,
    num_iters,
    warmup_iters,
    trace_mode,
    function_level_defaults,
):
    # On bh-llmbox (4,1 mesh), use 2 devices to avoid fabric routing issues
    # On other machines, use all devices in first dimension
    if bh_2d_mesh_device.shape == ttnn.MeshShape(4, 1):
        num_devices = 2
    else:
        num_devices = bh_2d_mesh_device.shape[0]
    cluster_axis = 0

    validate_test(num_devices, ttnn.Topology.Linear, bh_2d_mesh_device.shape, cluster_axis)
    submesh_device = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((num_devices, 1)))

    if output_shape == [1, 1, 32, 16 * 1024] and input_dtype == ttnn.bfloat16:
        pytest.skip("Skipping LM Head test with bfloat16 due to OOM")

    profiler = BenchmarkProfiler()

    run_all_reduce_impl(
        submesh_device,
        output_shape,
        cluster_axis,
        input_dtype,
        num_links,
        input_num_cores,
        input_core_range_set,
        output_num_cores,
        output_core_range_set,
        num_iters=num_iters,
        warmup_iters=warmup_iters,
        trace_mode=trace_mode,
        validate_all=False,
        profiler=profiler,
        cluster_shape=(num_devices, 1),
    )

    time_taken = profiler.get_duration("all-reduce-async-trace") - profiler.get_duration(
        "all-reduce-async-trace-warmup"
    )
    effective_iter = num_iters - warmup_iters
    latency_us = time_taken / effective_iter * 1e6
    logger.info(f"Time taken: {time_taken} s")
    logger.info(f"Time per iter: {latency_us} us")
