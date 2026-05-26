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
@pytest.mark.parametrize("output_shape", [[1, 1, 32, 1280]])
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize("input_num_cores", [24])
@pytest.mark.parametrize("input_core_range_set", [RING_CRS])
@pytest.mark.parametrize("output_num_cores", [10])
@pytest.mark.parametrize("output_core_range_set", [QKV_CRS])
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("num_iters, warmup_iters", [(20, 5)])
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
@pytest.mark.parametrize("cluster_axis", [0, 1], ids=["axis_0_row", "axis_1_col"])
def test_all_reduce_2d_fabric_turning(
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
    """
    Turning test that exercises FABRIC_2D routing along both cluster_axis 0 (row) and 1 (column).
    This tests the 2D fabric's ability to route data in both dimensions, requiring "turning corners"
    in the routing network - behavior that is unique to FABRIC_2D and cannot be tested with FABRIC_1D.
    """

    mesh_shape = bh_2d_mesh_device.shape

    if cluster_axis == 0:
        # On bh-llmbox (4,1 mesh), use 2 devices to avoid fabric routing issues
        if mesh_shape == ttnn.MeshShape(4, 1):
            num_devices = 2
        else:
            # On other machines, use all devices in first dimension
            num_devices = mesh_shape[0]
        submesh_shape = ttnn.MeshShape(num_devices, 1)
        cluster_shape_tuple = (num_devices, 1)
    else:  # cluster_axis == 1
        num_devices = mesh_shape[1]
        submesh_shape = ttnn.MeshShape(1, num_devices)
        cluster_shape_tuple = (1, num_devices)

    validate_test(num_devices, ttnn.Topology.Linear, mesh_shape, cluster_axis)
    submesh_device = bh_2d_mesh_device.create_submesh(submesh_shape)

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
        cluster_shape=cluster_shape_tuple,
    )

    time_taken = profiler.get_duration("all-reduce-async-trace") - profiler.get_duration(
        "all-reduce-async-trace-warmup"
    )
    effective_iter = num_iters - warmup_iters
    latency_us = time_taken / effective_iter * 1e6
    logger.info(f"Time taken: {time_taken} s")
    logger.info(f"Time per iter: {latency_us} us")
