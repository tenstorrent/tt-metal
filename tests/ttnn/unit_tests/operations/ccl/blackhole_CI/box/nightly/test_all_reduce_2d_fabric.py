# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger
import ttnn

from models.common.utility_functions import skip_for_n_or_less_dev
from tests.ttnn.unit_tests.operations.ccl.blackhole_CI.box.nightly.test_all_gather_nightly import validate_test

from models.perf.benchmarking_utils import BenchmarkProfiler
from tests.ttnn.unit_tests.operations.ccl.blackhole_CI._all_reduce_helpers import (
    SUB_DEVICE_CRS,
    QKV_CRS,
    RING_CRS,
    FF1_CRS,
    FF1_CRS_RS_OUT,
    NORM_CRS,
    NORM_CRS_QWEN,
    LM_HEAD_CRS,
)


# Import the actual run_all_reduce_impl from the parent test file
from tests.ttnn.unit_tests.operations.ccl.blackhole_CI.box.nightly.test_new_all_reduce import run_all_reduce_impl


@skip_for_n_or_less_dev(1)
@pytest.mark.parametrize("output_shape", [[1, 1, 32, 1280]])
@pytest.mark.parametrize("cluster_axis", [0])
@pytest.mark.parametrize("num_links", [2])
@pytest.mark.parametrize("input_num_cores", [24])
@pytest.mark.parametrize("input_core_range_set", [RING_CRS])
@pytest.mark.parametrize("output_num_cores", [10])
@pytest.mark.parametrize("output_core_range_set", [QKV_CRS])
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16])
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
