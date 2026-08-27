# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from functools import partial

import pytest
from loguru import logger
import ttnn
from models.common.utility_functions import skip_for_n_or_less_dev
from models.perf.benchmarking_utils import BenchmarkProfiler
from tests.ttnn.unit_tests.operations.ccl.blackhole_CI._all_reduce_helpers import (
    run_all_reduce_impl as _run_all_reduce_impl,
    SUB_DEVICE_CRS,
    QKV_CRS,
    RING_CRS,
    FF1_CRS,
    FF1_CRS_RS_OUT,
    NORM_CRS,
    NORM_CRS_QWEN,
    LM_HEAD_CRS,
)

# Pre-refactor behavior of this file: CacheEntriesCounter attached to the mesh
# device, with each run_op call measured as its own section.
run_all_reduce_impl = partial(_run_all_reduce_impl, cache_counter_mode="device_attr_sections")


@skip_for_n_or_less_dev(3)
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
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        }
    ],
    indirect=True,
)
def test_all_reduce(
    bh_1d_mesh_device,
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
    submesh_device = bh_1d_mesh_device.create_submesh(ttnn.MeshShape((4, 1)))

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
    )

    time_taken = profiler.get_duration("all-reduce-async-trace") - profiler.get_duration(
        "all-reduce-async-trace-warmup"
    )
    effective_iter = num_iters - warmup_iters
    latency_us = time_taken / effective_iter * 1e6
    logger.info(f"Time taken: {time_taken} s")
    logger.info(f"Time per iter: {latency_us} us")


@skip_for_n_or_less_dev(3)
@pytest.mark.parametrize(
    "output_shape, cluster_axis, num_links, input_num_cores, input_core_range_set, output_num_cores, output_core_range_set",
    [
        ([1, 1, 32, 3584], 0, 1, 24, RING_CRS, 28, FF1_CRS),  # FF1 all reduce
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
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        }
    ],
    indirect=True,
)
def test_all_reduce_loopback(
    bh_1d_mesh_device,
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
    submesh_device = bh_1d_mesh_device.create_submesh(ttnn.MeshShape((4, 1)))
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
        loopback_size=4,
        num_iters=num_iters,
        warmup_iters=warmup_iters,
        trace_mode=trace_mode,
        validate_all=False,
    )
