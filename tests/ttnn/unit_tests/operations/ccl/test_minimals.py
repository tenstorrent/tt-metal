# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

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

from tests.ttnn.unit_tests.operations.ccl.fusion_subtests.concat_fuse_test import (
    run_concat_fuse_impl,
    run_gather_concat_impl,
)

from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler


# Enumerate the post-commit cases explicitly
@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_devices, output_shape, dim, layout, input_shard_shape, input_shard_grid, output_shard_shape, output_shard_grid, tensor_mem_layout",
    [
        # Before Concat Heads
        (
            4,
            [1, 32, 32, 128],
            1,
            ttnn.TILE_LAYOUT,
            (32, 128),
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 1)),
                    ttnn.CoreRange(ttnn.CoreCoord(1, 2), ttnn.CoreCoord(2, 2)),
                }
            ),
            (32, 128),
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 1)),
                    ttnn.CoreRange(ttnn.CoreCoord(1, 2), ttnn.CoreCoord(2, 2)),
                }
            ),
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ),
    ],
    ids=[
        "concat_heads",
    ],
)
@pytest.mark.parametrize("num_links", [3])
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,
        # ttnn.bfloat8_b,
    ],
)
@pytest.mark.parametrize("num_iters, warmup_iters", [[3, 0]])
@pytest.mark.parametrize("enable_async", [True])
@pytest.mark.parametrize("trace_mode", [False])
@pytest.mark.parametrize(
    "device_params",
    [{"trace_region_size": 23887872}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4_grid")], indirect=True)
def test_concat_fuse(
    mesh_device,
    num_devices,
    output_shape,
    dim,
    num_links,
    input_dtype,
    layout,
    num_iters,
    warmup_iters,
    use_program_cache,
    function_level_defaults,
    enable_async,
    input_shard_shape,
    input_shard_grid,
    output_shard_shape,
    output_shard_grid,
    tensor_mem_layout,
    trace_mode,
):
    profiler = BenchmarkProfiler()
    run_concat_fuse_impl(
        mesh_device,
        num_devices,
        output_shape,
        dim,
        num_links,
        input_dtype,
        layout,
        use_program_cache,
        function_level_defaults,
        input_shard_shape,
        input_shard_grid,
        all_gather_topology=ttnn.Topology.Linear,
        warmup_iters=warmup_iters,
        num_iters=num_iters,
        enable_async=enable_async,
        output_shard_shape=output_shard_shape,
        output_shard_grid=output_shard_grid,
        tensor_mem_layout=tensor_mem_layout,
        trace_mode=trace_mode,
        profiler=profiler,
    )
