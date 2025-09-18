# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
from models.utility_functions import skip_for_grayskull
from tests.ttnn.unit_tests.operations.ccl.test_new_all_gather import (
    run_all_gather_impl,
)
from tests.ttnn.unit_tests.operations.ccl.test_reduce_scatter_async import (
    run_reduce_scatter_test,
)
from tests.ttnn.unit_tests.operations.ccl.test_all_gather_TG_post_commit import (
    run_line_all_gather_on_TG_with_mesh_tensor_along_rows,
)


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_devices, num_links, output_shape, dim, layout",
    [
        (4, 1, [1, 1, 64, 512], 3, ttnn.TILE_LAYOUT),
        (4, 1, [1, 1, 32, 1280], 0, ttnn.TILE_LAYOUT),
        (4, 1, [1, 1, 32, 7168], 0, ttnn.TILE_LAYOUT),
        (4, 1, [1, 1, 32, 3584], 0, ttnn.TILE_LAYOUT),
        (4, 1, [1, 1, 32, 32], 0, ttnn.TILE_LAYOUT),
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
@pytest.mark.parametrize("num_iters", [20])
@pytest.mark.parametrize("device_params", [{"trace_region_size": 1824800}], indirect=True)
def test_all_gather_async_t3000(
    t3k_mesh_device,
    num_devices,
    output_shape,
    dim,
    num_links,
    input_dtype,
    layout,
    mem_config,
    num_iters,
    function_level_defaults,
):
    output_shape[dim] *= num_devices
    run_all_gather_impl(
        t3k_mesh_device,
        num_devices,
        output_shape,
        dim,
        num_links,
        input_dtype,
        layout,
        function_level_defaults,
        all_gather_topology=ttnn.Topology.Ring,
        num_iters=num_iters,
        rand_tensor=True,
        mem_config=mem_config,
        trace_mode=True,
    )


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_devices, num_links, per_chip_output_shape, dim, layout",
    [
        (8, 1, [1, 8, 32, 1280], 1, ttnn.TILE_LAYOUT),
        (8, 1, [8, 1, 32, 1280], 0, ttnn.TILE_LAYOUT),
        (8, 1, [1, 8, 32, 2048], 1, ttnn.TILE_LAYOUT),
        (8, 1, [1, 8, 32, 2304], 1, ttnn.TILE_LAYOUT),
        (8, 1, [1, 8, 32, 4096], 1, ttnn.TILE_LAYOUT),
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
@pytest.mark.parametrize("replication_factor", [4])
@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4_grid")], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 1824800}], indirect=True)
def test_all_gather_async_tg(
    mesh_device,
    num_devices,
    per_chip_output_shape,
    dim,
    num_links,
    input_dtype,
    layout,
    buffer_type,
    function_level_defaults,
    replication_factor,
    num_iters=1,
):
    if mesh_device.get_num_devices() != 32:
        pytest.skip("Not TG!")
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
        function_level_defaults,
        num_iters=num_iters,
        num_all_gather_instances=replication_factor,
        cluster_axis=0,
        use_all_gather_async=True,
        trace_mode=True,
    )


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_devices, num_links",
    [
        (4, 1),
    ],
)
@pytest.mark.parametrize(
    "per_chip_output_shape, dim, layout",
    [
        ([1, 1, 32, 32], 3, ttnn.TILE_LAYOUT),
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
@pytest.mark.parametrize("math_op", [ttnn.ReduceType.Sum])
@pytest.mark.parametrize("trace_mode", [True])
@pytest.mark.parametrize("device_params", [{"trace_region_size": 1824800}], indirect=True)
def test_reduce_scatter_async_t3000(
    t3k_mesh_device,
    num_devices,
    per_chip_output_shape,
    dim,
    num_links,
    math_op,
    input_dtype,
    layout,
    mem_config,
    function_level_defaults,
    trace_mode,
    num_iters=20,
):
    run_reduce_scatter_test(
        t3k_mesh_device,
        num_devices,
        per_chip_output_shape,
        dim,
        num_links,
        math_op,
        input_dtype,
        layout,
        mem_config,
        function_level_defaults,
        num_iters=num_iters,
        topology=ttnn.Topology.Linear,
        trace_mode=trace_mode,
    )
