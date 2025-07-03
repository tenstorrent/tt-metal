# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
from models.utility_functions import skip_for_grayskull
from tests.ttnn.unit_tests.operations.ccl.test_all_gather import (
    run_all_gather_on_n300_impl,
    run_all_gather_on_t3000_impl_tight_loop,
)
from tests.ttnn.unit_tests.operations.ccl.test_reduce_scatter_post_commit import (
    run_reduce_scatter_test,
)
from tests.ttnn.unit_tests.operations.ccl.test_all_gather_TG_post_commit import (
    run_line_all_gather_on_TG_with_mesh_tensor_along_rows,
)
from tests.ttnn.unit_tests.operations.ccl.test_reduce_scatter_TG_nightly import (
    run_line_reduce_scatter_on_TG_with_mesh_tensor_along_rows,
)


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_devices, num_links, output_shape, dim, layout",
    [
        (2, 1, [1, 1, 64, 16384], 3, ttnn.TILE_LAYOUT),
        (2, 1, [8, 5, 32, 768], 3, ttnn.TILE_LAYOUT),
        (2, 1, [1, 1, 32, 736], 3, ttnn.TILE_LAYOUT),
        (2, 1, [1, 1, 32, 704], 3, ttnn.TILE_LAYOUT),
        (2, 1, [1, 1, 64, 704], 3, ttnn.TILE_LAYOUT),
        (2, 1, [1, 1, 32, 736], 3, ttnn.ROW_MAJOR_LAYOUT),
        (2, 1, [1, 1, 32, 704], 3, ttnn.ROW_MAJOR_LAYOUT),
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
def test_all_gather_on_n300(
    n300_mesh_device,
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
    run_all_gather_on_n300_impl(
        n300_mesh_device,
        num_devices,
        output_shape,
        dim,
        num_links,
        input_dtype,
        layout,
        mem_config,
        function_level_defaults,
        all_gather_topology=ttnn.Topology.Ring,
        num_iters=num_iters,
        trace_mode=True,
    )


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_devices, num_links, input_shape, dim, layout",
    [
        (8, 1, [8, 1, 256, 32], 0, ttnn.TILE_LAYOUT),
        (8, 1, [1, 1, 32, 16384], 3, ttnn.TILE_LAYOUT),
        (8, 1, [8, 1, 256, 32], 0, ttnn.ROW_MAJOR_LAYOUT),
        (8, 1, [1, 1, 32, 16384], 3, ttnn.ROW_MAJOR_LAYOUT),
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
@pytest.mark.parametrize("topology", [ttnn.Topology.Ring, ttnn.Topology.Linear])
@pytest.mark.parametrize("device_params", [{"trace_region_size": 266240}], indirect=True)
def test_all_gather_on_t3000(
    t3k_mesh_device,
    num_devices,
    input_shape,
    dim,
    num_links,
    topology,
    input_dtype,
    layout,
    mem_config,
    num_iters,
    function_level_defaults,
):
    run_all_gather_on_t3000_impl_tight_loop(
        t3k_mesh_device,
        num_devices,
        input_shape,
        dim,
        num_links,
        input_dtype,
        layout,
        mem_config,
        function_level_defaults,
        all_gather_topology=topology,
        num_iters=num_iters,
        trace_mode=True,
    )


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_devices, num_links",
    [
        (8, 1),
    ],
)
@pytest.mark.parametrize(
    "per_chip_output_shape, dim, layout",
    [
        ([1, 8, 1024, 1024], 3, ttnn.TILE_LAYOUT),
        ([1, 4, 1024, 1024], 3, ttnn.TILE_LAYOUT),
        ([1, 4, 2048, 1024], 3, ttnn.TILE_LAYOUT),
        ([1, 1, 32, 32 * 8], 3, ttnn.TILE_LAYOUT),
        ([1, 1, 32, 64 * 8], 3, ttnn.TILE_LAYOUT),
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
@pytest.mark.parametrize("math_op", [ttnn.ReduceType.Sum])
@pytest.mark.parametrize("topology", [ttnn.Topology.Linear, ttnn.Topology.Ring])
@pytest.mark.parametrize("device_params", [{"trace_region_size": 266240}], indirect=True)
def test_reduce_scatter_on_t3000(
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
    num_iters,
    topology,
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
        topology=topology,
        trace_mode=True,
    )


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_devices, num_links",
    [
        (2, 1),
    ],
)
@pytest.mark.parametrize(
    "per_chip_output_shape, dim, layout",
    [
        ([1, 1, 32, 4096], 3, ttnn.TILE_LAYOUT),
        ([1, 1, 32, 2048], 3, ttnn.TILE_LAYOUT),
        ([1, 1, 32, 1024], 3, ttnn.TILE_LAYOUT),
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
@pytest.mark.parametrize("num_iters", [20])
@pytest.mark.parametrize("math_op", [ttnn.ReduceType.Sum])
@pytest.mark.parametrize("device_params", [{"trace_region_size": 266240}], indirect=True)
def test_reduce_scatter_on_n300(
    n300_mesh_device,
    num_devices,
    per_chip_output_shape,
    dim,
    num_links,
    math_op,
    input_dtype,
    layout,
    mem_config,
    function_level_defaults,
    num_iters,
):
    run_reduce_scatter_test(
        n300_mesh_device,
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
        trace_mode=True,
    )


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_devices, num_links, per_chip_output_shape, dim, layout",
    [
        (4, 3, [4, 1, 32, 1280], 0, ttnn.TILE_LAYOUT),
        (4, 3, [1, 1, 32, 16384 * 4], 3, ttnn.TILE_LAYOUT),
        (4, 3, [1, 4, 32, 6656], 1, ttnn.TILE_LAYOUT),
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
@pytest.mark.parametrize("replication_factor", [8])
@pytest.mark.parametrize("num_iters", [20])
@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4_grid")], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 532480}], indirect=True)
def test_all_gather_on_tg(
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
    num_iters,
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
        cluster_axis=1,
        trace_mode=True,
    )


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_devices, num_links, per_chip_output_shape, dim, layout",
    [
        (4, 2, [1, 4, 32, 2304], 1, ttnn.TILE_LAYOUT),
        (4, 2, [1, 4, 64, 2304], 1, ttnn.TILE_LAYOUT),
        (4, 2, [1, 4, 64, 6656], 1, ttnn.TILE_LAYOUT),
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
        ttnn.BufferType.DRAM,
        ttnn.BufferType.L1,
    ],
)
@pytest.mark.parametrize("replication_factor", [8])
@pytest.mark.parametrize("num_iters", [20])
@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4_grid")], indirect=True)
@pytest.mark.parametrize("math_op", [ttnn.ReduceType.Sum])
@pytest.mark.parametrize("device_params", [{"trace_region_size": 10281600}], indirect=True)
def test_reduce_scatter_on_tg(
    mesh_device,
    num_devices,
    per_chip_output_shape,
    dim,
    num_links,
    math_op,
    input_dtype,
    layout,
    buffer_type,
    function_level_defaults,
    replication_factor,
    num_iters,
):
    run_line_reduce_scatter_on_TG_with_mesh_tensor_along_rows(
        mesh_device,
        num_devices,
        per_chip_output_shape,
        ttnn.TensorMemoryLayout.INTERLEAVED,
        dim,
        num_links,
        math_op,
        input_dtype,
        layout,
        buffer_type,
        function_level_defaults,
        num_iters=num_iters,
        num_reduce_scatter_instances=replication_factor,
        cluster_axis=1,
        trace_mode=True,
    )
