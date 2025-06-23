# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler
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
from tests.ttnn.unit_tests.operations.ccl.test_new_all_reduce import (
    run_all_reduce_impl,
    RING_CRS,
    NORM_CRS,
    LM_HEAD_CRS,
    QKV_CRS,
    FF1_CRS,
)


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_devices, num_links, output_shape, dim, layout",
    [
        (4, 4, [1, 32, 32, 1280], 1, ttnn.TILE_LAYOUT),
        # (4, 1, [1, 1, 32, 1280], 0, ttnn.TILE_LAYOUT),
        # (4, 1, [1, 1, 32, 7168], 0, ttnn.TILE_LAYOUT),
        # (4, 1, [1, 1, 32, 3584], 0, ttnn.TILE_LAYOUT),
        # (4, 1, [1, 1, 32, 32], 0, ttnn.TILE_LAYOUT),
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
@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4_grid")], indirect=True)
@pytest.mark.parametrize(
    "device_params", [{"trace_region_size": 23887872, "fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True
)
def test_all_gather_async_t3000(
    mesh_device,
    num_devices,
    output_shape,
    dim,
    num_links,
    input_dtype,
    layout,
    mem_config,
    num_iters,
    use_program_cache,
    function_level_defaults,
):
    submesh = mesh_device.create_submesh(ttnn.MeshShape(1, 4), ttnn.MeshCoordinate(0, 0))

    # output_shape[dim] *= num_devices
    run_all_gather_impl(
        submesh,
        num_devices,
        output_shape,
        dim,
        num_links,
        input_dtype,
        layout,
        use_program_cache,
        function_level_defaults,
        all_gather_topology=ttnn.Topology.Linear,
        num_iters=num_iters,
        rand_tensor=True,
        mem_config=mem_config,
        trace_mode=True,
    )


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_devices, num_links, per_chip_output_shape, dim, layout, input_dtype",
    (
        (4, 1, [1, 32, 32, 128], 1, ttnn.TILE_LAYOUT, ttnn.bfloat8_b),
        (4, 1, [1, 32, 32, 128], 1, ttnn.TILE_LAYOUT, ttnn.bfloat16),
        (4, 2, [1, 32, 32, 128], 1, ttnn.TILE_LAYOUT, ttnn.bfloat16),
        (4, 2, [1, 32, 32, 128], 1, ttnn.TILE_LAYOUT, ttnn.bfloat8_b),
        (4, 4, [1, 32, 32, 128], 1, ttnn.TILE_LAYOUT, ttnn.bfloat16),
        (4, 4, [1, 32, 32, 128], 1, ttnn.TILE_LAYOUT, ttnn.bfloat8_b),
        (8, 2, [1, 32, 32, 128], 1, ttnn.TILE_LAYOUT, ttnn.bfloat16),
        (8, 2, [1, 32, 32, 128], 1, ttnn.TILE_LAYOUT, ttnn.bfloat8_b),
        (8, 4, [1, 32, 32, 128], 1, ttnn.TILE_LAYOUT, ttnn.bfloat16),
        (8, 4, [1, 32, 32, 128], 1, ttnn.TILE_LAYOUT, ttnn.bfloat8_b),
        (8, 1, [8, 1, 32, 1280], 0, ttnn.TILE_LAYOUT, ttnn.bfloat16),
        (8, 1, [8, 1, 32, 1280], 0, ttnn.TILE_LAYOUT, ttnn.bfloat8_b),
        (8, 1, [8, 1, 64, 1280], 0, ttnn.TILE_LAYOUT, ttnn.bfloat16),
        (8, 1, [8, 1, 64, 1280], 0, ttnn.TILE_LAYOUT, ttnn.bfloat8_b),
        (8, 1, [8, 1, 128, 1280], 0, ttnn.TILE_LAYOUT, ttnn.bfloat16),
        (8, 1, [8, 1, 128, 1280], 0, ttnn.TILE_LAYOUT, ttnn.bfloat8_b),
        (8, 2, [8, 1, 32, 1280], 0, ttnn.TILE_LAYOUT, ttnn.bfloat16),
        (8, 2, [8, 1, 32, 1280], 0, ttnn.TILE_LAYOUT, ttnn.bfloat8_b),
        (8, 2, [8, 1, 64, 1280], 0, ttnn.TILE_LAYOUT, ttnn.bfloat16),
        (8, 2, [8, 1, 64, 1280], 0, ttnn.TILE_LAYOUT, ttnn.bfloat8_b),
        (8, 2, [8, 1, 128, 1280], 0, ttnn.TILE_LAYOUT, ttnn.bfloat16),
        (8, 2, [8, 1, 128, 1280], 0, ttnn.TILE_LAYOUT, ttnn.bfloat8_b),
        (8, 4, [8, 1, 32, 1280], 0, ttnn.TILE_LAYOUT, ttnn.bfloat16),
        (8, 4, [8, 1, 32, 1280], 0, ttnn.TILE_LAYOUT, ttnn.bfloat8_b),
        (8, 4, [8, 1, 64, 1280], 0, ttnn.TILE_LAYOUT, ttnn.bfloat16),
        (8, 4, [8, 1, 64, 1280], 0, ttnn.TILE_LAYOUT, ttnn.bfloat8_b),
        (8, 4, [8, 1, 128, 1280], 0, ttnn.TILE_LAYOUT, ttnn.bfloat16),
        (8, 4, [8, 1, 128, 1280], 0, ttnn.TILE_LAYOUT, ttnn.bfloat8_b),
    ),
    ids=[
        "4_1_1_32_32_128_1_Linear_bfloat8_b",
        "4_1_1_32_32_128_1_Linear_bfloat16",
        "4_2_1_32_32_128_1_Linear_bfloat16",
        "4_2_1_32_32_128_1_Linear_bfp8_b",
        "4_4_1_32_32_128_1_Linear_bfloat16",
        "4_4_1_32_32_128_1_Linear_bfp8_b",
        "8_2_1_32_32_128_1_Linear_bfloat16",
        "8_2_1_32_32_128_1_Linear_bfp8_b",
        "8_4_1_32_32_128_1_Linear_bfloat16",
        "8_4_1_32_32_128_1_Linear_bfp8_b",
        "8_1_8_1_32_1280_0_Linear_bfloat16",
        "8_1_8_1_32_1280_0_Linear_bfp8_b",
        "8_1_8_1_64_1280_0_Linear_bfloat16",
        "8_1_8_1_64_1280_0_Linear_bfp8_b",
        "8_1_8_1_128_1280_0_Linear_bfloat16",
        "8_1_8_1_128_1280_0_Linear_bfp8_b",
        "8_2_8_1_32_1280_0_Linear_bfloat16",
        "8_2_8_1_32_1280_0_Linear_bfp8_b",
        "8_2_8_1_64_1280_0_Linear_bfloat16",
        "8_2_8_1_64_1280_0_Linear_bfp8_b",
        "8_2_8_1_128_1280_0_Linear_bfloat16",
        "8_2_8_1_128_1280_0_Linear_bfp8_b",
        "8_4_8_1_32_1280_0_Linear_bfloat16",
        "8_4_8_1_32_1280_0_Linear_bfp8_b",
        "8_4_8_1_64_1280_0_Linear_bfloat16",
        "8_4_8_1_64_1280_0_Linear_bfp8_b",
        "8_4_8_1_128_1280_0_Linear_bfloat16",
        "8_4_8_1_128_1280_0_Linear_bfp8_b",
    ],
)
@pytest.mark.parametrize(
    "num_iters, warmup_iters",
    [
        (3, 1),
    ],
)
@pytest.mark.parametrize(
    "buffer_type",
    [
        ttnn.BufferType.DRAM,
    ],
)
@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4_grid")], indirect=True)
@pytest.mark.parametrize(
    "device_params", [{"trace_region_size": 17068032, "fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True
)
def test_all_gather_async_tg(
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
    num_iters,
    warmup_iters,
):
    if num_devices == 4:
        replication_factor = 8
        cluster_axis = 1
    else:
        replication_factor = 4
        cluster_axis = 0

    submesh = mesh_device.create_submesh(ttnn.MeshShape(1, 4), ttnn.MeshCoordinate(0, 0))
    replication_factor = 1
    # devices = mesh_device.get_device_ids()
    # print(f"devices: {devices}")
    # ttnn.visualize_mesh_device(mesh_device)
    # exit(0)

    if mesh_device.get_num_devices() != 32:
        pytest.skip("Not TG!")
    profiler = BenchmarkProfiler()
    run_line_all_gather_on_TG_with_mesh_tensor_along_rows(
        submesh,
        # mesh_device,
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
        num_iters=num_iters,
        warmup_iters=warmup_iters,
        num_all_gather_instances=replication_factor,
        cluster_axis=cluster_axis,
        profiler=profiler,
        trace_mode=False,
        use_all_gather_async=True,
        use_persistent_output=True,
        topology=ttnn.Topology.Linear,
    )


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_devices, num_links, per_chip_output_shape, dim, layout",
    [
        (8, 1, [8, 1, 32, 1280], 0, ttnn.TILE_LAYOUT),
        (8, 1, [8, 1, 64, 1280], 0, ttnn.TILE_LAYOUT),
        (8, 1, [8, 1, 128, 1280], 0, ttnn.TILE_LAYOUT),
        (8, 2, [8, 1, 32, 1280], 0, ttnn.TILE_LAYOUT),
        (8, 2, [8, 1, 64, 1280], 0, ttnn.TILE_LAYOUT),
        (8, 2, [8, 1, 128, 1280], 0, ttnn.TILE_LAYOUT),
        (8, 4, [8, 1, 32, 1280], 0, ttnn.TILE_LAYOUT),
        (8, 4, [8, 1, 64, 1280], 0, ttnn.TILE_LAYOUT),
        (8, 4, [8, 1, 128, 1280], 0, ttnn.TILE_LAYOUT),
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
        # ttnn.BufferType.L1,
    ],
)
@pytest.mark.parametrize("replication_factor", [4])
@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4_grid")], indirect=True)
@pytest.mark.parametrize(
    "device_params", [{"trace_region_size": 1824800, "fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True
)
def test_all_reduce_async_tg(
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
    replication_factor,
    num_iters=1,
):
    if mesh_device.get_num_devices() != 32:
        pytest.skip("Not TG!")
    run_all_reduce_impl(
        mesh_device,
        output_shape,
        cluster_axis,
        input_dtype,
        num_links,
        input_num_cores,
        input_core_range_set,
        output_num_cores,
        output_core_range_set,
        output_dtype=input_dtype,
        num_iters=num_iters,
        warmup_iters=warmup_iters,
        trace_mode=trace_mode,
        validate_all=False,
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
    use_program_cache,
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
        use_program_cache,
        function_level_defaults,
        num_iters=num_iters,
        topology=ttnn.Topology.Linear,
        trace_mode=trace_mode,
    )
