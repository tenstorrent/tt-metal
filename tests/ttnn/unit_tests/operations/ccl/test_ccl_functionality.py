# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from models.utility_functions import skip_for_grayskull

from tests.ttnn.unit_tests.operations.ccl.test_new_all_gather import (
    run_all_gather_impl,
)

from tests.ttnn.unit_tests.operations.ccl.test_all_reduce_async import run_all_reduce_test

from tests.ttnn.unit_tests.operations.ccl.test_reduce_scatter_async import (
    run_reduce_scatter_test,
)


def create_submeshes(mesh_device, data_parallel):
    if not isinstance(mesh_device, ttnn.MeshDevice) or data_parallel == 1:
        return [mesh_device]

    num_rows, num_cols = mesh_device.shape
    num_devices = num_rows * num_cols
    assert num_devices % data_parallel == 0, f"Unsupported device split: {num_devices} devices, {data_parallel} groups"

    # Check if the mesh is 8x4 (expected shape for TG) and perfer row split
    # Submeshes with 8 devices are expected to be in ring topology hence the row split
    if num_rows == 8 and num_cols == 4 and num_rows % data_parallel == 0:
        submeshes = mesh_device.create_submeshes(ttnn.MeshShape(num_rows // data_parallel, num_cols))
        for submesh in submeshes:
            submesh.reshape(ttnn.MeshShape(1, num_devices // data_parallel))
        return submeshes

    return mesh_device.create_submeshes(ttnn.MeshShape(1, num_devices // data_parallel))


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_links, output_shape, dim, layout",
    [
        (1, [1, 1, 64, 32], 3, ttnn.TILE_LAYOUT),
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
@pytest.mark.parametrize("num_iters", [10])
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("submesh_split_direction", ["row", "col"])
@pytest.mark.parametrize("topology", [ttnn.Topology.Linear])
def test_all_gather_async_t3000(
    mesh_device,
    output_shape,
    dim,
    num_links,
    input_dtype,
    layout,
    mem_config,
    num_iters,
    function_level_defaults,
    submesh_split_direction,
    topology,
):
    ttnn.visualize_mesh_device(mesh_device)

    mesh_shape = mesh_device.shape
    mesh_rows = mesh_shape[0]
    mesh_cols = mesh_shape[1]

    if submesh_split_direction == "row":
        submesh_shape = ttnn.MeshShape(1, mesh_cols)
        num_submeshes = mesh_rows
    elif submesh_split_direction == "col":
        submesh_shape = ttnn.MeshShape(mesh_rows, 1)
        num_submeshes = mesh_cols
    else:
        raise ValueError("Invalid submesh_split_direction")

    num_devices = submesh_shape[0] * submesh_shape[1]
    scaled_output_shape = list(output_shape)
    scaled_output_shape[dim] *= num_devices

    submesh_devices = mesh_device.create_submeshes(submesh_shape)
    assert len(submesh_devices) == num_submeshes, f"Expected {num_submeshes} submeshes but got {len(submesh_devices)}"

    for submesh_device in submesh_devices:
        ttnn.visualize_mesh_device(submesh_device)
        print(f"{submesh_device.get_device_ids()}")
        run_all_gather_impl(
            submesh_device,
            num_devices,
            scaled_output_shape,
            dim,
            num_links,
            input_dtype,
            layout,
            function_level_defaults,
            all_gather_topology=topology,
            num_iters=num_iters,
            rand_tensor=True,
            mem_config=mem_config,
        )


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_links, per_chip_output_shape, dim, layout",
    [
        (1, [1, 8, 32, 1280], 3, ttnn.TILE_LAYOUT),
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
@pytest.mark.parametrize("num_iters", [10])
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("math_op", [ttnn.ReduceType.Sum])
@pytest.mark.parametrize("submesh_split_direction", ["row", "col"])
@pytest.mark.parametrize("topology", [ttnn.Topology.Linear])
def test_all_reduce_async_t3000(
    mesh_device,
    per_chip_output_shape,
    dim,
    num_links,
    math_op,
    input_dtype,
    layout,
    mem_config,
    num_iters,
    function_level_defaults,
    submesh_split_direction,
    topology,
):
    ttnn.visualize_mesh_device(mesh_device)

    mesh_shape = mesh_device.shape
    mesh_rows = mesh_shape[0]
    mesh_cols = mesh_shape[1]

    if submesh_split_direction == "row":
        submesh_shape = ttnn.MeshShape(1, mesh_cols)
        num_submeshes = mesh_rows
    elif submesh_split_direction == "col":
        submesh_shape = ttnn.MeshShape(mesh_rows, 1)
        num_submeshes = mesh_cols
    else:
        raise ValueError("Invalid submesh_split_direction")

    num_devices = submesh_shape[0] * submesh_shape[1]

    submesh_devices = mesh_device.create_submeshes(submesh_shape)
    assert len(submesh_devices) == num_submeshes, f"Expected {num_submeshes} submeshes but got {len(submesh_devices)}"

    for submesh_device in submesh_devices:
        ttnn.visualize_mesh_device(submesh_device)
        print(f"{submesh_device.get_device_ids()}")
        run_all_reduce_test(
            submesh_device,
            num_devices,
            per_chip_output_shape,
            num_links,
            math_op,
            input_dtype,
            layout,
            mem_config,
            function_level_defaults,
            num_iters=num_iters,
            topology=topology,
        )


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_links, per_chip_output_shape, dim, layout",
    [
        (1, [1, 8, 32, 1280], 3, ttnn.TILE_LAYOUT),
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
@pytest.mark.parametrize("num_iters", [10])
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("math_op", [ttnn.ReduceType.Sum])
@pytest.mark.parametrize("submesh_split_direction", ["row", "col"])
@pytest.mark.parametrize("topology", [ttnn.Topology.Linear])
def test_reduce_scatter_async_t3000(
    mesh_device,
    per_chip_output_shape,
    dim,
    num_links,
    math_op,
    input_dtype,
    layout,
    mem_config,
    num_iters,
    function_level_defaults,
    submesh_split_direction,
    topology,
):
    ttnn.visualize_mesh_device(mesh_device)

    mesh_shape = mesh_device.shape
    mesh_rows = mesh_shape[0]
    mesh_cols = mesh_shape[1]

    if submesh_split_direction == "row":
        submesh_shape = ttnn.MeshShape(1, mesh_cols)
        num_submeshes = mesh_rows
    elif submesh_split_direction == "col":
        submesh_shape = ttnn.MeshShape(mesh_rows, 1)
        num_submeshes = mesh_cols
    else:
        raise ValueError("Invalid submesh_split_direction")

    num_devices = submesh_shape[0] * submesh_shape[1]

    submesh_devices = mesh_device.create_submeshes(submesh_shape)
    assert len(submesh_devices) == num_submeshes, f"Expected {num_submeshes} submeshes but got {len(submesh_devices)}"

    for submesh_device in submesh_devices:
        ttnn.visualize_mesh_device(submesh_device)
        print(f"{submesh_device.get_device_ids()}")
        run_reduce_scatter_test(
            submesh_device,
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
        )


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_links, output_shape, dim, layout",
    [
        (1, [1, 1, 64, 32], 3, ttnn.TILE_LAYOUT),
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
@pytest.mark.parametrize("num_iters", [10])
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize("num_devices", [2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("offset_val", list(range(8)))
@pytest.mark.parametrize("topology", [ttnn.Topology.Linear])
def test_all_gather_async_arbitrary_t3000(
    mesh_device,
    num_devices,
    offset_val,
    output_shape,
    dim,
    num_links,
    input_dtype,
    layout,
    mem_config,
    num_iters,
    function_level_defaults,
    topology,
):
    if offset_val > 8 - num_devices:
        pytest.skip(f"offset {offset_val} is too large for num_devices={num_devices}")
    ttnn.visualize_mesh_device(mesh_device)

    mesh_shape = mesh_device.shape
    mesh_rows = mesh_shape[0]
    mesh_cols = mesh_shape[1]
    submesh_shape = ttnn.MeshShape(mesh_rows, num_devices)

    max_offset = mesh_cols - num_devices

    offset = ttnn.MeshCoordinate(0, offset_val)
    submesh_device = mesh_device.create_submesh(submesh_shape, offset)
    device_ids = submesh_device.get_device_ids()
    print(f"Testing submesh with offset {offset}, IDs: {device_ids}")

    scaled_output_shape = list(output_shape)
    scaled_output_shape[dim] *= num_devices

    ttnn.visualize_mesh_device(submesh_device)

    run_all_gather_impl(
        submesh_device,
        num_devices,
        scaled_output_shape,
        dim,
        num_links,
        input_dtype,
        layout,
        function_level_defaults,
        all_gather_topology=topology,
        num_iters=num_iters,
        rand_tensor=True,
        mem_config=mem_config,
    )


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_links, per_chip_output_shape, dim, layout",
    [
        (1, [1, 8, 32, 26880], 3, ttnn.TILE_LAYOUT),
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
@pytest.mark.parametrize("num_iters", [10])
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize("num_devices", [2, 3, 4, 5, 6, 7])
@pytest.mark.parametrize("offset_val", list(range(8)))
@pytest.mark.parametrize("math_op", [ttnn.ReduceType.Sum])
@pytest.mark.parametrize("topology", [ttnn.Topology.Linear])
def test_all_reduce_async_arbitrary_t3000(
    mesh_device,
    num_devices,
    offset_val,
    per_chip_output_shape,
    dim,
    num_links,
    math_op,
    input_dtype,
    layout,
    mem_config,
    num_iters,
    function_level_defaults,
    topology,
):
    if offset_val > 8 - num_devices:
        pytest.skip(f"offset {offset_val} is too large for num_devices={num_devices}")

    ttnn.visualize_mesh_device(mesh_device)

    mesh_shape = mesh_device.shape
    mesh_rows = mesh_shape[0]
    mesh_cols = mesh_shape[1]
    submesh_shape = ttnn.MeshShape(mesh_rows, num_devices)

    max_offset = mesh_cols - num_devices

    # for col_offset in range(max_offset + 1):
    offset = ttnn.MeshCoordinate(0, offset_val)
    submesh_device = mesh_device.create_submesh(submesh_shape, offset)
    device_ids = submesh_device.get_device_ids()
    print(f"Testing submesh with offset {offset}, IDs: {device_ids}")

    ttnn.visualize_mesh_device(submesh_device)
    print(f"{submesh_device.get_device_ids()}")
    run_all_reduce_test(
        submesh_device,
        num_devices,
        per_chip_output_shape,
        num_links,
        math_op,
        input_dtype,
        layout,
        mem_config,
        function_level_defaults,
        num_iters=num_iters,
        topology=topology,
    )
    ttnn.close_mesh_device(submesh_device)


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_links, per_chip_output_shape, dim, layout",
    [
        (1, [1, 8, 32, 1280], 3, ttnn.TILE_LAYOUT),
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
@pytest.mark.parametrize("num_iters", [10])
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize("num_devices", [2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("offset_val", list(range(8)))
@pytest.mark.parametrize("math_op", [ttnn.ReduceType.Sum])
@pytest.mark.parametrize("topology", [ttnn.Topology.Linear])
def test_reduce_scatter_async_arbitrary_t3000(
    mesh_device,
    num_devices,
    offset_val,
    per_chip_output_shape,
    dim,
    num_links,
    math_op,
    input_dtype,
    layout,
    mem_config,
    num_iters,
    function_level_defaults,
    topology,
):
    if offset_val > 8 - num_devices:
        pytest.skip(f"offset {offset_val} is too large for num_devices={num_devices}")
    ttnn.visualize_mesh_device(mesh_device)

    mesh_shape = mesh_device.shape
    mesh_rows = mesh_shape[0]
    mesh_cols = mesh_shape[1]
    submesh_shape = ttnn.MeshShape(mesh_rows, num_devices)

    max_offset = mesh_cols - num_devices

    offset = ttnn.MeshCoordinate(0, offset_val)
    submesh_device = mesh_device.create_submesh(submesh_shape, offset)
    device_ids = submesh_device.get_device_ids()
    print(f"Testing submesh with offset {offset}, IDs: {device_ids}")
    ttnn.visualize_mesh_device(submesh_device)
    print(f"{submesh_device.get_device_ids()}")
    run_reduce_scatter_test(
        submesh_device,
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
    )


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_links, output_shape, dim, layout",
    [
        (1, [1, 1, 64, 32], 3, ttnn.TILE_LAYOUT),
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
@pytest.mark.parametrize("num_iters", [10])
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("offset", [2])
@pytest.mark.parametrize("topology", [ttnn.Topology.Ring])
def test_all_gather_async_2x2_t3000(
    mesh_device,
    offset,
    output_shape,
    dim,
    num_links,
    input_dtype,
    layout,
    mem_config,
    num_iters,
    function_level_defaults,
    topology,
):
    ttnn.visualize_mesh_device(mesh_device)

    num_devices = 4
    scaled_output_shape = list(output_shape)
    scaled_output_shape[dim] *= num_devices

    submesh_shape = ttnn.MeshShape(2, 2)
    submesh = mesh_device.create_submesh(submesh_shape, offset=ttnn.MeshCoordinate(0, offset))
    submesh.reshape(ttnn.MeshShape(1, num_devices))
    ttnn.visualize_mesh_device(submesh)
    print(f"{submesh.get_device_ids()}")
    run_all_gather_impl(
        submesh,
        num_devices,
        scaled_output_shape,
        dim,
        num_links,
        input_dtype,
        layout,
        function_level_defaults,
        all_gather_topology=topology,
        num_iters=num_iters,
        rand_tensor=True,
        mem_config=mem_config,
    )


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_links, per_chip_output_shape, dim, layout",
    [
        (1, [1, 8, 32, 1280], 3, ttnn.TILE_LAYOUT),
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
@pytest.mark.parametrize("num_iters", [10])
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("offset", [2])
@pytest.mark.parametrize("math_op", [ttnn.ReduceType.Sum])
@pytest.mark.parametrize("topology", [ttnn.Topology.Ring])
def test_all_reduce_async_2x2_t3000(
    mesh_device,
    offset,
    per_chip_output_shape,
    dim,
    num_links,
    math_op,
    input_dtype,
    layout,
    mem_config,
    num_iters,
    function_level_defaults,
    topology,
):
    ttnn.visualize_mesh_device(mesh_device)

    num_devices = 4

    submesh_shape = ttnn.MeshShape(2, 2)
    submesh = mesh_device.create_submesh(submesh_shape, offset=ttnn.MeshCoordinate(0, offset))
    submesh.reshape(ttnn.MeshShape(1, num_devices))
    ttnn.visualize_mesh_device(submesh)
    print(f"{submesh.get_device_ids()}")
    run_all_reduce_test(
        submesh,
        num_devices,
        per_chip_output_shape,
        num_links,
        math_op,
        input_dtype,
        layout,
        mem_config,
        function_level_defaults,
        num_iters=num_iters,
        topology=topology,
    )


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_links, per_chip_output_shape, dim, layout",
    [
        (1, [1, 8, 32, 1280], 3, ttnn.TILE_LAYOUT),
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
@pytest.mark.parametrize("num_iters", [10])
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("offset", [0, 1, 2])
@pytest.mark.parametrize("math_op", [ttnn.ReduceType.Sum])
@pytest.mark.parametrize("topology", [ttnn.Topology.Linear, ttnn.Topology.Ring])
def test_reduce_scatter_async_2x2_t3000(
    mesh_device,
    offset,
    per_chip_output_shape,
    dim,
    num_links,
    math_op,
    input_dtype,
    layout,
    mem_config,
    num_iters,
    function_level_defaults,
    topology,
):
    ttnn.visualize_mesh_device(mesh_device)

    num_devices = 4

    submesh_shape = ttnn.MeshShape(2, 2)
    submesh = mesh_device.create_submesh(submesh_shape, offset=ttnn.MeshCoordinate(0, offset))
    submesh.reshape(ttnn.MeshShape(1, num_devices))
    ttnn.visualize_mesh_device(submesh)
    print(f"{submesh.get_device_ids()}")

    run_reduce_scatter_test(
        submesh,
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
    )


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_links, per_chip_output_shape, dim, layout",
    [
        (1, [1, 8, 32, 1280], 3, ttnn.TILE_LAYOUT),
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
@pytest.mark.parametrize("num_iters", [10])
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True)
@pytest.mark.parametrize("mesh_device2", [(1, 4)], indirect=True)
@pytest.mark.parametrize("math_op", [ttnn.ReduceType.Sum])
@pytest.mark.parametrize("topology", [ttnn.Topology.Linear, ttnn.Topology.Ring])
def test_reduce_scatter_async_uncontiguous_t3000(
    mesh_device2,
    per_chip_output_shape,
    dim,
    num_links,
    math_op,
    input_dtype,
    layout,
    mem_config,
    num_iters,
    function_level_defaults,
    topology,
):
    ttnn.visualize_mesh_device(mesh_device2)
    num_devices = 4
    print(f"{mesh_device2.get_device_ids()}")
    run_reduce_scatter_test(
        mesh_device2,
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
    )


# Galaxy


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_links, output_shape, dim, layout",
    [
        (1, [1, 1, 64, 32], 3, ttnn.TILE_LAYOUT),
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
@pytest.mark.parametrize("num_iters", [10])
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("submesh_split_direction", ["row", "col"])
@pytest.mark.parametrize("topology", [ttnn.Topology.Linear])
def test_all_gather_async_galaxy(
    mesh_device,
    output_shape,
    dim,
    num_links,
    input_dtype,
    layout,
    mem_config,
    num_iters,
    function_level_defaults,
    submesh_split_direction,
    topology,
):
    ttnn.visualize_mesh_device(mesh_device)

    mesh_shape = mesh_device.shape
    mesh_rows = mesh_shape[0]
    mesh_cols = mesh_shape[1]

    if submesh_split_direction == "row":
        submesh_shape = ttnn.MeshShape(1, mesh_cols)
        num_submeshes = mesh_rows
    elif submesh_split_direction == "col":
        submesh_shape = ttnn.MeshShape(mesh_rows, 1)
        num_submeshes = mesh_cols
    else:
        raise ValueError("Invalid submesh_split_direction")

    num_devices = submesh_shape[0] * submesh_shape[1]
    scaled_output_shape = list(output_shape)
    scaled_output_shape[dim] *= num_devices

    submesh_devices = mesh_device.create_submeshes(submesh_shape)
    assert len(submesh_devices) == num_submeshes, f"Expected {num_submeshes} submeshes but got {len(submesh_devices)}"

    for submesh_device in submesh_devices:
        ttnn.visualize_mesh_device(submesh_device)
        print(f"{submesh_device.get_device_ids()}")
        run_all_gather_impl(
            submesh_device,
            num_devices,
            scaled_output_shape,
            dim,
            num_links,
            input_dtype,
            layout,
            function_level_defaults,
            all_gather_topology=topology,
            num_iters=num_iters,
            rand_tensor=True,
            mem_config=mem_config,
        )


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_links, per_chip_output_shape, dim, layout",
    [
        (1, [1, 8, 32, 1280], 3, ttnn.TILE_LAYOUT),
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
@pytest.mark.parametrize("num_iters", [10])
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("math_op", [ttnn.ReduceType.Sum])
@pytest.mark.parametrize("submesh_split_direction", ["row", "col"])
@pytest.mark.parametrize("topology", [ttnn.Topology.Linear])
def test_all_reduce_async_t3000(
    mesh_device,
    per_chip_output_shape,
    dim,
    num_links,
    math_op,
    input_dtype,
    layout,
    mem_config,
    num_iters,
    function_level_defaults,
    submesh_split_direction,
    topology,
):
    ttnn.visualize_mesh_device(mesh_device)

    mesh_shape = mesh_device.shape
    mesh_rows = mesh_shape[0]
    mesh_cols = mesh_shape[1]

    if submesh_split_direction == "row":
        submesh_shape = ttnn.MeshShape(1, mesh_cols)
        num_submeshes = mesh_rows
    elif submesh_split_direction == "col":
        submesh_shape = ttnn.MeshShape(mesh_rows, 1)
        num_submeshes = mesh_cols
    else:
        raise ValueError("Invalid submesh_split_direction")

    num_devices = submesh_shape[0] * submesh_shape[1]

    submesh_devices = mesh_device.create_submeshes(submesh_shape)
    assert len(submesh_devices) == num_submeshes, f"Expected {num_submeshes} submeshes but got {len(submesh_devices)}"

    for submesh_device in submesh_devices:
        ttnn.visualize_mesh_device(submesh_device)
        print(f"{submesh_device.get_device_ids()}")
        run_all_reduce_test(
            submesh_device,
            num_devices,
            per_chip_output_shape,
            num_links,
            math_op,
            input_dtype,
            layout,
            mem_config,
            function_level_defaults,
            num_iters=num_iters,
            topology=topology,
        )


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_links, per_chip_output_shape, dim, layout",
    [
        (1, [1, 8, 32, 1280], 3, ttnn.TILE_LAYOUT),
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
@pytest.mark.parametrize("num_iters", [10])
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("math_op", [ttnn.ReduceType.Sum])
@pytest.mark.parametrize("submesh_split_direction", ["row", "col"])
@pytest.mark.parametrize("topology", [ttnn.Topology.Linear])
def test_reduce_scatter_async_t3000(
    mesh_device,
    per_chip_output_shape,
    dim,
    num_links,
    math_op,
    input_dtype,
    layout,
    mem_config,
    num_iters,
    function_level_defaults,
    submesh_split_direction,
    topology,
):
    ttnn.visualize_mesh_device(mesh_device)

    mesh_shape = mesh_device.shape
    mesh_rows = mesh_shape[0]
    mesh_cols = mesh_shape[1]

    if submesh_split_direction == "row":
        submesh_shape = ttnn.MeshShape(1, mesh_cols)
        num_submeshes = mesh_rows
    elif submesh_split_direction == "col":
        submesh_shape = ttnn.MeshShape(mesh_rows, 1)
        num_submeshes = mesh_cols
    else:
        raise ValueError("Invalid submesh_split_direction")

    num_devices = submesh_shape[0] * submesh_shape[1]

    submesh_devices = mesh_device.create_submeshes(submesh_shape)
    assert len(submesh_devices) == num_submeshes, f"Expected {num_submeshes} submeshes but got {len(submesh_devices)}"

    for submesh_device in submesh_devices:
        ttnn.visualize_mesh_device(submesh_device)
        print(f"{submesh_device.get_device_ids()}")
        run_reduce_scatter_test(
            submesh_device,
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
        )


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_links, output_shape, dim, layout",
    [
        (1, [1, 1, 64, 32], 3, ttnn.TILE_LAYOUT),
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
@pytest.mark.parametrize("num_iters", [10])
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize("num_devices", [2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("offset_val", list(range(8)))
@pytest.mark.parametrize("topology", [ttnn.Topology.Linear])
def test_all_gather_async_arbitrary_t3000(
    mesh_device,
    num_devices,
    offset_val,
    output_shape,
    dim,
    num_links,
    input_dtype,
    layout,
    mem_config,
    num_iters,
    function_level_defaults,
    topology,
):
    if offset_val > 8 - num_devices:
        pytest.skip(f"offset {offset_val} is too large for num_devices={num_devices}")
    ttnn.visualize_mesh_device(mesh_device)

    mesh_shape = mesh_device.shape
    mesh_rows = mesh_shape[0]
    mesh_cols = mesh_shape[1]
    submesh_shape = ttnn.MeshShape(mesh_rows, num_devices)

    max_offset = mesh_cols - num_devices

    offset = ttnn.MeshCoordinate(0, offset_val)
    submesh_device = mesh_device.create_submesh(submesh_shape, offset)
    device_ids = submesh_device.get_device_ids()
    print(f"Testing submesh with offset {offset}, IDs: {device_ids}")

    scaled_output_shape = list(output_shape)
    scaled_output_shape[dim] *= num_devices

    ttnn.visualize_mesh_device(submesh_device)

    run_all_gather_impl(
        submesh_device,
        num_devices,
        scaled_output_shape,
        dim,
        num_links,
        input_dtype,
        layout,
        function_level_defaults,
        all_gather_topology=topology,
        num_iters=num_iters,
        rand_tensor=True,
        mem_config=mem_config,
    )


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_links, per_chip_output_shape, dim, layout",
    [
        (1, [1, 8, 32, 26880], 3, ttnn.TILE_LAYOUT),
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
@pytest.mark.parametrize("num_iters", [10])
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize("num_devices", [2, 3, 4, 5, 6, 7])
@pytest.mark.parametrize("offset_val", list(range(8)))
@pytest.mark.parametrize("math_op", [ttnn.ReduceType.Sum])
@pytest.mark.parametrize("topology", [ttnn.Topology.Linear])
def test_all_reduce_async_arbitrary_t3000(
    mesh_device,
    num_devices,
    offset_val,
    per_chip_output_shape,
    dim,
    num_links,
    math_op,
    input_dtype,
    layout,
    mem_config,
    num_iters,
    function_level_defaults,
    topology,
):
    if offset_val > 8 - num_devices:
        pytest.skip(f"offset {offset_val} is too large for num_devices={num_devices}")

    ttnn.visualize_mesh_device(mesh_device)

    mesh_shape = mesh_device.shape
    mesh_rows = mesh_shape[0]
    mesh_cols = mesh_shape[1]
    submesh_shape = ttnn.MeshShape(mesh_rows, num_devices)

    max_offset = mesh_cols - num_devices

    # for col_offset in range(max_offset + 1):
    offset = ttnn.MeshCoordinate(0, offset_val)
    submesh_device = mesh_device.create_submesh(submesh_shape, offset)
    device_ids = submesh_device.get_device_ids()
    print(f"Testing submesh with offset {offset}, IDs: {device_ids}")

    ttnn.visualize_mesh_device(submesh_device)
    print(f"{submesh_device.get_device_ids()}")
    run_all_reduce_test(
        submesh_device,
        num_devices,
        per_chip_output_shape,
        num_links,
        math_op,
        input_dtype,
        layout,
        mem_config,
        function_level_defaults,
        num_iters=num_iters,
        topology=topology,
    )
    ttnn.close_mesh_device(submesh_device)


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_links, per_chip_output_shape, dim, layout",
    [
        (1, [1, 8, 32, 1280], 3, ttnn.TILE_LAYOUT),
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
@pytest.mark.parametrize("num_iters", [10])
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize("num_devices", [2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("offset_val", list(range(8)))
@pytest.mark.parametrize("math_op", [ttnn.ReduceType.Sum])
@pytest.mark.parametrize("topology", [ttnn.Topology.Linear])
def test_reduce_scatter_async_arbitrary_t3000(
    mesh_device,
    num_devices,
    offset_val,
    per_chip_output_shape,
    dim,
    num_links,
    math_op,
    input_dtype,
    layout,
    mem_config,
    num_iters,
    function_level_defaults,
    topology,
):
    if offset_val > 8 - num_devices:
        pytest.skip(f"offset {offset_val} is too large for num_devices={num_devices}")
    ttnn.visualize_mesh_device(mesh_device)

    mesh_shape = mesh_device.shape
    mesh_rows = mesh_shape[0]
    mesh_cols = mesh_shape[1]
    submesh_shape = ttnn.MeshShape(mesh_rows, num_devices)

    max_offset = mesh_cols - num_devices

    offset = ttnn.MeshCoordinate(0, offset_val)
    submesh_device = mesh_device.create_submesh(submesh_shape, offset)
    device_ids = submesh_device.get_device_ids()
    print(f"Testing submesh with offset {offset}, IDs: {device_ids}")
    ttnn.visualize_mesh_device(submesh_device)
    print(f"{submesh_device.get_device_ids()}")
    run_reduce_scatter_test(
        submesh_device,
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
    )


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_links, output_shape, dim, layout",
    [
        (1, [1, 1, 64, 32], 3, ttnn.TILE_LAYOUT),
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
@pytest.mark.parametrize("num_iters", [10])
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("offset", [2])
@pytest.mark.parametrize("topology", [ttnn.Topology.Ring])
def test_all_gather_async_2x2_t3000(
    mesh_device,
    offset,
    output_shape,
    dim,
    num_links,
    input_dtype,
    layout,
    mem_config,
    num_iters,
    function_level_defaults,
    topology,
):
    ttnn.visualize_mesh_device(mesh_device)

    num_devices = 4
    scaled_output_shape = list(output_shape)
    scaled_output_shape[dim] *= num_devices

    submesh_shape = ttnn.MeshShape(2, 2)
    submesh = mesh_device.create_submesh(submesh_shape, offset=ttnn.MeshCoordinate(0, offset))
    submesh.reshape(ttnn.MeshShape(1, num_devices))
    ttnn.visualize_mesh_device(submesh)
    print(f"{submesh.get_device_ids()}")
    run_all_gather_impl(
        submesh,
        num_devices,
        scaled_output_shape,
        dim,
        num_links,
        input_dtype,
        layout,
        function_level_defaults,
        all_gather_topology=topology,
        num_iters=num_iters,
        rand_tensor=True,
        mem_config=mem_config,
    )


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_links, per_chip_output_shape, dim, layout",
    [
        (1, [1, 8, 32, 1280], 3, ttnn.TILE_LAYOUT),
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
@pytest.mark.parametrize("num_iters", [10])
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("offset", [2])
@pytest.mark.parametrize("math_op", [ttnn.ReduceType.Sum])
@pytest.mark.parametrize("topology", [ttnn.Topology.Ring])
def test_all_reduce_async_2x2_t3000(
    mesh_device,
    offset,
    per_chip_output_shape,
    dim,
    num_links,
    math_op,
    input_dtype,
    layout,
    mem_config,
    num_iters,
    function_level_defaults,
    topology,
):
    ttnn.visualize_mesh_device(mesh_device)

    num_devices = 4

    submesh_shape = ttnn.MeshShape(2, 2)
    submesh = mesh_device.create_submesh(submesh_shape, offset=ttnn.MeshCoordinate(0, offset))
    submesh.reshape(ttnn.MeshShape(1, num_devices))
    ttnn.visualize_mesh_device(submesh)
    print(f"{submesh.get_device_ids()}")
    run_all_reduce_test(
        submesh,
        num_devices,
        per_chip_output_shape,
        num_links,
        math_op,
        input_dtype,
        layout,
        mem_config,
        function_level_defaults,
        num_iters=num_iters,
        topology=topology,
    )


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_links, per_chip_output_shape, dim, layout",
    [
        (1, [1, 8, 32, 1280], 3, ttnn.TILE_LAYOUT),
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
@pytest.mark.parametrize("num_iters", [10])
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("offset", [0, 1, 2])
@pytest.mark.parametrize("math_op", [ttnn.ReduceType.Sum])
@pytest.mark.parametrize("topology", [ttnn.Topology.Linear, ttnn.Topology.Ring])
def test_reduce_scatter_async_2x2_t3000(
    mesh_device,
    offset,
    per_chip_output_shape,
    dim,
    num_links,
    math_op,
    input_dtype,
    layout,
    mem_config,
    num_iters,
    function_level_defaults,
    topology,
):
    ttnn.visualize_mesh_device(mesh_device)

    num_devices = 4

    submesh_shape = ttnn.MeshShape(2, 2)
    submesh = mesh_device.create_submesh(submesh_shape, offset=ttnn.MeshCoordinate(0, offset))
    submesh.reshape(ttnn.MeshShape(1, num_devices))
    ttnn.visualize_mesh_device(submesh)
    print(f"{submesh.get_device_ids()}")

    run_reduce_scatter_test(
        submesh,
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
    )


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_links, per_chip_output_shape, dim, layout",
    [
        (1, [1, 8, 32, 1280], 3, ttnn.TILE_LAYOUT),
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
@pytest.mark.parametrize("num_iters", [10])
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True)
@pytest.mark.parametrize("mesh_device2", [(1, 4)], indirect=True)
@pytest.mark.parametrize("math_op", [ttnn.ReduceType.Sum])
@pytest.mark.parametrize("topology", [ttnn.Topology.Linear, ttnn.Topology.Ring])
def test_reduce_scatter_async_uncontiguous_t3000(
    mesh_device2,
    per_chip_output_shape,
    dim,
    num_links,
    math_op,
    input_dtype,
    layout,
    mem_config,
    num_iters,
    function_level_defaults,
    topology,
):
    ttnn.visualize_mesh_device(mesh_device2)
    num_devices = 4
    print(f"{mesh_device2.get_device_ids()}")
    run_reduce_scatter_test(
        mesh_device2,
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
    )
