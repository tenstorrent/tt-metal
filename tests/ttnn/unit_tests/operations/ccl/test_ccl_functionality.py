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

from tests.ttnn.unit_tests.operations.ccl.test_send_recv_async import (
    run_send_recv_test,
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
        ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1),
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
@pytest.mark.parametrize("topology", [ttnn.Topology.Linear])
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


def _row_submesh(mesh_device, row_idx):
    rows, cols = mesh_device.shape
    assert 0 <= row_idx < rows
    return mesh_device.create_submesh(ttnn.MeshShape(1, cols), ttnn.MeshCoordinate(row_idx, 0))


def _col_submesh(mesh_device, col_idx):
    rows, cols = mesh_device.shape
    assert 0 <= col_idx < cols
    return mesh_device.create_submesh(ttnn.MeshShape(rows, 1), ttnn.MeshCoordinate(0, col_idx))


def _single_dev(mesh_device, r, c):
    return mesh_device.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(r, c))


def _submesh(mesh_device, start_r, start_c, h, w):
    rows, cols = mesh_device.shape
    assert 0 <= start_r < rows and 0 <= start_c < cols, "start coords OOB"
    assert start_r + h <= rows and start_c + w <= cols, "submesh shape OOB"
    return mesh_device.create_submesh(
        ttnn.MeshShape(h, w),
        ttnn.MeshCoordinate(start_r, start_c),
    )


def _overlap_2x2(a_r, a_c, b_r, b_c):
    a_rows = {a_r, a_r + 1}
    a_cols = {a_c, a_c + 1}
    b_rows = {b_r, b_r + 1}
    b_cols = {b_c, b_c + 1}
    return (a_rows & b_rows) and (a_cols & b_cols)


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.timeout(300)
@pytest.mark.parametrize(
    "per_chip_shape",
    [
        # ([1, 1, 32, 4096]),
        ([1, 1, 64, 8192]),
    ],
)
@pytest.mark.parametrize(
    "layout",
    [
        ttnn.TILE_LAYOUT,
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.bfloat16,
        # ttnn.bfloat8_b,
    ],
)
@pytest.mark.parametrize(
    "mem_config",
    [
        ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM),
        # ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1),
    ],
)
@pytest.mark.parametrize(
    "socket_storage_type",
    [
        ttnn.BufferType.DRAM,
        # ttnn.BufferType.L1,
    ],
)
@pytest.mark.parametrize(
    "socket_fifo_size",
    [
        10 * 1024,
    ],
)
@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_DYNAMIC}], indirect=True)
@pytest.mark.parametrize(
    "s, r",
    [(s, r) for s in range(2) for r in range(2) if s != r],  # rows=2 기준
)
def test_send_recv_rows_t3000(
    mesh_device, per_chip_shape, layout, dtype, mem_config, socket_storage_type, socket_fifo_size, s, r
):
    ttnn.visualize_mesh_device(mesh_device)
    rows, cols = mesh_device.shape
    row_meshes = [_row_submesh(mesh_device, rr) for rr in range(rows)]

    send_mesh = row_meshes[s]
    recv_mesh = row_meshes[r]

    logger.info(f"=== CASE: send row {s} -> recv row {r} ===")
    logger.info(f"Send IDs: {send_mesh.get_device_ids()}")
    ttnn.visualize_mesh_device(send_mesh)
    logger.info(f"Recv IDs: {recv_mesh.get_device_ids()}")
    ttnn.visualize_mesh_device(recv_mesh)

    run_send_recv_test(
        send_mesh,
        recv_mesh,
        socket_storage_type,
        socket_fifo_size,
        per_chip_shape,
        mem_config,
        dtype,
        layout,
    )


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.timeout(300)
@pytest.mark.parametrize(
    "per_chip_shape",
    [
        # ([1, 1, 32, 4096]),
        ([1, 1, 64, 8192]),
    ],
)
@pytest.mark.parametrize(
    "layout",
    [
        ttnn.TILE_LAYOUT,
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.bfloat16,
        # ttnn.bfloat8_b,
    ],
)
@pytest.mark.parametrize(
    "mem_config",
    [
        ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM),
        # ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1),
    ],
)
@pytest.mark.parametrize(
    "socket_storage_type",
    [
        ttnn.BufferType.DRAM,
        # ttnn.BufferType.L1,
    ],
)
@pytest.mark.parametrize(
    "socket_fifo_size",
    [
        10 * 1024,
    ],
)
@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_DYNAMIC}], indirect=True)
@pytest.mark.parametrize(
    "s, r",
    [(s, r) for s in range(4) for r in range(4) if s != r],  # cols=4 기준
)
def test_send_recv_cols_t3000(
    mesh_device, per_chip_shape, layout, dtype, mem_config, socket_storage_type, socket_fifo_size, s, r
):
    ttnn.visualize_mesh_device(mesh_device)
    rows, cols = mesh_device.shape
    col_meshes = [_col_submesh(mesh_device, c) for c in range(cols)]

    send_mesh = col_meshes[s]
    recv_mesh = col_meshes[r]

    logger.info(f"=== CASE: send col {s} -> recv col {r} ===")
    logger.info(f"Send Mesh Device: {send_mesh.get_device_ids()}")
    ttnn.visualize_mesh_device(send_mesh)
    logger.info(f"Receive Mesh Device: {recv_mesh.get_device_ids()}")
    ttnn.visualize_mesh_device(recv_mesh)

    run_send_recv_test(
        send_mesh,
        recv_mesh,
        socket_storage_type,
        socket_fifo_size,
        per_chip_shape,
        mem_config,
        dtype,
        layout,
    )


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.timeout(300)
@pytest.mark.parametrize(
    "per_chip_shape",
    [
        # ([1, 1, 32, 4096]),
        ([1, 1, 64, 8192]),
    ],
)
@pytest.mark.parametrize(
    "layout",
    [
        ttnn.TILE_LAYOUT,
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.bfloat16,
        # ttnn.bfloat8_b,
    ],
)
@pytest.mark.parametrize(
    "mem_config",
    [
        ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM),
        # ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1),
    ],
)
@pytest.mark.parametrize(
    "socket_storage_type",
    [
        ttnn.BufferType.DRAM,
        # ttnn.BufferType.L1,
    ],
)
@pytest.mark.parametrize(
    "socket_fifo_size",
    [
        10 * 1024,
    ],
)
@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_DYNAMIC}], indirect=True)
@pytest.mark.parametrize(
    "s, r",
    [(s, r) for s in range(8) for r in range(8) if s != r],  # (2x4) -> 8 singles
)
def test_send_recv_single_to_single_t3000(
    mesh_device, per_chip_shape, layout, dtype, mem_config, socket_storage_type, socket_fifo_size, s, r
):
    ttnn.visualize_mesh_device(mesh_device)

    rows, cols = mesh_device.shape
    singles = [_single_dev(mesh_device, rr, cc) for rr in range(rows) for cc in range(cols)]

    send_mesh = singles[s]
    recv_mesh = singles[r]

    logger.info(f"=== CASE: send {s} -> recv {r} ===")
    logger.info(f"Send Mesh Device: {send_mesh.get_device_ids()}")
    ttnn.visualize_mesh_device(send_mesh)
    logger.info(f"Receive Mesh Device: {recv_mesh.get_device_ids()}")
    ttnn.visualize_mesh_device(recv_mesh)

    run_send_recv_test(
        send_mesh,
        recv_mesh,
        socket_storage_type,
        socket_fifo_size,
        per_chip_shape,
        mem_config,
        dtype,
        layout,
    )


# =========================
# ========= Galaxy =========
# =========================


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.timeout(300)
@pytest.mark.parametrize(
    "per_chip_shape",
    [
        # ([1, 1, 32, 4096]),
        ([1, 1, 64, 8192]),
    ],
)
@pytest.mark.parametrize(
    "layout",
    [
        ttnn.TILE_LAYOUT,
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.bfloat16,
        # ttnn.bfloat8_b,
    ],
)
@pytest.mark.parametrize(
    "mem_config",
    [
        ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM),
        # ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1),
    ],
)
@pytest.mark.parametrize(
    "socket_storage_type",
    [
        ttnn.BufferType.DRAM,
        # ttnn.BufferType.L1,
    ],
)
@pytest.mark.parametrize(
    "socket_fifo_size",
    [
        10 * 1024,
    ],
)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_DYNAMIC}], indirect=True)
@pytest.mark.parametrize(
    "s, r",
    [(s, r) for s in range(4) for r in range(4) if s != r],  # rows=4
)
def test_send_recv_rows_galaxy(
    mesh_device, per_chip_shape, layout, dtype, mem_config, socket_storage_type, socket_fifo_size, s, r
):
    ttnn.visualize_mesh_device(mesh_device)
    rows, cols = mesh_device.shape
    row_meshes = [_row_submesh(mesh_device, rr) for rr in range(rows)]

    send_mesh = row_meshes[s]
    recv_mesh = row_meshes[r]

    logger.info(f"=== CASE: send row {s} -> recv row {r} ===")
    logger.info(f"Send Mesh Device: {send_mesh.get_device_ids()}")
    ttnn.visualize_mesh_device(send_mesh)
    logger.info(f"Receive Mesh Device: {recv_mesh.get_device_ids()}")
    ttnn.visualize_mesh_device(recv_mesh)

    run_send_recv_test(
        send_mesh,
        recv_mesh,
        socket_storage_type,
        socket_fifo_size,
        per_chip_shape,
        mem_config,
        dtype,
        layout,
    )


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.timeout(300)
@pytest.mark.parametrize(
    "per_chip_shape",
    [
        # ([1, 1, 32, 4096]),
        ([1, 1, 64, 8192]),
    ],
)
@pytest.mark.parametrize(
    "layout",
    [
        ttnn.TILE_LAYOUT,
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.bfloat16,
        # ttnn.bfloat8_b,
    ],
)
@pytest.mark.parametrize(
    "mem_config",
    [
        ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM),
        # ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1),
    ],
)
@pytest.mark.parametrize(
    "socket_storage_type",
    [
        ttnn.BufferType.DRAM,
        # ttnn.BufferType.L1,
    ],
)
@pytest.mark.parametrize(
    "socket_fifo_size",
    [
        10 * 1024,
    ],
)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_DYNAMIC}], indirect=True)
@pytest.mark.parametrize(
    "s, r",
    [(s, r) for s in range(8) for r in range(8) if s != r],  # cols=8
)
def test_send_recv_cols_galaxy(
    mesh_device, per_chip_shape, layout, dtype, mem_config, socket_storage_type, socket_fifo_size, s, r
):
    ttnn.visualize_mesh_device(mesh_device)
    rows, cols = mesh_device.shape
    col_meshes = [_col_submesh(mesh_device, c) for c in range(cols)]

    send_mesh = col_meshes[s]
    recv_mesh = col_meshes[r]

    logger.info(f"=== CASE: send col {s} -> recv col {r} ===")
    logger.info(f"Send Mesh Device: {send_mesh.get_device_ids()}")
    ttnn.visualize_mesh_device(send_mesh)
    logger.info(f"Receive Mesh Device: {recv_mesh.get_device_ids()}")
    ttnn.visualize_mesh_device(recv_mesh)

    run_send_recv_test(
        send_mesh,
        recv_mesh,
        socket_storage_type,
        socket_fifo_size,
        per_chip_shape,
        mem_config,
        dtype,
        layout,
    )


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.timeout(300)
@pytest.mark.parametrize(
    "per_chip_shape",
    [
        # ([1, 1, 32, 4096]),
        ([1, 1, 64, 8192]),
    ],
)
@pytest.mark.parametrize(
    "layout",
    [
        ttnn.TILE_LAYOUT,
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.bfloat16,
        # ttnn.bfloat8_b,
    ],
)
@pytest.mark.parametrize(
    "mem_config",
    [
        ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM),
        # ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1),
    ],
)
@pytest.mark.parametrize(
    "socket_storage_type",
    [
        ttnn.BufferType.DRAM,
        # ttnn.BufferType.L1,
    ],
)
@pytest.mark.parametrize(
    "socket_fifo_size",
    [
        10 * 1024,
    ],
)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_DYNAMIC}], indirect=True)
@pytest.mark.parametrize(
    "s, r",
    [(s, r) for s in range(32) for r in range(32) if s != r],  # 4x8 -> 32 singles
)
def test_send_recv_single_to_single_galaxy(
    mesh_device, per_chip_shape, layout, dtype, mem_config, socket_storage_type, socket_fifo_size, s, r
):
    ttnn.visualize_mesh_device(mesh_device)
    rows, cols = mesh_device.shape
    singles = [_single_dev(mesh_device, rr, cc) for rr in range(rows) for cc in range(cols)]

    send_mesh = singles[s]
    recv_mesh = singles[r]

    logger.info(f"=== CASE: send {s} -> recv {r} ===")
    logger.info(f"Send Mesh Device: {send_mesh.get_device_ids()}")
    ttnn.visualize_mesh_device(send_mesh)
    logger.info(f"Receive Mesh Device: {recv_mesh.get_device_ids()}")
    ttnn.visualize_mesh_device(recv_mesh)

    run_send_recv_test(
        send_mesh,
        recv_mesh,
        socket_storage_type,
        socket_fifo_size,
        per_chip_shape,
        mem_config,
        dtype,
        layout,
    )


# ---------- generate disjoint cases ----------
# 사용자가 예로 든 상단 2행(0,1행)만 쓰는 2x2 블록
# mesh: (4, 8) 가정, 2x2의 좌상단 col 범위는 0..6
ROWS, COLS = 4, 8
SEND_POS = [(0, 0), (0, 1)]  # 요청에 나온 두 송신 시작점
CASES = []
for send_r, send_c in SEND_POS:
    for recv_c in range(0, COLS - 1):  # 2x2 시작 col
        recv_r = 0  # 예시와 동일하게 상단 행만 사용
        if (recv_r, recv_c) == (send_r, send_c):
            continue
        if _overlap_2x2(send_r, send_c, recv_r, recv_c):
            continue  # 단 한 칩이라도 겹치면 제외
        desc = f"send({send_r},{send_c}) 2x2 -> recv({recv_r},{recv_c}) 2x2 (disjoint)"
        CASES.append((send_r, send_c, recv_r, recv_c, desc))

# 사람이 보기 좋게 id 붙이기
CASE_IDS = [c[-1] for c in CASES]


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.timeout(300)
@pytest.mark.parametrize(
    "per_chip_shape",
    [
        # ([1, 1, 32, 4096]),
        ([1, 1, 64, 8192]),
    ],
)
@pytest.mark.parametrize(
    "layout",
    [
        ttnn.TILE_LAYOUT,
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.bfloat16,
        # ttnn.bfloat8_b,
    ],
)
@pytest.mark.parametrize(
    "mem_config",
    [
        ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM),
        # ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1),
    ],
)
@pytest.mark.parametrize(
    "socket_storage_type",
    [
        ttnn.BufferType.DRAM,
        # ttnn.BufferType.L1,
    ],
)
@pytest.mark.parametrize(
    "socket_fifo_size",
    [
        10 * 1024,
    ],
)
@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_DYNAMIC}], indirect=True)
@pytest.mark.parametrize("send_r, send_c, recv_r, recv_c, desc", CASES, ids=CASE_IDS)
def test_send2x2_recv2x2_disjoint_only(
    mesh_device,
    per_chip_shape,
    layout,
    dtype,
    mem_config,
    socket_storage_type,
    socket_fifo_size,
    send_r,
    send_c,
    recv_r,
    recv_c,
    desc,
):
    """
    요구사항: send/recv 2x2 서브메시가 단 하나의 디바이스도 공유하지 않도록.
    상단 2행(0,1행)을 커버하는 2x2 블록만 대상으로, (0,0)과 (0,1) 두 송신 시작점에서
    가능한 모든 비겹침 수신 위치를 자동 생성해 테스트한다.
    """
    # sanity: 좌표 기준으로도 겹치지 않음 확인
    assert not _overlap_2x2(send_r, send_c, recv_r, recv_c)

    send_mesh = _submesh(mesh_device, send_r, send_c, 2, 2)
    recv_mesh = _submesh(mesh_device, recv_r, recv_c, 2, 2)

    # 안전하게 장치 ID 기준으로도 교집합 없는지 체크 (겹치면 테스트 실패)
    send_ids = set(send_mesh.get_device_ids())
    recv_ids = set(recv_mesh.get_device_ids())
    assert send_ids.isdisjoint(recv_ids), "send/recv submesh share at least one device!"

    logger.info(f"=== CASE: {desc} ===")
    logger.info(f"Send IDs: {sorted(send_ids)}")
    ttnn.visualize_mesh_device(send_mesh)
    logger.info(f"Recv IDs: {sorted(recv_ids)}")
    ttnn.visualize_mesh_device(recv_mesh)

    run_send_recv_test(
        send_mesh,
        recv_mesh,
        socket_storage_type,
        socket_fifo_size,
        per_chip_shape,
        mem_config,
        dtype,
        layout,
    )


def _submesh_1x2(mesh_device, start_c):
    # 1x2 연속 칩 블록 (행은 0 고정)
    return mesh_device.create_submesh(
        ttnn.MeshShape(1, 2),
        ttnn.MeshCoordinate(0, start_c),
    )


def _gen_disjoint_pairs_1xN(num_cols):
    # (0,1) ... (N-2, N-1) 모든 1x2 윈도우 쌍 중 겹치지 않는 조합만 생성
    pairs = []
    for i in range(0, num_cols - 1):  # send 시작 col
        for j in range(0, num_cols - 1):  # recv 시작 col
            if abs(j - i) >= 2:  # 겹치지 않음
                desc = f"send({i},{i+1})->recv({j},{j+1})"
                pairs.append((i, j, desc))
    return pairs


CASES_1x8 = _gen_disjoint_pairs_1xN(8)
CASE_IDS_1x8 = [c[2] for c in CASES_1x8]


@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_DYNAMIC}], indirect=True)
@pytest.mark.parametrize("per_chip_shape", [([1, 1, 64, 8192])])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("mem_config", [ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM)])
@pytest.mark.parametrize("socket_storage_type", [ttnn.BufferType.DRAM])
@pytest.mark.parametrize("socket_fifo_size", [10 * 1024])
@pytest.mark.parametrize("send_c, recv_c, _desc", CASES_1x8, ids=CASE_IDS_1x8)
def test_send_recv_1x2_disjoint_pairs_on_1x8(
    mesh_device,
    per_chip_shape,
    layout,
    dtype,
    mem_config,
    socket_storage_type,
    socket_fifo_size,
    send_c,
    recv_c,
    _desc,
):
    ttnn.visualize_mesh_device(mesh_device)
    """
    (1x8)에서 1x2 윈도우 (i,i+1) → (j,j+1) 모든 비겹침 케이스 테스트.
    예)
      (0,1) → (2,3),(3,4),...,(6,7)
      (1,2) → (3,4),...,(6,7)
      ...
      (6,7) → (0,1),...,(4,5)
    """
    send_mesh = _submesh_1x2(mesh_device, send_c)
    recv_mesh = _submesh_1x2(mesh_device, recv_c)

    # 안전장치: 실제 디바이스 ID도 교집합 없음 확인
    send_ids = set(send_mesh.get_device_ids())
    recv_ids = set(recv_mesh.get_device_ids())
    assert send_ids.isdisjoint(recv_ids), f"overlap detected: {send_ids & recv_ids}"

    logger.info(f"=== CASE: {_desc} ===")
    logger.info(f"Send IDs: {sorted(send_ids)}")
    ttnn.visualize_mesh_device(send_mesh)
    logger.info(f"Recv IDs: {sorted(recv_ids)}")
    ttnn.visualize_mesh_device(recv_mesh)

    run_send_recv_test(
        send_mesh,
        recv_mesh,
        socket_storage_type,
        socket_fifo_size,
        per_chip_shape,
        mem_config,
        dtype,
        layout,
    )


# 1x32 전용 케이스 생성
CASES_1x32 = _gen_disjoint_pairs_1xN(32)
CASE_IDS_1x32 = [c[2] for c in CASES_1x32]


@pytest.mark.parametrize("mesh_device", [(1, 32)], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_DYNAMIC}], indirect=True)
@pytest.mark.parametrize("per_chip_shape", [([1, 1, 64, 8192])])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("mem_config", [ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM)])
@pytest.mark.parametrize("socket_storage_type", [ttnn.BufferType.DRAM])
@pytest.mark.parametrize("socket_fifo_size", [10 * 1024])
@pytest.mark.parametrize("send_c, recv_c, _desc", CASES_1x32, ids=CASE_IDS_1x32)
def test_send_recv_1x2_disjoint_pairs_on_1x32(
    mesh_device,
    per_chip_shape,
    layout,
    dtype,
    mem_config,
    socket_storage_type,
    socket_fifo_size,
    send_c,
    recv_c,
    _desc,
):
    """
    (1x32)에서 1x2 윈도우 (i,i+1) → (j,j+1) 모든 비겹침 케이스 테스트.
    """
    ttnn.visualize_mesh_device(mesh_device)

    send_mesh = _submesh_1x2(mesh_device, send_c)
    recv_mesh = _submesh_1x2(mesh_device, recv_c)

    # 디바이스 ID 겹침 검사
    send_ids = set(send_mesh.get_device_ids())
    recv_ids = set(recv_mesh.get_device_ids())
    assert send_ids.isdisjoint(recv_ids), f"overlap detected: {send_ids & recv_ids}"

    logger.info(f"=== CASE: {_desc} ===")
    logger.info(f"Send IDs: {sorted(send_ids)}")
    ttnn.visualize_mesh_device(send_mesh)
    logger.info(f"Recv IDs: {sorted(recv_ids)}")
    ttnn.visualize_mesh_device(recv_mesh)

    run_send_recv_test(
        send_mesh,
        recv_mesh,
        socket_storage_type,
        socket_fifo_size,
        per_chip_shape,
        mem_config,
        dtype,
        layout,
    )
