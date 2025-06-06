from math import prod
from time import sleep


import pytest
import torch
from loguru import logger

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from tests.scripts.common import get_updated_device_params, reset_fabric, set_fabric


@pytest.fixture(scope="function")
def mesh_device_with_subdevices(request, silicon_arch_name, device_params):
    """
    Based off of conftest.mesh_device

    This version is needed because submeshes need to be manually closed and `del`d in the same scope is the parent mesh
    prior to fabric reset, otherwise we get hangs.
    """
    device_ids = ttnn.get_device_ids()

    param = request.param

    grid_dims, offset0, offset1 = param
    assert len(grid_dims) == 2, "Device mesh grid shape should have exactly two elements."
    num_devices_requested = grid_dims[0] * grid_dims[1]
    if num_devices_requested > len(device_ids):
        pytest.skip("Requested more devices than available. Test not applicable for machine")
    mesh_shape = ttnn.MeshShape(*grid_dims)
    assert num_devices_requested <= len(device_ids), "Requested more devices than available."

    request.node.pci_ids = [ttnn.GetPCIeDeviceID(i) for i in device_ids[:num_devices_requested]]

    updated_device_params = get_updated_device_params(device_params)
    fabric_config = updated_device_params.pop("fabric_config", None)

    set_fabric(fabric_config)
    mesh_device = ttnn.open_mesh_device(mesh_shape=mesh_shape, **updated_device_params)

    logger.debug(f"multidevice with {mesh_device.get_num_devices()} devices is created")

    coord0 = ttnn.MeshCoordinate(offset0)
    coord1 = ttnn.MeshCoordinate(offset1)

    submesh0 = mesh_device.create_submesh(ttnn.MeshShape(1, 1), offset=coord0)
    submesh1 = mesh_device.create_submesh(ttnn.MeshShape(1, 1), offset=coord1)

    yield mesh_device, (coord0, submesh0), (coord1, submesh1)

    ttnn.DumpDeviceProfiler(mesh_device)

    # uncommenting these lines will cause a hang
    # ttnn.close_mesh_device(submesh0)
    # ttnn.close_mesh_device(submesh1)
    ttnn.close_mesh_device(mesh_device)

    # del submesh0
    # del submesh1
    del mesh_device

    # reset_fabric(fabric_config)


# TEST_SHAPES = [(1, 1, 1, 16),(1, 1, 2, 16), (100, 1, 1, 16),(1000, 1, 1, 16), (1, 1, 1, 17)]
TEST_SHAPES = [(1, 1, 1, 16)]


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device_with_subdevices", [((1, 2), (0, 0), (0, 1))], indirect=True)
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])  # ,ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("shape", TEST_SHAPES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_send_receive(mesh_device_with_subdevices, shape, layout, dtype):
    mesh_device, (coord0, submesh0), (coord1, submesh1) = mesh_device_with_subdevices
    input_tensor_torch = torch.linspace(1, prod(shape), prod(shape)).reshape(shape).to(dtype=dtype)

    input_tensor = ttnn.from_torch(input_tensor_torch, layout=layout).to(submesh0)

    compute_grid_size = submesh1.compute_with_storage_grid_size()
    cores = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
    )
    send_semaphore = ttnn.create_global_semaphore(submesh0, cores, 0)
    receiver_semaphore = ttnn.create_global_semaphore(submesh1, cores, 0)

    sent_tensor = ttnn.point_to_point(
        input_tensor,
        coord0,
        coord1,
        submesh1,
        ttnn.Topology.Linear,
        mesh_device,
        receiver_semaphore,
    )
    assert_with_pcc(ttnn.to_torch(sent_tensor), input_tensor_torch)

    return_tensor = ttnn.point_to_point(
        sent_tensor, coord1, coord0, submesh0, ttnn.Topology.Linear, mesh_device, send_semaphore
    )

    # print(input_tensor_torch)
    # print(ttnn.to_torch(return_tensor))

    assert_with_pcc(ttnn.to_torch(return_tensor), input_tensor_torch)
