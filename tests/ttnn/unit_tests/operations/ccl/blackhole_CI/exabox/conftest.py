# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch
from loguru import logger

import ttnn
from tests.scripts.common import get_updated_device_params


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_device(device_types): mark test to run only on specified device types. "
        "device_types is a list of strings, e.g. ['DUAL_BH', 'QUAD_BH']. "
        "Checked against the MESH_DEVICE environment variable.",
    )


def pytest_collection_modifyitems(config, items):
    mesh_device_env = os.getenv("MESH_DEVICE", "").upper()
    if not mesh_device_env:
        return

    for item in items:
        marker = item.get_closest_marker("requires_device")
        if marker:
            device_types = marker.args[0] if marker.args else []
            if isinstance(device_types, str):
                device_types = [device_types]
            if mesh_device_env not in device_types:
                item.add_marker(
                    pytest.mark.skip(
                        reason=f"Test requires device type(s) {device_types}, but MESH_DEVICE={mesh_device_env}"
                    )
                )


@pytest.fixture(scope="function")
def mesh_device(request, device_params):
    """Exabox mesh device fixture for multi-galaxy configurations (DUAL_BH / QUAD_BH).

    Takes mesh shape from request.param (e.g. (16, 4) or (32, 4)) and handles
    fabric config, dispatch core axis, and torch thread restoration for MPI envs.
    """
    request.node.pci_ids = ttnn.get_pcie_device_ids()

    try:
        param = request.param
    except (ValueError, AttributeError):
        param = ttnn._ttnn.multi_device.SystemMeshDescriptor().shape().mesh_size()

    if isinstance(param, tuple):
        assert len(param) == 2
        num_devices_requested = param[0] * param[1]
        if not ttnn.using_distributed_env() and num_devices_requested > ttnn.get_num_devices():
            pytest.skip("Requested more devices than available. Test not applicable for machine")
        mesh_shape = ttnn.MeshShape(*param)
    else:
        if not ttnn.using_distributed_env() and param > ttnn.get_num_devices():
            pytest.skip("Requested more devices than available. Test not applicable for machine")
        mesh_shape = ttnn.MeshShape(1, param)

    device_params.setdefault("dispatch_core_axis", ttnn.DispatchCoreAxis.COL)
    updated_device_params = get_updated_device_params(device_params)

    fabric_config = updated_device_params.pop("fabric_config", None)
    if fabric_config:
        ttnn.set_fabric_config(fabric_config)

    mesh_device = ttnn.open_mesh_device(mesh_shape=mesh_shape, **updated_device_params)

    # MPI_Init_thread sets OpenMP threads to 1; restore for torch reference computations
    if ttnn.using_distributed_env():
        num_torch_threads = max(1, os.cpu_count())
        logger.info(f"Restoring torch num_threads to {num_torch_threads}")
        torch.set_num_threads(num_torch_threads)

    logger.debug(f"Exabox mesh device with {mesh_device.get_num_devices()} devices created, shape {mesh_device.shape}")
    yield mesh_device

    try:
        for submesh in mesh_device.get_submeshes():
            ttnn.close_mesh_device(submesh)
        ttnn.close_mesh_device(mesh_device)
    finally:
        if fabric_config:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        del mesh_device
