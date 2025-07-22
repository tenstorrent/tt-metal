# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0
import os
import tempfile
from pathlib import Path

import pytest
from loguru import logger
from transformers import AutoConfig

import ttnn
from tests.scripts.common import get_updated_device_params


@pytest.fixture(scope="function")
def galaxy_or_t3k_mesh(request, device_params):
    """
    Pytest fixture to set up a device mesh for Deepseek tests.
    Many Deepseek submodules operate on a single row of devices,
    so we are happy to run those on a TG or T3K. Others need
    the full Galaxy mesh in (rows=4, cols=8) format.

    If a galaxy is available, it returns a mesh of 4x8 devices.
    If a t3k is available, it returns a mesh of 1x8 devices.
    If no galaxy or t3k is available, it returns a mesh of 1x1 devices.

    Yields:
        mesh_device: Initialized device mesh object.
    """
    import ttnn

    device_ids = ttnn.get_device_ids()

    if len(device_ids) == 32:
        mesh_shape = ttnn.MeshShape(4, 8)
    elif len(device_ids) == 8:
        mesh_shape = ttnn.MeshShape(1, 8)
    else:
        mesh_shape = ttnn.MeshShape(1, 1)
        request.node.pci_ids = [ttnn.GetPCIeDeviceID(i) for i in device_ids[:1]]

    updated_device_params = get_updated_device_params(device_params)
    fabric_config = updated_device_params.pop("fabric_config", None)
    reliability_mode = updated_device_params.pop("reliability_mode", None)
    if fabric_config:
        ttnn.set_fabric_config(fabric_config)
    mesh_device = ttnn.open_mesh_device(mesh_shape=mesh_shape, **updated_device_params)

    logger.debug(f"multidevice with {mesh_device.get_num_devices()} devices is created with shape {mesh_device.shape}")
    yield mesh_device

    for submesh in mesh_device.get_submeshes():
        ttnn.close_mesh_device(submesh)

    ttnn.close_mesh_device(mesh_device)
    if fabric_config:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    del mesh_device


@pytest.fixture
def mesh_row(galaxy_or_t3k_mesh):
    """
    DeepSeek runs many modules on a single 8-device row of a Galaxy system.
    This can be emulated on a T3K or by selecting a single submesh of a TG.

    For Galaxy+ systems (32+ devices), creates a submesh with shape (1, 8)
    and returns the first row. Otherwise, returns the original mesh_device.
    """
    if ttnn.get_num_devices() >= 32:
        rows = galaxy_or_t3k_mesh.create_submeshes(ttnn.MeshShape(1, 8))
        yield rows[0]
    else:
        yield galaxy_or_t3k_mesh


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def hf_config():
    """Load DeepSeek config for testing."""
    path = os.getenv("HF_MODEL", "/proj_sw/user_dev/deepseek-ai/DeepSeek-R1-0528")
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    return config


@pytest.fixture
def hf_config_single_layer(hf_config):
    """Load DeepSeek config with a single layerfor testing."""
    hf_config.num_hidden_layers = 1
    return hf_config
