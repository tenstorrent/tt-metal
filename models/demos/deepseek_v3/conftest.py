# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0
import json
import os
from pathlib import Path

import pytest
import safetensors.torch
import torch
from loguru import logger
from transformers import AutoConfig

import ttnn
from models.demos.deepseek_v3.scripts.generate_test_inputs_outputs import __file__ as REFERENCE_IO_SCRIPT_NAME
from tests.scripts.common import get_updated_device_params


@pytest.fixture(scope="function")
def mesh_device(request, device_params):
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


@pytest.fixture(scope="session")
def model_path():
    return Path(os.getenv("HF_MODEL", "models/demos/deepseek_v3/reference"))


@pytest.fixture
def reference_io(request):
    param = getattr(request, "param", None)
    if (
        param is None
        or not isinstance(param, tuple)
        or len(param) != 2
        or param[0] not in ["prefill", "decode"]
        or not isinstance(param[1], str)
    ):
        raise ValueError(
            "Reference IO fixture requires a mode ('prefill', 'decode') and a module path to load the reference IO for."
        )
    mode, module = param
    path = (
        Path(os.getenv("DEEPSEEK_V3_CACHE", "/proj_sw/user_dev/deepseek-v3-cache"))
        / f"test_io_cache/{mode}.{module}.pt"
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"Reference IO cache file not found at {path}. Please run the {REFERENCE_IO_SCRIPT_NAME} script to create it. Did you set the 'HF_MODEL' environment variable coorectly?"
        )
    return torch.load(path)


@pytest.fixture(scope="session")
def hf_config(model_path):
    """Load DeepSeek config for testing"""
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    return config


@pytest.fixture
def state_dict(request, model_path):
    param = getattr(request, "param", None)
    if param is None or not isinstance(param, str):
        raise ValueError(
            "State dict fixture requires either a module path to load the weights for (empty string to load the entire state dict)."
        )

    if param:
        param += "."  # So that the later matches include the separating dot

    weight_paths = json.load(open(model_path / "model.safetensors.index.json", "r"))["weight_map"]
    per_safetensor_weights = {}

    for weight_name in weight_paths.keys():
        if not weight_name.startswith(param):
            continue
        per_safetensor_weights.setdefault(weight_paths[weight_name], []).append(weight_name)

    return {
        weight_name[len(param) :]: safetensor_state_dict[weight_name]
        for safetensor_file_path, weight_names in per_safetensor_weights.items()
        for safetensor_state_dict in [safetensors.torch.load_file(model_path / safetensor_file_path)]
        for weight_name in weight_names
    }


@pytest.fixture
def mesh_row(mesh_device):
    """
    DeepSeek runs many modules on a single 8-device row of a Galaxy system.
    This can be emulated on a T3K or by selecting a single submesh of a TG.
    For Galaxy+ systems (32+ devices), creates a submesh with shape (1, 8)
    and returns the first row. Otherwise, returns the original mesh_device.
    """
    if ttnn.get_num_devices() >= 32:
        rows = mesh_device.create_submeshes(ttnn.MeshShape(1, 8))
        yield rows[0]
    else:
        yield mesh_device
