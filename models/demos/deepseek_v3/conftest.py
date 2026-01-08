# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0
import os
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from loguru import logger
from transformers import AutoConfig

import ttnn
from models.demos.deepseek_v3.tt.ccl import CCL
from models.demos.deepseek_v3.utils.test_utils import load_state_dict, system_name_to_mesh_shape
from tests.scripts.common import get_updated_device_params

RESET_WEIGHT_CACHE_OPTION = "--recalculate-weights"

# Shared test parametrization constants
# Prefill sequence lengths: powers of 2 from 128 to 128K
PREFILL_SEQ_LENS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]


def pytest_addoption(parser):
    parser.addoption(
        RESET_WEIGHT_CACHE_OPTION,
        action="store_true",
        help="Reset weight configs for tests",
    )


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

    request.node.pci_ids = ttnn.get_pcie_device_ids()

    # Override mesh shape based on MESH_DEVICE environment variable
    requested_system_name = os.getenv("MESH_DEVICE")
    if requested_system_name is None:
        raise ValueError("Environment variable $MESH_DEVICE is not set. Please set it to T3K, DUAL, QUAD, or TG.")

    mesh_shape = system_name_to_mesh_shape(requested_system_name.upper())
    logger.info(f"Selected MESH_DEVICE: '{requested_system_name}' - mesh shape will be set to: {mesh_shape}")

    updated_device_params = get_updated_device_params(device_params)

    fabric_config = updated_device_params.pop("fabric_config", None)
    if fabric_config:
        ttnn.set_fabric_config(fabric_config)

    updated_device_params.setdefault("mesh_shape", mesh_shape)
    mesh_device = ttnn.open_mesh_device(**updated_device_params)

    logger.debug(f"Mesh device with {mesh_device.get_num_devices()} devices is created with shape {mesh_device.shape}")
    yield mesh_device

    for submesh in mesh_device.get_submeshes():
        ttnn.close_mesh_device(submesh)

    ttnn.close_mesh_device(mesh_device)
    if fabric_config:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

    del mesh_device


@pytest.fixture(scope="session")
def model_path():
    return Path(os.getenv("DEEPSEEK_V3_HF_MODEL", "models/demos/deepseek_v3/reference"))


@pytest.fixture(scope="session")
def hf_config(model_path):
    """Load DeepSeek config for testing"""
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    return config


@pytest.fixture(scope="session")
def state_dict(model_path):
    yield load_state_dict(model_path, "")


@pytest.fixture(scope="function", autouse=True)
def clear_state_dict_cache(state_dict):
    """
    Clear the LazyStateDict cache after each test to prevent memory accumulation.
    This preserves file handles (mmap benefits) while freeing tensor memory.
    """
    yield
    state_dict.clear_cache()


@pytest.fixture(scope="session")
def hf_config_short(request, hf_config):
    hf_config_out = deepcopy(hf_config)
    hf_config_out.num_hidden_layers = getattr(request, "param", 1)
    hf_config_out.max_seq_len = 3 * 1024
    return hf_config_out


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


@pytest.fixture
def ccl(mesh_device):
    """
    Fixture to create a CCL instance for testing.
    This is used to test distributed operations in DeepSeek modules.
    """
    return CCL(mesh_device)


@pytest.fixture(scope="function")
def set_deterministic_env():
    """
    Fixture to set seeds and enable deterministic algorithms for DeepSeek tests.
    This ensures reproducible results across test runs.
    """
    torch.manual_seed(5)
    torch.use_deterministic_algorithms(True)


@pytest.fixture(scope="session")
def force_recalculate_weight_config(request):
    """
    Fixture to control whether weight configuration files should be recalculated.
    """
    return request.config.getoption(RESET_WEIGHT_CACHE_OPTION)


@pytest.fixture(scope="session")
def cache_path():
    try:
        default_cache = f"/localdev/{os.getlogin()}/deepseek-v3-cache"
    except OSError:
        default_cache = "/proj_sw/user_dev/deepseek-v3-cache"
    return Path(os.getenv("DEEPSEEK_V3_CACHE", default_cache))
