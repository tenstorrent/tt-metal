# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from conftest import is_galaxy


def pytest_addoption(parser):
    parser.addoption(
        "--start-from",
        action="store",
        default=0,
        help="Start from prompt number (0-4999)",
    )
    parser.addoption(
        "--num-prompts",
        action="store",
        default=5000,
        help="Number of prompts to process (default: 5000)",
    )
    parser.addoption(
        "--reset-bool",
        action="store",
        type=int,
        default=1,
        help="Whether to reset periodically (1 or 0), default: 1",
    )
    parser.addoption(
        "--reset-period",
        action="store",
        default=200,
        type=int,
        help="How often to reset (default: 200 (images))",
    )
    parser.addoption(
        "--loop-iter-num",
        action="store",
        default=10,
        help="Number of iterations of denoising loop (default: 10)",
    )
    parser.addoption(
        "--debug-mode",
        action="store_true",
        default=False,
        help="Run SDXL in debug mode (default: False)",
    )


@pytest.fixture
def evaluation_range(request):
    start_from = request.config.getoption("--start-from")
    num_prompts = request.config.getoption("--num-prompts")
    if start_from is not None:
        start_from = int(start_from)
    else:
        start_from = 0
    if num_prompts is not None:
        num_prompts = int(num_prompts)
    else:
        num_prompts = 5000
    return start_from, num_prompts


@pytest.fixture
def reset_config(request):
    reset_bool_val = request.config.getoption("--reset-bool")
    reset_period = request.config.getoption("--reset-period")
    if reset_bool_val is not None:
        reset_bool = bool(reset_bool_val)
    else:
        reset_bool = True
    if reset_period is not None:
        reset_period = int(reset_period)
    else:
        reset_period = 200
    return reset_bool, reset_period


def get_device_name():
    import ttnn

    num_devices = ttnn.GetNumAvailableDevices()
    if is_galaxy():
        return "galaxy"
    elif num_devices == 0:
        return "cpu"
    elif num_devices == 1:
        return "n150"
    elif num_devices == 2:
        return "n300"
    elif num_devices == 8:
        return "t3k"


@pytest.fixture
def loop_iter_num(request):
    return int(request.config.getoption("--loop-iter-num"))


@pytest.fixture
def debug_mode(request):
    return request.config.getoption("--debug-mode")


@pytest.fixture(scope="function")
def validate_fabric_compatibility(request):
    """
    Validate that fabric configuration is compatible with the requested mesh device configuration.
    This fixture runs before mesh_device creation to catch incompatibilities early.
    It is needed to be able to gracefully fail if the configuration is not possible.
    """
    import ttnn

    params = getattr(request.node, "callspec", {}).params
    use_cfg_parallel = params.get("use_cfg_parallel", None)
    mesh_device_param = params.get("mesh_device", None)

    if not use_cfg_parallel:
        return

    if mesh_device_param is not None:
        total_devices = ttnn.GetNumAvailableDevices()

        if isinstance(mesh_device_param, int):
            requested_devices = mesh_device_param
        elif isinstance(mesh_device_param, tuple):
            requested_devices = mesh_device_param[0] * mesh_device_param[1]
        else:
            requested_devices = total_devices

        assert requested_devices == total_devices, "Requested devices must be equal to total devices"
