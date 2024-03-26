# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import random
import os
import numpy as np
from functools import partial
from itertools import chain
from operator import contains, eq, getitem
from pathlib import Path
import json

from loguru import logger

from tests.scripts.common import run_process_and_get_result


@pytest.fixture(scope="function")
def reset_seeds():
    torch.manual_seed(213919)
    np.random.seed(213919)
    random.seed(213919)

    yield


@pytest.fixture(scope="function")
def function_level_defaults(reset_seeds):
    yield


@pytest.fixture(scope="session")
def model_location_generator():
    def model_location_generator_(model_version, model_subdir=""):
        model_folder = Path("tt_dnn-models") / model_subdir
        internal_weka_path = Path("/mnt/MLPerf") / model_folder / model_version
        has_internal_weka = internal_weka_path.exists()
        internal_cache_path = Path("/opt/tt-metal-models") / model_folder / model_version
        has_internal_cache = internal_cache_path.exists()
        if has_internal_weka:
            return internal_weka_path
        elif has_internal_cache:
            return internal_cache_path
        else:
            return model_version

    return model_location_generator_


@pytest.fixture(scope="session")
def get_tt_cache_path():
    def get_tt_cache_path_(model_version, model_subdir="", default_dir=""):
        model_folder = Path("tt_dnn-models/tt") / model_subdir
        internal_weka_path = Path("/mnt/MLPerf") / model_folder / model_version
        has_internal_weka = internal_weka_path.exists()
        internal_cache_path = Path("/opt/tt-metal-models") / model_folder / model_version
        has_internal_cache = internal_cache_path.exists()
        if has_internal_weka:
            return internal_weka_path
        elif has_internal_cache:
            return internal_cache_path
        else:
            default_path = Path(default_dir) / model_folder / model_version
            default_path.mkdir(parents=True, exist_ok=True)
            return default_path

    return get_tt_cache_path_


ALL_ARCHS = set(
    [
        "grayskull",
        "wormhole_b0",
    ]
)


def pytest_addoption(parser):
    parser.addoption(
        "--tt-arch",
        choices=[*ALL_ARCHS],
        default=os.environ.get("ARCH_NAME", "grayskull"),
        help="Target arch, ex. grayskull, wormhole_b0",
    )
    parser.addoption(
        "--device-id",
        type=int,
        default=0,
        help="Target device id",
    )
    parser.addoption(
        "--input-method",
        action="store",
        choices=["json", "cli"],
        default=None,
        help="Choose input method: 1) json or 2) cli",
    )
    parser.addoption(
        "--input-path",
        action="store",
        default="",
        help="Path to json file with inputs",
    )
    parser.addoption("--cli-input", action="store", default=None, help="Enter prompt if --input-method=cli")


def pytest_generate_tests(metafunc):
    """
    This is not a standard docstring.

    We will explain the non-standard fixtures that pytest_generate_tests is
    creating here.

    silicon_arch_name and silicon_arch_<ARCH_NAME>
    ----------------------------------------------

    This is how tests should be requesting accelerator architecture names.
    Tests which aim to run on silicon should request a silicon_arch_name
    fixture. Just that single fixture will parametrize the test to run on the
    provided architecture name from the command line through the --tt-arch
    option. The value of the fixture will be the string value of the
    architecture name. For example,

    @pytest.mark.post_commit
    def test_model_silicon(silicon_arch_name):
        # silicon_arch_name will be one of grayskull, wormhole_b0 etc.
        run_model_on_silicon(silicon_arch_name)
        ...

    If you want to restrict a test to only a specific architecture, you can
    provide an additional fixture in the form of silicon_arch_<ARCH_NAME>. This
    will limit the range of possible values for silicon_arch_name to only be
    ARCH_NAME.

    @pytest.mark.post_commit
    def test_model_silicon_grayskull_only(
        silicon_arch_name,
        silicon_arch_grayskull,
    ):
        # silicon_arch_name can only be grayskull or empty
        run_model_on_silicon(silicon_arch_name)
        ...

    If --tt-arch specifies an architecture that's not ARCH_NAME, the test will
    be skipped. We ensure skipping by providing an empty list parametrization
    for silicon_arch_name, and with the empty_parameter_set_mark config option
    for pytest, will skip any tests with an empty list parametrization.

    Note that you must provide silicon_arch_name as a fixture if you want to
    use the silicon_arch_<ARCH_NAME> fixture.

    Note that if tests want to use the ARCH value from the API, tests should
    create their own separate fixture which will convert the string value
    provided from silicon_arch_name into ARCH. We keep it as strings here
    because these fixtures will be used in tests which do not have access to
    any Python APIs.
    """

    tt_arch = metafunc.config.getoption("--tt-arch")

    silicon_arch_specific_fixture_name_to_avail_archs = {
        "silicon_arch_grayskull": set(
            [
                "grayskull",
            ]
        ),
        "silicon_arch_wormhole_b0": set(
            [
                "wormhole_b0",
            ]
        ),
    }

    check_uses_silicon_arch_specific_fixture = partial(contains, silicon_arch_specific_fixture_name_to_avail_archs)
    test_requested_silicon_arch_fixtures = tuple(
        filter(check_uses_silicon_arch_specific_fixture, metafunc.fixturenames)
    )
    is_test_requesting_specific_silicon_archs = len(test_requested_silicon_arch_fixtures) > 0
    get_archs_for_silicon_arch_specific_fixture = partial(getitem, silicon_arch_specific_fixture_name_to_avail_archs)
    test_requested_silicon_archs = ALL_ARCHS.intersection(
        *map(
            get_archs_for_silicon_arch_specific_fixture,
            test_requested_silicon_arch_fixtures,
        )
    )

    available_archs = test_requested_silicon_archs if is_test_requesting_specific_silicon_archs else ALL_ARCHS
    matches_user_requested_silicon_arch = partial(eq, tt_arch)
    available_archs = tuple(filter(matches_user_requested_silicon_arch, available_archs))

    uses_silicon_arch = "silicon_arch_name" in metafunc.fixturenames

    # sanity
    if is_test_requesting_specific_silicon_archs and not uses_silicon_arch:
        raise Exception(
            f"{metafunc.function} requesting a specific silicon target, but doesn't use silicon_arch_name fixture"
        )

    if uses_silicon_arch:
        metafunc.parametrize("silicon_arch_name", available_archs)
        for test_requested_silicon_arch_fixture in test_requested_silicon_arch_fixtures:
            # The values of these arch-specific fixtures should not be used in
            # the test function, so use any parameters, like [True]
            metafunc.parametrize(test_requested_silicon_arch_fixture, [True])

    input_method = metafunc.config.getoption("--input-method")
    if input_method == "json":
        json_path = metafunc.config.getoption("--input-path")
        if not json_path:
            raise ValueError("Please provide a valid JSON path using --input-path option.")
        with open(json_path, "r") as f:
            data = json.load(f)
        metafunc.parametrize("user_input", [data])
    elif input_method == "cli":
        cli_input = metafunc.config.getoption("--cli-input")
        if not cli_input:
            raise ValueError("Please provide input using --cli-input option.")
        metafunc.parametrize("user_input", [[cli_input]])


# Report stashing to get outcomes etc
phase_report_key = pytest.StashKey()


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    # execute all other hooks to obtain the report object
    outcome = yield
    rep = outcome.get_result()

    # store test results for each phase of a call, which can
    # be "setup", "call", "teardown"
    item.stash.setdefault(phase_report_key, {})[rep.when] = rep


@pytest.fixture(scope="function")
def reset_tensix(request, silicon_arch_name):
    yield

    report = request.node.stash[phase_report_key]

    test_failed = ("call" not in report) or report["call"].failed

    if test_failed:
        logger.debug("Test failed - resetting with smi")
        if silicon_arch_name == "grayskull":
            result = run_process_and_get_result("tt-smi -tr all")
        elif silicon_arch_name == "wormhole_b0":
            result = run_process_and_get_result("tt-smi -wr all")
        else:
            raise Exception(f"Unrecognized arch for tensix-reset: {silicon_arch_name}")
        assert result.returncode == 0, "Tensix reset script raised error"


@pytest.fixture(scope="function")
def device_init_destroy(request):
    import tt_lib as ttl

    device_id = request.config.getoption("device_id")

    num_devices = ttl.device.GetNumPCIeDevices()
    assert device_id < num_devices, "CreateDevice not supported for non-mmio device"

    device = ttl.device.CreateDevice(device_id)
    ttl.device.SetDefaultDevice(device)

    yield device

    ttl.device.Synchronize(device)
    ttl.device.CloseDevice(device)


@pytest.fixture(scope="function")
def device(device_init_destroy):
    import tt_lib as ttl

    device = ttl.device.GetDefaultDevice()
    yield device
    ttl.device.DumpDeviceProfiler(device, True)
    ttl.device.DeallocateBuffers(device)


@pytest.fixture(scope="function")
def pcie_devices(request):
    import tt_lib as ttl

    num_devices = ttl.device.GetNumPCIeDevices()

    # Get only physical devices
    devices = ttl.device.CreateDevices([i for i in range(num_devices)])

    yield [devices[i] for i in range(num_devices)]

    for device in devices.values():
        ttl.device.DumpDeviceProfiler(device, True)
        ttl.device.DeallocateBuffers(device)

    ttl.device.CloseDevices(devices)


@pytest.fixture(scope="function")
def all_devices(request):
    import tt_lib as ttl

    num_devices = ttl.device.GetNumAvailableDevices()

    # Get only physical devices
    devices = ttl.device.CreateDevices([i for i in range(num_devices)])

    yield [devices[i] for i in range(num_devices)]

    for device in devices.values():
        ttl.device.DumpDeviceProfiler(device, True)
        ttl.device.DeallocateBuffers(device)

    ttl.device.CloseDevices(devices)


@pytest.fixture(scope="function")
def device_mesh(request, silicon_arch_name, silicon_arch_wormhole_b0):
    import ttnn

    device_ids = ttnn.get_device_ids()
    try:
        num_devices_requested = min(request.param, len(device_ids))
    except (ValueError, AttributeError):
        num_devices_requested = len(device_ids)

    if num_devices_requested <= 1:
        pytest.skip("Requires multiple devices to run")
    device_mesh = ttnn.open_device_mesh(ttnn.DeviceGrid(1, len(device_ids)), device_ids)

    device_mesh = ttnn.open_device_mesh(ttnn.DeviceGrid(1, num_devices_requested), device_ids[:num_devices_requested])

    logger.info(f"multidevice with {device_mesh.get_num_devices()} devices is created")
    yield device_mesh

    ttnn.close_device_mesh(device_mesh)
    del device_mesh


@pytest.fixture(scope="function")
def pcie_device_mesh(request, silicon_arch_name, silicon_arch_wormhole_b0):
    import ttnn

    device_ids = ttnn.get_pcie_device_ids()
    try:
        num_pcie_devices_requested = min(request.param, len(device_ids))
    except (ValueError, AttributeError):
        num_pcie_devices_requested = len(device_ids)

    if num_pcie_devices_requested <= 1:
        pytest.skip("Requires multiple devices to run")

    device_mesh = ttnn.open_device_mesh(
        ttnn.DeviceGrid(1, num_pcie_devices_requested), device_ids[:num_pcie_devices_requested]
    )

    logger.info(f"multidevice with {device_mesh.get_num_devices()} devices is created")
    yield device_mesh

    ttnn.close_device_mesh(device_mesh)
    del device_mesh


@pytest.fixture()
def clear_compile_cache():
    yield
    import tt_lib as ttl

    ttl.device.DisablePersistentKernelCache()


@pytest.fixture(autouse=True)
def reset_default_device():
    import tt_lib as ttl

    device = ttl.device.GetDefaultDevice()
    yield
    ttl.device.SetDefaultDevice(device)


@pytest.fixture(scope="function")
def use_program_cache(request):
    import tt_lib as ttl

    if "device" in request.fixturenames:
        dev = ttl.device.GetDefaultDevice()
        dev.enable_program_cache()
    elif "all_devices" in request.fixturenames:
        devices = request.getfixturevalue("all_devices")
        for dev in devices:
            dev.enable_program_cache()
    yield


@pytest.fixture(scope="function")
def tracy_profile():
    from tracy import Profiler

    profiler = Profiler()

    profiler.enable()
    yield
    profiler.disable()


@pytest.fixture
def input_path(request):
    return request.config.getoption("--input-path")
