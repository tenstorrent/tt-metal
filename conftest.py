# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import random
import os
import numpy as np
from functools import partial
from operator import contains, eq, getitem
from pathlib import Path
import json
import multiprocess
import signal
import time
import psutil
from datetime import datetime

from loguru import logger

from tests.scripts.common import run_process_and_get_result
from tests.scripts.common import get_updated_device_params

# Constants for device configurations
GALAXY_NUM_DEVICES = 32
TG_NUM_PCIE_DEVICES = 4
SIX_U_NUM_PCIE_DEVICES = 32


@pytest.fixture(scope="function")
def reset_seeds():
    torch.manual_seed(213919)
    np.random.seed(213919)
    random.seed(213919)

    yield


@pytest.fixture(scope="function")
def function_level_defaults(reset_seeds):
    yield


@pytest.fixture(scope="function")
def is_ci_env():
    if os.getenv("CI") == "true":
        return True
    return False


@pytest.fixture(scope="function")
def is_single_card_n300(device):
    import ttnn

    num_pcie = ttnn.GetNumPCIeDevices()
    num_devices = ttnn.GetNumAvailableDevices()
    # N150 has 1 chip; N300 has 2 chips (1 pcie); T3000 has 8 chips (4 pcie)
    return num_pcie == 1 and num_devices == 2 and device.arch().name == "WORMHOLE_B0"


@pytest.fixture(scope="function")
def galaxy_type():
    if is_6u():
        return "6U"
    elif is_tg_cluster():
        return "4U"
    else:
        return None


def is_galaxy():
    import ttnn

    num_devices = ttnn.GetNumAvailableDevices()
    # Galaxy systems have 32 devices
    return num_devices == GALAXY_NUM_DEVICES


# TODO: Remove this when TG clusters are deprecated.
def is_6u():
    import ttnn

    # 6U has 32 PCIe devices
    return is_galaxy() and ttnn.GetNumPCIeDevices() == SIX_U_NUM_PCIE_DEVICES


# TODO: Remove this when TG clusters are deprecated.
def is_tg_cluster():
    import ttnn

    # TG has 4 PCIe devices
    return is_galaxy() and ttnn.GetNumPCIeDevices() == TG_NUM_PCIE_DEVICES


def first_available_tg_device():
    assert is_tg_cluster()
    # The id of the first user exposed device for a TG cluster is 4
    return 4


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
            logger.debug(f"Using internal weka path: {internal_weka_path}")
            return internal_weka_path
        elif has_internal_cache:
            logger.debug(f"Using internal cache path: {internal_cache_path}")
            return internal_cache_path
        else:
            default_path = Path(default_dir) / model_folder / model_version
            default_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Using default cache path: {default_path}")
            return default_path

    return get_tt_cache_path_


@pytest.fixture(scope="function")
def device_params(request):
    return getattr(request, "param", {})


@pytest.fixture(scope="function")
def device(request, device_params):
    import ttnn

    device_id = request.config.getoption("device_id")
    request.node.pci_ids = [ttnn.GetPCIeDeviceID(device_id)]

    # When initializing a single device on a TG system, we want to
    # target the first user exposed device, not device 0 (one of the
    # 4 gateway devices)
    if is_tg_cluster() and not device_id:
        device_id = first_available_tg_device()

    updated_device_params = get_updated_device_params(device_params)
    device = ttnn.CreateDevice(device_id=device_id, **updated_device_params)
    ttnn.SetDefaultDevice(device)

    yield device

    ttnn.DumpDeviceProfiler(device)
    ttnn.close_device(device)


@pytest.fixture(scope="function")
def pcie_devices(request, device_params):
    import ttnn

    num_devices = ttnn.GetNumPCIeDevices()
    device_ids = [i for i in range(num_devices)]
    request.node.pci_ids = device_ids

    # Get only physical devices
    updated_device_params = get_updated_device_params(device_params)
    devices = ttnn.CreateDevices(device_ids, **updated_device_params)

    yield [devices[i] for i in range(num_devices)]

    for device in devices.values():
        ttnn.DumpDeviceProfiler(device)

    ttnn.CloseDevices(devices)


@pytest.fixture(scope="function")
def all_devices(request, device_params):
    import ttnn

    num_devices = ttnn.GetNumAvailableDevices()
    device_ids = [i for i in range(num_devices)]
    request.node.pci_ids = [ttnn.GetPCIeDeviceID(i) for i in device_ids]

    # Get only physical devices
    updated_device_params = get_updated_device_params(device_params)
    devices = ttnn.CreateDevices(device_ids, **updated_device_params)

    yield [devices[i] for i in range(num_devices)]

    for device in devices.values():
        ttnn.DumpDeviceProfiler(device)

    ttnn.CloseDevices(devices)


# Reset fabric config to DISABLED if not None, and do nothing otherwise
# Temporarily require previous state to be passed in as even setting it to DISABLED might be unstable
# This is to ensure that we don't propagate the instability to the rest of CI
def reset_fabric(fabric_config):
    import ttnn

    if fabric_config:
        ttnn.initialize_fabric_config(ttnn.FabricConfig.DISABLED)


# Set fabric config to passed in value
# Do nothing if not set
# Must be called before creating the mesh device
def set_fabric(fabric_config):
    import ttnn

    # If fabric_config is not None, set it to fabric_config
    if fabric_config:
        ttnn.initialize_fabric_config(fabric_config)


@pytest.fixture(scope="function")
def mesh_device(request, silicon_arch_name, device_params):
    """
    Pytest fixture to set up a device mesh for tests.

    If `request.param` is an integer, it specifies the number of devices to use (up to available devices).
    If `request.param` is a tuple, it defines the 2D grid dimensions (rows, columns) for TG, e.g., (8, 4) creates
    a devish mesh grid of 8 rows and 4 columns, totaling 32 devices. The total number of devices should not exceed available devices.

    Args:
        request: Pytest request object.
        silicon_arch_name: Name of the silicon architecture.
        device_params: Additional device configuration parameters.

    Yields:
        mesh_device: Initialized device mesh object.
    """
    import ttnn

    device_ids = ttnn.get_device_ids()

    try:
        param = request.param
    except (ValueError, AttributeError):
        param = len(device_ids)  # Default to using all available devices

    if isinstance(param, tuple):
        grid_dims = param
        assert len(grid_dims) == 2, "Device mesh grid shape should have exactly two elements."
        num_devices_requested = grid_dims[0] * grid_dims[1]
        if num_devices_requested > len(device_ids):
            pytest.skip("Requested more devices than available. Test not applicable for machine")
        mesh_shape = ttnn.MeshShape(*grid_dims)
        assert num_devices_requested <= len(device_ids), "Requested more devices than available."
    else:
        num_devices_requested = min(param, len(device_ids))
        mesh_shape = ttnn.MeshShape(1, num_devices_requested)

    request.node.pci_ids = [ttnn.GetPCIeDeviceID(i) for i in device_ids[:num_devices_requested]]

    updated_device_params = get_updated_device_params(device_params)
    fabric_config = updated_device_params.pop("fabric_config", None)
    set_fabric(fabric_config)
    mesh_device = ttnn.open_mesh_device(mesh_shape=mesh_shape, **updated_device_params)

    logger.debug(f"multidevice with {mesh_device.get_num_devices()} devices is created")
    yield mesh_device

    ttnn.DumpDeviceProfiler(mesh_device)

    ttnn.close_mesh_device(mesh_device)
    reset_fabric(fabric_config)
    del mesh_device


@pytest.fixture(scope="function")
def pcie_mesh_device(request, silicon_arch_name, silicon_arch_wormhole_b0, device_params):
    import ttnn

    device_ids = ttnn.get_pcie_device_ids()
    try:
        num_pcie_devices_requested = min(request.param, len(device_ids))
    except (ValueError, AttributeError):
        num_pcie_devices_requested = len(device_ids)

    if num_pcie_devices_requested != 4:
        pytest.skip("Only 4 PCIe devices are supported for testing")

    request.node.pci_ids = device_ids[:num_pcie_devices_requested]

    updated_device_params = get_updated_device_params(device_params)
    fabric_config = updated_device_params.pop("fabric_config", None)
    set_fabric(fabric_config)
    mesh_device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(2, 2),
        **updated_device_params,
        offset=ttnn.MeshCoordinate(0, 1),
    )
    mesh_device.reshape(ttnn.MeshShape(1, 4))

    logger.debug(f"multidevice with {mesh_device.get_num_devices()} devices is created")
    yield mesh_device

    ttnn.DumpDeviceProfiler(mesh_device)

    ttnn.close_mesh_device(mesh_device)
    reset_fabric(fabric_config)
    del mesh_device


@pytest.fixture(scope="function")
def n300_mesh_device(request, silicon_arch_name, silicon_arch_wormhole_b0, device_params):
    import ttnn

    if ttnn.get_num_devices() < 2:
        pytest.skip()

    updated_device_params = get_updated_device_params(device_params)
    fabric_config = updated_device_params.pop("fabric_config", None)
    set_fabric(fabric_config)
    mesh_device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, 2),
        **updated_device_params,
    )

    logger.debug(f"multidevice with {mesh_device.get_num_devices()} devices is created")
    yield mesh_device

    ttnn.DumpDeviceProfiler(mesh_device)

    ttnn.close_mesh_device(mesh_device)
    reset_fabric(fabric_config)
    del mesh_device


@pytest.fixture(scope="function")
def t3k_mesh_device(request, silicon_arch_name, silicon_arch_wormhole_b0, device_params):
    import ttnn

    if ttnn.get_num_devices() < 8:
        pytest.skip()

    request.node.pci_ids = ttnn.get_pcie_device_ids()
    updated_device_params = get_updated_device_params(device_params)
    fabric_config = updated_device_params.pop("fabric_config", None)
    set_fabric(fabric_config)
    mesh_device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, 8),
        **updated_device_params,
    )

    logger.debug(f"multidevice with {mesh_device.get_num_devices()} devices is created")
    yield mesh_device

    ttnn.DumpDeviceProfiler(mesh_device)

    ttnn.close_mesh_device(mesh_device)
    reset_fabric(fabric_config)
    del mesh_device


@pytest.fixture()
def ensure_devices_tg():
    import ttnn

    device_ids = ttnn.get_device_ids()
    assert len(device_ids) == 32, f"Expected 32 devices, got {len(device_ids)}"


@pytest.fixture()
def clear_compile_cache():
    yield
    import ttnn

    ttnn.device.DisablePersistentKernelCache()


@pytest.fixture(autouse=True)
def reset_default_device():
    import ttnn

    device = ttnn.GetDefaultDevice()
    yield
    ttnn.SetDefaultDevice(device)


def get_devices(request):
    if "device" in request.fixturenames:
        devices = [request.getfixturevalue("device")]
    elif "all_devices" in request.fixturenames:
        devices = request.getfixturevalue("all_devices")
    elif "pcie_devices" in request.fixturenames:
        devices = request.getfixturevalue("pcie_devices")
    elif "mesh_device" in request.fixturenames:
        devices = [request.getfixturevalue("mesh_device")]
    elif "n300_mesh_device" in request.fixturenames:
        devices = [request.getfixturevalue("n300_mesh_device")]
    elif "t3k_mesh_device" in request.fixturenames:
        devices = [request.getfixturevalue("t3k_mesh_device")]
    elif "pcie_mesh_device" in request.fixturenames:
        devices = [request.getfixturevalue("pcie_mesh_device")]
    else:
        devices = []
    return devices


@pytest.fixture(scope="function")
def use_program_cache(request):
    devices = get_devices(request)
    if not devices:
        logger.warning("No device fixture found to apply program cache to: PROGRAM CACHE DISABLED")
    for dev in devices:
        dev.enable_program_cache()
    yield
    for dev in devices:
        dev.disable_and_clear_program_cache()


@pytest.fixture(scope="function")
def tracy_profile():
    from tracy import Profiler

    profiler = Profiler()

    profiler.enable()
    yield
    profiler.disable()


###############################
# Modifying pytest hooks
###############################
ALL_ARCHS = set(
    [
        "grayskull",
        "wormhole_b0",
        "blackhole",
    ]
)


def pytest_addoption(parser):
    import ttnn

    parser.addoption(
        "--tt-arch",
        choices=[*ALL_ARCHS],
        default=ttnn.get_arch_name(),
        help="Target arch, ex. grayskull, wormhole_b0, blackhole",
    )
    parser.addoption(
        "--pipeline-type",
        default="",
        help="Only `models_device_performance_bare_metal` should run `pytest_runtest_teardown`",
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
    parser.addoption(
        "--metal-timeout",
        action="store",
        default=None,
        help="Enable process timeout",
    )


@pytest.fixture
def input_path(request):
    return request.config.getoption("--input-path")


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
        "silicon_arch_blackhole": set(
            [
                "blackhole",
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
        metafunc.parametrize("silicon_arch_name", available_archs, scope="session")
        for test_requested_silicon_arch_fixture in test_requested_silicon_arch_fixtures:
            # The values of these arch-specific fixtures should not be used in
            # the test function, so use any parameters, like [True]
            metafunc.parametrize(test_requested_silicon_arch_fixture, [True], scope="session")

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


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_teardown(item, nextitem):
    yield
    metal_timeout_enabled = item.config.getoption("--metal-timeout")
    using_xdist = int(os.getenv("PYTEST_XDIST_WORKER_COUNT", "0"))

    if metal_timeout_enabled is not None or using_xdist:
        report = item.stash[phase_report_key]
        test_failed = report.get("call", None) and report["call"].failed
        if test_failed:
            logger.info(f"In custom teardown, open device ids: {set(item.pci_ids)}")
            reset_tensix(set(item.pci_ids))


# This is overriding the timer setup hook from pytest-timeout
# If --metal-timeout is passed, we define a new timeout method that spawns a timer process
# At timeout, the process kills it's parent (the test process) and then itself
@pytest.hookimpl(tryfirst=True)
def pytest_timeout_set_timer(item, settings):
    metal_timeout_enabled = item.config.getoption("--metal-timeout")
    using_xdist = int(os.getenv("PYTEST_XDIST_WORKER_COUNT", "0"))

    if metal_timeout_enabled is not None or using_xdist:
        parent_pid = os.getpid()
        logger.info(f"Metal timeout {settings.timeout} seconds {parent_pid} for {item.nodeid}")

        def get_parent_status():
            try:
                parent = psutil.Process(parent_pid)
            except:
                return "already dead"
            return parent.status()

        def run_timer(settings):
            logger.info(f"Timer started for {item.nodeid}")
            dead_status = ["zombie", "dead", "already dead"]
            timeout = settings.timeout
            parent_status = "running"
            while parent_status not in dead_status and timeout > 0:
                time.sleep(5)
                timeout -= 5
                parent_status = get_parent_status()
            if parent_status != "already dead":
                logger.warning(f"This test seems to have hung... Timing out test case")
                os.kill(parent_pid, signal.SIGKILL)
            logger.info(f"Killing timer")
            os._exit(1)

        def cancel():
            logger.info(f"Cancelling timer")
            metal_timer.terminate()

        metal_timer = multiprocess.Process(target=run_timer, args=(settings,), daemon=True)
        item.cancel_timeout = cancel
        metal_timer.start()
    return True


# This is a hook used in pytest-xdist to handle when a worker crashes out
# In our case, combined with pytest-timeout thread method, the worker will crash out for a hang and
# then it should get cleaned up by the controller through this fixture
@pytest.hookimpl(tryfirst=True)
def pytest_handlecrashitem(crashitem, report, sched):
    reset_tensix()


def reset_tensix(tt_open_devices=None):
    import shutil

    if is_galaxy():
        logger.info("Skipping reset for Galaxy systems, need a new reset.json scheme")
        return

    # Check if tt-smi exists
    if not shutil.which("tt-smi"):
        logger.error("tt-smi command not found. Cannot reset devices. Please install tt-smi.")
        return

    if tt_open_devices is None:
        logger.info(f"Running reset for all pci devices")
        smi_reset_result = run_process_and_get_result(f"tt-smi -r")
    else:
        tt_open_devices_str = ",".join([str(i) for i in tt_open_devices])
        logger.info(f"Running reset for pci devices: {tt_open_devices_str}")
        smi_reset_result = run_process_and_get_result(f"tt-smi -r {tt_open_devices_str}")

    logger.info(f"tt-smi reset status: {smi_reset_result.returncode}")


@pytest.hookimpl(tryfirst=True)
def pytest_xdist_auto_num_workers(config):
    return 1


@pytest.fixture(scope="function", autouse=True)
def record_test_timestamp(record_property):
    start_timestamp = datetime.strftime(datetime.now(), "%Y-%m-%dT%H:%M:%S%z")
    record_property("start_timestamp", start_timestamp)
    yield
    end_timestamp = datetime.strftime(datetime.now(), "%Y-%m-%dT%H:%M:%S%z")
    record_property("end_timestamp", end_timestamp)


def pytest_configure(config):
    xmlpath = config.option.xmlpath
    # https://github.com/tenstorrent/tt-metal/pull/18372
    # Only override the xmlpath if it's set, and we're in a CI env (GHA)
    # Problem: t3k unit tests run pytest multiple times overwriting the junit xml file each time, so the generated xml artifact only contains test case info from the last running testsuite.
    # Fix: when running in CI env, override config.option.xmlpath to rename the xml filepath to include timestamp, so that serial pytest invocations running in scripts do not clobber the junit xml test report
    if xmlpath and os.getenv("CI") == "true":
        # Get the dir and filename for the generated xml
        directory, filename = os.path.split(xmlpath)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Append timestamp to the end of the xml filename
        # This avoids clobbering the xml file when pytest is invoked multiple times during a test script
        new_filename = f"{os.path.splitext(filename)[0]}_{timestamp}{os.path.splitext(filename)[1]}"
        new_xmlpath = os.path.join(directory, new_filename)
        config.option.xmlpath = new_xmlpath
