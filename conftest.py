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
import subprocess
from datetime import datetime

from loguru import logger

from tests.scripts.common import run_process_and_get_result
from tests.scripts.common import get_updated_device_params

# Constants for device configurations
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

    return num_pcie == 1 and ttnn.cluster.get_cluster_type() == ttnn.cluster.ClusterType.N300


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

    return (
        ttnn.cluster.get_cluster_type() == ttnn.cluster.ClusterType.GALAXY
        or ttnn.cluster.get_cluster_type() == ttnn.cluster.ClusterType.TG
    )


# TODO: Remove this when TG clusters are deprecated.
def is_6u():
    import ttnn

    return ttnn.cluster.get_cluster_type() == ttnn.cluster.ClusterType.GALAXY


# TODO: Remove this when TG clusters are deprecated.
def is_tg_cluster():
    import ttnn

    return ttnn.cluster.get_cluster_type() == ttnn.cluster.ClusterType.TG


def first_available_tg_device():
    assert is_tg_cluster()
    # The id of the first user exposed device for a TG cluster is 4
    return 4


@pytest.fixture(scope="session")
def is_ci_v2_env():
    yield "TT_GH_CI_INFRA" in os.environ


# We don't want other people using this stuff... wonder if we should just stuff it in the fixture that's calling it instead
class CIv2ModelDownloadUtils_:
    @staticmethod
    def download_from_ci_v2_cache(
        model_path,
        timeout_in_s,
        download_dir_suffix="",
        endpoint_prefix="http://large-file-cache.large-file-cache.svc.cluster.local//mldata/model_checkpoints/pytorch/huggingface",
    ):
        assert model_path, f"model_path cannot be empty when downloading - what is wrong with you?: {model_path}"

        assert isinstance(
            timeout_in_s, int
        ), f"{timeout_in_s} is not an integer, which it should be because it's a timeout duration"

        # RK: Will this be portable? LOL
        download_dir = Path("/tmp/ttnn_model_cache/") / download_dir_suffix

        download_dir.mkdir(parents=True, exist_ok=True)

        download_dir_str = str(download_dir)

        # Add trailing slash to model_path if it doesn't have one, as wget
        # seems to not download recursively via subprocess if it doesn't have
        # it
        if model_path and not model_path.endswith("/"):
            model_path = model_path + "/"

        endpoint = f"{endpoint_prefix}/{model_path}"

        try:
            # TODO: How do we add a timeout here without relying on native timeout command?
            subprocess.run(
                [
                    "wget",
                    "-r",
                    "-nH",
                    "-x",
                    "--cut-dirs=5",
                    "-np",
                    "--progress=dot:giga",
                    "-R",
                    "index.html*",
                    "-P",
                    download_dir_str,
                    endpoint,
                ],
                check=True,
                text=True,
                timeout=timeout_in_s,
            )
        except subprocess.TimeoutExpired as err:
            logger.error(f"Timeout of {timeout_in_s} seconds occurred while downloading from {endpoint}.")
            raise err
        except Exception as err:
            logger.error(
                f"Unknown error occurred while trying to download from {endpoint}. Check above logs from wget call."
            )
            logger.error(err)
            raise err

        return download_dir / Path(model_path)


@pytest.fixture(scope="session")
def model_location_generator(is_ci_v2_env):
    """
    Returns a function that will determine the appropriate file path for a
    model based on available locations.

    This function locates model files by checking several possible locations in the following order:
    1. CIv2 cache if running in CI environment and the user requests CIv2
       resources via setting download_if_ci_v2 to True.
       If we're in a CIv2 environment and download_if_ci_v2 is True, that means
       the model is requesting files from CIv2. However, we will error out if
       the files are not available because that means the responsible developer
       did not properly uploaded the requested files.
    2. Cloud MLPerf path if available, which is virtually all cases for CIv1
    3. Default to the model_version string, which means downloading to the
       local Huggingface cache directory (HF_HOME, or ~/.cache/huggingface by
       default)

    For CIv2 specifically
    ---------------------

    The expected directory structure in the single source of truth datastore
    should be:

    lfc://mldata/model_checkpoints/pytorch/huggingface/pytorch
    ├── huggingface
    └── hf_repo_owner/hf_repo
        ├── weight1.bin
        ├── weight2.bin
        ├── ...
        └── T3K
            ├── T3K_ttnn_tensor1.bin
            ├── T3K_ttnn_tensor2.bin
            └── ...
        └── N300
            ├── N300_ttnn_tensor1.bin
            ├── N300_ttnn_tensor2.bin
            └── ...

    Why couple the TT-NN tensor binaries into the Huggingface model's folder?
    This is because tensors are generated on a per-model basis, so in terms
    of folder organization there isn't too much benefit from having a separate
    place for HF weights and a separate place for the bins.

    What's nice about this is this makes it clear which HF model corresponds
    to which set of tensor binaries, which is useful for engineers to quickly
    see which model is generating which binaries.

    Note that the logic for all of this is in CIv2ModelDownloadUtils_.

    :param model_version: The version identifier of the model to locate
    :type model_version: str
    :param model_subdir: Subdirectory within the model folder structure.
                         Default is empty string.
                         Note: Nested subdirectories (model_subdir) are not
                         supported in CIv2 cache.
    :type model_subdir: str
    :param download_if_ci_v2: Whether to download from CI v2 cache if in a CI v2 environment
    :type download_if_ci_v2: bool
    :param ci_v2_timeout_in_s: Timeout for download from CI v2 cache in seconds
    :type ci_v2_timeout_in_s: int

    :return: The path to the model files (internal MLPerf path, CI v2 cache
             path, or just model_version which uses HF_HOME)
    :rtype: os.PathLike (str, pathlib.Path etc.)

    :raises AssertionError: If trying to run in CIv2 environment with MLPerf
    files which is impossible, or if model_subdir contains unsupported
    directory structure
    """

    def model_location_generator_(
        model_version,
        model_subdir="",
        download_if_ci_v2=False,
        ci_v2_timeout_in_s=300,
        endpoint_prefix="http://large-file-cache.large-file-cache.svc.cluster.local//mldata/model_checkpoints/pytorch/huggingface",
        download_dir_suffix="model_weights",
    ):
        model_folder = Path("tt_dnn-models") / model_subdir
        internal_weka_path = Path("/mnt/MLPerf") / model_folder / model_version
        has_internal_weka = internal_weka_path.exists()

        download_from_ci_v2 = download_if_ci_v2 and is_ci_v2_env

        if download_from_ci_v2:
            assert (
                not has_internal_weka
            ), "For some reason, we see a file existing at the expected MLPerf location: {internal_weka_path} on CIv2. Please use the opportunity to clean up your model and get rid of MLPerf if you're moving to CIv2"
            assert (
                not model_subdir
            ), f"model_subdir is set to {model_subdir}, but we don't support further levels of directories in the large file cache in CIv2"
            civ2_download_path = CIv2ModelDownloadUtils_.download_from_ci_v2_cache(
                model_version,
                download_dir_suffix=download_dir_suffix,
                timeout_in_s=ci_v2_timeout_in_s,
                endpoint_prefix=endpoint_prefix,
            )
            logger.info(f"For model location, using CIv2 large file cache: {civ2_download_path}")
            return civ2_download_path
        elif has_internal_weka:
            logger.info(f"For model location, using internal MLPerf path: {internal_weka_path}")
            return internal_weka_path
        else:
            logger.info(
                f"For model location, local copy not found, so likely downloading straight from HF: {model_version}"
            )
            return model_version

    return model_location_generator_


@pytest.fixture(scope="session")
def get_tt_cache_path():
    def get_tt_cache_path_(model_version, model_subdir="", default_dir=""):
        model_folder = Path("tt_dnn-models/tt") / model_subdir
        internal_weka_path = Path("/mnt/MLPerf") / model_folder / model_version
        has_internal_weka = internal_weka_path.exists()
        if has_internal_weka:
            logger.debug(f"Using internal MLPerf path: {internal_weka_path}")
            return internal_weka_path
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

    ttnn.close_device(device)


# Reset fabric config to DISABLED if not None, and do nothing otherwise
# Temporarily require previous state to be passed in as even setting it to DISABLED might be unstable
# This is to ensure that we don't propagate the instability to the rest of CI
def reset_fabric(fabric_config):
    import ttnn

    if fabric_config:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


# Set fabric config to passed in value
# Do nothing if not set
# Must be called before creating the mesh device
def set_fabric(fabric_config, reliability_mode=None, fabric_tensix_config=None):
    import ttnn

    # If fabric_config is not None, set it to fabric_config
    if fabric_config:
        if reliability_mode is None:
            reliability_mode = ttnn.FabricReliabilityMode.STRICT_INIT

        # Apply default logic for fabric_tensix_config,
        # fabric_tensix_config is used for enabling tensix extensions for the fabric router,
        # some sender channels in the fabric router are moved to the fabric tensix extension
        # (currently the extension is mux kernel, can have other kernels in future as well).
        if fabric_tensix_config is None:
            fabric_tensix_config = get_default_fabric_tensix_config()

        ttnn.set_fabric_config(fabric_config, reliability_mode, None, fabric_tensix_config)  # num_planes


def get_default_fabric_tensix_config():
    import ttnn

    # Default to DISABLED for all architectures
    return ttnn.FabricTensixConfig.DISABLED


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

    request.node.pci_ids = ttnn.get_pcie_device_ids()

    try:
        param = request.param
    except (ValueError, AttributeError):
        # Get number of devices from the system mesh descriptor.
        param = ttnn._ttnn.multi_device.SystemMeshDescriptor().shape().mesh_size()

    if isinstance(param, tuple):
        grid_dims = param
        assert len(grid_dims) == 2, "Device mesh grid shape should have exactly two elements."
        num_devices_requested = grid_dims[0] * grid_dims[1]
        if not ttnn.using_distributed_env() and num_devices_requested > ttnn.get_num_devices():
            pytest.skip("Requested more devices than available. Test not applicable for machine")
        mesh_shape = ttnn.MeshShape(*grid_dims)
    else:
        if not ttnn.using_distributed_env() and param > ttnn.get_num_devices():
            pytest.skip("Requested more devices than available. Test not applicable for machine")
        mesh_shape = ttnn.MeshShape(1, param)

    updated_device_params = get_updated_device_params(device_params)
    fabric_config = updated_device_params.pop("fabric_config", None)
    fabric_tensix_config = updated_device_params.pop("fabric_tensix_config", None)
    reliability_mode = updated_device_params.pop("reliability_mode", None)
    set_fabric(fabric_config, reliability_mode, fabric_tensix_config)
    mesh_device = ttnn.open_mesh_device(mesh_shape=mesh_shape, **updated_device_params)

    logger.debug(f"multidevice with {mesh_device.get_num_devices()} devices is created")
    yield mesh_device

    for submesh in mesh_device.get_submeshes():
        ttnn.close_mesh_device(submesh)

    ttnn.close_mesh_device(mesh_device)
    reset_fabric(fabric_config)
    del mesh_device


@pytest.fixture(scope="function")
def t3k_single_board_mesh_device(request, silicon_arch_name, silicon_arch_wormhole_b0, device_params):
    import ttnn

    device_ids = ttnn.get_device_ids()

    assert len(device_ids) == 8, "This fixture is only applicable for T3K systems"

    try:
        pcie_id = request.param
    except (ValueError, AttributeError):
        pcie_id = 0  # Default to using first board

    assert pcie_id < 4, "Requested board id is out of range"

    mesh_device_ids = [device_ids[pcie_id], device_ids[pcie_id + 4]]
    mesh_shape = ttnn.MeshShape(1, 2)
    mesh_device = ttnn.open_mesh_device(
        mesh_shape, mesh_device_ids, dispatch_core_type=ttnn.device.DispatchCoreType.WORKER, **device_params
    )

    logger.debug(f"multidevice with {mesh_device.get_num_devices()} devices is created")
    yield mesh_device

    ttnn.close_mesh_device(mesh_device)
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
    fabric_tensix_config = updated_device_params.pop("fabric_tensix_config", None)
    reliability_mode = updated_device_params.pop("reliability_mode", None)
    set_fabric(fabric_config, reliability_mode, fabric_tensix_config)
    mesh_device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(2, 2),
        **updated_device_params,
        offset=ttnn.MeshCoordinate(0, 1),
    )
    mesh_device.reshape(ttnn.MeshShape(1, 4))

    logger.debug(f"multidevice with {mesh_device.get_num_devices()} devices is created")
    yield mesh_device

    for submesh in mesh_device.get_submeshes():
        ttnn.close_mesh_device(submesh)

    ttnn.close_mesh_device(mesh_device)
    reset_fabric(fabric_config)
    del mesh_device


@pytest.fixture(scope="function")
def t3k_mesh_device(request, silicon_arch_name, silicon_arch_wormhole_b0, device_params):
    import ttnn

    if ttnn.get_num_devices() < 8:
        pytest.skip("Not enough devices to run test")

    request.node.pci_ids = ttnn.get_pcie_device_ids()
    updated_device_params = get_updated_device_params(device_params)
    fabric_config = updated_device_params.pop("fabric_config", None)
    fabric_tensix_config = updated_device_params.pop("fabric_tensix_config", None)
    reliability_mode = updated_device_params.pop("reliability_mode", None)
    set_fabric(fabric_config, reliability_mode, fabric_tensix_config)
    mesh_device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, 8),
        **updated_device_params,
    )

    logger.debug(f"multidevice with {mesh_device.get_num_devices()} devices is created")
    yield mesh_device

    for submesh in mesh_device.get_submeshes():
        ttnn.close_mesh_device(submesh)

    ttnn.close_mesh_device(mesh_device)
    reset_fabric(fabric_config)
    del mesh_device


@pytest.fixture(scope="function")
def bh_1d_mesh_device(request, silicon_arch_name, silicon_arch_blackhole, device_params):
    # Generic blackhole configuration
    # This configures an [m,n] blackhole mesh device to appear as a [1,m*n] line or ring
    # Implements wraparound in rackboxes
    import ttnn

    if ttnn.get_num_devices() not in [1, 2, 4, 8, 32]:
        pytest.skip()

    request.node.pci_ids = ttnn.get_pcie_device_ids()
    updated_device_params = get_updated_device_params(device_params)
    fabric_config = updated_device_params.pop("fabric_config", None)
    fabric_tensix_config = updated_device_params.pop("fabric_tensix_config", None)
    reliability_mode = updated_device_params.pop("reliability_mode", None)
    set_fabric(fabric_config, reliability_mode, fabric_tensix_config)

    mesh_device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(ttnn.get_num_devices(), 1),
        **updated_device_params,
    )
    logger.debug(f"multidevice with {mesh_device.get_num_devices()} devices is created")
    yield mesh_device

    for submesh in mesh_device.get_submeshes():
        ttnn.close_mesh_device(submesh)

    ttnn.close_mesh_device(mesh_device)
    reset_fabric(fabric_config)
    del mesh_device


@pytest.fixture(scope="function")
def bh_2d_mesh_device(request, silicon_arch_name, silicon_arch_blackhole, device_params):
    # Generic blackhole configuration
    # This preserves the 2D mesh configuration in rackbox and galaxy
    import ttnn

    if ttnn.get_num_devices() not in [1, 2, 4, 8, 32]:
        pytest.skip()

    request.node.pci_ids = ttnn.get_pcie_device_ids()
    updated_device_params = get_updated_device_params(device_params)
    fabric_config = updated_device_params.pop("fabric_config", None)
    fabric_tensix_config = updated_device_params.pop("fabric_tensix_config", None)
    reliability_mode = updated_device_params.pop("reliability_mode", None)
    set_fabric(fabric_config, reliability_mode, fabric_tensix_config)
    if ttnn.get_num_devices() == 8:
        mesh_device = ttnn.open_mesh_device(
            mesh_shape=ttnn.MeshShape(4, 2),
            **updated_device_params,
        )
    elif ttnn.get_num_devices() == 32:
        mesh_device = ttnn.open_mesh_device(
            mesh_shape=ttnn.MeshShape(4, 8),
            **updated_device_params,
        )
    else:
        mesh_device = ttnn.open_mesh_device(
            mesh_shape=ttnn.MeshShape(ttnn.get_num_devices(), 1),
            **updated_device_params,
        )
    logger.debug(f"multidevice with {mesh_device.get_num_devices()} devices is created")
    yield mesh_device

    for submesh in mesh_device.get_submeshes():
        ttnn.close_mesh_device(submesh)

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
def reset_default_device(request):
    import ttnn

    # Skip applying the fixture logic for this test
    if "no_reset_default_device" in request.keywords:
        yield
        return
    device = ttnn.GetDefaultDevice()
    yield
    ttnn.SetDefaultDevice(device)


def get_devices(request):
    if "device" in request.fixturenames:
        devices = [request.getfixturevalue("device")]
    elif "mesh_device" in request.fixturenames:
        devices = [request.getfixturevalue("mesh_device")]
    elif "t3k_mesh_device" in request.fixturenames:
        devices = [request.getfixturevalue("t3k_mesh_device")]
    elif "pcie_mesh_device" in request.fixturenames:
        devices = [request.getfixturevalue("pcie_mesh_device")]
    elif "t3k_single_board_mesh_device" in request.fixturenames:
        devices = request.getfixturevalue("t3k_single_board_mesh_device").get_devices()
    else:
        devices = []
    return devices


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
    parser.addoption(
        "--didt-workload-iterations",
        action="store",
        default=None,
        help="Number of workload iterations to run for didt tests",
    )
    parser.addoption(
        "--determinism-check-interval",
        action="store",
        default=None,
        help="Check determinism every nth iteration",
    )
    parser.addoption(
        "--grid-size",
        action="store",
        default=None,
        help="Size of chip grid for the test to run on. Grid size is defined by number of cores in row x number of cores in column, e.g., 8x8",
    )


@pytest.fixture
def grid_size(request):
    """
    Fixture to set the chip grid size for the test to run on.
    If --grid-size is provided, it returns a tuple of integers (rows, columns).
    If not provided, it defaults to None.
    """
    grid_size_str = request.config.getoption("--grid-size")
    if grid_size_str:
        try:
            rows, cols = map(int, grid_size_str.split("x"))
            return (rows, cols)
        except ValueError:
            raise ValueError(f"Invalid grid size format: {grid_size_str}. Use format 'rows x cols'.")
    return None


# Indicates the iteration interval at which determinism is verified for the op output
@pytest.fixture
def determinism_check_interval(request):
    iterations = request.config.getoption("--determinism-check-interval")
    if iterations is not None:
        # this will throw an error if bad value is passed
        return int(iterations)
    return -1


# Indicated the number of workload iterations to run within didt tests
@pytest.fixture
def didt_workload_iterations(request):
    iterations = request.config.getoption("--didt-workload-iterations")
    if iterations is not None:
        # this will throw an error if bad value is passed
        return int(iterations)
    # default is 100000
    return 100000


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


def _metal_alarm_handler(signum, frame):
    """Alarm handler for test timeouts; collects debug info then force-kills the process."""
    try:
        logger.warning("This test seems to have hung... Timing out test case")
        run_debug_script()
    except Exception as e:
        logger.error(f"Failed to run debug script after timeout: {e}")
    finally:
        # Ensure the process is terminated even if debug collection fails
        os.kill(os.getpid(), signal.SIGKILL)


# This overrides the timer setup hook from pytest-timeout.
# If --metal-timeout is passed or when using xdist, we arm a POSIX itimer.
# On expiry, the alarm handler runs debug collection and kills the process.
@pytest.hookimpl(tryfirst=True)
def pytest_timeout_set_timer(item, settings):
    metal_timeout_enabled = item.config.getoption("--metal-timeout")
    using_xdist = int(os.getenv("PYTEST_XDIST_WORKER_COUNT", "0"))

    if metal_timeout_enabled is not None or using_xdist:
        current_pid = os.getpid()
        logger.debug(f"Metal timeout {settings.timeout} seconds {current_pid} for {item.nodeid}")

        # Install handler (idempotent is fine)
        signal.signal(signal.SIGALRM, _metal_alarm_handler)

        # Arm the REAL timer for this test
        secs = float(settings.timeout)
        signal.setitimer(signal.ITIMER_REAL, secs, 0.0)

        # Provide a canceller for pytest-timeout
        def cancel():
            logger.debug("Cancelling timer")
            # Disarm the timer
            signal.setitimer(signal.ITIMER_REAL, 0.0, 0.0)

        item.cancel_timeout = cancel
    return True


# This is a hook used in pytest-xdist to handle when a worker crashes out
# In our case, combined with pytest-timeout thread method, the worker will crash out for a hang and
# then it should get cleaned up by the controller through this fixture
@pytest.hookimpl(tryfirst=True)
def pytest_handlecrashitem(crashitem, report, sched):
    reset_tensix()


def run_debug_script():
    """Run the tt-triage.py debug script to check system state before cleanup."""

    # Check if ttexalens module is available
    try:
        import ttexalens
    except ImportError:
        logger.warning(
            "ttexalens module not found. Debug script requires ttexalens to be installed. Skipping debug collection."
        )
        return

    debug_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools", "tt-triage.py")

    if not os.path.exists(debug_script_path):
        logger.warning(f"Debug script not found at {debug_script_path}. Skipping debug collection.")
        return

    try:
        logger.info("Running debug script to check system state")
        # Remove LD_LIBRARY_PATH to avoid conflicts with prebuilt libraries
        extra_env = {
            "LD_LIBRARY_PATH": None,
        }
        debug_result = run_process_and_get_result(f"python {debug_script_path}", extra_env)

        logger.info(f"Debug script status: {debug_result.returncode}")
        if debug_result.stdout:
            logger.info(f"Debug script output: {debug_result.stdout.decode('utf-8')}")
        if debug_result.stderr:
            logger.info(f"Debug script stderr: {debug_result.stderr.decode('utf-8')}")
    except Exception as e:
        logger.error(f"Failed to run debug script: {e}")


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
