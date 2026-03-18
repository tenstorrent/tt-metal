# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0
import os
import socket
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from loguru import logger
from transformers import AutoConfig

import ttnn
from models.demos.deepseek_v3.tt.ccl import CCL
from models.demos.deepseek_v3.utils.test_utils import get_valid_system_names, load_state_dict, system_name_to_mesh_shape
from tests.scripts.common import get_updated_device_params

RESET_WEIGHT_CACHE_OPTION = "--recalculate-weights"
RECALCULATE_WEIGHT_CACHE_DEMO_TIMEOUT_SECONDS = 6 * 60 * 60
DEEPSEEK_DEMO_NODEID_PREFIX = "models/demos/deepseek_v3/demo/"
DEEPSEEK_TESTS_NODEID_PREFIX = "models/demos/deepseek_v3/tests/"
MULTIHOST_WEIGHT_CACHE_TEST_TIMEOUT_SECONDS = 30 * 60

# Shared test parametrization constants
# Prefill sequence lengths: powers of 2 from 128 to 128K
PREFILL_SEQ_LENS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]


def pytest_addoption(parser):
    parser.addoption(
        RESET_WEIGHT_CACHE_OPTION,
        action="store_true",
        help="Reset weight configs for tests",
    )
    parser.addoption(
        "--bitsculpt-trace",
        action="store_true",
        help="Enable the manual BitSculpt-compatible DeepSeek trace harness test.",
    )
    parser.addoption(
        "--bitsculpt-trace-prompt",
        action="store",
        default="why is aho so sus",
        help="Raw prompt text for the BitSculpt trace harness.",
    )
    parser.addoption(
        "--bitsculpt-trace-run-tag",
        action="store",
        default="tt_moconnor",
        help="Run tag subdirectory for the BitSculpt trace harness output.",
    )
    parser.addoption(
        "--bitsculpt-trace-output-dir",
        action="store",
        default=str(
            Path(__file__).resolve().parents[3].parent / "bit_sculpt" / "results" / "deepseek-r1-0528" / "debug_trace"
        ),
        help="Base output directory for the BitSculpt trace harness.",
    )
    parser.addoption(
        "--bitsculpt-trace-model-id",
        action="store",
        default="deepseek-ai/DeepSeek-R1-0528",
        help="Model ID recorded into BitSculpt trace metadata.",
    )
    parser.addoption(
        "--bitsculpt-trace-tokenizer",
        action="store",
        default=None,
        help="Optional tokenizer path or repo ID for the BitSculpt trace harness.",
    )
    parser.addoption(
        "--bitsculpt-trace-save-dtype",
        action="store",
        default="bfloat16",
        choices=("bfloat16", "float32"),
        help="Save dtype for the BitSculpt trace harness.",
    )
    parser.addoption(
        "--bitsculpt-trace-max-tokens",
        action="store",
        type=int,
        default=200,
        help="Maximum allowed prompt token count for the BitSculpt trace harness.",
    )


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    setattr(item, f"rep_{report.when}", report)


def automatically_detect_current_device_type() -> str:
    """
    Automatically detect device type based on cluster type and device count.

    Returns:
        str: One of "N150", "N300", "T3K", "TG", "DUAL", "QUAD"

    Raises:
        ValueError: If the device type cannot be determined from the current configuration
    """
    cluster_type = ttnn.cluster.get_cluster_type()
    num_devices = ttnn.get_num_devices()

    # Check cluster type first
    if cluster_type == ttnn.cluster.ClusterType.T3K:
        if num_devices == 8:
            return "T3K"
        else:
            raise ValueError(f"T3K cluster type detected but unexpected device count: {num_devices} (expected 8)")
    elif cluster_type == ttnn.cluster.ClusterType.TG:
        if num_devices == 32:
            return "TG"
        else:
            raise ValueError(f"TG cluster type detected but unexpected device count: {num_devices} (expected 32)")
    elif cluster_type == ttnn.cluster.ClusterType.GALAXY:
        # Galaxy can be TG (32 devices), DUAL (64 devices), or QUAD (128 devices)
        if num_devices == 32:
            return "TG"
        elif num_devices == 64:
            return "DUAL"
        elif num_devices == 128:
            return "QUAD"
        else:
            raise ValueError(
                f"GALAXY cluster type detected but unexpected device count: {num_devices} "
                f"(expected 32 for TG, 64 for DUAL, or 128 for QUAD)"
            )
    elif cluster_type == ttnn.cluster.ClusterType.N150:
        if num_devices == 1:
            return "N150"
        else:
            raise ValueError(f"N150 cluster type detected but unexpected device count: {num_devices} (expected 1)")
    elif cluster_type == ttnn.cluster.ClusterType.N300:
        if num_devices == 1 or num_devices == 2:
            return "N300"
        else:
            raise ValueError(f"N300 cluster type detected but unexpected device count: {num_devices} (expected 1 or 2)")

    raise ValueError(
        f"Unable to determine device type: cluster_type={cluster_type}, "
        f"num_devices={num_devices}, arch_name={ttnn.get_arch_name()}"
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

    def get_mesh_shape(system_name: str) -> ttnn.MeshShape:
        if system_name.upper() == "AUTO":
            detected_name = get_current_device_type()
            logger.info(
                f"Selected MESH_DEVICE: 'AUTO' - detected device type: '{detected_name}' - mesh shape will be set to: {system_name_to_mesh_shape(detected_name)}"
            )
            return system_name_to_mesh_shape(detected_name)
        else:
            logger.info(
                f"Selected MESH_DEVICE: '{system_name}' - mesh shape will be set to: {system_name_to_mesh_shape(system_name)}"
            )
            return system_name_to_mesh_shape(system_name.upper())

    mesh_shape = get_mesh_shape(requested_system_name)
    updated_device_params = get_updated_device_params(device_params)

    fabric_config = updated_device_params.pop("fabric_config", None)
    if fabric_config:
        ttnn.set_fabric_config(fabric_config)

    updated_device_params.setdefault("mesh_shape", mesh_shape)
    mesh_device = ttnn.open_mesh_device(**updated_device_params)

    # MPI_Init_thread (triggered by open_mesh_device in multi-host configs) sets OpenMP threads to 1,
    # torch inherits this setting, which makes CPU-side reference model computations extremely slow.
    # We restore a reasonable thread count for torch.
    if requested_system_name.upper() in ("DUAL", "QUAD"):
        num_torch_threads = max(1, os.cpu_count())
        logger.info(f"Restoring torch num_threads to {num_torch_threads}")
        torch.set_num_threads(num_torch_threads)

    logger.debug(f"Mesh device with {mesh_device.get_num_devices()} devices is created with shape {mesh_device.shape}")
    yield mesh_device

    for phase in ("setup", "call"):
        report = getattr(request.node, f"rep_{phase}", None)
        if report is not None and report.failed:
            rank = "unknown"
            if ttnn.distributed_context_is_initialized():
                try:
                    rank = int(ttnn.distributed_context_get_rank())
                except Exception:
                    pass
            logger.error(
                "Rank {} on {} is tearing down mesh after {} failure in {}.\n{}",
                rank,
                socket.gethostname().split(".", 1)[0],
                phase,
                request.node.nodeid,
                report.longreprtext,
            )
            break

    for submesh in mesh_device.get_submeshes():
        ttnn.close_mesh_device(submesh)

    ttnn.close_mesh_device(mesh_device)
    if fabric_config:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

    del mesh_device


@pytest.fixture(scope="session")
def model_path():
    """Get model path and resolve symlinks to ensure all operations can find files."""
    path = Path(os.getenv("DEEPSEEK_V3_HF_MODEL", "models/demos/deepseek_v3/reference"))
    # Resolve symlinks to ensure AutoConfig and other operations can find config.json and other files
    return path.resolve()


@pytest.fixture(scope="session")
def hf_config(model_path):
    """Load DeepSeek config for testing"""
    # model_path is already resolved in the fixture
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    return config


@pytest.fixture(scope="session")
def state_dict(model_path):
    yield load_state_dict(model_path, "")


@pytest.fixture(scope="function", autouse=True)
def clear_state_dict_cache(request):
    """
    Clear the LazyStateDict cache after each test to prevent memory accumulation.
    This preserves file handles (mmap benefits) while freeing tensor memory.
    """
    # Check if state_dict is requested by this test
    if "state_dict" not in request.fixturenames:
        yield
        return

    state_dict = request.getfixturevalue("state_dict")
    yield
    state_dict.clear_cache()


@pytest.fixture(scope="session")
def hf_config_short(request, hf_config):
    """
    Build a shortened DeepSeek config for tests.

    Environment variables:
        DEEPSEEK_MAX_SEQ_LEN_OVERRIDE: Optional override for `hf_config_short.max_seq_len`.
            When set (e.g. "32768"), tests that read `hf_config_short.max_seq_len`
            can exercise longer sequence lengths without modifying code.
    """
    hf_config_out = deepcopy(hf_config)
    hf_config_out.num_hidden_layers = getattr(request, "param", 1)
    max_seq_len_override = os.getenv("DEEPSEEK_MAX_SEQ_LEN_OVERRIDE")
    if max_seq_len_override is not None:
        hf_config_out.max_seq_len = int(max_seq_len_override)
    else:
        hf_config_out.max_seq_len = 128
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


def get_current_device_type() -> str:
    """
    Determine the current device type based on cluster type and device count.
    Honors MESH_DEVICE environment variable when set, falling back to hardware detection otherwise.

    Returns:
        str: One of "N150", "N300", "T3K", "TG", "DUAL", "QUAD"

    Raises:
        ValueError: If the device type cannot be determined from the current configuration
    """

    valid_system_names = get_valid_system_names()
    requested_system_name = os.getenv("MESH_DEVICE")
    if requested_system_name is None:
        raise ValueError(
            f"Invalid system name: {requested_system_name}. Must be one of {', '.join(valid_system_names)} or AUTO"
        )

    upper_name = requested_system_name.upper()
    if upper_name == "AUTO":
        system_name = automatically_detect_current_device_type()
        logger.warning(f"MESH_DEVICE was set to 'AUTO' - detected device type: '{system_name}'")
    else:
        system_name = upper_name

    if system_name not in valid_system_names:
        raise ValueError(f"Invalid system name: {system_name}. Must be one of {', '.join(valid_system_names)}")

    return system_name


def pytest_configure(config):
    # Register the requires_device marker
    config.addinivalue_line(
        "markers",
        "requires_device(device_types): mark test to run only on specified device types. "
        "device_types can be a single string or list of strings from: N150, N300, T3K, TG, DUAL, QUAD. "
        "Example: @pytest.mark.requires_device(['T3K', 'TG'])",
    )


def _get_timeout_override_marker(item, timeout_seconds: float) -> pytest.MarkDecorator:
    timeout_marker = item.get_closest_marker("timeout")
    if timeout_marker is None:
        return pytest.mark.timeout(timeout_seconds)

    args = list(timeout_marker.args)
    kwargs = dict(timeout_marker.kwargs)
    if args:
        args[0] = timeout_seconds
    elif "timeout" in kwargs:
        kwargs["timeout"] = timeout_seconds
    else:
        args = [timeout_seconds]

    return pytest.mark.timeout(*args, **kwargs)


def _maybe_extend_demo_timeout_for_weight_recalculation(config: pytest.Config, item: pytest.Item) -> None:
    if not config.getoption(RESET_WEIGHT_CACHE_OPTION):
        return
    if not item.nodeid.startswith(DEEPSEEK_DEMO_NODEID_PREFIX):
        return
    if "force_recalculate_weight_config" not in item.fixturenames:
        return

    timeout_marker = item.get_closest_marker("timeout")
    current_timeout = None
    if timeout_marker is not None:
        current_timeout = timeout_marker.args[0] if timeout_marker.args else timeout_marker.kwargs.get("timeout")
        try:
            current_timeout = float(current_timeout)
        except (TypeError, ValueError):
            current_timeout = None

    if current_timeout is not None and current_timeout >= RECALCULATE_WEIGHT_CACHE_DEMO_TIMEOUT_SECONDS:
        return

    # Cold-cache demo weight generation can take hours, so relax the timeout only
    # when the user explicitly asks pytest to recalculate cached weights.
    item.add_marker(
        _get_timeout_override_marker(item, RECALCULATE_WEIGHT_CACHE_DEMO_TIMEOUT_SECONDS),
        append=False,
    )


def _maybe_extend_test_timeout_for_multihost_weight_cache(current_device: str, item: pytest.Item) -> None:
    if current_device not in {"TG", "DUAL", "QUAD"}:
        return
    if not item.nodeid.startswith(DEEPSEEK_TESTS_NODEID_PREFIX):
        return
    if "force_recalculate_weight_config" not in item.fixturenames:
        return

    timeout_marker = item.get_closest_marker("timeout")
    current_timeout = None
    if timeout_marker is not None:
        current_timeout = timeout_marker.args[0] if timeout_marker.args else timeout_marker.kwargs.get("timeout")
        try:
            current_timeout = float(current_timeout)
        except (TypeError, ValueError):
            current_timeout = None

    if current_timeout is not None and current_timeout >= MULTIHOST_WEIGHT_CACHE_TEST_TIMEOUT_SECONDS:
        return

    # Shared-cache publication/visibility for multihost DeepSeek tests regularly
    # exceeds pytest's default 300s even when the code under test is healthy.
    item.add_marker(
        _get_timeout_override_marker(item, MULTIHOST_WEIGHT_CACHE_TEST_TIMEOUT_SECONDS),
        append=False,
    )


def pytest_collection_modifyitems(config, items):
    """
    Check if tests have requires_device marker and skip them during collection if current device doesn't match.
    """
    try:
        current_device = get_current_device_type()
        logger.debug(f"Current detected device type: {current_device}")
    except Exception as e:
        pytest.exit(f"Could not determine device type during collection: {e}", returncode=1)

    for item in items:
        _maybe_extend_demo_timeout_for_weight_recalculation(config, item)
        _maybe_extend_test_timeout_for_multihost_weight_cache(current_device, item)

        if item.nodeid.endswith("models/demos/deepseek_v3/tests/test_bitsculpt_trace.py::test_bitsculpt_trace"):
            if not config.getoption("bitsculpt_trace"):
                item.add_marker(
                    pytest.mark.skip(reason="BitSculpt trace harness is manual-only; pass --bitsculpt-trace to run it.")
                )
                continue

        marker = item.get_closest_marker("requires_device")
        if marker:
            # Get device_types from marker - can be single value or list
            device_types = marker.args[0] if marker.args else marker.kwargs.get("device_types", [])

            # Normalize to list
            if isinstance(device_types, str):
                device_types = [device_types]
            elif not isinstance(device_types, (list, tuple)):
                device_types = [device_types]

            # Check if current device is in the allowed list
            if current_device not in device_types:
                # Add skip marker during collection - this will make skips collapse like @pytest.mark.skip()
                skip_reason = (
                    f"Test case requires device type(s) {device_types}, but current device is {current_device}"
                )
                item.add_marker(pytest.mark.skip(reason=skip_reason))
