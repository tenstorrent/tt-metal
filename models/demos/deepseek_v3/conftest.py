# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0
import os
import time
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import torch
from loguru import logger
from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.deepseek_v3.tt.ccl import CCL
from models.demos.deepseek_v3.utils.test_utils import load_state_dict, system_name_to_mesh_shape
from tests.scripts.common import get_updated_device_params

RESET_WEIGHT_CACHE_OPTION = "--recalculate-weights"
USE_RANDOM_WEIGHTS_OPTION = "--use-random-weights"


# =============================================================================
# TIMING INSTRUMENTATION
# =============================================================================
@contextmanager
def timed_section(name: str):
    """Context manager to time a section of code."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    logger.info(f"⏱️  {name}: {elapsed:.3f}s")


class TestTimer:
    """Accumulator for timing multiple sections within a test."""

    def __init__(self):
        self.times = {}
        self._start_times = {}

    @contextmanager
    def time(self, name: str):
        """Context manager to time a named section."""
        start = time.perf_counter()
        yield
        elapsed = time.perf_counter() - start
        self.times[name] = self.times.get(name, 0) + elapsed
        logger.info(f"⏱️  {name}: {elapsed:.3f}s")

    def summary(self):
        """Print a summary of all timed sections."""
        if not self.times:
            return
        total = sum(self.times.values())
        logger.info(f"\n{'='*60}")
        logger.info(f"TIMING SUMMARY (total: {total:.3f}s)")
        logger.info(f"{'='*60}")
        for name, t in sorted(self.times.items(), key=lambda x: -x[1]):
            pct = (t / total) * 100 if total > 0 else 0
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            logger.info(f"  {bar} {pct:5.1f}% | {t:7.3f}s | {name}")
        logger.info(f"{'='*60}\n")


@pytest.fixture(scope="function")
def test_timer():
    """Fixture to provide a timer for tests."""
    timer = TestTimer()
    yield timer
    timer.summary()


# =============================================================================
# UNIFIED TEST CACHE MANAGER
# =============================================================================
@dataclass
class TestCacheManager:
    """Unified cache manager for all test resources."""

    # Weight caches
    dequantized_state_dicts: dict[str, dict[str, torch.Tensor]] = field(default_factory=dict)
    ttnn_weight_tensors: dict[str, ttnn.Tensor] = field(default_factory=dict)

    # Model caches
    reference_models: dict[str, torch.nn.Module] = field(default_factory=dict)

    # Mode flag
    use_random_weights: bool = False

    def get_dequantized_state_dict(
        self, module_path: str, state_dict: dict[str, torch.Tensor], hf_config: PretrainedConfig
    ) -> dict[str, torch.Tensor]:
        """Get dequantized state dict with session-level caching."""
        if module_path in self.dequantized_state_dicts:
            logger.info(f"Cache hit: dequantized weights for {module_path}")
            return self.dequantized_state_dicts[module_path]

        logger.info(f"Cache miss: dequantizing weights for {module_path}")
        from models.demos.deepseek_v3.utils.config_helpers import sub_state_dict
        from models.demos.deepseek_v3.utils.test_utils import dequantize_state_dict

        sub_dict = sub_state_dict(state_dict, module_path + ".")
        dequantized = dequantize_state_dict(sub_dict, hf_config)
        self.dequantized_state_dicts[module_path] = dequantized
        return dequantized

    def get_ttnn_weight(self, weight_path: Path, mesh_device: ttnn.Device, saved_weight: Any) -> ttnn.Tensor:
        """Get TTNN weight tensor with caching."""
        path_str = str(weight_path)
        if path_str in self.ttnn_weight_tensors:
            logger.info(f"Cache hit: TTNN weight {path_str}")
            return self.ttnn_weight_tensors[path_str]

        logger.info(f"Cache miss: loading TTNN weight {path_str}")
        from models.demos.deepseek_v3.utils.run_config import load_weight

        tensor = load_weight(saved_weight, mesh_device)
        self.ttnn_weight_tensors[path_str] = tensor
        return tensor

    def get_reference_model(
        self, model_class: type, hf_config: PretrainedConfig, module_path: str | None, cache_key: str
    ) -> torch.nn.Module:
        """Get reference model with caching."""
        if cache_key in self.reference_models:
            logger.info(f"Cache hit: reference model {cache_key}")
            return self.reference_models[cache_key]

        logger.info(f"Cache miss: creating reference model {cache_key}")
        # Check if model class needs layer_idx parameter
        import inspect

        sig = inspect.signature(model_class.__init__)
        if "layer_idx" in sig.parameters:
            # Try to extract layer_idx from cache_key if present
            layer_idx = 0  # Default
            if ":" in cache_key:
                parts = cache_key.split(":")
                for part in parts:
                    if part.isdigit():
                        layer_idx = int(part)
                        break
            model = model_class(hf_config, layer_idx=layer_idx).eval().to(torch.bfloat16)
        else:
            model = model_class(hf_config).eval().to(torch.bfloat16)
        self.reference_models[cache_key] = model
        return model

    def clear_ttnn_cache(self):
        """Clear TTNN weight cache to free memory."""
        for tensor in self.ttnn_weight_tensors.values():
            try:
                ttnn.deallocate(tensor)
            except Exception as e:
                logger.warning(f"Failed to deallocate tensor: {e}")
        self.ttnn_weight_tensors.clear()


def pytest_addoption(parser):
    parser.addoption(
        RESET_WEIGHT_CACHE_OPTION,
        action="store_true",
        help="Reset weight configs for tests",
    )
    parser.addoption(
        USE_RANDOM_WEIGHTS_OPTION,
        action="store_true",
        help="Use random weights for all tests (fastest, but no accuracy validation)",
    )


@pytest.fixture(scope="session")
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

    with timed_section("FIXTURE: mesh_device setup"):
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

        logger.debug(
            f"Mesh device with {mesh_device.get_num_devices()} devices is created with shape {mesh_device.shape}"
        )

    yield mesh_device

    with timed_section("FIXTURE: mesh_device teardown"):
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
    with timed_section("FIXTURE: hf_config loading"):
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    return config


@pytest.fixture(scope="session")
def state_dict(model_path, request):
    # Skip loading state dict if using random weights
    use_random = request.config.getoption(USE_RANDOM_WEIGHTS_OPTION, default=False)
    if use_random:
        logger.info("Skipping state_dict loading (using random weights)")
        yield {}
        return

    with timed_section("FIXTURE: state_dict loading"):
        sd = load_state_dict(model_path, "")
    yield sd


@pytest.fixture(scope="function", autouse=True)
def clear_state_dict_cache(state_dict):
    """
    Clear the LazyStateDict cache after each test to prevent memory accumulation.
    This preserves file handles (mmap benefits) while freeing tensor memory.
    """
    yield
    if hasattr(state_dict, "clear_cache"):
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
    with timed_section("FIXTURE: ccl creation"):
        ccl_instance = CCL(mesh_device)
    return ccl_instance


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


@pytest.fixture(scope="session")
def test_cache_manager(request) -> TestCacheManager:
    """
    Unified cache manager for all test resources.

    Supports two modes:
    - Random weights: use_random_weights=True (fastest, no I/O)
    - Real weights: use_random_weights=False (cached setup)
    """
    use_random = request.config.getoption(USE_RANDOM_WEIGHTS_OPTION, default=False)
    manager = TestCacheManager(use_random_weights=use_random)
    yield manager

    # Cleanup: deallocate TTNN tensors
    manager.clear_ttnn_cache()
