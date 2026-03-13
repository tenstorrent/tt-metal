# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""conftest.py — pytest fixtures for Kimi K2.5 tests.

This mirrors the structure of models/demos/deepseek_v3/conftest.py, adapted for
Kimi K2.5. Key differences:

* ``hf_config`` loads from ``KIMI_HF_MODEL`` (real weights) if set, otherwise
  falls back to :class:`~models.demos.kimi_k25.utils.config_adapter.KimiK25Config`
  (the hardcoded fixture — no HF download needed for unit tests).
* ``cache_path`` defaults to ``generated/kimi_k25_test_cache`` or
  ``KIMI_K25_CACHE`` env var.
* ``mesh_device`` uses the same ``MESH_DEVICE`` env-var-driven logic as DSV3.
* ``state_dict`` loads via :class:`KimiLazyStateDict` (INT4 → BF16 transparent).
"""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from loguru import logger

from models.demos.kimi_k25.utils.config_adapter import KimiK25Config

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_kimi_hf_config_fixture() -> KimiK25Config:
    """Return a KimiK25Config usable as ``hf_config`` in all DSV3 module tests.

    ``KimiK25Config`` is a plain Python dataclass.  DSV3 modules access config
    fields by attribute (``hf_config.n_group``, etc.) which works identically on
    a dataclass.  No ``transformers.PretrainedConfig`` is required for unit tests.
    """
    return KimiK25Config.from_fixture()


def _make_kimi_hf_config_from_real(model_path: Path) -> object:
    """Load HF config from a local Kimi K2.5 checkpoint directory.

    Returns the raw transformers config (or a validated KimiK25Config
    derived from it).  Falls back to fixture if load fails.
    """
    try:
        from transformers import AutoConfig

        hf_cfg = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)
        # Validate and log — raises ValueError on arch mismatch
        kimi_cfg = KimiK25Config.from_hf_config(hf_cfg)
        logger.info(f"Kimi conftest: loaded real HF config from {model_path}\n{kimi_cfg.summary()}")
        return kimi_cfg
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            f"Kimi conftest: failed to load real HF config from {model_path!r}: {exc}. " "Falling back to fixture."
        )
        return _make_kimi_hf_config_fixture()


# ---------------------------------------------------------------------------
# Session-scoped fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def model_path() -> Path:
    """Path to a local Kimi K2.5 checkpoint, or None if not available."""
    raw = os.getenv("KIMI_HF_MODEL")
    if raw is None:
        return None
    p = Path(raw).resolve()
    if not p.exists():
        logger.warning(f"Kimi conftest: KIMI_HF_MODEL={raw!r} does not exist.")
        return None
    return p


@pytest.fixture(scope="session")
def hf_config(model_path: Path) -> KimiK25Config:
    """Kimi K2.5 architecture config.

    Uses real HF config when ``KIMI_HF_MODEL`` is set; otherwise uses the
    hardcoded fixture (suitable for all unit tests that don't need real weights).
    """
    if model_path is not None:
        return _make_kimi_hf_config_from_real(model_path)
    logger.info("Kimi conftest: KIMI_HF_MODEL not set — using hardcoded fixture config.")
    return _make_kimi_hf_config_fixture()


@pytest.fixture(scope="session")
def hf_config_single_layer(hf_config: KimiK25Config) -> KimiK25Config:
    """One-layer Kimi config for fast single-layer tests."""
    cfg = deepcopy(hf_config)
    cfg.num_hidden_layers = 1
    return cfg


@pytest.fixture(scope="session")
def cache_path() -> Path:
    """Directory for converted TTNN weight tensors."""
    try:
        default = f"/localdev/{os.getlogin()}/kimi-k25-test-cache"
    except OSError:
        default = "/proj_sw/user_dev/kimi-k25-test-cache"
    return Path(os.getenv("KIMI_K25_CACHE", default))


@pytest.fixture(scope="session")
def state_dict(model_path: Path):
    """Kimi K2.5 lazy state dict (INT4 → BF16 transparent dequant).

    Skips the test if ``KIMI_HF_MODEL`` is not set.
    """
    if model_path is None:
        pytest.skip("KIMI_HF_MODEL not set — skipping real-weight test")
    from models.demos.kimi_k25.utils.weight_loader import KimiLazyStateDict

    return KimiLazyStateDict(model_path)


@pytest.fixture(scope="function", autouse=True)
def clear_state_dict_cache(request):
    """Clear KimiLazyStateDict tensor cache after each test to prevent OOM."""
    if "state_dict" not in request.fixturenames:
        yield
        return
    sd = request.getfixturevalue("state_dict")
    yield
    if hasattr(sd, "clear_cache"):
        sd.clear_cache()


# ---------------------------------------------------------------------------
# Function-scoped fixtures (device / determinism)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def force_recalculate_weight_config():
    return False


@pytest.fixture(scope="function")
def set_deterministic_env():
    """Seed torch and enable deterministic algorithms for reproducible tests."""
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)


# ---------------------------------------------------------------------------
# Mesh device — mirrors DSV3 conftest pattern
# ---------------------------------------------------------------------------


@pytest.fixture(scope="function")
def mesh_device(request):
    """TTNN mesh device fixture.

    Reads ``MESH_DEVICE`` env var (T3K / TG / DUAL / QUAD / AUTO).
    Skips if ttnn is not available (CPU-only environments).
    """
    try:
        import ttnn
    except ImportError:
        pytest.skip("ttnn not available — skipping hardware test")

    from tests.scripts.common import get_updated_device_params

    device_params = getattr(request, "param", {})
    requested_name = os.getenv("MESH_DEVICE")
    if requested_name is None:
        pytest.skip("MESH_DEVICE not set — skipping hardware test")

    from models.demos.deepseek_v3.utils.test_utils import system_name_to_mesh_shape

    try:
        mesh_shape = system_name_to_mesh_shape(requested_name.upper())
    except Exception as exc:
        pytest.skip(f"Unsupported MESH_DEVICE={requested_name!r}: {exc}")

    request.node.pci_ids = ttnn.get_pcie_device_ids()
    updated_params = get_updated_device_params(device_params)
    fabric_config = updated_params.pop("fabric_config", None)
    if fabric_config:
        ttnn.set_fabric_config(fabric_config)
    updated_params.setdefault("mesh_shape", mesh_shape)

    mesh = ttnn.open_mesh_device(**updated_params)
    logger.debug(f"Kimi conftest: mesh_device {mesh.shape} ({mesh.get_num_devices()} devices)")

    yield mesh

    for sub in mesh.get_submeshes():
        ttnn.close_mesh_device(sub)
    ttnn.close_mesh_device(mesh)
    if fabric_config:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    del mesh


@pytest.fixture
def ccl(mesh_device):
    """CCL instance for distributed tests."""
    from models.demos.deepseek_v3.tt.ccl import CCL

    return CCL(mesh_device)
