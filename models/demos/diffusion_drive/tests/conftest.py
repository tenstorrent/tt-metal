# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Pytest fixtures for DiffusionDrive TTNN bring-up tests.

Device fixture follows the granite_ttm_r1 pattern (single CQ, default params).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Generator

import pytest
import torch

import ttnn

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_GOLDEN_DIR = Path(__file__).resolve().parent.parent / "reference" / "golden"


# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def fixed_seed() -> None:
    """Pin random seeds for reproducible PCC values."""
    import numpy as np

    torch.manual_seed(42)
    np.random.seed(42)


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def device() -> Generator[ttnn.Device, None, None]:
    # l1_small_size=32768 is required by ttnn.conv2d (TTNN BasicBlock tests).
    # Setting it session-wide is harmless for tests that don't use conv2d.
    dev = ttnn.open_device(device_id=0, l1_small_size=32768)
    yield dev
    ttnn.close_device(dev)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def model_config():
    from models.demos.diffusion_drive.tt.config import ModelConfig

    cfg = ModelConfig()
    # Resolve the kmeans plan-anchor asset to an EXISTING file, else None so tests
    # that need it skip cleanly. ModelConfig.plan_anchor_path defaults to a path in
    # the gitignored data/ dir; if that file is absent the default is a stale,
    # non-None path that would otherwise fail mid-build with FileNotFoundError
    # (defeating each test's `if plan_anchor_path is None: skip` guard).
    # DD_DATA_ROOT mirrors the eval-asset layout (default /mnt/diffusion-drive); see README.
    _dd_data_root = os.environ.get("DD_DATA_ROOT", "/mnt/diffusion-drive")
    candidates = [
        os.environ.get("DD_ANCHOR_PATH"),  # explicit override
        _DATA_DIR / "kmeans_navsim_traj_20.npy",  # scripts/prepare_assets.py target
        f"{_dd_data_root}/resnet34/kmeans_navsim_traj_20.npy",  # staged eval-asset layout
        cfg.plan_anchor_path,  # ModelConfig default
    ]
    cfg.plan_anchor_path = next((str(p) for p in candidates if p and Path(p).exists()), None)
    return cfg


# ---------------------------------------------------------------------------
# CI asset gate
# ---------------------------------------------------------------------------


def _assets_required() -> bool:
    """CI sets DD_REQUIRE_ASSETS=1 once the model-test job has staged the assets."""
    return os.environ.get("DD_REQUIRE_ASSETS", "").strip().lower() in ("1", "true", "yes", "on")


def _checkpoint_path() -> str | None:
    data_root = os.environ.get("DD_DATA_ROOT", "/mnt/diffusion-drive")
    candidates = [
        os.environ.get("DD_CHECKPOINT_PATH"),  # explicit override
        _DATA_DIR / "diffusiondrive_navsim.pth",  # scripts/prepare_assets.py target
        f"{data_root}/weights/diffusiondrive_navsim_88p1_PDMS.pth",  # staged eval layout
    ]
    return next((str(p) for p in candidates if p and Path(p).exists()), None)


@pytest.fixture(scope="session")
def checkpoint_path() -> str | None:
    return _checkpoint_path()


@pytest.fixture(scope="session", autouse=True)
def require_ci_assets(model_config) -> None:
    """Under DD_REQUIRE_ASSETS=1 a missing asset fails the run instead of skipping it.

    Each asset-dependent test guards itself with ``pytest.skip`` so a plain checkout
    (no 730 MB checkpoint) stays usable.  That is the wrong behaviour in CI, where a
    gate that skips itself gates nothing — so the model-test job stages the assets
    with ``scripts/prepare_assets.py`` and sets this flag, and anything still missing
    is a hard failure here rather than a green run full of skips.
    """
    if not _assets_required():
        return
    missing = []
    if model_config.plan_anchor_path is None:
        missing.append("plan anchors (data/kmeans_navsim_traj_20.npy)")
    if _checkpoint_path() is None:
        missing.append("checkpoint (data/diffusiondrive_navsim.pth)")
    if missing:
        pytest.fail(
            "DD_REQUIRE_ASSETS=1 but these assets are absent: "
            + ", ".join(missing)
            + " — the CI job must run models/demos/diffusion_drive/scripts/prepare_assets.py first",
            pytrace=False,
        )


# ---------------------------------------------------------------------------
# Golden tensor loader
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def golden(request) -> dict:
    """Load golden tensors from reference/golden/<name>.pt if available."""
    name = getattr(request, "param", None)
    if name is None:
        return {}
    path = _GOLDEN_DIR / f"{name}.pt"
    if not path.exists():
        pytest.skip(f"Golden file not found: {path}")
    return torch.load(path, weights_only=True)
