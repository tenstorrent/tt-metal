# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Pytest fixtures for DiffusionDrive TTNN bring-up tests.

Device fixture follows the granite_ttm_r1 pattern (single CQ, default params).
"""

from __future__ import annotations

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
    dev = ttnn.open_device(device_id=0)
    yield dev
    ttnn.close_device(dev)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def model_config():
    from models.demos.diffusion_drive.tt.config import ModelConfig

    cfg = ModelConfig()
    anchors = _DATA_DIR / "kmeans_navsim_traj_20.npy"
    if anchors.exists():
        cfg.plan_anchor_path = str(anchors)
    return cfg


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
