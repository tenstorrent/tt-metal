# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Golden-artifact fixtures; PADDLEOCR_VL_GOLDEN_DIR overrides ../demo/golden."""
from __future__ import annotations

import json
import os

import pytest
import torch

from models.demos.blackhole.paddleocr_vl.tests._artifacts import missing_artifact

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN_DIR = os.environ.get("PADDLEOCR_VL_GOLDEN_DIR", os.path.join(HERE, "..", "demo", "golden"))


@pytest.fixture(scope="session")
def golden_dir():
    """Resolved golden dir (skips if absent)."""
    if not os.path.isdir(GOLDEN_DIR):
        missing_artifact(f"golden dir not found: {GOLDEN_DIR} (set PADDLEOCR_VL_GOLDEN_DIR)")
    return GOLDEN_DIR


@pytest.fixture(scope="session")
def golden(golden_dir):
    """Callable ``golden(name)`` -> the loaded ``intermediates_{name}.pt`` dict."""

    def _load(name: str) -> dict:
        path = os.path.join(golden_dir, f"intermediates_{name}.pt")
        if not os.path.isfile(path):
            missing_artifact(f"golden tensor not found: {path}")
        return torch.load(path, weights_only=True)

    return _load


@pytest.fixture(scope="session")
def hf_goldens(golden_dir):
    """The ``hf_goldens.json`` manifest: ``{"samples": [...]}``."""
    path = os.path.join(golden_dir, "hf_goldens.json")
    if not os.path.isfile(path):
        missing_artifact(f"golden manifest not found: {path}")
    with open(path) as f:
        return json.load(f)
