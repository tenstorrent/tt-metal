# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for the PaddleOCR-VL PCC / accuracy tests.

The tests compare the ttnn port against CPU-reference "golden" tensors and decoded
text captured by ``generate_goldens.py`` (gitignored, kept outside the repo). Both
are provided via env vars and the fixtures below skip cleanly when they are
absent, so the suite is safe to collect on a machine without the data (e.g. a
generic CI runner) and runs for real on a P150 box that has them staged. Set
``PADDLEOCR_VL_REQUIRE_ARTIFACTS=1`` on an unattended run to turn those skips into
failures on a runner that is supposed to have the data.

Env vars (all optional; default matches the layout ``generate_goldens.py`` writes):
  PADDLEOCR_VL_GOLDEN_DIR   dir with intermediates_*.pt / hf_goldens.json
                            (default: ../demo/golden next to this file)

  PADDLEOCR_VL_REQUIRE_ARTIFACTS=1  turn every "artifact missing" skip into a
                            FAILURE. Set this on any unattended run, so a
                            staging/dependency/API break fails loudly instead of
                            reporting a green skip. Regenerate with
                            generate_goldens.py.
"""
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
        return torch.load(path, weights_only=False)

    return _load


@pytest.fixture(scope="session")
def hf_goldens(golden_dir):
    """The ``hf_goldens.json`` manifest: ``{"samples": [...]}``."""
    path = os.path.join(golden_dir, "hf_goldens.json")
    if not os.path.isfile(path):
        missing_artifact(f"golden manifest not found: {path}")
    with open(path) as f:
        return json.load(f)
