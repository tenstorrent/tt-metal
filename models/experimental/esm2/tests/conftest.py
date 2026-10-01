# SPDX-License-Identifier: MIT
"""Shared fixtures for ESM-2 tests: checkpoint path, model config, device weights."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from models.experimental.esm2.tt.esm2.config import Esm2TTConfig
from models.experimental.esm2.tt.esm2.loader import load_canonical_weights


def _checkpoint_root() -> Path:
    """Resolve the checkpoint directory from env or the standard CI cache."""
    env = os.environ.get("ESM2_WEIGHTS") or os.environ.get("MODEL_DIR")
    if env:
        return Path(env)
    ci = Path("/mnt/MLPerf/huggingface/hub/models--facebook--esm2_t33_650M_UR50D/snapshots")
    if ci.is_dir():
        snapshots = sorted(ci.iterdir())
        if snapshots:
            return snapshots[-1]
    pytest.skip("Set ESM2_WEIGHTS to the checkpoint directory (config.json + model.safetensors)")


@pytest.fixture(scope="session")
def checkpoint(request):
    return _checkpoint_root()


@pytest.fixture(scope="session")
def config(checkpoint):
    return Esm2TTConfig.from_json_file(str(checkpoint / "config.json"))


@pytest.fixture(scope="session")
def weights(config, checkpoint):
    return load_canonical_weights(str(checkpoint), config)


@pytest.fixture(scope="module")
def device():
    """Open (and close) a single TT device for the test module."""
    import ttnn

    dev = ttnn.open_device(device_id=0)
    yield dev
    ttnn.close_device(dev)
