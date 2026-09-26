# SPDX-FileCopyrightText: © 2026 Abror Shopulatov

# SPDX-License-Identifier: Apache-2.0

"""Opt-in fixtures; collecting the tests never opens hardware.

CHRONOS2_CHECKPOINT   directory with config.json and model.safetensors
CHRONOS2_TEST_DEVICE  device id to run the TTNN tests on
"""

import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def checkpoint_dir():
    path = os.environ.get("CHRONOS2_CHECKPOINT")
    if not path or not (Path(path) / "config.json").exists() or not (Path(path) / "model.safetensors").exists():
        pytest.skip("set CHRONOS2_CHECKPOINT to a Chronos-2 checkpoint directory")
    return path


@pytest.fixture(scope="session")
def tt_device():
    """One TT device for the session, opened with the package DEVICE_OPTIONS."""
    device_id = os.environ.get("CHRONOS2_TEST_DEVICE")
    if device_id is None:
        pytest.skip("set CHRONOS2_TEST_DEVICE to opt in to TT inference")
    ttnn = pytest.importorskip("ttnn")
    from models.experimental.chronos2.tt import DEVICE_OPTIONS

    device = ttnn.open_device(device_id=int(device_id), **DEVICE_OPTIONS)
    yield device
    ttnn.close_device(device)
