# SPDX-FileCopyrightText: © 2026 Abror Shopulatov

# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures. Paths come from PARAKEET_WEIGHTS (default /weights) and PARAKEET_INPUT (default /input).

Device tests skip cleanly when ttnn is not importable or no TT device is present; CPU tests skip
when the pinned checkpoint is not available.
"""

import json
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

WEIGHTS = os.environ.get("PARAKEET_WEIGHTS", "/weights")
INPUT = os.environ.get("PARAKEET_INPUT", "/input")


def pytest_configure(config):
    config.addinivalue_line("markers", "device: needs a TT device (skipped automatically when none is present)")


def _num_tt_devices():
    try:
        import ttnn
    except Exception:
        return 0
    try:
        return int(ttnn.GetNumAvailableDevices())
    except Exception:
        return 0


@pytest.fixture(scope="session")
def weights_path():
    if not os.path.exists(os.path.join(WEIGHTS, "config.json")):
        pytest.skip(f"pinned checkpoint not found at {WEIGHTS} (set PARAKEET_WEIGHTS)")
    return WEIGHTS


@pytest.fixture(scope="session")
def hf_config(weights_path):
    with open(os.path.join(weights_path, "config.json")) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def reference(weights_path):
    pytest.importorskip("transformers")
    from reference import ParakeetReference

    return ParakeetReference(weights_path)


@pytest.fixture(scope="session")
def device():
    if _num_tt_devices() == 0:
        pytest.skip("no TT device available")
    from tt import DEVICE_OPTIONS

    import ttnn

    dev = ttnn.open_device(device_id=0, **DEVICE_OPTIONS)
    yield dev
    ttnn.close_device(dev)


@pytest.fixture(scope="session")
def tt_model(weights_path, hf_config, device):
    from tt import create_backend

    return create_backend(weights_path, hf_config, device, precision="bf16")
