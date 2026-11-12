# SPDX-FileCopyrightText: © 2026 Abror Shopulatov

# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for Parakeet tests.

Checkpoint resolved from PARAKEET_WEIGHTS env var or the CI HF cache.
Device tests skip cleanly when ttnn is not importable or no TT device is present.
"""

import json
import os

import numpy as np
import pytest


def _resolve_checkpoint():
    env = os.environ.get("PARAKEET_WEIGHTS")
    if env and os.path.exists(os.path.join(env, "config.json")):
        return env
    ci = "/mnt/MLPerf/huggingface/hub/models--nvidia--parakeet-tdt-0.6b-v3/snapshots"
    if os.path.isdir(ci):
        for snap in sorted(os.listdir(ci)):
            path = os.path.join(ci, snap)
            if os.path.exists(os.path.join(path, "config.json")):
                return path
    return None


WEIGHTS = _resolve_checkpoint()


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


def make_deterministic_mels():
    """Synthetic mel spectrograms for numerics tests (no real speech needed)."""
    rng = np.random.default_rng(0)
    mel = rng.standard_normal((2, 300, 128)).astype(np.float32)
    lens = np.array([300, 181], dtype=np.int64)
    mel[1, 181:] = 0.0
    return {"synthetic_b2": (mel, lens)}


@pytest.fixture(scope="session")
def weights_path():
    if WEIGHTS is None:
        pytest.skip("checkpoint not found (set PARAKEET_WEIGHTS)")
    return WEIGHTS


@pytest.fixture(scope="session")
def hf_config(weights_path):
    with open(os.path.join(weights_path, "config.json")) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def reference(weights_path):
    pytest.importorskip("transformers")
    from models.experimental.parakeet.reference.torch_parakeet import ParakeetReference

    return ParakeetReference(weights_path)


@pytest.fixture(scope="session")
def device():
    if _num_tt_devices() == 0:
        pytest.skip("no TT device available")
    import ttnn
    from models.experimental.parakeet.tt import DEVICE_OPTIONS

    dev = ttnn.open_device(device_id=0, **DEVICE_OPTIONS)
    yield dev
    ttnn.close_device(dev)


@pytest.fixture(scope="session")
def tt_model(weights_path, hf_config, device):
    from models.experimental.parakeet.tt import create_backend

    return create_backend(weights_path, hf_config, device, precision="bf16")
