# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Shared pytest fixtures for the HunyuanImage-3.0 PCC tests.

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[5]
PCC_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PCC_DIR) not in sys.path:
    sys.path.insert(0, str(PCC_DIR))


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "unit_host: host-only unit tests (mock logits); excluded from on-device PCC sweeps",
    )
    config.addinivalue_line(
        "markers",
        "e2e_random_inputs: integration test with random latent/text embeds; opt-in via HY_RUN_E2E_RANDOM=1",
    )


def pytest_collection_modifyitems(items):
    """Production slow tests (32L load) exceed the global 300s pytest.ini timeout."""
    for item in items:
        if "slow" in item.keywords and not any(m.name == "timeout" for m in item.iter_markers()):
            item.add_marker(pytest.mark.timeout(10800))


@pytest.fixture(scope="function")
def device():
    """Function-scoped device — safe for single-device and mesh tests."""
    import ttnn

    dev = ttnn.open_device(device_id=0, l1_small_size=32768)
    yield dev
    ttnn.close_device(dev)
