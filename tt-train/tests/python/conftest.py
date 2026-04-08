# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Pytest configuration for TTML Python tests."""

import sys
from pathlib import Path

import pytest

# Ensure local test helper modules (e.g., python_test_utils.py) are importable in CI.
TESTS_PYTHON_DIR = Path(__file__).resolve().parent
if str(TESTS_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_PYTHON_DIR))


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "requires_device: mark test as requiring a Tenstorrent device to run",
    )


def pytest_collection_modifyitems(config, items):
    """Skip device-requiring tests if no device is available."""
    # Check if device is available
    device_available = False
    try:
        import ttml

        auto_ctx = ttml.autograd.AutoContext.get_instance()
        auto_ctx.open_device()
        auto_ctx.close_device()
        device_available = True
    except Exception:  # noqa: S110 - intentionally ignore device check failures
        pass

    if not device_available:
        skip_device = pytest.mark.skip(reason="Tenstorrent device not available")
        for item in items:
            if "requires_device" in item.keywords:
                item.add_marker(skip_device)
