# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Pytest configuration for ttml smoke tests.

Smoke tests are designed to be fast (<30s total) and verify core functionality.
"""

import pytest


def pytest_configure(config):
    """Register custom markers for smoke tests."""
    config.addinivalue_line(
        "markers",
        "smoke: mark test as a smoke test (fast, core functionality)",
    )
    config.addinivalue_line(
        "markers",
        "requires_device: mark test as requiring a Tenstorrent device to run",
    )


def pytest_collection_modifyitems(config, items):
    """Skip device-requiring tests if no device is available."""
    device_available = False
    try:
        import ttml

        auto_ctx = ttml.autograd.AutoContext.get_instance()
        auto_ctx.open_device()
        auto_ctx.close_device()
        device_available = True
    except Exception:
        pass

    if not device_available:
        skip_device = pytest.mark.skip(reason="Tenstorrent device not available")
        for item in items:
            if "requires_device" in item.keywords:
                item.add_marker(skip_device)


@pytest.fixture(scope="session")
def auto_context():
    """Provide a shared AutoContext for device tests."""
    import ttml

    ctx = ttml.autograd.AutoContext.get_instance()
    ctx.open_device()
    yield ctx
    ctx.close_device()


@pytest.fixture(autouse=True)
def reset_graph_after_test():
    """Reset the autograd graph after each test to prevent state leakage."""
    yield
    try:
        import ttml

        ttml.autograd.AutoContext.get_instance().reset_graph()
    except Exception:
        pass
