# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Pytest configuration for rms_norm golden tests."""

import pytest
import ttnn


def pytest_configure(config):
    config.addinivalue_line("markers", "quick: fast tests for smoke checking (<5s each)")
    config.addinivalue_line("markers", "standard: standard shape coverage tests")
    config.addinivalue_line("markers", "large: large tensor tests (may be slow)")
    config.addinivalue_line("markers", "stress: stress tests with extreme values")
    config.addinivalue_line("markers", "validation: input validation tests")


@pytest.fixture(scope="module")
def device():
    device = ttnn.open_device(device_id=0)
    yield device
    ttnn.close_device(device)
