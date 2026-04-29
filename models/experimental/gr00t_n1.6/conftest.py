# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Pytest configuration for GR00T N1.6 tests."""

import os
import pytest


def pytest_collection_modifyitems(config, items):
    """Mark tests that require a Tenstorrent device."""
    for item in items:
        if "device" in item.fixturenames:
            item.add_marker(pytest.mark.requires_device)


@pytest.fixture(scope="session")
def model_id():
    """HuggingFace model ID for GR00T N1.6."""
    return os.environ.get("GROOT_MODEL_ID", "nvidia/GR00T-N1.6-3B")
