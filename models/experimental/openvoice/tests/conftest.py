# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI
# SPDX-License-Identifier: Apache-2.0

"""Shared pytest fixtures for OpenVoice tests."""

import pytest

import ttnn


@pytest.fixture(scope="module")
def device():
    """Get TTNN device."""
    try:
        dev = ttnn.open_device(device_id=0)
        yield dev
        ttnn.close_device(dev)
    except Exception as e:
        pytest.skip(f"Could not open TTNN device: {e}")
