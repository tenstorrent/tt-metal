# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Pytest fixtures for RVC TTNN tests."""

import pytest
import ttnn


# L1 small size needed for conv2d ops (same as Whisper: 16384)
RVC_L1_SMALL_SIZE = 16384


@pytest.fixture(scope="session")
def device():
    """Open a TT device for the entire test session with L1 small buffer."""
    dev = ttnn.open_device(device_id=0, l1_small_size=RVC_L1_SMALL_SIZE)
    yield dev
    ttnn.close_device(dev)
