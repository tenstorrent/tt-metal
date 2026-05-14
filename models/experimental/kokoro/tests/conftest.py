# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Device fixtures for Kokoro TTNN tests under ``models/experimental/kokoro/tests``."""

from __future__ import annotations

import pytest

import ttnn


@pytest.fixture(scope="function")
def device():
    dev = ttnn.open_device(device_id=0, l1_small_size=24576)
    yield dev
    ttnn.close_device(dev)
