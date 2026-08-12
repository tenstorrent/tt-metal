# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""One device for the whole correctness gate (this dir sits outside tests/, so it does
not pick up the repo's ttnn conftest)."""

import pytest
import ttnn


@pytest.fixture(scope="session")
def device():
    dev = ttnn.open_device(device_id=0)
    try:
        yield dev
    finally:
        ttnn.close_device(dev)
