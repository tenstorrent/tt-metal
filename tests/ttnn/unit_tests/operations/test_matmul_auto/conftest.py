# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn


@pytest.fixture(scope="module")
def device():
    """Shared single-device fixture for matmul auto-config tests."""
    device = ttnn.open_device(device_id=0)
    yield device
    ttnn.close_device(device)
