# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest

import ttnn


@pytest.fixture(scope="module")
def device():
    device = ttnn.open_device(device_id=0)
    yield device
    ttnn.close_device(device)


@pytest.fixture(scope="module")
def device_2cq():
    """Device opened with 2 command queues for double-buffering tests."""
    device = ttnn.open_device(device_id=0, num_command_queues=2)
    yield device
    ttnn.close_device(device)
