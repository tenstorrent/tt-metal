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
