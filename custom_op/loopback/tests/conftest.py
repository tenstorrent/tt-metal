# SPDX-License-Identifier: Apache-2.0
import pytest
import ttnn


@pytest.fixture(scope="session")
def device():
    dev = ttnn.open_device(device_id=0)
    yield dev
    ttnn.close_device(dev)
