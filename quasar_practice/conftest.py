# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest

import ttnn


@pytest.fixture(scope="session")
def device():
    """One device for the whole session, overriding the repo's per-test `device` fixture.

    On the craq-sim Quasar simulator, closing the device and opening it again in the same process
    hangs the first op after the reopen, so the per-test fixture hangs every test after the first.
    """
    dev = ttnn.open_device(device_id=0)
    yield dev
    ttnn.close_device(dev)
