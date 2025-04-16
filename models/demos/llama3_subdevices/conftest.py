# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import gc

import pytest


@pytest.fixture(autouse=True)
def ensure_gc():
    gc.collect()


@pytest.fixture(autouse=True)
def ensure_devices(ensure_devices_tg):
    pass
