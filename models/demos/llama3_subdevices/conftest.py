# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import pytest
import gc
import ttnn


@pytest.fixture(autouse=True)
def ensure_gc():
    gc.collect()


@pytest.fixture(autouse=True)
def ensure_devices():
    device_ids = ttnn.get_device_ids()
    assert len(device_ids) == 32, f"Expected 32 devices, got {len(device_ids)}"
