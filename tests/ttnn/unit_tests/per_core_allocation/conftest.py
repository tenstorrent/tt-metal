# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Local conftest that enables HYBRID allocator mode for all tests in this directory.
The env var is set before device creation and removed on teardown.
"""

import os
import pytest
import ttnn


@pytest.fixture(scope="module")
def device(request):
    """Module-scoped device with HYBRID allocator mode enabled via env var."""
    os.environ["TT_METAL_ALLOCATOR_MODE_HYBRID"] = "1"

    original_default_device = ttnn.GetDefaultDevice()

    device_id = request.config.getoption("device_id")
    device = ttnn.CreateDevice(device_id=device_id)
    ttnn.SetDefaultDevice(device)

    yield device

    ttnn.SetDefaultDevice(original_default_device)
    ttnn.close_device(device)
    os.environ.pop("TT_METAL_ALLOCATOR_MODE_HYBRID", None)


@pytest.fixture(scope="function")
def mesh_device():
    """Function-scoped mesh device with HYBRID allocator mode for multi-device tests."""
    os.environ["TT_METAL_ALLOCATOR_MODE_HYBRID"] = "1"

    num_devices = ttnn.get_num_devices()
    if num_devices < 2:
        pytest.skip("Multi-device per-core allocation tests require at least 2 devices")
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, min(num_devices, 2)))
    yield mesh
    ttnn.close_mesh_device(mesh)
    os.environ.pop("TT_METAL_ALLOCATOR_MODE_HYBRID", None)
