# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


from types import ModuleType
from loguru import logger
import pytest
import ttnn

from tests.scripts.common import get_updated_device_params


# A fixture for device reset. Borrows same logic to close and open device from device fixture.
# Useful if test needs to reset device in middle (ie. between capture and replay of light metal binary)
@pytest.fixture(scope="function")
def reset_device(request, device_params):
    def _reset_device(device):
        """Closes and reopens the device to ensure a fresh state."""
        ttnn.DumpDeviceProfiler(device)
        ttnn.close_device(device)

        # Reopen a new device instance
        device_id = request.config.getoption("device_id")
        request.node.pci_ids = [ttnn.GetPCIeDeviceID(device_id)]

        num_devices = ttnn.GetNumPCIeDevices()
        assert device_id < num_devices, "CreateDevice not supported for non-mmio device"
        updated_device_params = get_updated_device_params(device_params)
        new_device = ttnn.CreateDevice(device_id=device_id, **updated_device_params)
        ttnn.SetDefaultDevice(device)
        return new_device

    yield _reset_device  # Provides function to test
