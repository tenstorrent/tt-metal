// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <gtest/gtest.h>
#include <cstdlib>
#include <string>
#include <iostream>

namespace tt::tt_metal::distributed {

namespace {

TEST(VisibleDevicesMPTest, ValidateDeviceRatio) {
    // Validate the 2:1 ratio of total devices to PCIe devices
    size_t num_available = tt::tt_metal::GetNumAvailableDevices();
    size_t num_pcie = tt::tt_metal::GetNumPCIeDevices();

    // Each PCIe device should correspond to 2 chips (one with PCIe, one without)
    if (num_pcie > 0) {
        EXPECT_EQ(num_available, num_pcie * 2)
            << "Total available devices should be exactly 2x the number of PCIe devices";
    }

    // Log the device configuration for debugging
    const char* visible_devices_env = std::getenv("TT_METAL_VISIBLE_DEVICES");
    std::string visible_devices = visible_devices_env ? visible_devices_env : "<not set>";

    std::cout << "Device configuration summary:" << std::endl;
    std::cout << "  TT_METAL_VISIBLE_DEVICES: " << visible_devices << std::endl;
    std::cout << "  PCIe devices: " << num_pcie << std::endl;
    std::cout << "  Total available devices: " << num_available << std::endl;
    std::cout << "  Ratio: " << (num_pcie > 0 ? static_cast<double>(num_available) / num_pcie : 0) << std::endl;
}

}  // namespace

}  // namespace tt::tt_metal::distributed
