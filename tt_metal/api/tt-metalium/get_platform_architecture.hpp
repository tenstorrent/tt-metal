// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdlib>

#include "tt_backend_api_types.hpp"
#include "assert.hpp"
#include "umd/device/pci_device.hpp"
#include "umd/device/tt_soc_descriptor.h"

namespace tt::tt_metal {

/**
 * @brief Detects the platform architecture based on the environment or hardware.
 *
 * This function determines the platform architecture by inspecting the environment
 * variables or available physical devices. If the environment variable
 * `TT_METAL_SIMULATOR_EN` is set, the architecture is retrieved from the
 * `ARCH_NAME` environment variable. Otherwise, the architecture is deduced
 * by detecting available physical devices.
 *
 * @return tt::ARCH The detected platform architecture. Returns tt::ARCH::Invalid
 *                  if no valid architecture could be detected.
 *
 * @note
 * - If the system is in simulation mode (`TT_METAL_SIMULATOR_EN` is set),
 *   the `ARCH_NAME` environment variable must be defined.
 * - A fatal error occurs if multiple devices are detected with conflicting
 *   architectures.
 *
 * @exception std::runtime_error Throws a fatal error if:
 * - `ARCH_NAME` is not set when `TT_METAL_SIMULATOR_EN` is enabled.
 * - Multiple devices with inconsistent architectures are detected.
 *
 * Example usage:
 * @code
 * #include "tt_backend_api_types.hpp"
 *
 * tt::ARCH arch = tt::tt_metal::get_platform_architecture();
 * if (arch == tt::ARCH::Invalid) {
 *     std::cerr << "Failed to detect architecture!" << std::endl;
 * } else {
 *     std::cout << "Detected architecture: " << tt::arch_to_str(arch) << std::endl;
 * }
 * @endcode
 *
 * @see tt::get_arch_from_string
 * @see PCIDevice::enumerate_devices_info
 */
inline tt::ARCH get_platform_architecture() {
    auto arch = tt::ARCH::Invalid;
    if (std::getenv("TT_METAL_SIMULATOR_EN")) {
        auto arch_env = std::getenv("ARCH_NAME");
        TT_FATAL(arch_env, "ARCH_NAME env var needed for VCS");
        arch = tt::get_arch_from_string(arch_env);
    } else {

        // Issue tt_umd#361: tt_ClusterDescriptor::create() won't work here.
        // This map holds PCI info for each mmio chip.
        auto devices_info = PCIDevice::enumerate_devices_info();
        if (devices_info.size() > 0) {
            arch = devices_info.begin()->second.get_arch();
            for (auto &[device_id, device_info] : devices_info) {
                tt::ARCH detected_arch = device_info.get_arch();
                TT_FATAL(
                    arch == detected_arch,
                    "Expected all devices to be {} but device {} is {}",
                    tt::arch_to_str(arch),
                    device_id,
                    tt::arch_to_str(detected_arch));
            }
        }
    }

    return arch;
}

}  // namespace tt::tt_metal
