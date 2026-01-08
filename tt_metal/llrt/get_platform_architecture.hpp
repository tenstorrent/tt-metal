// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdlib>
#include <mutex>

#include <tt_stl/assert.hpp>
#include "llrt/rtoptions.hpp"
#include "tracy/Tracy.hpp"
#include <umd/device/pcie/pci_device.hpp>
#include <umd/device/soc_descriptor.hpp>
#include <umd/device/simulation/simulation_chip.hpp>

namespace tt::tt_metal {

inline tt::ARCH get_physical_architecture() {
    ZoneScoped;
    static tt::ARCH current_arch = tt::ARCH::Invalid;
    static std::once_flag current_arch_once_flag;
    std::call_once(current_arch_once_flag, []() {
        // Issue tt_umd#361: ClusterDescriptor::create() won't work here.
        // This map holds PCI info for each mmio chip.
        auto devices_info = umd::PCIDevice::enumerate_devices_info();
        if (!devices_info.empty()) {
            current_arch = devices_info.begin()->second.get_arch();
            for (auto& [device_id, device_info] : devices_info) {
                tt::ARCH detected_arch = device_info.get_arch();
                TT_FATAL(
                    current_arch == detected_arch,
                    "Expected all devices to be {} but device {} is {}",
                    tt::arch_to_str(current_arch),
                    device_id,
                    tt::arch_to_str(detected_arch));
            }
        }
    });
    return current_arch;
}

/**
 * @brief Detects the platform architecture based on the environment or hardware.
 *
 * This function determines the platform architecture by inspecting the environment
 * variables or available physical devices. If the environment variable
 * `TT_METAL_SIMULATOR` is set, the architecture is retrieved from simulator.
 * Otherwise, the architecture is deduced by detecting available physical devices.
 *
 * @return tt::ARCH The detected platform architecture. Returns tt::ARCH::Invalid
 *                  if no valid architecture could be detected.
 *
 * @note
 * - A fatal error occurs if multiple devices are detected with conflicting
 *   architectures.
 *
 * @exception std::runtime_error Throws a fatal error if:
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

inline tt::ARCH get_platform_architecture(const tt::llrt::RunTimeOptions& rtoptions) {
    auto arch = tt::ARCH::Invalid;
    // If running in mock mode, derive architecture from provided cluster descriptor
    if (rtoptions.get_target_device() == tt::TargetDevice::Mock) {
        auto cluster_desc = umd::ClusterDescriptor::create_from_yaml(rtoptions.get_mock_cluster_desc_path());
        if (cluster_desc && cluster_desc->get_number_of_chips() > 0) {
            auto chips = cluster_desc->get_all_chips();
            arch = cluster_desc->get_arch(*chips.begin());
        }
        return arch;
    }
    if (rtoptions.get_target_device() == tt::TargetDevice::Simulator) {
        auto soc_desc =
            umd::SimulationChip::get_soc_descriptor_path_from_simulator_path(rtoptions.get_simulator_path());
        arch = umd::SocDescriptor::get_arch_from_soc_descriptor_path(soc_desc);
    } else {
        arch = get_physical_architecture();
    }

    return arch;
}

}  // namespace tt::tt_metal
