// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdlib>

#include "tt_metal/common/tt_backend_api_types.hpp"
#include "tt_metal/common/assert.hpp"
#include "tt_metal/third_party/umd/device/cluster.h"

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
 * #include "tt_metal/common/tt_backend_api_types.hpp"
 *
 * tt::ARCH arch = tt::tt_metal::get_platform_architecture();
 * if (arch == tt::ARCH::Invalid) {
 *     std::cerr << "Failed to detect architecture!" << std::endl;
 * } else {
 *     std::cout << "Detected architecture: " << tt::get_arch_str(arch) << std::endl;
 * }
 * @endcode
 *
 * @see tt::get_arch_from_string
 * @see tt::umd::Cluster::detect_available_device_ids
 * @see detect_arch
 */
inline tt::ARCH get_platform_architecture() {
    auto arch = tt::ARCH::Invalid;
    if (std::getenv("TT_METAL_SIMULATOR_EN")) {
        auto arch_env = std::getenv("ARCH_NAME");
        TT_FATAL(arch_env, "ARCH_NAME env var needed for VCS");
        arch = tt::get_arch_from_string(arch_env);
    } else {
        std::vector<chip_id_t> physical_mmio_device_ids = tt::umd::Cluster::detect_available_device_ids();
        if (!physical_mmio_device_ids.empty()) {
            arch = detect_arch(physical_mmio_device_ids.at(0));
            for (int i = 1; i < physical_mmio_device_ids.size(); ++i) {
                chip_id_t device_id = physical_mmio_device_ids.at(i);
                tt::ARCH detected_arch = detect_arch(device_id);
                TT_FATAL(
                    arch == detected_arch,
                    "Expected all devices to be {} but device {} is {}",
                    get_arch_str(arch),
                    device_id,
                    get_arch_str(detected_arch));
            }
        }
    }

    return arch;
}

}  // namespace tt::tt_metal
