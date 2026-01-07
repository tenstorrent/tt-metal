// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file mock_device.hpp
 * @brief Mock Device API for hardware-free testing and development
 *
 * Enables running TT-Metal without physical hardware. Useful for graph capture,
 * allocator testing, and CI/CD pipelines. Configure mock mode before device creation,
 * then use MeshDevice::create() normally.
 */

#include <umd/device/types/arch.hpp>

#include <cstdint>
#include <optional>
#include <string>

namespace tt::tt_metal::experimental {

// arch: Target architecture (WORMHOLE_B0 or BLACKHOLE)
// num_chips: Number of chips to simulate (1, 2, 4, 8, 32, etc.)
void configure_mock_mode(tt::ARCH arch, uint32_t num_chips = 1);

// Auto-detect architecture from hardware and configure mock mode
// Only works on machines with TT hardware present
void configure_mock_mode_from_hw();

// Disable mock mode (return to hardware mode)
void disable_mock_mode();

// Check if mock mode has been registered via API
bool is_mock_mode_registered();

// Get the cluster descriptor filename for the registered mock config
// Returns nullopt if mock mode is not registered
// Returns just the filename (e.g., "blackhole_P150.yaml"), caller prepends base path
std::optional<std::string> get_mock_cluster_desc();

}  // namespace tt::tt_metal::experimental
