// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <umd/device/types/arch.hpp>

#include <cstdint>
#include <optional>
#include <string>

namespace tt::tt_metal::experimental {

// Configuration for mock device mode
struct MockDeviceConfig {
    tt::ARCH arch;
    uint32_t num_chips;
};

// Configure mock mode programmatically - call BEFORE MeshDevice::create()
// This allows creating mock devices without setting environment variables.
// arch: Target architecture (WORMHOLE_B0 or BLACKHOLE)
// num_chips: Number of chips to simulate (1, 2, 4, 8, 32, etc.)
void configure_mock_mode(tt::ARCH arch, uint32_t num_chips = 1);

// Configure mock mode by detecting current hardware topology
void configure_mock_mode_from_hw();

// Disable mock mode (return to hardware mode)
void disable_mock_mode();

// Check if mock mode has been registered via API
bool is_mock_mode_registered();

// Internal: Get the registered mock config (used by MetalContext)
std::optional<MockDeviceConfig> get_registered_mock_config();

// Internal: Convert config to cluster descriptor YAML path
std::string get_mock_cluster_desc_path(const MockDeviceConfig& config);

}  // namespace tt::tt_metal::experimental
