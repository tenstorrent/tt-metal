// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <memory>
#include <filesystem>
#include "tt_metal/api/tt-metalium/experimental/fabric/control_plane.hpp"
#include "tt_metal/api/tt-metalium/experimental/fabric/fabric_types.hpp"
#include "tt_metal/fabric/physical_system_descriptor.hpp"
#include "tt_metal/impl/context/metal_context.hpp"
#include <tt-metalium/mesh_coord.hpp>

namespace tt::scaleout_tools {

// ============================================================================
// Utility Functions
// ============================================================================

void log_output_rank(const std::string& message, std::optional<int> rank = std::nullopt);

// ============================================================================
// Fabric Configuration Functions
// ============================================================================

void configure_fabric_routing(
    const std::optional<tt::tt_fabric::FabricConfig>& fabric_config,
    const std::optional<tt::tt_fabric::FabricReliabilityMode>& reliability_mode,
    const std::optional<uint8_t>& num_routing_planes,
    const std::optional<tt::tt_fabric::FabricTensixConfig>& fabric_tensix_config,
    const std::optional<tt::tt_fabric::FabricUDMMode>& fabric_udm_mode,
    const tt::tt_fabric::FabricManagerMode& fabric_manager,
    const tt::tt_metal::distributed::MeshShape& mesh_shape,
    const std::filesystem::path& output_path);

// ============================================================================
// Fabric Information and Monitoring Functions
// ============================================================================

void log_fabric_status(const std::filesystem::path& output_path);

// ============================================================================
// Fabric Status and Health Functions
// ============================================================================

struct FabricStatus {
    bool fabric_configured = false;
    bool fabric_initialized = false;

    // SetFabricConfig parameters
    tt::tt_fabric::FabricConfig fabric_config = tt::tt_fabric::FabricConfig::DISABLED;
    tt::tt_fabric::FabricReliabilityMode reliability_mode =
        tt::tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE;
    uint8_t num_routing_planes = 0;
    tt::tt_fabric::FabricTensixConfig fabric_tensix_config = tt::tt_fabric::FabricTensixConfig::DISABLED;
    tt::tt_fabric::FabricUDMMode fabric_udm_mode = tt::tt_fabric::FabricUDMMode::DISABLED;
    tt::tt_fabric::FabricManagerMode fabric_manager = tt::tt_fabric::FabricManagerMode::DEFAULT;

    // Additional status info
    uint32_t num_active_ethernet_channels = 0;
    std::vector<std::string> active_hosts;
    std::vector<tt::ChipId> active_chips;
};

FabricStatus get_fabric_status();
void log_fabric_status_to_file(const FabricStatus& status, const std::filesystem::path& output_path);

}  // namespace tt::scaleout_tools
