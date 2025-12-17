// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tools/scaleout/fabric_manager/utils/fabric_manager_utils.hpp"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <chrono>

#include "tests/tt_metal/test_utils/test_common.hpp"
#include "tt_metal/impl/context/metal_context.hpp"
#include <tt-metalium/hal.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <enchantum/enchantum.hpp>
#include <llrt/tt_cluster.hpp>

namespace tt::scaleout_tools {

// ============================================================================
// Utility Functions
// ============================================================================

void log_output_rank(const std::string& message, std::optional<int> rank) {
    if (rank.has_value()) {
        if (*rank == *tt::tt_metal::MetalContext::instance().global_distributed_context().rank()) {
            log_info(tt::LogDistributed, "{}", message);
        }
    } else {
        log_info(tt::LogDistributed, "{}", message);
    }
}

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
    const std::filesystem::path& output_path) {
    log_output_rank("Configuring Fabric Routing");

    // Use provided enum values or defaults
    tt::tt_fabric::FabricConfig fabric_config_enum = fabric_config.value_or(tt::tt_fabric::FabricConfig::DISABLED);
    tt::tt_fabric::FabricReliabilityMode reliability_mode_enum =
        reliability_mode.value_or(tt::tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE);
    tt::tt_fabric::FabricTensixConfig fabric_tensix_config_enum =
        fabric_tensix_config.value_or(tt::tt_fabric::FabricTensixConfig::DISABLED);
    tt::tt_fabric::FabricUDMMode fabric_udm_mode_enum =
        fabric_udm_mode.value_or(tt::tt_fabric::FabricUDMMode::DISABLED);

    if (fabric_config.has_value()) {
        log_output_rank(fmt::format("Setting fabric config to: {}", enchantum::to_string(fabric_config_enum)), 0);
    }
    if (reliability_mode.has_value()) {
        log_output_rank(fmt::format("Setting reliability mode to: {}", enchantum::to_string(reliability_mode_enum)), 0);
    }
    if (fabric_tensix_config.has_value()) {
        log_output_rank(
            fmt::format("Setting fabric tensix config to: {}", enchantum::to_string(fabric_tensix_config_enum)), 0);
    }
    if (fabric_udm_mode.has_value()) {
        log_output_rank(fmt::format("Setting fabric UDM mode to: {}", enchantum::to_string(fabric_udm_mode_enum)), 0);
    }
    log_output_rank(fmt::format("Using mesh shape: {}x{}", mesh_shape[0], mesh_shape[1]), 0);

    tt::tt_fabric::SetFabricConfig(
        fabric_config_enum,
        reliability_mode_enum,
        num_routing_planes,
        fabric_tensix_config_enum,
        fabric_udm_mode_enum,
        fabric_manager);

    tt::tt_metal::distributed::MeshDeviceConfig mesh_device_config(mesh_shape);
    auto mesh_device = tt::tt_metal::distributed::MeshDevice::create(mesh_device_config);
    auto status = get_fabric_status();
    log_fabric_status_to_file(status, output_path);
    // After mesh_device goes out of scope, routing tables and fabric kernels are kept alive, but metal context is reset
}

// ============================================================================
// Fabric Information and Monitoring Functions
// ============================================================================

void log_fabric_status(const std::filesystem::path& output_path) {
    auto status = get_fabric_status();
    log_fabric_status_to_file(status, output_path);
}

// ============================================================================
// Fabric Status and Health Functions
// ============================================================================

FabricStatus get_fabric_status() {
    FabricStatus status;

    try {
        auto& context = tt::tt_metal::MetalContext::instance();

        // Get SetFabricConfig parameters from MetalContext
        status.fabric_config = context.get_fabric_config();
        status.fabric_configured = (status.fabric_config != tt::tt_fabric::FabricConfig::DISABLED);
        status.fabric_tensix_config = context.get_fabric_tensix_config();
        status.fabric_udm_mode = context.get_fabric_udm_mode();
        status.fabric_manager = context.get_fabric_manager();

        // Print fabric node IDs
        auto& control_plane = context.get_control_plane();
        const auto& cluster = tt_metal::MetalContext::instance().get_cluster();

        log_output_rank("Fabric Node IDs:");
        for (auto chip_id : cluster.all_chip_ids()) {
            auto fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(chip_id);
            log_output_rank(fmt::format("  Chip ID: {}, Fabric Node ID: {}", chip_id, fabric_node_id));
        }

    } catch (const std::exception& e) {
        log_output_rank("Error getting fabric status: " + std::string(e.what()));
    }

    return status;
}

void log_fabric_status_to_file(const FabricStatus& status, const std::filesystem::path& output_path) {
    std::filesystem::path status_file = output_path / "fabric_status.txt";
    std::ofstream file(status_file);

    if (file.is_open()) {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);

        file << "Fabric Status Report" << std::endl;
        file << "Generated: " << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S") << std::endl;
        file << "==========================================" << std::endl;
        file << "Fabric Configured: " << (status.fabric_configured ? "Yes" : "No") << std::endl;
        file << "Fabric Initialized: " << (status.fabric_initialized ? "Yes" : "No") << std::endl;
        file << std::endl;
        file << "SetFabricConfig Parameters:" << std::endl;
        file << "  Fabric Config: " << enchantum::to_string(status.fabric_config) << std::endl;
        file << "  Reliability Mode: " << enchantum::to_string(status.reliability_mode) << std::endl;
        file << "  Num Routing Planes: " << static_cast<int>(status.num_routing_planes) << std::endl;
        file << "  Fabric Tensix Config: " << enchantum::to_string(status.fabric_tensix_config) << std::endl;
        file << "  Fabric UDM Mode: " << enchantum::to_string(status.fabric_udm_mode) << std::endl;
        file << "  Fabric Manager: " << enchantum::to_string(status.fabric_manager) << std::endl;
        file << std::endl;
        file << "Additional Status:" << std::endl;
        file << "  Number of Active Ethernet Channels: " << status.num_active_ethernet_channels << std::endl;
        file << "  Active Chips: " << status.active_chips.size() << std::endl;
        for (const auto& chip_id : status.active_chips) {
            file << "    Chip ID: " << chip_id << std::endl;
        }
        file << "  Active Hosts: " << status.active_hosts.size() << std::endl;
        for (const auto& host : status.active_hosts) {
            file << "    Host: " << host << std::endl;
        }

        file.close();
        log_output_rank("✓ Fabric status written to: " + status_file.string());
    } else {
        log_output_rank("✗ Warning: Could not open status file for writing: " + status_file.string());
    }
}

}  // namespace tt::scaleout_tools
