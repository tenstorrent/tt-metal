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

namespace tt::scaleout_tools {

// ============================================================================
// Utility Functions
// ============================================================================

void log_output_rank0(const std::string& message) {
    if (*tt::tt_metal::MetalContext::instance().global_distributed_context().rank() == 0) {
        log_info(tt::LogDistributed, "{}", message);
    }
}

// ============================================================================
// Fabric Configuration Functions
// ============================================================================

void configure_fabric_routing(
    const std::optional<std::string>& fabric_config,
    const std::optional<std::string>& reliability_mode,
    const std::optional<uint8_t>& num_routing_planes,
    const std::optional<std::string>& fabric_tensix_config,
    const std::filesystem::path& output_path) {
    log_output_rank0("Configuring Fabric Routing");

    // Parse configuration parameters
    FabricConfig fabric_config_enum = FabricConfig::DISABLED;
    if (fabric_config.has_value()) {
        fabric_config_enum = parse_fabric_config(fabric_config.value());
        log_output_rank0("Setting fabric config to: " + fabric_config_to_string(fabric_config_enum));
    }

    FabricReliabilityMode reliability_mode_enum = FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE;
    if (reliability_mode.has_value()) {
        reliability_mode_enum = parse_reliability_mode(reliability_mode.value());
        log_output_rank0("Setting reliability mode to: " + reliability_mode_to_string(reliability_mode_enum));
    }

    FabricTensixConfig fabric_tensix_config_enum = FabricTensixConfig::DISABLED;
    if (fabric_tensix_config.has_value()) {
        fabric_tensix_config_enum = parse_fabric_tensix_config(fabric_tensix_config.value());
        log_output_rank0(
            "Setting fabric tensix config to: " + fabric_tensix_config_to_string(fabric_tensix_config_enum));
    }

    tt::tt_fabric::SetFabricConfig(
        fabric_config_enum, reliability_mode_enum, num_routing_planes, fabric_tensix_config_enum);

    tt::tt_metal::distributed::MeshDeviceConfig mesh_device_config(tt::tt_metal::distributed::MeshShape(8, 4));
    auto mesh_device = tt::tt_metal::distributed::MeshDevice::create(mesh_device_config);
}

// ============================================================================
// Fabric Information and Monitoring Functions
// ============================================================================

void print_fabric_information(
    bool print_routing_tables,
    bool print_ethernet_channels,
    bool print_active_connections,
    bool print_all_connections,
    const std::filesystem::path& output_path) {
    /*
     auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();

     if (print_routing_tables) {
         log_output_rank0("Printing Fabric Routing Tables");
         control_plane.print_routing_tables();
     }

     if (print_ethernet_channels) {
         log_output_rank0("Printing Ethernet Channel Information");
         control_plane.print_ethernet_channels();
     }

     if (print_active_connections) {
         log_output_rank0("Printing Active Ethernet Connections");
         control_plane.print_active_ethernet_connections();
     }

     if (print_all_connections) {
         log_output_rank0("Printing All Ethernet Connections");
         control_plane.print_all_ethernet_connections();
     }
     */
    // Log fabric status to file
    log_fabric_status(output_path);
}

void log_fabric_status(const std::filesystem::path& output_path) {
    auto status = get_fabric_status();
    log_fabric_status_to_file(status, output_path);
}

// ============================================================================
// Fabric Configuration Helper Functions
// ============================================================================

FabricConfig parse_fabric_config(const std::string& config_str) {
    if (config_str == "DISABLED") {
        return FabricConfig::DISABLED;
    }
    if (config_str == "FABRIC_1D") {
        return FabricConfig::FABRIC_1D;
    }
    if (config_str == "FABRIC_1D_RING") {
        return FabricConfig::FABRIC_1D_RING;
    }
    if (config_str == "FABRIC_2D") {
        return FabricConfig::FABRIC_2D;
    }
    if (config_str == "FABRIC_2D_TORUS_X") {
        return FabricConfig::FABRIC_2D_TORUS_X;
    }
    if (config_str == "FABRIC_2D_TORUS_Y") {
        return FabricConfig::FABRIC_2D_TORUS_Y;
    }
    if (config_str == "FABRIC_2D_TORUS_XY") {
        return FabricConfig::FABRIC_2D_TORUS_XY;
    }
    if (config_str == "CUSTOM") {
        return FabricConfig::CUSTOM;
    }

    TT_FATAL(false, "Unknown fabric config: {}", config_str);
    return FabricConfig::DISABLED;
}

FabricReliabilityMode parse_reliability_mode(const std::string& mode_str) {
    if (mode_str == "STRICT_SYSTEM_HEALTH_SETUP_MODE") {
        return FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE;
    }
    if (mode_str == "RELAXED_SYSTEM_HEALTH_SETUP_MODE") {
        return FabricReliabilityMode::RELAXED_SYSTEM_HEALTH_SETUP_MODE;
    }

    TT_FATAL(false, "Unknown reliability mode: {}", mode_str);
    return FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE;
}

FabricTensixConfig parse_fabric_tensix_config(const std::string& config_str) {
    if (config_str == "DISABLED") {
        return FabricTensixConfig::DISABLED;
    }
    if (config_str == "MUX") {
        return FabricTensixConfig::MUX;
    }

    TT_FATAL(false, "Unknown fabric tensix config: {}", config_str);
    return FabricTensixConfig::DISABLED;
}

std::string fabric_config_to_string(FabricConfig config) {
    switch (config) {
        case FabricConfig::DISABLED: return "DISABLED";
        case FabricConfig::FABRIC_1D: return "FABRIC_1D";
        case FabricConfig::FABRIC_1D_RING: return "FABRIC_1D_RING";
        case FabricConfig::FABRIC_2D: return "FABRIC_2D";
        case FabricConfig::FABRIC_2D_TORUS_X: return "FABRIC_2D_TORUS_X";
        case FabricConfig::FABRIC_2D_TORUS_Y: return "FABRIC_2D_TORUS_Y";
        case FabricConfig::FABRIC_2D_TORUS_XY: return "FABRIC_2D_TORUS_XY";
        case FabricConfig::CUSTOM: return "CUSTOM";
        default: return "UNKNOWN";
    }
}

std::string reliability_mode_to_string(FabricReliabilityMode mode) {
    switch (mode) {
        case FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE: return "STRICT_SYSTEM_HEALTH_SETUP_MODE";
        case FabricReliabilityMode::RELAXED_SYSTEM_HEALTH_SETUP_MODE: return "RELAXED_SYSTEM_HEALTH_SETUP_MODE";
        default: return "UNKNOWN";
    }
}

std::string fabric_tensix_config_to_string(FabricTensixConfig config) {
    switch (config) {
        case FabricTensixConfig::DISABLED: return "DISABLED";
        case FabricTensixConfig::MUX: return "MUX";
        default: return "UNKNOWN";
    }
}

// ============================================================================
// Fabric Status and Health Functions
// ============================================================================

FabricStatus get_fabric_status() {
    FabricStatus status;

    try {
        auto& context = tt::tt_metal::MetalContext::instance();
        // auto& control_plane = context.get_control_plane();

        // Check if fabric is configured
        auto fabric_config = context.get_fabric_config();
        status.fabric_configured = (fabric_config != FabricConfig::DISABLED);
        status.current_fabric_config = fabric_config_to_string(fabric_config);

        // Check if fabric is initialized
        try {
            //  auto& fabric_context = control_plane.get_fabric_context();
            status.fabric_initialized = true;
        } catch (...) {
            status.fabric_initialized = false;
        }

        // Get current configuration strings
        status.current_reliability_mode = "UNKNOWN";      // Would need to add getter to MetalContext
        status.current_fabric_tensix_config = "UNKNOWN";  // Would need to add getter to MetalContext

        // Get active hosts and chips
        auto& cluster = context.get_cluster();
        for (auto chip_id : cluster.all_chip_ids()) {
            status.active_chips.push_back(chip_id);
        }

        // Get active hosts from physical system descriptor
        // This would require access to the physical system descriptor
        status.active_hosts = {"localhost"};  // Placeholder

    } catch (const std::exception& e) {
        log_output_rank0("Error getting fabric status: " + std::string(e.what()));
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
        file << "Current Fabric Config: " << status.current_fabric_config << std::endl;
        file << "Current Reliability Mode: " << status.current_reliability_mode << std::endl;
        file << "Current Fabric Tensix Config: " << status.current_fabric_tensix_config << std::endl;
        file << "Number of Active Routing Planes: " << status.num_active_routing_planes << std::endl;
        file << "Number of Active Ethernet Channels: " << status.num_active_ethernet_channels << std::endl;
        file << "Active Chips: " << status.active_chips.size() << std::endl;
        for (const auto& chip_id : status.active_chips) {
            file << "  Chip ID: " << chip_id << std::endl;
        }
        file << "Active Hosts: " << status.active_hosts.size() << std::endl;
        for (const auto& host : status.active_hosts) {
            file << "  Host: " << host << std::endl;
        }

        file.close();
        log_output_rank0("✓ Fabric status written to: " + status_file.string());
    } else {
        log_output_rank0("✗ Warning: Could not open status file for writing: " + status_file.string());
    }
}

}  // namespace tt::scaleout_tools
