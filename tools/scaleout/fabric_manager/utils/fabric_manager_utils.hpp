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
#include <optional>
#include "tt_metal/fabric/physical_system_descriptor.hpp"
#include "tt_metal/impl/context/metal_context.hpp"
#include "tt_metal/api/tt-metalium/control_plane.hpp"
#include "tt_metal/api/tt-metalium/fabric_types.hpp"

namespace tt::scaleout_tools {

using tt::ChipId;
using tt::tt_fabric::FabricConfig;
using tt::tt_fabric::FabricReliabilityMode;
using tt::tt_fabric::FabricTensixConfig;
using tt::tt_metal::PhysicalSystemDescriptor;

// ============================================================================
// Utility Functions
// ============================================================================

void log_output_rank0(const std::string& message);

// ============================================================================
// Fabric Configuration Functions
// ============================================================================

void configure_fabric_routing(
    PhysicalSystemDescriptor& physical_system_descriptor,
    const std::optional<std::string>& fabric_config,
    const std::optional<std::string>& reliability_mode,
    const std::optional<uint8_t>& num_routing_planes,
    const std::optional<std::string>& fabric_tensix_config,
    const std::filesystem::path& output_path);

void initialize_fabric_system(
    PhysicalSystemDescriptor& physical_system_descriptor, const std::filesystem::path& output_path);

// ============================================================================
// Fabric Information and Monitoring Functions
// ============================================================================

void print_fabric_information(
    bool print_routing_tables,
    bool print_ethernet_channels,
    bool print_active_connections,
    bool print_all_connections,
    const std::filesystem::path& output_path);

void log_fabric_status(const std::filesystem::path& output_path);

// ============================================================================
// Fabric Configuration Helper Functions
// ============================================================================

FabricConfig parse_fabric_config(const std::string& config_str);
FabricReliabilityMode parse_reliability_mode(const std::string& mode_str);
FabricTensixConfig parse_fabric_tensix_config(const std::string& config_str);

std::string fabric_config_to_string(FabricConfig config);
std::string reliability_mode_to_string(FabricReliabilityMode mode);
std::string fabric_tensix_config_to_string(FabricTensixConfig config);

// ============================================================================
// Fabric Status and Health Functions
// ============================================================================

struct FabricStatus {
    bool fabric_configured = false;
    bool fabric_initialized = false;
    std::string current_fabric_config;
    std::string current_reliability_mode;
    std::string current_fabric_tensix_config;
    uint32_t num_active_routing_planes = 0;
    uint32_t num_active_ethernet_channels = 0;
    std::vector<std::string> active_hosts;
    std::vector<ChipId> active_chips;
};

FabricStatus get_fabric_status();
void log_fabric_status_to_file(const FabricStatus& status, const std::filesystem::path& output_path);

}  // namespace tt::scaleout_tools
