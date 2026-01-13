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
#include "protobuf/fabric_status.pb.h"
#include <google/protobuf/text_format.h>

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

    // Capture fabric status including mesh shape and cores
    auto status = get_fabric_status();
    status.mesh_shape = mesh_shape;

    // Collect fabric program cores from each device in the mesh
    for (auto* device : mesh_device->get_devices()) {
        auto device_cores = device->get_fabric_program_cores();
        if (!device_cores.empty()) {
            status.fabric_program_cores[device->id()] = device_cores;
        }
    }

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

        // Also write protobuf formats
        write_fabric_status_textproto(status, output_path);
        write_fabric_status_binary(status, output_path);
    } else {
        log_output_rank("✗ Warning: Could not open status file for writing: " + status_file.string());
    }
}

// ============================================================================
// Protobuf Serialization Functions
// ============================================================================

tt::fabric::proto::FabricStatus to_protobuf(const FabricStatus& status) {
    tt::fabric::proto::FabricStatus proto_status;

    // Set configuration state
    proto_status.set_fabric_configured(status.fabric_configured);
    proto_status.set_fabric_initialized(status.fabric_initialized);

    // Convert enums
    proto_status.set_fabric_config(
        static_cast<tt::fabric::proto::FabricConfig>(static_cast<uint32_t>(status.fabric_config)));
    proto_status.set_reliability_mode(
        static_cast<tt::fabric::proto::FabricReliabilityMode>(static_cast<uint32_t>(status.reliability_mode)));
    proto_status.set_num_routing_planes(status.num_routing_planes);
    proto_status.set_fabric_tensix_config(
        static_cast<tt::fabric::proto::FabricTensixConfig>(static_cast<uint32_t>(status.fabric_tensix_config)));
    proto_status.set_fabric_udm_mode(
        static_cast<tt::fabric::proto::FabricUDMMode>(static_cast<uint32_t>(status.fabric_udm_mode)));
    proto_status.set_fabric_manager(
        static_cast<tt::fabric::proto::FabricManagerMode>(static_cast<uint32_t>(status.fabric_manager)));

    // Set mesh shape
    auto* mesh_shape_proto = proto_status.mutable_mesh_shape();
    for (size_t i = 0; i < status.mesh_shape.dims(); ++i) {
        mesh_shape_proto->add_dims(status.mesh_shape[i]);
    }

    // Set fabric program cores (per device)
    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    size_t total_cores = 0;

    for (const auto& [device_id, device_cores] : status.fabric_program_cores) {
        auto* device_fabric_cores = proto_status.add_fabric_cores();
        device_fabric_cores->set_device_id(device_id);

        for (size_t core_type_idx = 0; core_type_idx < device_cores.size(); ++core_type_idx) {
            const auto& cores = device_cores[core_type_idx];

            if (cores.empty()) {
                continue;  // Skip empty core types
            }

            auto* core_type_info = device_fabric_cores->add_core_types();
            core_type_info->set_type_index(core_type_idx);
            core_type_info->set_count(cores.size());
            total_cores += cores.size();

            // Get core type name
            std::string core_type_name = "UNKNOWN";
            if (core_type_idx < hal.get_programmable_core_type_count()) {
                CoreType core_type = hal.get_core_type(core_type_idx);
                switch (core_type) {
                    case CoreType::WORKER: core_type_name = "TENSIX"; break;
                    case CoreType::ETH: core_type_name = "ETHERNET"; break;
                    case CoreType::DRAM: core_type_name = "DRAM"; break;
                    default: core_type_name = "CORE_TYPE_" + std::to_string(static_cast<int>(core_type)); break;
                }
            }
            core_type_info->set_type_name(core_type_name);

            // Add individual cores
            for (const auto& core : cores) {
                auto* core_coord = core_type_info->add_cores();
                core_coord->set_x(core.x);
                core_coord->set_y(core.y);
            }
        }
    }

    proto_status.set_total_cores_used(total_cores);

    // Set additional metadata
    proto_status.set_num_active_ethernet_channels(status.num_active_ethernet_channels);

    for (const auto& host : status.active_hosts) {
        proto_status.add_active_hosts(host);
    }

    for (const auto& chip : status.active_chips) {
        proto_status.add_active_chips(chip);
    }

    // Set timestamp
    proto_status.set_timestamp(std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()));

    return proto_status;
}

void write_fabric_status_textproto(const FabricStatus& status, const std::filesystem::path& output_path) {
    try {
        auto proto_status = to_protobuf(status);

        std::filesystem::path textproto_file = output_path / "fabric_status.textproto";
        std::ofstream file(textproto_file);

        if (!file.is_open()) {
            log_output_rank("✗ Warning: Could not open textproto file: " + textproto_file.string());
            return;
        }

        std::string textproto_output;
        google::protobuf::TextFormat::PrintToString(proto_status, &textproto_output);

        file << "# Fabric Status Report\n";
        file << "# Generated by run_fabric_manager\n\n";
        file << textproto_output;
        file.close();

        log_output_rank("✓ Fabric status textproto written to: " + textproto_file.string());
    } catch (const std::exception& e) {
        log_output_rank("✗ Error writing textproto: " + std::string(e.what()));
    }
}

void write_fabric_status_binary(const FabricStatus& status, const std::filesystem::path& output_path) {
    try {
        auto proto_status = to_protobuf(status);

        std::filesystem::path binary_file = output_path / "fabric_status.pb";
        std::ofstream file(binary_file, std::ios::binary);

        if (!file.is_open()) {
            log_output_rank("✗ Warning: Could not open binary file: " + binary_file.string());
            return;
        }

        if (!proto_status.SerializeToOstream(&file)) {
            log_output_rank("✗ Failed to serialize protobuf");
            return;
        }

        file.close();
        log_output_rank("✓ Fabric status protobuf written to: " + binary_file.string());
    } catch (const std::exception& e) {
        log_output_rank("✗ Error writing binary protobuf: " + std::string(e.what()));
    }
}

}  // namespace tt::scaleout_tools
