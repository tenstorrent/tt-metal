// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_fabric_test_eth_readback.hpp"
#include "impl/context/metal_context.hpp"

// Include the necessary headers for the implementation
#include <tt-metalium/hal.hpp>
#include <tt-metalium/fabric.hpp>
#include <tt-metalium/mesh_coord.hpp>

// Include the test device and fixture headers
#include "tt_fabric_test_device_setup.hpp"
#include "tt_fabric_test_interfaces.hpp"

EthCoreBufferReadback::EthCoreBufferReadback(
    const std::unordered_map<MeshCoordinate, TestDevice>& test_devices,
    TestFixture& fixture)
    : test_devices_(test_devices), fixture_(fixture) {
}

void EthCoreBufferReadback::clear_buffer(uint32_t address, size_t buffer_size) {
    auto& ctx = tt::tt_metal::MetalContext::instance();
    auto& cluster = ctx.get_cluster();
    auto& control_plane = ctx.get_control_plane();

    // Create zero vector for the buffer
    std::vector<uint8_t> zero_vec(buffer_size, 0);

    for (const auto& [coord, test_device] : test_devices_) {
        auto device_id = test_device.get_node_id();
        auto physical_chip_id = control_plane.get_physical_chip_id_from_fabric_node_id(device_id);
        auto active_eth_cores = control_plane.get_active_ethernet_cores(physical_chip_id);
        for (const auto& eth_core : active_eth_cores) {
            if (!cluster.is_ethernet_link_up(physical_chip_id, eth_core)) {
                continue;
            }

            std::vector<CoreCoord> cores = {eth_core};
            fixture_.write_buffer_to_ethernet_cores(coord, cores, address, zero_vec);
        }
    }
}

std::vector<EthCoreBufferResult> EthCoreBufferReadback::read_buffer(uint32_t address, size_t buffer_size) {
    auto& ctx = tt::tt_metal::MetalContext::instance();
    auto& cluster = ctx.get_cluster();
    auto& control_plane = ctx.get_control_plane();

    auto results_num_elements = tt::align(buffer_size, sizeof(uint32_t)) / sizeof(uint32_t);

    // Build metadata and temporary buffer map for batch reading
    std::vector<EthCoreBufferResult> flat_results;
    std::unordered_map<FabricNodeId, std::unordered_map<CoreCoord, std::vector<uint32_t>>> temp_buffer_map;

    // First pass: collect metadata and initialize temp buffer storage
    for (const auto& [coord, test_device] : test_devices_) {
        auto device_id = test_device.get_node_id();
        auto physical_chip_id = control_plane.get_physical_chip_id_from_fabric_node_id(device_id);
        auto& soc_desc = cluster.get_soc_desc(physical_chip_id);
        auto fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(physical_chip_id);

        for (const auto& [direction, link_indices] : test_device.get_used_fabric_connections()) {
            const auto& eth_cores =
                control_plane.get_active_fabric_eth_channels_in_direction(fabric_node_id, direction);
            for (const auto& link_index : link_indices) {
                const tt::tt_fabric::chan_id_t eth_channel = eth_cores.at(link_index);
                const CoreCoord& eth_core = soc_desc.get_eth_core_for_channel(eth_channel, CoordSystem::LOGICAL);

                // Create result entry with metadata using aggregate initialization
                flat_results.push_back(
                    {.coord = coord,
                     .fabric_node_id = fabric_node_id,
                     .eth_core = eth_core,
                     .eth_channel = eth_channel,
                     .direction = direction,
                     .link_index = link_index,
                     .buffer_data = std::vector<uint32_t>(results_num_elements, 0)});

                // Initialize temp buffer for reading
                temp_buffer_map[fabric_node_id][eth_core] = std::vector<uint32_t>(results_num_elements, 0);
            }
        }
    }

    // Second pass: perform buffer reads
    for (const auto& [coord, test_device] : test_devices_) {
        auto device_id = test_device.get_node_id();
        auto physical_chip_id = control_plane.get_physical_chip_id_from_fabric_node_id(device_id);
        auto& soc_desc = cluster.get_soc_desc(physical_chip_id);
        auto fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(physical_chip_id);

        for (const auto& [direction, link_indices] : test_device.get_used_fabric_connections()) {
            const auto& eth_cores =
                control_plane.get_active_fabric_eth_channels_in_direction(fabric_node_id, direction);
            for (const auto& link_index : link_indices) {
                const tt::tt_fabric::chan_id_t eth_channel = eth_cores.at(link_index);
                const CoreCoord& eth_core = soc_desc.get_eth_core_for_channel(eth_channel, CoordSystem::LOGICAL);

                TT_FATAL(
                    cluster.is_ethernet_link_up(physical_chip_id, eth_core),
                    "Ethernet link is not up for {}",
                    eth_core);

                std::vector<CoreCoord> cores = {eth_core};
                fixture_.read_buffer_from_ethernet_cores(
                    coord, cores, address, buffer_size, false, temp_buffer_map[fabric_node_id]);
            }
        }
    }

    fixture_.barrier_reads();

    // Third pass: copy buffer data from temp map to flat results
    for (auto& result : flat_results) {
        result.buffer_data = temp_buffer_map[result.fabric_node_id][result.eth_core];
    }

    return flat_results;
}
