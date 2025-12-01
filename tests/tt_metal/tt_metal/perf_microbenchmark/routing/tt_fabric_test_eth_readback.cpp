// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_fabric_test_eth_readback.hpp"
#include "impl/context/metal_context.hpp"

// Include the necessary headers for the implementation
#include <tt-metalium/hal.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
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

std::vector<EthCoreBufferResult> EthCoreBufferReadback::read_buffer(uint32_t address, size_t buffer_size, bool read_all_active) {
    auto& ctx = tt::tt_metal::MetalContext::instance();
    auto& cluster = ctx.get_cluster();
    auto& control_plane = ctx.get_control_plane();

    auto results_num_elements = tt::align(buffer_size, sizeof(uint32_t)) / sizeof(uint32_t);

    std::vector<EthCoreBufferResult> flat_results;
    std::unordered_map<FabricNodeId, std::unordered_map<CoreCoord, std::vector<uint32_t>>> temp_buffer_map;

    // Lambda that processes a core and adds it to results (idempotent - won't duplicate)
    auto process_core = [&](const MeshCoordinate& coord,
                            FabricNodeId fabric_node_id,
                            CoreCoord eth_core,
                            tt::tt_fabric::chan_id_t eth_channel,
                            RoutingDirection direction,
                            uint32_t link_index,
                            uint32_t physical_chip_id) {
        // Skip if link is down or already in temp_buffer_map
        if (!cluster.is_ethernet_link_up(physical_chip_id, eth_core) ||
            temp_buffer_map[fabric_node_id].count(eth_core)) {
            return;
        }

        flat_results.push_back(
            {.coord = coord,
             .fabric_node_id = fabric_node_id,
             .eth_core = eth_core,
             .eth_channel = eth_channel,
             .direction = direction,
             .link_index = link_index,
             .buffer_data = std::vector<uint32_t>(results_num_elements, 0)});

        temp_buffer_map[fabric_node_id][eth_core] = std::vector<uint32_t>(results_num_elements, 0);
    };

    // Collect all cores to read
    for (const auto& [coord, test_device] : test_devices_) {
        auto device_id = test_device.get_node_id();
        auto physical_chip_id = control_plane.get_physical_chip_id_from_fabric_node_id(device_id);
        const auto& soc_desc = cluster.get_soc_desc(physical_chip_id);
        auto fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(physical_chip_id);

        // Process registered fabric connections
        for (const auto& [direction, link_indices] : test_device.get_used_fabric_connections()) {
            const auto& eth_cores =
                control_plane.get_active_fabric_eth_channels_in_direction(fabric_node_id, direction);
            for (const auto& link_index : link_indices) {
                const auto eth_channel = eth_cores.at(link_index);
                const auto& eth_core = soc_desc.get_eth_core_for_channel(eth_channel, tt::CoordSystem::LOGICAL);
                process_core(coord, fabric_node_id, CoreCoord(eth_core.x, eth_core.y), eth_channel, direction, link_index, physical_chip_id);
            }
        }

        // Process remaining active cores if requested
        if (read_all_active) {
            for (const auto& eth_core_xy : control_plane.get_active_ethernet_cores(physical_chip_id)) {
                CoreCoord eth_core(eth_core_xy.x, eth_core_xy.y);
                tt::umd::CoreCoord umd_eth_core(eth_core_xy, tt::CoreType::ETH, tt::CoordSystem::LOGICAL);
                auto eth_channel = soc_desc.get_eth_channel_for_core(umd_eth_core, tt::CoordSystem::LOGICAL);
                process_core(coord, fabric_node_id, eth_core, eth_channel, RoutingDirection::N, 0, physical_chip_id);
            }
        }
    }

    // Read all buffers
    for (const auto& result : flat_results) {
        fixture_.read_buffer_from_ethernet_cores(
            result.coord, {result.eth_core}, address, buffer_size, false, temp_buffer_map[result.fabric_node_id]);
    }

    fixture_.barrier_reads();

    // Copy buffer data to results
    for (auto& result : flat_results) {
        result.buffer_data = std::move(temp_buffer_map[result.fabric_node_id][result.eth_core]);
    }

    return flat_results;
}
