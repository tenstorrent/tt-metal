// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "fabric_fixture.hpp"
#include "utils.hpp"
#include <tt-metalium/device_pool.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "tt_metal/test_utils/env_vars.hpp"
#include <tt-metalium/tt_backend_api_types.hpp>
#include "impl/context/metal_context.hpp"

namespace tt::tt_fabric {
namespace fabric_router_tests {

// Find a device with enough neighbours in the specified direction
bool find_device_with_neighbor_in_multi_direction(
    BaseFabricFixture* fixture,
    FabricNodeId& src_fabric_node_id,
    std::unordered_map<RoutingDirection, std::vector<FabricNodeId>>& dst_fabric_node_ids_by_dir,
    chip_id_t& src_physical_device_id,
    std::unordered_map<RoutingDirection, std::vector<chip_id_t>>& dst_physical_device_ids_by_dir,
    const std::unordered_map<RoutingDirection, uint32_t>& mcast_hops,
    std::optional<RoutingDirection> incoming_direction) {
    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();

    auto devices = fixture->get_devices();
    // Find a device with enough neighbours in the specified direction
    bool connection_found = false;
    for (auto* device : devices) {
        src_fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(device->id());
        if (incoming_direction.has_value()) {
            if (!control_plane.get_intra_chip_neighbors(src_fabric_node_id, incoming_direction.value()).size()) {
                // This potential source will not have the requested incoming direction, skip
                continue;
            }
        }
        std::unordered_map<RoutingDirection, std::vector<FabricNodeId>> temp_end_fabric_node_ids_by_dir;
        std::unordered_map<RoutingDirection, std::vector<chip_id_t>> temp_physical_end_device_ids_by_dir;
        connection_found = true;
        for (auto [routing_direction, num_hops] : mcast_hops) {
            bool direction_found = true;
            auto& temp_end_fabric_node_ids = temp_end_fabric_node_ids_by_dir[routing_direction];
            auto& temp_physical_end_device_ids = temp_physical_end_device_ids_by_dir[routing_direction];
            auto curr_fabric_node_id = src_fabric_node_id;
            for (uint32_t i = 0; i < num_hops; i++) {
                auto neighbors = control_plane.get_intra_chip_neighbors(curr_fabric_node_id, routing_direction);
                if (neighbors.size() > 0) {
                    temp_end_fabric_node_ids.emplace_back(curr_fabric_node_id.mesh_id, neighbors[0]);
                    temp_physical_end_device_ids.push_back(
                        control_plane.get_physical_chip_id_from_fabric_node_id(temp_end_fabric_node_ids.back()));
                    curr_fabric_node_id = temp_end_fabric_node_ids.back();
                } else {
                    direction_found = false;
                    break;
                }
            }
            if (!direction_found) {
                connection_found = false;
                break;
            }
        }
        if (connection_found) {
            src_physical_device_id = device->id();
            dst_fabric_node_ids_by_dir = std::move(temp_end_fabric_node_ids_by_dir);
            dst_physical_device_ids_by_dir = std::move(temp_physical_end_device_ids_by_dir);
            break;
        }
    }
    return connection_found;
}

bool find_device_with_neighbor_in_direction(
    BaseFabricFixture* fixture,
    FabricNodeId& src_fabric_node_id,
    FabricNodeId& dst_fabric_node_id,
    chip_id_t& src_physical_device_id,
    chip_id_t& dst_physical_device_id,
    RoutingDirection direction) {
    auto& control_plane= tt::tt_metal::MetalContext::instance().get_control_plane();
    auto devices = fixture->get_devices();
    for (auto* device : devices) {
        src_fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(device->id());

        // Get neighbours within a mesh in the given direction
        auto neighbors = control_plane.get_intra_chip_neighbors(src_fabric_node_id, direction);
        if (neighbors.size() > 0) {
            src_physical_device_id = device->id();
            dst_fabric_node_id = FabricNodeId(src_fabric_node_id.mesh_id, neighbors[0]);
            dst_physical_device_id = control_plane.get_physical_chip_id_from_fabric_node_id(dst_fabric_node_id);
            return true;
        }
    }
    return false;
}

std::map<FabricNodeId, chip_id_t> get_physical_chip_mapping_from_eth_coords_mapping(
    const std::vector<std::vector<eth_coord_t>>& mesh_graph_eth_coords) {
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    std::map<FabricNodeId, chip_id_t> physical_chip_ids_mapping;
    for (std::uint32_t mesh_id = 0; mesh_id < mesh_graph_eth_coords.size(); mesh_id++) {
        for (std::uint32_t chip_id = 0; chip_id < mesh_graph_eth_coords[mesh_id].size(); chip_id++) {
            const auto& eth_coord = mesh_graph_eth_coords[mesh_id][chip_id];
            physical_chip_ids_mapping.insert(
                {FabricNodeId(MeshId{mesh_id}, chip_id), cluster.get_physical_chip_id_from_eth_coord(eth_coord)});
        }
    }
    return physical_chip_ids_mapping;
}

}  // namespace fabric_router_tests
}  // namespace tt::tt_fabric
