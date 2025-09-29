// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>
#include <stdexcept>
#include "tt_metal/fabric/serialization/router_port_directions.hpp"
#include "protobuf/router_port_directions.pb.h"

namespace tt::tt_fabric {

std::vector<uint8_t> serialize_router_port_directions_to_bytes(
    const RouterPortDirectionsData& router_port_directions_data) {
    tt::tt_fabric::protobuf::RouterPortDirectionsMap proto_msg;

    // Set local mesh id
    auto* mesh_id = proto_msg.mutable_local_mesh_id();
    mesh_id->set_value(*router_port_directions_data.local_mesh_id);

    // Set local host rank id
    auto* host_rank_id = proto_msg.mutable_local_host_rank_id();
    host_rank_id->set_value(*router_port_directions_data.local_host_rank_id);

    // Process router port directions map
    for (const auto& [fabric_node_id, direction_map] : router_port_directions_data.router_port_directions_map) {
        auto* fabric_node_map = proto_msg.add_fabric_node_maps();

        // Set fabric node id
        auto* proto_fabric_node = fabric_node_map->mutable_fabric_node();
        proto_fabric_node->set_mesh_id(*fabric_node_id.mesh_id);
        proto_fabric_node->set_chip_id(fabric_node_id.chip_id);

        // Process direction entries
        for (const auto& [direction, channels] : direction_map) {
            auto* direction_entry = fabric_node_map->add_direction_entries();
            direction_entry->set_direction(static_cast<uint32_t>(direction));

            // Add channels
            for (const auto& channel : channels) {
                direction_entry->add_channels(channel);
            }
        }
    }

    // Serialize to bytes
    std::vector<uint8_t> serialized_data(proto_msg.ByteSizeLong());
    if (!proto_msg.SerializeToArray(serialized_data.data(), serialized_data.size())) {
        throw std::runtime_error("Failed to serialize RouterPortDirectionsMap to protobuf");
    }

    return serialized_data;
}

RouterPortDirectionsData deserialize_router_port_directions_from_bytes(const std::vector<uint8_t>& data) {
    RouterPortDirectionsData result;

    tt::tt_fabric::protobuf::RouterPortDirectionsMap proto_msg;
    if (!proto_msg.ParseFromArray(data.data(), data.size())) {
        throw std::runtime_error("Failed to parse protobuf data");
    }

    // Extract local mesh id
    if (proto_msg.has_local_mesh_id()) {
        result.local_mesh_id = MeshId{proto_msg.local_mesh_id().value()};
    }

    // Extract local host rank id
    if (proto_msg.has_local_host_rank_id()) {
        result.local_host_rank_id = MeshHostRankId{proto_msg.local_host_rank_id().value()};
    }

    // Extract router port directions map
    for (const auto& fabric_node_map : proto_msg.fabric_node_maps()) {
        FabricNodeId fabric_node_id(MeshId{0}, 0);

        if (fabric_node_map.has_fabric_node()) {
            fabric_node_id.mesh_id = MeshId{fabric_node_map.fabric_node().mesh_id()};
            fabric_node_id.chip_id = fabric_node_map.fabric_node().chip_id();
        }

        std::unordered_map<RoutingDirection, std::vector<chan_id_t>> direction_map;

        for (const auto& direction_entry : fabric_node_map.direction_entries()) {
            RoutingDirection direction = static_cast<RoutingDirection>(direction_entry.direction());

            std::vector<chan_id_t> channels;
            for (const auto& channel : direction_entry.channels()) {
                channels.push_back(channel);
            }

            direction_map[direction] = std::move(channels);
        }

        result.router_port_directions_map[fabric_node_id] = std::move(direction_map);
    }

    return result;
}

}  // namespace tt::tt_fabric
