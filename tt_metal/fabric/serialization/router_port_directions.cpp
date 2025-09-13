// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>
#include "tt_metal/fabric/serialization/router_port_directions.hpp"
#include "flatbuffers/router_port_directions_generated.h"

namespace tt::tt_fabric {

std::vector<uint8_t> serialize_router_port_directions_to_bytes(
    const RouterPortDirectionsData& router_port_directions_data) {
    flatbuffers::FlatBufferBuilder builder;

    auto mesh_id = tt::tt_fabric::flatbuffer::CreateMeshId(builder, *(router_port_directions_data.local_mesh_id));
    auto host_rank_id =
        tt::tt_fabric::flatbuffer::CreateMeshHostRankId(builder, *(router_port_directions_data.local_host_rank_id));

    // Create vector of FabricNodeDirectionsMap objects
    std::vector<flatbuffers::Offset<tt::tt_fabric::flatbuffer::FabricNodeDirectionsMap>> fabric_node_maps;
    fabric_node_maps.reserve(router_port_directions_data.router_port_directions_map.size());

    for (const auto& [fabric_node_id, direction_map] : router_port_directions_data.router_port_directions_map) {
        // Create FabricNodeId
        auto fabric_node =
            tt::tt_fabric::flatbuffer::CreateFabricNodeId(builder, *fabric_node_id.mesh_id, fabric_node_id.chip_id);

        // Create RoutingDirectionEntry objects for each direction
        std::vector<flatbuffers::Offset<tt::tt_fabric::flatbuffer::RoutingDirectionEntry>> direction_entries;
        direction_entries.reserve(direction_map.size());

        for (const auto& [direction, channels] : direction_map) {
            // Convert channel vector to flatbuffer vector
            auto channels_vector = builder.CreateVector(channels);

            auto direction_entry = tt::tt_fabric::flatbuffer::CreateRoutingDirectionEntry(
                builder, static_cast<uint8_t>(direction), channels_vector);

            direction_entries.push_back(direction_entry);
        }

        // Create vector of direction entries
        auto direction_entries_vector = builder.CreateVector(direction_entries);

        // Create FabricNodeDirectionsMap
        auto fabric_node_map =
            tt::tt_fabric::flatbuffer::CreateFabricNodeDirectionsMap(builder, fabric_node, direction_entries_vector);

        fabric_node_maps.push_back(fabric_node_map);
    }

    // Create vector of fabric node maps
    auto fabric_node_maps_vector = builder.CreateVector(fabric_node_maps);

    // Create the root RouterPortDirectionsMap
    auto serialized_router_port_directions = tt::tt_fabric::flatbuffer::CreateRouterPortDirectionsMap(
        builder, mesh_id, host_rank_id, fabric_node_maps_vector);

    // Finish the buffer
    builder.Finish(serialized_router_port_directions);

    // Return the serialized data
    return std::vector<uint8_t>(builder.GetBufferPointer(), builder.GetBufferPointer() + builder.GetSize());
}

RouterPortDirectionsData deserialize_router_port_directions_from_bytes(const std::vector<uint8_t>& data) {
    RouterPortDirectionsData result;

    auto verifier = flatbuffers::Verifier(data.data(), data.size());
    if (!tt::tt_fabric::flatbuffer::VerifyRouterPortDirectionsMapBuffer(verifier)) {
        throw std::runtime_error("Invalid FlatBuffer data");
    }

    auto router_port_directions = tt::tt_fabric::flatbuffer::GetRouterPortDirectionsMap(data.data());

    if (router_port_directions->local_mesh_id()) {
        result.local_mesh_id = MeshId{router_port_directions->local_mesh_id()->value()};
    }

    if (router_port_directions->local_host_rank_id()) {
        result.local_host_rank_id = MeshHostRankId{router_port_directions->local_host_rank_id()->value()};
    }

    // Extract router port directions map
    if (router_port_directions->fabric_node_maps()) {
        for (const auto* fabric_node_map : *router_port_directions->fabric_node_maps()) {
            FabricNodeId fabric_node_id(MeshId{0}, 0);
            if (fabric_node_map->fabric_node()) {
                fabric_node_id.mesh_id = MeshId{fabric_node_map->fabric_node()->mesh_id()};
                fabric_node_id.chip_id = fabric_node_map->fabric_node()->chip_id();
            }

            std::unordered_map<RoutingDirection, std::vector<chan_id_t>> direction_map;

            if (fabric_node_map->direction_entries()) {
                for (const auto* direction_entry : *fabric_node_map->direction_entries()) {
                    RoutingDirection direction = static_cast<RoutingDirection>(direction_entry->direction());

                    std::vector<chan_id_t> channels;
                    if (direction_entry->channels()) {
                        for (const auto& channel : *direction_entry->channels()) {
                            channels.push_back(channel);
                        }
                    }

                    direction_map[direction] = std::move(channels);
                }
            }

            result.router_port_directions_map[fabric_node_id] = std::move(direction_map);
        }
    }

    return result;
}

}  // namespace tt::tt_fabric
