// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <cstdint>
#include <map>
#include <unordered_map>
#include "tt-metalium/fabric_types.hpp"
#include "tt_metal/hostdevcommon/api/hostdevcommon/fabric_common.h"
#include <tt-metalium/routing_table_generator.hpp>

namespace tt::tt_fabric {

// Structure to hold router port directions data for serialization
struct RouterPortDirectionsData {
    MeshId local_mesh_id = MeshId{0};
    MeshHostRankId local_host_rank_id = MeshHostRankId{0};
    std::map<FabricNodeId, std::unordered_map<RoutingDirection, std::vector<chan_id_t>>> router_port_directions_map;

    bool operator==(const RouterPortDirectionsData& other) const {
        return local_mesh_id == other.local_mesh_id && local_host_rank_id == other.local_host_rank_id &&
               router_port_directions_map == other.router_port_directions_map;
    }
};

// Serialization functions
std::vector<uint8_t> serialize_router_port_directions_to_bytes(
    const RouterPortDirectionsData& router_port_directions_data);
RouterPortDirectionsData deserialize_router_port_directions_from_bytes(const std::vector<uint8_t>& data);

}  // namespace tt::tt_fabric
