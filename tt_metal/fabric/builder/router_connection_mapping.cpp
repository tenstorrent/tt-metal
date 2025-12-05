// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/builder/router_connection_mapping.hpp"

#include <tt-logger/tt-logger.hpp>

namespace tt::tt_fabric {

std::vector<ConnectionTarget> RouterConnectionMapping::get_downstream_targets(
    uint32_t vc, uint32_t sender_channel) const {
    
    SenderChannelKey key{vc, sender_channel};
    auto it = sender_to_targets_.find(key);
    
    if (it != sender_to_targets_.end()) {
        return it->second;
    }
    
    return {};  // No targets for this sender channel
}

bool RouterConnectionMapping::has_targets(uint32_t vc, uint32_t sender_channel) const {
    SenderChannelKey key{vc, sender_channel};
    return sender_to_targets_.find(key) != sender_to_targets_.end();
}

std::vector<SenderChannelKey> RouterConnectionMapping::get_all_sender_keys() const {
    std::vector<SenderChannelKey> keys;
    keys.reserve(sender_to_targets_.size());
    
    for (const auto& [key, _] : sender_to_targets_) {
        keys.push_back(key);
    }
    
    return keys;
}

void RouterConnectionMapping::add_target(
    uint32_t vc, uint32_t sender_channel, const ConnectionTarget& target) {
    
    SenderChannelKey key{vc, sender_channel};
    sender_to_targets_[key].push_back(target);
}

RoutingDirection RouterConnectionMapping::get_opposite_direction(RoutingDirection dir) {
    switch (dir) {
        case RoutingDirection::N: return RoutingDirection::S;
        case RoutingDirection::S: return RoutingDirection::N;
        case RoutingDirection::E: return RoutingDirection::W;
        case RoutingDirection::W: return RoutingDirection::E;
        default:
            TT_FATAL(false, "Invalid routing direction for opposite calculation: {}", static_cast<int>(dir));
            return dir;  // Unreachable
    }
}

RouterConnectionMapping RouterConnectionMapping::for_mesh_router(
    Topology topology,
    RoutingDirection direction,
    bool has_z) {
    
    RouterConnectionMapping mapping;
    
    // VC0 sender channels for mesh routers
    // Channel 0: Reserved for local/internal use
    // Channel 1: Primary inter-router connection (opposite direction)
    // Channels 2-3: Additional directions for 2D topology
    
    if (topology == Topology::Linear) {
        // 1D topology: Only channel 1 connects to opposite direction peer
        RoutingDirection opposite = get_opposite_direction(direction);
        mapping.add_target(
            0,  // VC0
            1,  // Sender channel 1
            ConnectionTarget(
                ConnectionType::INTRA_MESH,
                0,  // Target VC0
                0,  // Target sender channel (will be resolved by peer)
                opposite));
        
    } else if (topology == Topology::Mesh) {
        // 2D topology: Channels 1-3 connect to opposite direction peers
        // Channel 1: Primary direction (opposite of this router's direction)
        // Channels 2-3: Cross directions
        
        // Compute the 3 outbound directions for a 2D mesh router
        // For NORTH router: sends to SOUTH (primary), EAST, WEST
        // For EAST router: sends to WEST (primary), NORTH, SOUTH
        // For SOUTH router: sends to NORTH (primary), EAST, WEST
        // For WEST router: sends to EAST (primary), NORTH, SOUTH
        
        std::vector<RoutingDirection> outbound_directions;
        RoutingDirection opposite = get_opposite_direction(direction);
        outbound_directions.push_back(opposite);  // Primary (channel 1)
        
        // Add cross directions (channels 2-3)
        std::vector<RoutingDirection> all_directions = {
            RoutingDirection::N,
            RoutingDirection::E,
            RoutingDirection::S,
            RoutingDirection::W
        };
        
        for (auto dir : all_directions) {
            if (dir != direction && dir != opposite) {
                outbound_directions.push_back(dir);
            }
        }
        
        // Map sender channels 1-3 to outbound directions
        for (size_t i = 0; i < outbound_directions.size() && i < 3; ++i) {
            mapping.add_target(
                0,  // VC0
                1 + i,  // Sender channels 1, 2, 3
                ConnectionTarget(
                    ConnectionType::INTRA_MESH,
                    0,  // Target VC0
                    0,  // Target sender channel (resolved by peer)
                    outbound_directions[i]));
        }
    }
    
    // If this device has a Z router, add MESH_TO_Z connection
    // Mesh routers use their aggregation channel to send to Z router
    if (has_z) {
        // For mesh routers, the aggregation channel is the last sender channel
        // 1D: channel 2 (after channel 0=local, 1=peer)
        // 2D: channel 4 (after channels 0=local, 1-3=peers)
        uint32_t aggregation_channel = (topology == Topology::Linear) ? 2 : 4;
        
        mapping.add_target(
            0,  // VC0
            aggregation_channel,
            ConnectionTarget(
                ConnectionType::MESH_TO_Z,
                0,  // Target Z router VC0
                0,  // Target sender channel (resolved by Z router)
                RoutingDirection::Z));  // Target is Z router
    }
    
    return mapping;
}

RouterConnectionMapping RouterConnectionMapping::for_z_router() {
    RouterConnectionMapping mapping;
    
    // VC0: Standard mesh forwarding (if Z router participates in mesh routing)
    // For now, Z routers primarily use VC1, so VC0 may be unused or reserved
    // This can be extended later if needed
    
    // VC1: Multi-target Z_TO_MESH connections
    // Sender channels 0-3 map to N/E/S/W mesh routers
    // Each sender channel has intent to connect to a specific direction
    // Phase 5 orchestration will skip non-existent directions (2-4 mesh routers)
    
    std::vector<RoutingDirection> mesh_directions = {
        RoutingDirection::N,   // Sender channel 0
        RoutingDirection::E,    // Sender channel 1
        RoutingDirection::S,   // Sender channel 2
        RoutingDirection::W     // Sender channel 3
    };
    
    for (size_t i = 0; i < mesh_directions.size(); ++i) {
        mapping.add_target(
            1,  // VC1
            i,  // Sender channels 0-3
            ConnectionTarget(
                ConnectionType::Z_TO_MESH,
                0,  // Target mesh router VC0
                0,  // Target sender channel (resolved by mesh router)
                mesh_directions[i]));
    }
    
    return mapping;
}

}  // namespace tt::tt_fabric

