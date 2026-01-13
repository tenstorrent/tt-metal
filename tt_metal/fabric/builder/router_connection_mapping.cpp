// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/builder/router_connection_mapping.hpp"
#include "tt_metal/fabric/builder/fabric_builder_config.hpp"

#include <tt-logger/tt-logger.hpp>

namespace tt::tt_fabric {

std::vector<ConnectionTarget> RouterConnectionMapping::get_downstream_targets(
    uint32_t vc, uint32_t receiver_channel) const {
    ReceiverChannelKey key{vc, receiver_channel};
    auto it = receiver_to_targets_.find(key);

    if (it != receiver_to_targets_.end()) {
        return it->second;
    }

    return {};  // No targets for this sender channel
}

bool RouterConnectionMapping::has_targets(uint32_t vc, uint32_t receiver) const {
    ReceiverChannelKey key{vc, receiver};
    return receiver_to_targets_.contains(key);
}

std::vector<ReceiverChannelKey> RouterConnectionMapping::get_all_receiver_keys() const {
    std::vector<ReceiverChannelKey> keys;
    keys.reserve(receiver_to_targets_.size());

    for (const auto& [key, _] : receiver_to_targets_) {
        keys.push_back(key);
    }

    return keys;
}

void RouterConnectionMapping::add_target(uint32_t vc, uint32_t receiver_channel, const ConnectionTarget& target) {
    ReceiverChannelKey key{vc, receiver_channel};
    receiver_to_targets_[key].push_back(target);
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
    Topology topology, RoutingDirection direction, bool has_z, bool enable_vc1) {
    RouterConnectionMapping mapping;

    // VC0 receiver_channel channels for mesh routers
    // Channel 0: Reserved for local/internal use
    // Channel 1: Primary inter-router connection (opposite direction)
    // Channels 2-3: Additional directions for 2D topology

    if (topology == Topology::Linear || topology == Topology::Ring) {
        // 1D topology: Only channel 1 connects to opposite direction peer
        // Ring is also 1D but with wrap-around (handled by FabricBuilder connection logic)
        RoutingDirection opposite = get_opposite_direction(direction);
        mapping.add_target(
            0,  // VC0
            0,  // receiver channel 1
            ConnectionTarget(
                ConnectionType::INTRA_MESH,
                0,  // Target VC0
                1,  // Target sender channel (will be resolved by peer)
                opposite));

    } else if (topology == Topology::Mesh || topology == Topology::Torus) {
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

        // Map sender channels 1-3 to outbound directions on VC0
        // Map sender channels 0-2 to outbound directions on VC1
        //
        // IMPORTANT: INTRA_MESH connections are hardcoded to use VC0 only.
        // This is the intended behavior for the following reasons:
        //
        // 1. VC0 is reserved for intra-mesh traffic (chip-to-chip communication within a single mesh)
        //    All locally generated traffic in a mesh whether destined for another chip in the mesh or exiting the mesh
        //    is transported over VC0. If traffic exits the mesh, the inter-mesh receiver router in the receiving mesh
        //    crosses over the traffic to VC1. In other words ALL traffic generated locally on a mesh is considered
        //    intra-mesh until it exits the mesh. If traffic exits the mesh, it is considered inter-mesh traffic (by the
        //    receiving mesh/router) and is transported over VC1.
        // 2. VC1 is reserved for inter-mesh traffic (Z-to-mesh, or mesh-to-mesh across different meshes)
        // 3. This separation ensures proper traffic isolation and prevents deadlocks in multi-mesh systems
        // 4. Even when VC1 is enabled on mesh routers (via IntermeshVCConfig), INTRA_MESH connections
        //    continue to use VC0, while inter-mesh connections use VC1
        // 5. The VC assignment is determined by the connection type, not the router capabilities
        //
        TT_FATAL(outbound_directions.size() <= builder_config::num_downstream_edms_2d_vc0, "Outbound directions size must be less than or equal to num_downstream_edms_2d_vc0");

        // Add VC0 targets for intra-mesh traffic
        for (size_t i = 0; i < outbound_directions.size(); ++i) {
            mapping.add_target(
                0,  // VC0 - for intra-mesh traffic
                0,  // Receiver channel 0
                ConnectionTarget(
                    ConnectionType::INTRA_MESH,
                    0,      // Target VC0
                    i + 1,  // Target sender channel
                    outbound_directions[i]));
        }

        // Add VC1 targets for intra-mesh routers (to forward inter-mesh traffic)
        // VC1 connections are only for intra-mesh routers in multi-mesh topologies
        // They forward inter-mesh traffic that was received via VC1
        if (enable_vc1) {
            for (size_t i = 0; i < outbound_directions.size(); ++i) {
                mapping.add_target(
                    1,  // VC1 - for inter-mesh traffic forwarding
                    0,  // Receiver channel 0 (VC1 only has one receiver channel)
                    ConnectionTarget(
                        ConnectionType::INTRA_MESH,
                        1,  // Target VC1
                        i,  // Target sender channel (0-2 for VC1)
                        outbound_directions[i]));
            }
        }
    }

    // If this device has a Z router, add MESH_TO_Z connection
    if (has_z) {
        // Mesh routers use the next available sender channel (after base mesh channels) for MESH_TO_Z
        // 1D: base channels from builder_config::num_sender_channels_1d_linear
        // 2D: base channels from builder_config::num_sender_channels_2d_mesh
        uint32_t base_channels = (topology == Topology::Linear || topology == Topology::Ring)
                                     ? builder_config::num_sender_channels_1d_linear
                                     : builder_config::num_sender_channels_2d_mesh;
        uint32_t mesh_to_z_channel = base_channels;

        mapping.add_target(
            0,  // VC0
            0,  // Receiver channel 0
            ConnectionTarget(
                ConnectionType::MESH_TO_Z,
                0,  // Target Z router VC0
                mesh_to_z_channel,  // Target sender channel (resolved by Z router)
                RoutingDirection::Z));  // Target is Z router
    }

    return mapping;
}

RouterConnectionMapping RouterConnectionMapping::for_z_router() {
    RouterConnectionMapping mapping;

    std::vector<RoutingDirection> vc1_outbound_directions = {
        RoutingDirection::E,  // Forward to EAST mesh router
        RoutingDirection::W,  // Forward to WEST mesh router
        RoutingDirection::N,  // Forward to NORTH mesh router
        RoutingDirection::S   // Forward to SOUTH mesh router
    };

    for (size_t i = 0; i < vc1_outbound_directions.size(); ++i) {
        mapping.add_target(
            1,  // VC1
            0,  // Receiver channel 0
            ConnectionTarget(
                ConnectionType::Z_TO_MESH,
                1,  // Target mesh router VC1
                i,  // Target sender channel on mesh router (0-3, no worker)
                vc1_outbound_directions[i]));
    }

    return mapping;
}

}  // namespace tt::tt_fabric
