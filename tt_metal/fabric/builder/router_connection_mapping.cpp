// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/builder/router_connection_mapping.hpp"
#include "tt_metal/fabric/builder/fabric_builder_config.hpp"

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
    Topology topology,
    RoutingDirection direction,
    bool has_z,
    bool enable_vc1,
    bool enable_mesh_pass_through,
    bool has_intra_mesh_z) {
    RouterConnectionMapping mapping;

    // VC0 receiver_channel channels for mesh routers
    // Channel 0: Reserved for local/internal use
    // Channel 1: Primary inter-router connection (opposite direction)
    // Channels 2-3: Additional directions for 2D topology

    if (topology == Topology::Linear || topology == Topology::Ring) {
        // 1D topology: Only channel 1 connects to opposite direction peer
        // Ring is also 1D but with wrap-around (handled by FabricBuilder connection logic)
        //
        // Intra-mesh Z (skip-link) routers are not supported under 1D (Linear/Ring) routing yet: a Z
        // endpoint has no single "opposite" direction and the 1D connection layout has no slot for a
        // skip-link peer. Fail clearly here rather than hitting get_opposite_direction()'s default-case
        // TT_FATAL ("Invalid routing direction for opposite calculation: 4"), which is opaque.
        TT_FATAL(
            direction != RoutingDirection::Z,
            "Intra-mesh Z (skip-link) routers are unsupported for 1D (Linear/Ring) topology. Skip links require "
            "2D (Mesh/Torus) routing; run this skip-link mesh graph descriptor under a Mesh/Torus topology, or "
            "drop the skip_links for 1D runs.");
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
        std::vector<RoutingDirection> all_directions = {
            RoutingDirection::N,
            RoutingDirection::E,
            RoutingDirection::S,
            RoutingDirection::W
        };

        const bool is_intra_mesh_z_router = (direction == RoutingDirection::Z);
        if (is_intra_mesh_z_router) {
            // Intra-mesh Z (sub-torus skip-link) endpoint: a MESH-variant router whose own eth link is the
            // skip link. It has no "opposite" mesh direction; it forwards skip-link traffic to / receives
            // from all four mesh-direction routers on this device. (get_opposite_direction has no Z case.)
            outbound_directions = all_directions;
        } else {
            RoutingDirection opposite = get_opposite_direction(direction);
            outbound_directions.push_back(opposite);  // Primary (channel 1)

            // Add cross directions (channels 2-3)
            for (auto dir : all_directions) {
                if (dir != direction && dir != opposite) {
                    outbound_directions.push_back(dir);
                }
            }
        }

        // VC0 outbound directions are the mesh directions plus, for sub-torus skip links, intra-mesh Z.
        // Intra-mesh Z maps to VC0 target sender channel 4 (the slot reserved by the 5-wide
        // intra-mesh-Z VC0 layout), letting intra-mesh Z traffic ride VC0 alongside E/W/N/S.
        // NOTE: VC0-only — VC1 (inter-mesh) keeps using the mesh directions exclusively (see below).
        // The intra-mesh Z router itself only forwards to the four mesh directions (already covered above);
        // it does not get an additional Z outbound (it would be a self-edge).
        std::vector<RoutingDirection> vc0_outbound_directions = outbound_directions;
        if (has_intra_mesh_z && !is_intra_mesh_z_router) {
            vc0_outbound_directions.push_back(RoutingDirection::Z);
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
        const size_t max_vc0_outbound =
            builder_config::get_vc0_downstream_edm_count(/*is_2D_routing=*/true, has_intra_mesh_z);
        TT_FATAL(
            vc0_outbound_directions.size() <= max_vc0_outbound,
            "VC0 outbound directions size ({}) must be <= VC0 downstream EDM count ({}) (has_intra_mesh_z={})",
            vc0_outbound_directions.size(),
            max_vc0_outbound,
            has_intra_mesh_z);

        // Add VC0 targets for intra-mesh traffic (mesh directions + intra-mesh Z when present)
        for (size_t i = 0; i < vc0_outbound_directions.size(); ++i) {
            mapping.add_target(
                0,  // VC0 - for intra-mesh traffic
                0,  // Receiver channel 0
                ConnectionTarget(
                    ConnectionType::INTRA_MESH,
                    0,      // Target VC0
                    i + 1,  // Target sender channel
                    vc0_outbound_directions[i]));
        }

        // Add VC1 targets for intra-mesh routers (to forward inter-mesh traffic)
        // VC1 connections are only for intra-mesh routers in multi-mesh topologies
        // They forward inter-mesh traffic that was received via VC1.
        // NOTE: intra-mesh Z is VC0-only, so VC1 keeps iterating the mesh directions exclusively, and the
        // intra-mesh Z router itself (direction == Z) does not service VC1.
        if (enable_vc1 && !is_intra_mesh_z_router) {
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

        // EXPERIMENTAL: inter-mesh pass-through (A->B->C).
        // In the default (full_mesh) mode, inter-mesh traffic that has been crossed over to VC1
        // sinks within the receiving mesh. To let it pass through toward a further mesh, the mesh
        // router must also forward VC1 traffic to the local Z router, which re-exports it across
        // the next inter-mesh link. This wires the 4th VC1 sender channel that the channel mapping
        // already reserves on Z-stacked devices (has_z => mesh_vc1_sender_count == 4).
        // NOTE: only valid in pass-through mode; in full_mesh the Z router does not service VC1
        // (FABRIC_2D_VC1_SERVICED is gated on requires_vc1_mesh_pass_through for inter-mesh routers),
        // so feeding its VC1 sender otherwise would create an undrained channel.
        if (enable_vc1 && enable_mesh_pass_through && (topology == Topology::Mesh || topology == Topology::Torus)) {
            // VC1 sender channels are 0-based (no local worker channel); slot 3 is reserved for Z.
            constexpr uint32_t vc1_mesh_to_z_channel = builder_config::num_downstream_edms_2d_vc1;  // 3
            mapping.add_target(
                1,  // VC1
                0,  // Receiver channel 0 (VC1 only has one receiver channel)
                ConnectionTarget(
                    ConnectionType::MESH_TO_Z,
                    1,                      // Target Z router VC1
                    vc1_mesh_to_z_channel,  // Target sender channel (resolved by Z router)
                    RoutingDirection::Z));  // Target is Z router
        }
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
