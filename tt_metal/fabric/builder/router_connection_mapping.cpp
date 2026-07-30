// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/builder/router_connection_mapping.hpp"
#include "tt_metal/fabric/builder/fabric_builder_config.hpp"
#include "tt_metal/fabric/fabric_builder_context.hpp"

#include <algorithm>
#include <array>

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

std::vector<RoutingDirection> RouterConnectionMapping::express_outbound_directions(
    RoutingDirection direction, EdgeCapability ingress_capability) {
    static constexpr std::array<RoutingDirection, 4> k_cardinals = {
        RoutingDirection::N, RoutingDirection::E, RoutingDirection::S, RoutingDirection::W};

    // A boundary landing is a route root rather than a packet already in its X phase, so it may begin
    // Y even when its local port is E or W. Every non-self output stays available to it.
    const bool is_landing = ingress_capability == EdgeCapability::INTERMESH;

    std::vector<RoutingDirection> outbound;
    if (direction == RoutingDirection::Z) {
        // Arrived over an express chord and still in the Y phase: continue Y cardinally, or turn onto
        // X. Z itself is excluded because that would return the packet over the link it arrived on.
        outbound.assign(k_cardinals.begin(), k_cardinals.end());
        return outbound;
    }

    if (is_x_axis_direction(direction) && !is_landing) {
        // Ordinary X ingress: dimension order forbids returning to Y, so the only legal output is the
        // opposite X direction, continuing around the X ring.
        outbound.push_back(get_opposite_direction(direction));
        return outbound;
    }

    // Y ingress, or an intermesh landing: continue on the opposite cardinal first, then the two
    // orthogonal turns, then the express chord.
    outbound.push_back(get_opposite_direction(direction));
    for (const auto candidate : k_cardinals) {
        if (candidate != direction && candidate != get_opposite_direction(direction)) {
            outbound.push_back(candidate);
        }
    }
    outbound.push_back(RoutingDirection::Z);
    return outbound;
}

std::vector<RoutingDirection> RouterConnectionMapping::wired_express_outbound_directions(
    RoutingDirection direction, EdgeCapability ingress_capability, bool has_intramesh_express) {
    auto outbound = express_outbound_directions(direction, ingress_capability);

    // A Z output is realizable only when this chip terminates an intramesh express chord. On a
    // chip whose only Z edge crosses a mesh boundary, a Z target would resolve to the intermesh
    // Z router and leak same-mesh traffic onto the boundary link; the MESH_TO_Z template is the
    // only correct way to reach that router. On a chip with no Z edge the target would match
    // nothing at connection resolution, but not emitting it keeps the transition set honest --
    // and keeps the pass-through VC1 MESH_TO_Z channel (slot 3) free of aliases.
    if (!has_intramesh_express) {
        std::erase(outbound, RoutingDirection::Z);
    }
    return outbound;
}

bool RouterConnectionMapping::is_express_producer_wired(
    RoutingDirection producer_direction,
    EdgeCapability producer_capability,
    RoutingDirection egress_direction,
    bool has_intramesh_express) {
    const auto outbound =
        wired_express_outbound_directions(producer_direction, producer_capability, has_intramesh_express);
    return std::find(outbound.begin(), outbound.end(), egress_direction) != outbound.end();
}

RouterConnectionMapping::PerDirectionCapabilities RouterConnectionMapping::canonical_express_endpoint_capabilities() {
    PerDirectionCapabilities caps;
    for (size_t i = 0; i < caps.size() - 1; ++i) {
        caps[i] = EdgeCapability::INTRAMESH_CARDINAL;
    }
    caps[static_cast<size_t>(RoutingDirection::Z)] = EdgeCapability::INTRAMESH_EXPRESS;
    return caps;
}

uint32_t RouterConnectionMapping::express_vc0_producer_arity(
    RoutingDirection direction, const PerDirectionCapabilities& caps) {
    static constexpr std::array<RoutingDirection, 5> k_all_directions = {
        RoutingDirection::N, RoutingDirection::E, RoutingDirection::S, RoutingDirection::W, RoutingDirection::Z};

    // The chord state the wiring rule consults is this chip's own Z-edge capability, not a global.
    const bool has_chord = caps[static_cast<size_t>(RoutingDirection::Z)] == EdgeCapability::INTRAMESH_EXPRESS;

    uint32_t count = 1;  // sender channel 0 is the local worker
    for (const auto producer : k_all_directions) {
        if (producer == direction) {
            continue;
        }
        const auto& capability = caps[static_cast<size_t>(producer)];
        if (!capability.has_value()) {
            continue;  // direction absent on this chip: no producer to wire
        }
        if (is_express_producer_wired(producer, *capability, direction, has_chord)) {
            ++count;
        }
    }
    return count;
}

uint32_t RouterConnectionMapping::express_mesh_vc0_sender_count() {
    static constexpr std::array<RoutingDirection, 5> k_all_directions = {
        RoutingDirection::N, RoutingDirection::E, RoutingDirection::S, RoutingDirection::W, RoutingDirection::Z};

    // The canonical endpoint chip attains the structural ceiling: every Y and X producer wires
    // into an E/W facing under any capability assignment, so no per-chip set produces a wider one.
    const auto caps = canonical_express_endpoint_capabilities();
    uint32_t count = 0;
    for (const auto direction : k_all_directions) {
        count = std::max(count, express_vc0_producer_arity(direction, caps));
    }
    return count;
}

uint32_t RouterConnectionMapping::express_mesh_vc1_sender_count() {
    // VC1 forwards the same wired producer set as VC0 on every facing, minus the worker slot, so
    // vc1_arity(f) == vc0_arity(f) - 1 for every facing f. The family max therefore commutes with
    // the subtraction: max_f(vc0_arity(f)) - 1 == max_f(vc0_arity(f) - 1).
    return express_mesh_vc0_sender_count() - 1;
}

uint32_t RouterConnectionMapping::mesh_router_vc0_sender_count(
    bool /*has_intermesh_z_edge*/, bool express_routing_enabled) {
    if (express_routing_enabled) {
        return express_mesh_vc0_sender_count();  // family max over facing of the express rule
    }
    // Legacy rule: the worker plus every non-self cardinal producer. An intermesh Z edge adds
    // nothing on VC0: the boundary's VC0 receiver forwards nowhere, so no from-Z producer exists
    // on VC0 regardless of has_intermesh_z_edge.
    return 1 + builder_config::num_downstream_edms_2d_vc0;
}

uint32_t RouterConnectionMapping::mesh_router_vc1_sender_count(
    bool has_intermesh_z_edge, bool express_routing_enabled) {
    if (has_intermesh_z_edge) {
        // Legacy non-self downstreams plus the from-Z slot for the boundary's VC1 fanout.
        return builder_config::num_downstream_edms_2d_vc1 + 1;
    }
    if (express_routing_enabled) {
        return express_mesh_vc1_sender_count();  // family max over facing of the express rule
    }
    // Legacy rule: every non-self cardinal receiver forwards into this router on VC1.
    return builder_config::num_downstream_edms_2d_vc1;
}

namespace {

// Emit the flat-index prefix sums and enforce the capacity ceilings at the one construction
// site. The ceiling checks are what turn the "express with VC2 reaches it exactly (5+4+1)"
// comment into a guarantee: every family's shape passes through here, with zero margin on
// senders and on the 32 stream registers the flat space maps onto.
void finalize_vc_shape_bases(RouterConnectionMapping::RouterVcShape& shape) {
    uint32_t sender_base = 0;
    uint32_t receiver_base = 0;
    for (size_t vc = 0; vc < shape.sender_counts.size(); ++vc) {
        shape.sender_flat_base[vc] = sender_base;
        sender_base += shape.sender_counts[vc];
        shape.receiver_flat_base[vc] = receiver_base;
        receiver_base += shape.receiver_counts[vc];
    }
    TT_FATAL(
        sender_base <= builder_config::num_max_sender_channels,
        "Router shape needs {} sender channels, over the {}-channel ceiling",
        sender_base,
        builder_config::num_max_sender_channels);
    TT_FATAL(
        receiver_base <= builder_config::num_max_receiver_channels,
        "Router shape needs {} receiver channels, over the {}-channel ceiling",
        receiver_base,
        builder_config::num_max_receiver_channels);
}

}  // namespace

RouterConnectionMapping::RouterVcShape RouterConnectionMapping::router_vc_shape(
    Topology topology,
    RoutingDirection facing,
    EdgeCapability edge_capability,
    bool has_intermesh_z_edge,
    bool express_routing_enabled,
    const IntermeshVCConfig* vc_config) {
    const bool z_boundary = (facing == RoutingDirection::Z && edge_capability == EdgeCapability::INTERMESH);
    const bool requires_vc1 = vc_config && vc_config->requires_vc1;
    const bool requires_vc2 = vc_config && vc_config->requires_vc2;

    RouterVcShape shape{};

    // num_vcs is config-only: the identical answer for 1D and 2D, by design. A 1D router can
    // therefore report more VCs than it has channels for, since VC1/VC2 channels are never
    // created on 1D. That oddity is preserved, not fixed -- get_all_sender_mappings() already
    // tolerates counts exceeding created channels. Whether any 1D configuration should ever set
    // requires_vc1 is a separate question, deliberately out of scope here.
    shape.num_vcs = requires_vc2 ? 3 : (requires_vc1 ? 2 : 1);

    // The intermesh boundary family is 2D-only by construction. Today the boundary VC0 arm of the
    // channel mapping precedes the 2D check, so a 1D Z boundary would silently report 5 -- with
    // topology in hand that becomes an explicit configuration error instead.
    TT_FATAL(!z_boundary || is_2D_topology(topology), "A Z-facing intermesh boundary router requires a 2D topology");

    if (!is_2D_topology(topology)) {
        // 1D counts: worker, plus one forwarding peer on Linear/Ring. No VC1 or VC2 channels are
        // ever created, independent of the num_vcs answer above.
        shape.sender_counts = {builder_config::get_num_used_sender_channel_count(topology), 0, 0};
        shape.receiver_counts = {1, 0, 0};
        finalize_vc_shape_bases(shape);
        return shape;
    }

    // A Z-facing boundary cannot exist without VC1: its whole shape is the from-boundary VC1
    // fanout. This is where the construction error lives now (moved from the channel mapping).
    TT_FATAL(!z_boundary || requires_vc1, "A Z-facing intermesh boundary router cannot be constructed without VC1");

    // VC0: worker + wired producers.
    const uint32_t vc0_senders = z_boundary
                                     ? builder_config::num_sender_channels_intermesh_z_boundary_vc0
                                     : mesh_router_vc0_sender_count(has_intermesh_z_edge, express_routing_enabled);

    // VC1: wired producers, when the VC exists.
    const uint32_t vc1_senders = !requires_vc1 ? 0
                                 : z_boundary
                                     ? builder_config::num_sender_channels_intermesh_z_boundary_vc1
                                     : mesh_router_vc1_sender_count(has_intermesh_z_edge, express_routing_enabled);

    // VC2: one sender by VC2's own definition.
    const uint32_t vc2_senders = requires_vc2 ? 1 : 0;

    // Receivers: one per active carrier VC. The boundary services no VC2 receiver.
    const uint32_t vc0_receivers = 1;
    const uint32_t vc1_receivers = requires_vc1 ? 1 : 0;
    const uint32_t vc2_receivers = (requires_vc2 && !z_boundary) ? 1 : 0;

    shape.sender_counts = {vc0_senders, vc1_senders, vc2_senders};
    shape.receiver_counts = {vc0_receivers, vc1_receivers, vc2_receivers};
    finalize_vc_shape_bases(shape);
    return shape;
}

RouterConnectionMapping RouterConnectionMapping::for_mesh_router(
    Topology topology,
    RoutingDirection direction,
    bool has_z,
    bool enable_vc1,
    bool enable_mesh_pass_through,
    bool express_routing_enabled,
    EdgeCapability ingress_capability,
    bool has_intramesh_express) {
    RouterConnectionMapping mapping;

    // One direction-parameterized generator for every router. The edge's capability selects the
    // template; the direction selects the slot arithmetic. A Z-facing router is therefore not a
    // variant of its own: an intermesh Z edge gets the from-boundary fanout, and an express chord
    // is an ordinary mesh-like forwarding direction handled by the express path below.
    if (direction == RoutingDirection::Z) {
        if (ingress_capability == EdgeCapability::INTERMESH) {
            return z_intermesh_boundary_fanout();
        }
        TT_FATAL(
            ingress_capability == EdgeCapability::INTRAMESH_EXPRESS,
            "A same-mesh Z edge is an express chord and must carry INTRAMESH_EXPRESS capability, got {}. "
            "An ordinary cardinal-capability Z edge cannot exist.",
            to_string(ingress_capability));
        TT_FATAL(
            express_routing_enabled && (topology == Topology::Mesh || topology == Topology::Torus),
            "An express (Z) chord requires 2D Mesh/Torus routing with express routing enabled");
        // INTRAMESH_EXPRESS: fall through to the express path below, where the chord router is
        // wired to all four cardinals like any express-facing router.
    }

    // Express routing changes which local transitions are legal, so it gets its own construction.
    // Without it the wiring below is left exactly as it was: today's 2D routing is already
    // dimension-ordered, so its wired-but-unused X->Y arcs are harmless, and removing them would
    // change downstream counts, stream assignment, and L1 layout on every existing 2D configuration.
    if (express_routing_enabled && (topology == Topology::Mesh || topology == Topology::Torus)) {
        // Wired outputs, not merely legal ones: a Z target exists only where the chip terminates
        // the chord, so an intermesh Z router can only be reached through the MESH_TO_Z template.
        const auto outbound = wired_express_outbound_directions(direction, ingress_capability, has_intramesh_express);

        for (const auto egress : outbound) {
            TT_FATAL(
                egress != direction,
                "Router facing {} would be wired back over the link it arrived on",
                static_cast<int>(direction));
        }

        const size_t vc0_limit =
            builder_config::get_vc0_downstream_edm_count(/*is_2D_routing=*/true, /*express=*/true);
        TT_FATAL(
            outbound.size() <= vc0_limit,
            "Express VC0 outbound direction count ({}) exceeds the downstream EDM count ({})",
            outbound.size(),
            vc0_limit);

        // Cardinal and express outputs are realized on every carrier VC, not just VC0. Traffic that
        // has crossed a mesh boundary stays on VC1 through every later mesh, and a landed carrier can
        // still decode a Z action, so a Z output with no VC1 express sender would be unroutable.
        // target_vc always equals the source VC here: there is no VC1->VC0 landing crossover.
        for (uint32_t vc : {0u, 1u}) {
            if (vc == 1 && !enable_vc1) {
                continue;
            }
            // VC0 reserves sender channel 0 for the local worker; VC1 has no worker channel.
            const uint32_t target_channel_base = (vc == 0) ? 1 : 0;
            for (size_t i = 0; i < outbound.size(); ++i) {
                mapping.add_target(
                    vc,
                    0,  // single receiver channel per VC
                    ConnectionTarget(
                        ConnectionType::INTRA_MESH,
                        vc,
                        static_cast<uint32_t>(target_channel_base + i),
                        outbound[i]));
            }
        }

        // An intermesh Z router is a different edge from an express chord, so it keeps its own
        // capability-specific template rather than being folded into the shared maps above.
        if (has_z) {
            add_mesh_to_z_targets(mapping, topology, enable_vc1, enable_mesh_pass_through);
        }
        return mapping;
    }

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
        add_mesh_to_z_targets(mapping, topology, enable_vc1, enable_mesh_pass_through);
    }

    return mapping;
}

void RouterConnectionMapping::add_mesh_to_z_targets(
    RouterConnectionMapping& mapping, Topology topology, bool enable_vc1, bool enable_mesh_pass_through) {
    {
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
}

RouterConnectionMapping RouterConnectionMapping::z_intermesh_boundary_fanout() {
    RouterConnectionMapping mapping;

    // The boundary's VC1 receiver landing from the remote mesh fans out to every mesh direction.
    // Its VC0 senders are fed by the mesh routers' MESH_TO_Z targets (emitted on their maps), so
    // there is no VC0 map here. Constructed without inputs so no caller can pass a fact this
    // template ignores.
    std::vector<RoutingDirection> vc1_outbound_directions = {
        RoutingDirection::E,
        RoutingDirection::W,
        RoutingDirection::N,
        RoutingDirection::S,
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

RouterConnectionMapping RouterConnectionMapping::for_z_router() {
    // Alias kept for existing callers: the same construction the unified generator's capability
    // dispatch reaches for (Z, INTERMESH).
    return z_intermesh_boundary_fanout();
}

}  // namespace tt::tt_fabric
