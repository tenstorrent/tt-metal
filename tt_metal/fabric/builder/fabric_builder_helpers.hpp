// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <optional>
#include <utility>
#include <vector>

#include <tt_stl/small_vector.hpp>

#include "hostdevcommon/fabric_common.h"
#include "tt_metal/fabric/builder/fabric_builder_config.hpp"
#include "tt_metal/fabric/fabric_context.hpp"

namespace tt::tt_fabric {

namespace builder {

bool is_east_or_west(eth_chan_directions direction);
bool is_north_or_south(eth_chan_directions direction);

// RoutingDirection and eth_chan_directions order their compass differently; map explicitly.
inline eth_chan_directions routing_direction_to_eth_direction(RoutingDirection direction) {
    switch (direction) {
        case RoutingDirection::E: return eth_chan_directions::EAST;
        case RoutingDirection::W: return eth_chan_directions::WEST;
        case RoutingDirection::N: return eth_chan_directions::NORTH;
        case RoutingDirection::S: return eth_chan_directions::SOUTH;
        case RoutingDirection::Z: return eth_chan_directions::Z;
        default: TT_FATAL(false, "routing_direction_to_eth_direction: not a port direction");
    }
}

// Inverse of routing_direction_to_eth_direction.
inline RoutingDirection eth_direction_to_routing_direction(eth_chan_directions direction) {
    switch (direction) {
        case eth_chan_directions::EAST: return RoutingDirection::E;
        case eth_chan_directions::WEST: return RoutingDirection::W;
        case eth_chan_directions::NORTH: return RoutingDirection::N;
        case eth_chan_directions::SOUTH: return RoutingDirection::S;
        case eth_chan_directions::Z: return RoutingDirection::Z;
        default: TT_FATAL(false, "eth_direction_to_routing_direction: not a port direction");
    }
}

// The canonical direction <-> slot bijection for 2D mesh-like routers.
//
// A router facing D has four non-self directions; the compact index is a direction's rank among
// them in enum order (E, W, N, S, Z) with D removed. Connection placement
// (get_downstream_sender_channel), producer naming (get_sender_channel_direction), and receiver
// downstream indexing (get_receiver_channel_compact_index) all derive from this one relation, so
// the slot a producer is placed into and the producer named by that slot cannot drift apart.
//
// Precondition (checked by the wrappers, not here): producer != facing and both in [E, Z].
constexpr size_t direction_compact_index(eth_chan_directions producer, eth_chan_directions facing) {
    const size_t p = static_cast<size_t>(producer);
    const size_t f = static_cast<size_t>(facing);
    return p < f ? p : p - 1;
}

// Inverse of direction_compact_index: the direction at `compact` among facing's non-self
// directions, in enum order.
constexpr eth_chan_directions direction_from_compact_index(eth_chan_directions facing, size_t compact) {
    const size_t f = static_cast<size_t>(facing);
    return static_cast<eth_chan_directions>(compact < f ? compact : compact + 1);
}

// The producer slots of one router: which sender channel each non-self direction feeds, and which
// direction feeds each channel. One rule, both directions, the VC worker offset included.
//
// eth vocabulary deliberately: the compact ranking is defined by eth_chan_directions order
// (E,W,N,S,Z), which is NOT RoutingDirection order (N,E,S,W,Z). Callers reasoning in
// RoutingDirection convert at the boundary, which is where that difference belongs.
//
// The ERISC kernel encodes the same ranking in its own tables (cross-area); if this changes, they
// must change with it.
class RouterProducerSlots {
public:
    struct Slot {
        uint32_t channel;
        eth_chan_directions producer;
    };

    // The VC routing modes the two-argument form assumes: VC0 with turns, VC1 the inter-mesh VC, VC2 a line where it
    // has a forwarding channel (a neighbour-mode VC2 has only its worker, which the line rule gives too).
    static constexpr builder_config::VcRoutings default_routing = {
        builder_config::VcRouting::FULL_2D, builder_config::VcRouting::INTER_MESH, builder_config::VcRouting::LINE};

    // Per router, not per VC: the ranking depends only on facing, and the VC contributes its worker offset and whether
    // it routes as a line (its routing mode). Takes this router's own sender counts so slot ranges are its, not the
    // family max's.
    RouterProducerSlots(
        eth_chan_directions facing,
        const std::array<uint32_t, builder_config::MAX_NUM_VCS>& sender_counts,
        const builder_config::VcRoutings& routing = default_routing);

    // Channel 0 on a VC whose mode has a worker channel (VC0 always; VC2; a 1D fabric's line VCs); nullopt on the
    // inter-mesh VC1, which has no worker. Whether that channel carries an injection guard is not decided here.
    std::optional<uint32_t> worker_channel(uint32_t vc) const;

    // The producer feeding `channel` on `vc`, or nullopt: the worker slot and a channel this router
    // does not have have no producer. A line VC's one forwarding channel (channel 1, after its
    // worker) is fed by the router facing the opposite way; a neighbour-mode VC has only its worker.
    std::optional<eth_chan_directions> producer_at(uint32_t vc, uint32_t channel) const;

    // The channel `producer` feeds on `vc`, or nullopt for the facing direction itself, a producer
    // with no slot on the VC (on a line VC all but the opposite direction), or a slot this router does
    // not have.
    std::optional<uint32_t> channel_for(uint32_t vc, eth_chan_directions producer) const;

    // Only the producer slots this router actually has (bounded by its own count), in eth order;
    // on a line VC at most the opposite direction's.
    ttsl::SmallVector<Slot, 4> producer_slots(uint32_t vc) const;

    uint32_t sender_count(uint32_t vc) const;

private:
    eth_chan_directions facing_;
    std::array<uint32_t, builder_config::MAX_NUM_VCS> sender_counts_;
    builder_config::VcRoutings routing_;

    // The ONE place the VC rule lives: a VC whose mode has a worker channel starts its forwarding channels after it.
    uint32_t worker_offset(uint32_t vc) const { return builder_config::vc_routing_has_worker(routing_[vc]) ? 1 : 0; }
    // A VC that routes straight on (line) or not at all past its worker (neighbour).
    bool routes_straight(uint32_t vc) const {
        return routing_[vc] == builder_config::VcRouting::LINE || routing_[vc] == builder_config::VcRouting::NEIGHBOUR;
    }
    // A line VC's one producer is the direction opposite the facing (none for Z).
    std::optional<eth_chan_directions> line_producer() const;
};

eth_chan_directions get_sender_channel_direction(eth_chan_directions my_direction, size_t sender_channel_index);

// The sender channel on the downstream router that a producer facing `upstream_direction` feeds,
// on the given VC: the producer's compact index among the downstream router's non-self directions,
// plus the VC0 worker offset (VC0 sender channel 0 is the local worker). This is the production
// placement rule, shared so tests can compute the same slot rather than relying on stored
// bookkeeping channels.
inline uint32_t get_downstream_sender_channel_for_vc(
    bool is_2d_routing, uint32_t vc, eth_chan_directions upstream_direction, eth_chan_directions downstream_direction) {
    if (!is_2d_routing) {
        return 1;  // 1D: sender channel 1 for forwarding
    }
    const size_t compact = direction_compact_index(upstream_direction, downstream_direction);
    return vc == 0 ? static_cast<uint32_t>(1 + compact) : static_cast<uint32_t>(compact);
}

// Helper function to determine perpendicular directions
// E/W direction returns N/S as perpendicular; N/S direction returns E/W as perpendicular
std::pair<eth_chan_directions, eth_chan_directions> get_perpendicular_directions(eth_chan_directions direction);

// Helper function to get directions for inter-mux connections
// Returns all directions except the current direction
// exclude_z: if true, excludes Z direction (for modes that don't support Z like MUX)
std::vector<eth_chan_directions> get_all_other_directions(eth_chan_directions direction, bool exclude_z = false);

}  // namespace builder

inline uint32_t get_worker_connected_sender_channel() {
    // Sender channel 0 is always for local worker in the new design
    return 0;
}

// A receiver channel has up to 4 downstream EDMs (3 mesh directions + optional Z)
// This helper returns the index at which the receiver channel should store the downstream EDMs information.
// The index is 0-2 for mesh directions (N/E/S/W excluding own direction), 3 for Z direction.
inline size_t get_receiver_channel_compact_index(
    const eth_chan_directions receiver_direction, const eth_chan_directions downstream_direction) {
    // The downstream view of the same canonical bijection: the downstream direction's rank among
    // this router's non-self directions.
    const size_t compact_index = builder::direction_compact_index(downstream_direction, receiver_direction);
    if (downstream_direction == eth_chan_directions::Z) {
        TT_FATAL(
            compact_index == 3,
            "Z is the last direction in enum order, so its compact index must be 3; got {} instead, "
            "which means the direction<->slot bijection no longer holds",
            compact_index);
    }
    return compact_index;
}
}  // namespace tt::tt_fabric
