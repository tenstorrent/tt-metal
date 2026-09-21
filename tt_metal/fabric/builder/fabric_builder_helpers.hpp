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
constexpr inline size_t direction_compact_index(eth_chan_directions producer, eth_chan_directions facing) {
    const size_t p = static_cast<size_t>(producer);
    const size_t f = static_cast<size_t>(facing);
    return p < f ? p : p - 1;
}

// Inverse of direction_compact_index: the direction at `compact` among facing's non-self
// directions, in enum order.
constexpr inline eth_chan_directions direction_from_compact_index(eth_chan_directions facing, size_t compact) {
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

    // Per router, not per VC: the ranking depends only on facing, and the VC contributes only the
    // worker offset. Takes this router's own sender counts so slot ranges are its, not the family
    // max's.
    RouterProducerSlots(
        eth_chan_directions facing, const std::array<uint32_t, builder_config::MAX_NUM_VCS>& sender_counts);

    // Channel 0 on a VC whose channel 0 is worker-type (VC0 always; VC2 when this router has a VC2
    // sender); nullopt on VC1, which has no worker. Whether that channel carries an injection
    // guard is not decided here.
    std::optional<uint32_t> worker_channel(uint32_t vc) const;

    // The producer feeding `channel` on `vc`, or nullopt: the worker slot, a channel this router
    // does not have, and VC2 (whose single sender is worker-type) have no producer.
    std::optional<eth_chan_directions> producer_at(uint32_t vc, uint32_t channel) const;

    // The channel `producer` feeds on `vc`, or nullopt for the facing direction itself, a VC with
    // no producer mapping (VC2), or a slot this router does not have.
    std::optional<uint32_t> channel_for(uint32_t vc, eth_chan_directions producer) const;

    // Only the producer slots this router actually has (bounded by its own count), in eth order;
    // empty on VC2.
    ttsl::SmallVector<Slot, 4> producer_slots(uint32_t vc) const;

    uint32_t sender_count(uint32_t vc) const;

private:
    eth_chan_directions facing_;
    std::array<uint32_t, builder_config::MAX_NUM_VCS> sender_counts_;

    static constexpr uint32_t worker_offset(uint32_t vc) { return vc == 0 ? 1 : 0; }  // the ONE place the VC rule lives
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
