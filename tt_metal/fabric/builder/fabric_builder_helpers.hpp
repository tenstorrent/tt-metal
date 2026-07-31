// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <utility>
#include <vector>

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
