// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/builder/fabric_master_router_selection.hpp"

#include <algorithm>
#include <tuple>

#include <tt_stl/assert.hpp>

namespace tt::tt_fabric {

namespace {

template <eth_chan_directions FACING>
uint32_t count_live_2d_downstreams(uint32_t downstream_count, bool turns_packets) {
    constexpr bool east_west = FACING == eth_chan_directions::EAST || FACING == eth_chan_directions::WEST;
    constexpr auto downstream_directions = Routing2DCodec::fwd_dirs<FACING>();

    const size_t entries = std::min<size_t>(downstream_count, downstream_directions.size());
    uint32_t live = 0;
    for (size_t entry = 0; entry < entries; entry++) {
        const auto name = downstream_directions[entry];
        const bool is_north_south = name == eth_chan_directions::NORTH || name == eth_chan_directions::SOUTH;
        if (!(east_west && !turns_packets && is_north_south)) {
            live++;
        }
    }
    return live;
}

// The number of downstreams a router facing `facing` forwards to on a VC.
uint32_t count_live_downstreams(
    eth_chan_directions facing,
    bool is_2d_fabric,
    uint32_t downstream_count,
    bool is_intermesh_router,
    bool is_intramesh_router_on_edge) {
    if (!is_2d_fabric) {
        return downstream_count > 0 ? 1 : 0;
    }
    const bool turns_packets = is_intermesh_router || is_intramesh_router_on_edge;
    switch (facing) {
        case eth_chan_directions::EAST:
            return count_live_2d_downstreams<eth_chan_directions::EAST>(downstream_count, turns_packets);
        case eth_chan_directions::WEST:
            return count_live_2d_downstreams<eth_chan_directions::WEST>(downstream_count, turns_packets);
        case eth_chan_directions::NORTH:
            return count_live_2d_downstreams<eth_chan_directions::NORTH>(downstream_count, turns_packets);
        case eth_chan_directions::SOUTH:
            return count_live_2d_downstreams<eth_chan_directions::SOUTH>(downstream_count, turns_packets);
        case eth_chan_directions::Z:
            return count_live_2d_downstreams<eth_chan_directions::Z>(downstream_count, turns_packets);
        case eth_chan_directions::COUNT: break;
    }
    TT_THROW("Router facing {} has no downstream order", static_cast<uint32_t>(facing));
}

// The live downstreams of every VC the router forwards on, added up.
uint32_t count_forwarding_downstreams(const MasterRouterCandidate& candidate) {
    uint32_t total = 0;
    for (const uint32_t downstream_count : candidate.downstream_count_by_vc) {
        total += count_live_downstreams(
            candidate.facing,
            candidate.is_2d_fabric,
            downstream_count,
            candidate.is_intermesh_router,
            candidate.is_intramesh_router_on_edge);
    }
    return total;
}

}  // namespace

chan_id_t pick_local_handshake_master_router(std::span<const MasterRouterCandidate> candidates) {
    TT_FATAL(!candidates.empty(), "A chip's local handshake master is chosen from at least one router");

    const auto size_order = [](const MasterRouterCandidate& candidate) {
        return std::make_tuple(
            candidate.is_intramesh_router_on_edge, count_forwarding_downstreams(candidate), candidate.eth_chan);
    };

    const auto smallest = std::min_element(
        candidates.begin(), candidates.end(), [&](const MasterRouterCandidate& a, const MasterRouterCandidate& b) {
            // Two step comparison:
            // 1. Is the router an intramesh edge router? If so, it has a larger kernel due to the extra checks
            // for leaving the mesh.
            // 2. Which router has the fewest forwarding downstreams? Routers with fewer downstreams have less instances
            // of code for forwarding packets.
            return size_order(a) < size_order(b);
        });
    return smallest->eth_chan;
}

}  // namespace tt::tt_fabric
