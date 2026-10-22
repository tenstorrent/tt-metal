// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Which fabric links are host-bridged, from TT_METAL_FABRIC_HOST_BRIDGE: a comma-separated list of
//   all          every fabric link
//   intermesh    every link that leaves its mesh
//   NS | EW      the intra-mesh links on that axis
//   m:c:ch       one router: mesh m, chip c, eth channel ch (its peer router is bridged too)
// Read by the router builder (to build the router bridged) and by FabricHostBridge (to open the same links).
#pragma once

#include <algorithm>
#include <cstdio>
#include <string>
#include <string_view>
#include <vector>

#include <tt-metalium/experimental/fabric/control_plane.hpp>

namespace tt::tt_fabric::host_bridge {

namespace detail {

inline bool chan_in(const std::vector<chan_id_t>& chans, chan_id_t chan) {
    return std::find(chans.begin(), chans.end(), chan) != chans.end();
}

inline bool explicit_token_matches(std::string_view token, FabricNodeId node, chan_id_t eth_chan) {
    unsigned m = 0, c = 0, ch = 0;
    return std::sscanf(std::string(token).c_str(), "%u:%u:%u", &m, &c, &ch) == 3 && m == *node.mesh_id &&
           c == node.chip_id && ch == eth_chan;
}

// Directional tokens are evaluated on the local end only: both ends of a link lie on the same axis, and
// both or neither leave the mesh, so the far end gives the same answer.
inline bool token_matches(
    std::string_view token, const ControlPlane& control_plane, FabricNodeId node, chan_id_t eth_chan) {
    if (token == "all") {
        return true;
    }
    if (token == "intermesh") {
        return chan_in(control_plane.get_intermesh_facing_eth_chans(node), eth_chan);
    }
    if (token == "NS" || token == "EW") {
        if (!chan_in(control_plane.get_intramesh_facing_eth_chans(node), eth_chan)) {
            return false;
        }
        const auto dir = control_plane.get_eth_chan_direction(node, eth_chan);
        return token == "NS" ? (dir == eth_chan_directions::NORTH || dir == eth_chan_directions::SOUTH)
                             : (dir == eth_chan_directions::EAST || dir == eth_chan_directions::WEST);
    }
    return explicit_token_matches(token, node, eth_chan);
}

template <typename Match>
inline bool any_token(const std::string& spec, Match&& match) {
    std::string_view rest(spec);
    while (!rest.empty()) {
        const auto comma = rest.find(',');
        const auto token = rest.substr(0, comma);
        rest = comma == std::string_view::npos ? std::string_view{} : rest.substr(comma + 1);
        if (!token.empty() && match(token)) {
            return true;
        }
    }
    return false;
}

}  // namespace detail

inline bool link_is_bridged(
    const std::string& spec, const ControlPlane& control_plane, FabricNodeId node, chan_id_t eth_chan) {
    if (spec.empty()) {
        return false;
    }
    if (detail::any_token(
            spec, [&](std::string_view t) { return detail::token_matches(t, control_plane, node, eth_chan); })) {
        return true;
    }
    // An explicit m:c:ch names one router; its peer across the cable must build bridged as well.
    const auto peer = control_plane.try_get_connected_mesh_chip_chan_ids(node, eth_chan);
    return peer.has_value() && detail::any_token(spec, [&](std::string_view t) {
               return detail::explicit_token_matches(t, peer->first, peer->second);
           });
}

}  // namespace tt::tt_fabric::host_bridge
