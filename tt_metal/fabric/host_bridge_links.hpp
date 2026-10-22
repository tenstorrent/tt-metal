// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Which fabric links are host-bridged, from TT_METAL_FABRIC_HOST_BRIDGE:
//   "intermesh"              every link that leaves its mesh
//   "m:c:ch[,m:c:ch...]"     explicit (mesh_id, chip_id, eth_chan) routers
// Read by the router builder (to build the router bridged) and by the host bridge (to open the same links).
#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>
#include <vector>

#include <tt-metalium/experimental/fabric/control_plane.hpp>

namespace tt::tt_fabric::host_bridge {

inline bool link_is_bridged(
    const std::string& spec, const ControlPlane& control_plane, FabricNodeId node, chan_id_t eth_chan) {
    if (spec.empty()) {
        return false;
    }
    if (spec == "intermesh") {
        const auto chans = control_plane.get_intermesh_facing_eth_chans(node);
        return std::find(chans.begin(), chans.end(), eth_chan) != chans.end();
    }
    std::string_view rest(spec);
    while (!rest.empty()) {
        const auto comma = rest.find(',');
        const auto item = rest.substr(0, comma);
        rest = comma == std::string_view::npos ? std::string_view{} : rest.substr(comma + 1);
        unsigned m = 0, c = 0, ch = 0;
        if (std::sscanf(std::string(item).c_str(), "%u:%u:%u", &m, &c, &ch) == 3 && m == *node.mesh_id &&
            c == node.chip_id && ch == eth_chan) {
            return true;
        }
    }
    return false;
}

}  // namespace tt::tt_fabric::host_bridge
