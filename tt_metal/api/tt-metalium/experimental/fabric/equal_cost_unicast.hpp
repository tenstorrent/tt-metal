// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Experimental API; subject to change without notice.

#include <array>
#include <cstdint>
#include <optional>
#include <vector>
#include <hostdevcommon/fabric_common.h>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>

namespace tt::tt_metal::experimental::fabric {

// Opaque route arguments produced by Fabric. A default object requests canonical routing.
class UnicastRoute {
public:
    UnicastRoute() = default;
    tt::tt_fabric::eth_chan_directions initial_direction() const {
        return static_cast<tt::tt_fabric::eth_chan_directions>(args_[0]);
    }
    uint32_t num_hops() const { return args_[1]; }
    const std::vector<uint32_t>& runtime_args() const { return args_; }

private:
    friend std::optional<std::array<UnicastRoute, 2>> get_equal_cost_unicast_routes(
        const tt::tt_fabric::FabricNodeId&,
        const tt::tt_fabric::FabricNodeId&,
        const std::array<tt::tt_fabric::FabricNodeId, 2>&);
    std::vector<uint32_t> args_{tt::tt_fabric::eth_chan_directions::COUNT, 0};
};

// Opt-in load balancing for an intra-mesh unicast. Return two supported routes, one
// through each supplied direct neighbor, only when both use distinct physical egresses
// and match the canonical route's hop count. Fabric owns path selection, validation,
// header capacity checks, and encoding. Unsupported paths return nullopt; retain
// canonical routing in that case. Routes are valid until Fabric is closed.
std::optional<std::array<UnicastRoute, 2>> get_equal_cost_unicast_routes(
    const tt::tt_fabric::FabricNodeId& source,
    const tt::tt_fabric::FabricNodeId& destination,
    const std::array<tt::tt_fabric::FabricNodeId, 2>& neighbors);

}  // namespace tt::tt_metal::experimental::fabric
