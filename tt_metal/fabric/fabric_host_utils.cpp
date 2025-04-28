// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/fabric_edm_types.hpp>
#include <tt-metalium/fabric_types.hpp>
#include <tt-metalium/assert.hpp>
#include <magic_enum/magic_enum.hpp>
#include <umd/device/types/cluster_descriptor_types.h>  // chip_id_t
#include <tt-metalium/metal_soc_descriptor.h>
#include "impl/context/metal_context.hpp"
#include <set>
#include <vector>
#include <algorithm>
#include "tt_metal/hostdevcommon/api/hostdevcommon/fabric_routing_mode.h"

namespace tt::tt_fabric {

bool is_1d_fabric_config(const tt::tt_metal::FabricConfig& fabric_config) {
    return fabric_config == tt::tt_metal::FabricConfig::FABRIC_1D ||
           fabric_config == tt::tt_metal::FabricConfig::FABRIC_1D_RING;
}

bool is_2d_fabric_config(const tt::tt_metal::FabricConfig& fabric_config) {
    return fabric_config == tt::tt_metal::FabricConfig::FABRIC_2D ||
           fabric_config == tt::tt_metal::FabricConfig::FABRIC_2D_PUSH;
}

Topology get_1d_topology(const tt::tt_metal::FabricConfig& fabric_config) {
    switch (fabric_config) {
        case tt::tt_metal::FabricConfig::FABRIC_1D: return tt::tt_fabric::Topology::Linear;
        case tt::tt_metal::FabricConfig::FABRIC_1D_RING: return tt::tt_fabric::Topology::Ring;
        case tt::tt_metal::FabricConfig::DISABLED:
        case tt::tt_metal::FabricConfig::FABRIC_2D:
        case tt::tt_metal::FabricConfig::FABRIC_2D_PUSH:
        case tt::tt_metal::FabricConfig::CUSTOM:
            TT_THROW("Unsupported fabric config for 1D: {}", magic_enum::enum_name(fabric_config));
    }
    return tt::tt_fabric::Topology::Linear;
}

std::vector<chan_id_t> get_ordered_fabric_eth_chans(chip_id_t chip_id, const std::set<chan_id_t>& eth_chans) {
    std::vector<std::pair<chan_id_t, CoreCoord>> ordered_eth_chans_cores;
    auto soc_desc = tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(chip_id);
    for (const auto& chan : eth_chans) {
        ordered_eth_chans_cores.push_back(
            std::make_pair(chan, soc_desc.get_eth_core_for_channel(chan, CoordSystem::VIRTUAL)));
    }

    std::sort(ordered_eth_chans_cores.begin(), ordered_eth_chans_cores.end(), [](const auto& a, const auto& b) {
        return a.second.x < b.second.x;
    });

    std::vector<chan_id_t> ordered_eth_chans;
    for (const auto& [chan, _] : ordered_eth_chans_cores) {
        ordered_eth_chans.push_back(chan);
    }
    return ordered_eth_chans;
}

void set_routing_mode(RoutingMode routing_mode) {
    // override for forced routing mode
    uint16_t mode = (uint16_t)routing_mode;
    if (routing_mode != RoutingMode::Undefined) {
        return;
    }

    // Validate dimension flags are orthogonal (only one can be set)
    TT_ASSERT(
        __builtin_popcount(mode & (ROUTING_MODE_1D | ROUTING_MODE_2D | ROUTING_MODE_3D)) == 1,
        "Only one dimension mode (1D, 2D, 3D) can be active at once");

    // Validate topology flags are orthogonal
    TT_ASSERT(
        __builtin_popcount(mode & (ROUTING_MODE_RING | ROUTING_MODE_LINE | ROUTING_MODE_MESH | ROUTING_MODE_TORUS)) ==
            1,
        "Only one topology mode (RING, LINE, MESH, TORUS) can be active at once");

    // Validate push/pull flags are orthogonal
    TT_ASSERT(
        __builtin_popcount(mode & (ROUTING_MODE_PUSH | ROUTING_MODE_PULL)) <= 1,
        "PUSH and PULL routing modes cannot be used together");

    // Validate push/pull flags are only for 2D
    TT_ASSERT(
        !(mode & (ROUTING_MODE_PUSH | ROUTING_MODE_PULL)) || (mode & ROUTING_MODE_2D),
        "PUSH and PULL routing modes can only be used with 2D topology");

    // Validate 1D can't be used with MESH or TORUS
    TT_ASSERT(
        !(mode & ROUTING_MODE_1D) || !(mode & (ROUTING_MODE_MESH | ROUTING_MODE_TORUS)),
        "1D routing mode cannot be combined with MESH or TORUS topology");

    // Validate 2D can't be used with LINE or RING
    TT_ASSERT(
        !(mode & ROUTING_MODE_2D) || !(mode & (ROUTING_MODE_LINE | ROUTING_MODE_RING)),
        "2D routing mode cannot be combined with LINE or RING topology");

    auto control_plane = tt::tt_metal::MetalContext::instance().get_cluster().get_control_plane();
    SET_ROUTING_MODE(control_plane, routing_mode);
}

void set_routing_mode(Topology topology, uint32_t dimension /*, take more*/) {
    // TODO: take more parameters to set detail routing mode
    TT_ASSERT(
        dimension == 1 || dimension == 2 || dimension == 3,
        "Invalid dimension {}. Supported dimensions are 1, 2, or 3",
        dimension);

    uint16_t mode = (dimension == 1 ? ROUTING_MODE_1D : dimension == 2 ? ROUTING_MODE_2D : ROUTING_MODE_3D);
    if (topology == Topology::Ring) {
        mode |= ROUTING_MODE_RING;
    } else if (topology == Topology::Linear) {
        mode |= ROUTING_MODE_LINE;
    } else if (topology == Topology::Mesh) {
        mode |= ROUTING_MODE_MESH;
        // } else if (topology == Topology::Torus) {
        //     mode |= ROUTING_MODE_TORUS;
    }

    mode |= ROUTING_MODE_LOW_LATENCY;
    set_routing_mode((RoutingMode)mode);
}

}  // namespace tt::tt_fabric
