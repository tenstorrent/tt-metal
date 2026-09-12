// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/streaming_profiler/streaming_profiler_link_sync.hpp"

#include <algorithm>
#include <tuple>

#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>

#include "impl/context/metal_context.hpp"
#include "llrt/tt_cluster.hpp"

namespace tt::tt_metal::streaming_profiler::link_sync {

namespace {
// Whether an eth core can carry the sync: any connected core without fabric; with fabric, only a core the topology
// gave a router, since the router is what runs the link end.
bool eligible(const tt::Cluster& cluster, uint32_t chip, const CoreCoord& eth_logical) {
    auto& mc = MetalContext::instance();
    if (mc.get_fabric_config() == tt_fabric::FabricConfig::DISABLED) {
        return true;
    }
    const auto& cp = mc.get_control_plane();
    const auto node = cp.get_fabric_node_id_from_physical_chip_id(static_cast<ChipId>(chip));
    const auto& soc = cluster.get_soc_desc(static_cast<ChipId>(chip));
    for (const auto& [chan, direction] : cp.get_active_fabric_eth_channels(node)) {
        if (soc.get_eth_core_for_channel(chan, CoordSystem::LOGICAL) == eth_logical) {
            return true;
        }
    }
    return false;
}
}  // namespace

std::optional<Link> link_between(const tt::Cluster& cluster, uint32_t chip_x, uint32_t chip_y) {
    const uint32_t lo = std::min(chip_x, chip_y), hi = std::max(chip_x, chip_y);
    if (lo == hi) {
        return std::nullopt;
    }
    const auto by_peer = cluster.get_ethernet_cores_grouped_by_connected_chips(static_cast<ChipId>(lo));
    const auto it = by_peer.find(static_cast<ChipId>(hi));
    if (it == by_peer.end()) {
        return std::nullopt;
    }
    for (const CoreCoord& eth_a : it->second) {
        const CoreCoord eth_b =
            std::get<1>(cluster.get_connected_ethernet_core(std::make_tuple(static_cast<ChipId>(lo), eth_a)));
        if (eligible(cluster, lo, eth_a) && eligible(cluster, hi, eth_b)) {
            return Link{.chip_a = lo, .chip_b = hi, .eth_a = eth_a, .eth_b = eth_b};
        }
    }
    return std::nullopt;
}

Role role_of(const tt::Cluster& cluster, uint32_t chip, const CoreCoord& eth_logical) {
    const auto by_peer = cluster.get_ethernet_cores_grouped_by_connected_chips(static_cast<ChipId>(chip));
    for (const auto& [peer, cores] : by_peer) {
        if (std::find(cores.begin(), cores.end(), eth_logical) == cores.end()) {
            continue;
        }
        const auto link = link_between(cluster, chip, static_cast<uint32_t>(peer));
        if (!link) {
            return Role::None;
        }
        if (link->chip_a == chip && link->eth_a == eth_logical) {
            return Role::Sender;
        }
        if (link->chip_b == chip && link->eth_b == eth_logical) {
            return Role::Receiver;
        }
        return Role::None;
    }
    return Role::None;
}

}  // namespace tt::tt_metal::streaming_profiler::link_sync
