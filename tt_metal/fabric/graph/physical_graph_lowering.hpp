// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <set>
#include <vector>

#include <hostdevcommon/fabric_common.h>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>

#include "channel_graph_lowering.hpp"
#include "tt_metal/fabric/builder/fabric_static_sized_channels_allocator.hpp"

namespace tt::tt_fabric::graph {

struct ChannelAddress {
    uint32_t base = 0;
    uint32_t slots = 0;
};

// Two routers on one chip paired for a link: both use `noc_vc`, and core placement follows link `link` of
// `num_links`.
struct LinkPairing {
    RouterIndex a = 0;
    RouterIndex b = 0;
    uint32_t link = 0;
    uint32_t num_links = 0;
    uint32_t noc_vc = 0;
};

// Where every channel lives in L1, which eth channel each router runs on, and the NoC settings of each link
// pairing. Noc command buffers stay auto.
struct PhysicalGraph : ChannelGraph {
    std::vector<RouterChannelLayout> layouts;  // [router] every slot, legacy ones included
    std::vector<ChannelAddress> addresses;     // [channel]
    std::vector<chan_id_t> eth_chans;          // [router]
    std::vector<LinkPairing> links;
    std::map<FabricNodeId, RouterIndex> handshake_masters;  // per chip
};

// The channel allocator's layout for a router with these sender counts.
using ChannelAllocator =
    std::function<RouterChannelLayout(const std::array<size_t, builder_config::MAX_NUM_VCS>& sender_counts)>;

// Addresses from the allocator, one router at a time. Link pairings come from a separate pass.
PhysicalGraph lower_to_physical(ChannelGraph graph, const ChannelAllocator& allocate);

// The live eth channels facing a direction, by plane. On device:
// ControlPlane::get_active_fabric_eth_routing_planes_in_direction.
using LiveEthChannels = std::function<std::vector<chan_id_t>(const FabricNodeId&, RoutingDirection)>;

// Each router's eth channel: the live channel of its plane.
std::vector<chan_id_t> bind_eth_channels(const LogicalGraph& graph, const LiveEthChannels& live);

// Legacy pass: the chip's handshake master is the router stock's FabricBuilder puts first in its hash
// maps, so this fills the same maps the same way. Goes when the kernels agree on a rule.
std::map<FabricNodeId, RouterIndex> legacy_handshake_masters(
    const LogicalGraph& graph, const std::vector<chan_id_t>& eth_chans);

// Legacy pass: pair routers the way the stock builder does (every direction pair in 2D; opposite
// directions, or the two directions of a folded corner, in 1D), noc edge between them or not. Stock pairs
// them to keep its builder simple; this is here for parity and goes away a piece at a time.
std::vector<LinkPairing> legacy_link_pairings(
    const LogicalGraph& graph, Topology topology, const std::set<MeshId>& wrap_around_meshes);

}  // namespace tt::tt_fabric::graph
