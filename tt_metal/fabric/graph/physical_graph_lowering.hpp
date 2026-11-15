// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <set>
#include <vector>

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

// Where every channel lives in L1, and the NoC settings of each link pairing. Noc command buffers stay auto.
struct PhysicalGraph : ChannelGraph {
    std::vector<ChannelAddress> addresses;  // [channel]
    std::vector<LinkPairing> links;
};

// The channel allocator's layout for a router with these sender counts.
using ChannelAllocator =
    std::function<RouterChannelLayout(const std::array<size_t, builder_config::MAX_NUM_VCS>& sender_counts)>;

// Addresses from the allocator, one router at a time. Link pairings come from a separate pass.
PhysicalGraph lower_to_physical(ChannelGraph graph, const ChannelAllocator& allocate);

// Legacy pass: pair routers the way the stock builder does (every direction pair in 2D; opposite
// directions, or the two directions of a folded corner, in 1D), noc edge between them or not. Stock pairs
// them to keep its builder simple; this is here for parity and goes away a piece at a time.
std::vector<LinkPairing> legacy_link_pairings(
    const LogicalGraph& graph, Topology topology, const std::set<MeshId>& wrap_around_meshes);

}  // namespace tt::tt_fabric::graph
