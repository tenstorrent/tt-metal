// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <functional>
#include <vector>

#include "fabric_graph_types.hpp"
#include "fabric_wiring_rules.hpp"
#include "tt_metal/fabric/builder/protected_domain_effect.hpp"

namespace tt::tt_fabric::graph {

// A stock-sized sender slot no channel uses. Stock services it anyway, and flags it as if the producer
// stock names for it fed it. Part of the legacy pass below; goes away with #65720.
struct LegacySlot {
    RouterIndex router = 0;
    uint32_t vc = 0;
    uint32_t index = 0;
    bool enters_ring = false;
};

// Each router's channel slots: how many senders per VC, and the slot each channel takes. No addresses yet.
struct ChannelGraph {
    LogicalGraph logical;
    std::vector<std::vector<uint32_t>> sender_counts;  // [router][vc]
    std::vector<uint32_t> index_in_vc;                 // [channel]
    std::vector<bool> enters_ring;                     // [channel] sender that acquires a protected ring
    std::vector<LegacySlot> legacy_slots;
};

// The slot the kernel expects this channel in: the worker's, or the one stock gives the producer's direction.
uint32_t stock_channel_index(const LogicalGraph& graph, ChannelIndex channel, bool two_d);

// A router gets the senders it uses (at least one per fabric VC, so the kernel sees every VC).
ChannelGraph lower_to_channels(LogicalGraph logical, bool two_d);

// Legacy pass: widen every router to the stock shape (router_vc_shape), used slots or not, and keep the
// unused slots as legacy_slots. Stock sizes a router by its family, not by what feeds it; this is here for
// parity and goes away with #65720.
void legacy_stock_sender_counts(ChannelGraph& graph, const WiringOptions& options);

// The chip's protected-ring queries, for express meshes. On device: make_protected_ring_queries.
using RingQueries = std::function<ProtectedRingQueries(const FabricNodeId&)>;

// Which senders acquire a protected ring (the kernel's injection flag, for bubble flow control): the
// stock injection policy, asked about each channel's producer. Run after the legacy pass so its slots get
// flags too.
void mark_ring_entries(ChannelGraph& graph, const WiringOptions& options, const RingQueries& ring_queries);

}  // namespace tt::tt_fabric::graph
