// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

#include "fabric_graph_types.hpp"
#include "fabric_wiring_rules.hpp"

namespace tt::tt_fabric::graph {

// Each router's channel slots: how many senders per VC, and the slot each channel takes. No addresses yet.
struct ChannelGraph {
    LogicalGraph logical;
    std::vector<std::vector<uint32_t>> sender_counts;  // [router][vc]
    std::vector<uint32_t> index_in_vc;                 // [channel]
};

// The slot the kernel expects this channel in: the worker's, or the one stock gives the producer's direction.
uint32_t stock_channel_index(const LogicalGraph& graph, ChannelIndex channel, bool two_d);

// A router gets the senders it uses (at least one per fabric VC, so the kernel sees every VC).
ChannelGraph lower_to_channels(LogicalGraph logical, bool two_d);

// Legacy pass: widen every router to the stock shape (router_vc_shape), used slots or not. Stock sizes
// a router by its family, not by what feeds it; this is here for parity and goes away with #65720.
void legacy_stock_sender_counts(ChannelGraph& graph, const WiringOptions& options);

}  // namespace tt::tt_fabric::graph
