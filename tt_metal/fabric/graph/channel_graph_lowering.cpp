// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "channel_graph_lowering.hpp"

#include <algorithm>
#include <map>
#include <set>
#include <tuple>

#include <tt_stl/assert.hpp>

#include "tt_metal/fabric/builder/fabric_builder_helpers.hpp"
#include "tt_metal/fabric/builder/router_wiring_rules.hpp"

namespace tt::tt_fabric::graph {

uint32_t stock_channel_index(const LogicalGraph& g, ChannelIndex c, bool two_d) {
    const auto& ch = g.channels[c];
    if (ch.kind == ChannelKind::Receiver) {
        return 0;
    }
    if (ch.injection) {
        return get_worker_connected_sender_channel();
    }
    return builder::get_downstream_sender_channel_for_vc(
        two_d,
        ch.vc,
        builder::routing_direction_to_eth_direction(**ch.from),
        builder::routing_direction_to_eth_direction(*g.routers[ch.router].port.direction));
}

ChannelGraph lower_to_channels(LogicalGraph logical, bool two_d) {
    ChannelGraph cg;
    uint32_t num_vcs = 0;
    for (const auto& vc : logical.vcs) {
        num_vcs = std::max(num_vcs, vc.id + 1);
    }

    std::set<std::tuple<RouterIndex, uint32_t, ChannelKind, uint32_t>> taken;
    cg.index_in_vc.resize(logical.channels.size());
    cg.sender_counts.assign(logical.routers.size(), std::vector<uint32_t>(num_vcs, 0));
    for (ChannelIndex c = 0; c < logical.channels.size(); ++c) {
        const auto& ch = logical.channels[c];
        TT_FATAL(ch.vc < num_vcs, "fabric graph: channel on vc{} but only {} VCs", ch.vc, num_vcs);
        const uint32_t idx = stock_channel_index(logical, c, two_d);
        TT_FATAL(
            taken.emplace(ch.router, ch.vc, ch.kind, idx).second,
            "fabric graph: router {} has two {} channels at vc{} index {}",
            logical.routers[ch.router].id,
            ch.kind == ChannelKind::Sender ? "sender" : "receiver",
            ch.vc,
            idx);
        cg.index_in_vc[c] = idx;
        if (ch.kind == ChannelKind::Sender) {
            auto& n = cg.sender_counts[ch.router][ch.vc];
            n = std::max(n, idx + 1);
        }
    }
    for (auto& counts : cg.sender_counts) {
        for (const auto& vc : logical.vcs) {
            counts[vc.id] = std::max(counts[vc.id], 1u);
        }
    }
    cg.logical = std::move(logical);
    return cg;
}

void legacy_stock_sender_counts(ChannelGraph& cg, const WiringOptions& options) {
    const auto& g = cg.logical;
    std::map<FabricNodeId, PerDirectionCapabilities> chips;
    for (const auto& r : g.routers) {
        chips[r.port.node].at(*r.port.direction) = r.port.capability;
    }
    for (RouterIndex r = 0; r < g.routers.size(); ++r) {
        const auto& port = g.routers[r].port;
        const auto axis = options.express_axis.find(port.node.mesh_id);
        const auto shape = router_vc_shape(
            options.topology,
            *port.direction,
            chips.at(port.node),
            axis == options.express_axis.end() ? ExpressAxis::NONE : axis->second,
            &options.vc_config);
        auto& counts = cg.sender_counts[r];
        for (uint32_t vc = 0; vc < counts.size(); ++vc) {
            counts[vc] = std::max(counts[vc], shape.sender_counts[vc]);
        }
    }
}

}  // namespace tt::tt_fabric::graph
