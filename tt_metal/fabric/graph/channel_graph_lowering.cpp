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
#include "tt_metal/fabric/builder/injection_policy.hpp"
#include "tt_metal/fabric/builder/router_wiring_rules.hpp"

namespace tt::tt_fabric::graph {

namespace {

std::map<FabricNodeId, PerDirectionCapabilities> chip_capabilities(const LogicalGraph& g) {
    std::map<FabricNodeId, PerDirectionCapabilities> chips;
    for (const auto& r : g.routers) {
        chips[r.port.node].at(*r.port.direction) = r.port.capability;
    }
    return chips;
}

}  // namespace

uint32_t stock_channel_index(const LogicalGraph& g, ChannelIndex c, Topology topology) {
    const auto& ch = g.channels[c];
    if (ch.kind == ChannelKind::Receiver) {
        return 0;
    }
    if (ch.injection) {
        return get_worker_connected_sender_channel();
    }
    return builder::get_downstream_sender_channel_for_vc(
        is_2D_topology(topology),
        ch.vc,
        builder::routing_direction_to_eth_direction(**ch.from),
        builder::routing_direction_to_eth_direction(*g.routers[ch.router].port.direction));
}

ChannelGraph lower_to_channels(LogicalGraph logical, Topology topology) {
    ChannelGraph cg;
    uint32_t num_vcs = 0;
    for (const auto& vc : logical.vcs) {
        num_vcs = std::max(num_vcs, vc.id + 1);
    }

    std::set<std::tuple<RouterIndex, uint32_t, ChannelKind, uint32_t>> taken;
    cg.index_in_vc.resize(logical.channels.size());
    cg.enters_ring.assign(logical.channels.size(), false);
    cg.sender_counts.assign(logical.routers.size(), std::vector<uint32_t>(num_vcs, 0));
    for (ChannelIndex c = 0; c < logical.channels.size(); ++c) {
        const auto& ch = logical.channels[c];
        TT_FATAL(ch.vc < num_vcs, "fabric graph: channel on vc{} but only {} VCs", ch.vc, num_vcs);
        const uint32_t idx = stock_channel_index(logical, c, topology);
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
    cg.chips = chip_capabilities(logical);
    cg.logical = std::move(logical);
    return cg;
}

void mark_ring_entries(ChannelGraph& cg, const WiringOptions& options, const RingQueries& ring_queries) {
    const auto& g = cg.logical;
    for (RouterIndex r = 0; r < g.routers.size(); ++r) {
        const auto& port = g.routers[r].port;
        const auto& caps = cg.chips.at(port.node);
        const auto facing = builder::routing_direction_to_eth_direction(*port.direction);
        const bool express = options.express_axis.contains(port.node.mesh_id);
        const auto queries = express ? ring_queries(port.node) : ProtectedRingQueries{};
        const NonExpressInjectionPolicy non_express_policy(options.topology, facing);
        const ExpressInjectionPolicy express_policy(queries, caps, z_role_of(caps), *port.direction, port.capability);
        const InjectionPolicy& policy = express ? static_cast<const InjectionPolicy&>(express_policy)
                                                : static_cast<const InjectionPolicy&>(non_express_policy);
        // no producer: the worker
        const auto enters_ring = [&](uint32_t vc, std::optional<eth_chan_directions> producer) {
            if (!builder_config::bubble_flow_control_enabled_on_vc(vc) || !policy.participates(vc)) {
                return false;
            }
            return producer ? policy.producer_is_injection(vc, *producer) : policy.worker_is_injection(vc);
        };
        for (ChannelIndex c : g.channels_of(r, ChannelKind::Sender)) {
            const auto& ch = g.channels[c];
            cg.enters_ring[c] = enters_ring(
                ch.vc,
                ch.injection ? std::nullopt : std::optional{builder::routing_direction_to_eth_direction(**ch.from)});
        }
    }
}

}  // namespace tt::tt_fabric::graph
