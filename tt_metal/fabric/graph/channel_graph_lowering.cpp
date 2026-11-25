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
        chips[r.port.node].at(*r.port.direction) = stock_capability(r.port.capability);
    }
    return chips;
}

std::array<uint32_t, builder_config::MAX_NUM_VCS> counts_of(const ChannelGraph& cg, RouterIndex r) {
    std::array<uint32_t, builder_config::MAX_NUM_VCS> counts{};
    std::copy(cg.sender_counts[r].begin(), cg.sender_counts[r].end(), counts.begin());
    return counts;
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
    if (is_relay_sender(g, ch)) {
        TT_FATAL(ch.vc == 0, "fabric graph: relay sender on vc{}", ch.vc);
        return builder_config::inter_domain_relay_sender_channel;
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

void legacy_stock_sender_counts(ChannelGraph& cg, const WiringOptions& options) {
    const auto& g = cg.logical;
    for (RouterIndex r = 0; r < g.routers.size(); ++r) {
        const auto& port = g.routers[r].port;
        const auto axis = options.express_axis.find(port.node.mesh_id);
        const auto shape = router_vc_shape(
            options.topology,
            *port.direction,
            cg.chips.at(port.node),
            axis == options.express_axis.end() ? ExpressAxis::NONE : axis->second,
            &options.vc_config);
        auto& counts = cg.sender_counts[r];
        std::set<std::pair<uint32_t, uint32_t>> used;  // (vc, index)
        for (ChannelIndex c : g.channels_of(r, ChannelKind::Sender)) {
            used.emplace(g.channels[c].vc, cg.index_in_vc[c]);
        }
        // every perimeter router has the relay sender's slot, used or not: the two ends of an inter-domain link must
        // lay out L1 the same (each writes into the other's VC2 receiver at its own layout's address). Only where the
        // fabric sizes VC0 for it (FabricBuilderContext::compute_max_channel_counts): 2D, no VC1.
        if (port.capability == EdgeCapability::INTERDOMAIN) {
            if (is_2D_topology(options.topology) && !options.vc_config.requires_vc1) {
                counts[0] = std::max<uint32_t>(counts[0], builder_config::num_sender_channels_2d_vc0_inter_domain);
            } else {
                TT_FATAL(
                    !used.contains({0, builder_config::inter_domain_relay_sender_channel}),
                    "fabric graph: {} plane {} takes a relay or split on its relay sender, which a fabric with VC1 "
                    "can't (#67139)",
                    port.node,
                    port.plane);
            }
        }
        for (uint32_t vc = 0; vc < counts.size(); ++vc) {
            counts[vc] = std::max(counts[vc], shape.sender_counts[vc]);
            // no stock layout to match (a forwarding VC2, an inter-domain router): padding keeps positions, not service
            if ((vc == 2 && options.vc_config.vc2_forwards) || port.capability == EdgeCapability::INTERDOMAIN ||
                port.capability == EdgeCapability::INTRADOMAIN) {
                continue;
            }
            for (uint32_t i = 0; i < counts[vc]; ++i) {
                // an unused relay sender slot keeps its layout but is not serviced (no code for it)
                const bool relay_slot = port.capability == EdgeCapability::INTERDOMAIN && vc == 0 &&
                                        i == builder_config::inter_domain_relay_sender_channel;
                if (!used.contains({vc, i}) && !relay_slot) {
                    cg.legacy_slots.push_back(LegacySlot{.router = r, .vc = vc, .index = i});
                }
            }
        }
    }
}

void mark_ring_entries(ChannelGraph& cg, const WiringOptions& options, const RingQueries& ring_queries) {
    const auto& g = cg.logical;
    std::map<RouterIndex, std::vector<LegacySlot*>> legacy;
    for (auto& slot : cg.legacy_slots) {
        legacy[slot.router].push_back(&slot);
    }
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
            // a relay comes off a perimeter router, never a ring
            cg.enters_ring[c] =
                !is_relay_sender(g, ch) &&
                enters_ring(
                    ch.vc,
                    ch.injection ? std::nullopt
                                 : std::optional{builder::routing_direction_to_eth_direction(**ch.from)});
        }
        // a legacy slot takes the producer stock names for it
        const builder::RouterProducerSlots stock_slots(facing, counts_of(cg, r));
        for (LegacySlot* slot : legacy[r]) {
            const auto producer = stock_slots.producer_at(slot->vc, slot->index);
            slot->enters_ring = (producer || stock_slots.worker_channel(slot->vc) == slot->index) &&
                                enters_ring(slot->vc, producer);
        }
    }
}

}  // namespace tt::tt_fabric::graph
