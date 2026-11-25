// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "logical_graph_validation.hpp"

#include "logical_graph_lowering.hpp"

#include <map>
#include <set>
#include <tuple>
#include <utility>

#include <tt_stl/assert.hpp>

namespace tt::tt_fabric::graph {

std::vector<std::string> find_problems(
    const LogicalGraph& g, const WiringRules& rules, const ValidationLimits& limits) {
    std::vector<std::string> problems;
    auto where = [&](ChannelIndex c) {
        const auto& ch = g.channels[c];
        return g.routers[ch.router].id + ".vc" + std::to_string(ch.vc) + "." + channel_name(ch);
    };

    std::vector<std::vector<const LogicalEdge*>> link_out(g.channels.size());
    std::vector<bool> noc_fed(g.channels.size(), false);
    std::vector<size_t> relays_out(g.channels.size(), 0);
    std::map<ChannelIndex, std::set<RouterIndex>> relayed_by;  // relay sender -> the routers relaying into it
    for (const auto& e : g.edges) {
        if (e.transport == Transport::Noc) {
            noc_fed[e.to] = true;
            if (g.channels[e.from].kind == ChannelKind::Sender) {
                ++relays_out[e.from];
                relayed_by[e.to].insert(g.channels[e.from].router);
            }
        } else {
            link_out[e.from].push_back(&e);
        }
    }
    // every relay chain ends at a router whose link is up
    {
        std::map<std::tuple<FabricNodeId, PortDirection, uint32_t>, RouterIndex> by_key;
        for (RouterIndex r = 0; r < g.routers.size(); ++r) {
            const auto& p = g.routers[r].port;
            by_key.emplace(std::tuple{p.node, p.direction, p.plane}, r);
        }
        for (RouterIndex r = 0; r < g.routers.size(); ++r) {
            RouterIndex at = r;
            size_t hops = 0;
            while (g.routers[at].port.relay_plane && hops <= g.routers.size()) {
                const auto& p = g.routers[at].port;
                const auto it = by_key.find({p.node, p.direction, *p.relay_plane});
                if (it == by_key.end()) {
                    break;
                }
                at = it->second;
                ++hops;
            }
            if (g.routers[at].port.relay_plane) {
                problems.push_back(g.routers[r].id + ": relay chain does not end at a live link");
            }
        }
    }
    for (const auto& [relay, routers] : relayed_by) {
        if (routers.size() != 1) {
            problems.push_back(where(relay) + ": relayed into by " + std::to_string(routers.size()) + " routers");
        }
    }

    std::map<std::pair<RouterIndex, uint32_t>, std::pair<uint32_t, uint32_t>> counts;  // -> (senders, receivers)
    for (ChannelIndex c = 0; c < g.channels.size(); ++c) {
        const auto& ch = g.channels[c];
        const auto& router = g.routers[ch.router];
        auto& [senders, receivers] = counts[{ch.router, ch.vc}];
        if (ch.kind == ChannelKind::Receiver) {
            ++receivers;
            continue;
        }
        ++senders;
        if (!ch.injection && !noc_fed[c]) {
            problems.push_back(where(c) + ": forwarding sender with no inbound noc edge");
        }
        if (router.port.relay_plane) {
            if (!link_out[c].empty()) {
                problems.push_back(where(c) + ": relaying sender with a link edge");
            }
            if (relays_out[c] != 1) {
                problems.push_back(where(c) + ": expected one relay edge, found " + std::to_string(relays_out[c]));
            }
            continue;
        }
        if (relays_out[c] != 0) {
            problems.push_back(where(c) + ": relay edge from a router whose link is up");
        }
        const bool perimeter = router.port.capability == EdgeCapability::INTERDOMAIN;
        const Transport link = perimeter ? Transport::Dynamic : Transport::Eth;
        size_t links = 0;
        for (const auto* e : link_out[c]) {
            if (e->transport != link) {
                problems.push_back(
                    where(c) +
                    (perimeter ? ": perimeter sender with an eth edge" : ": dynamic edge off the perimeter"));
                continue;
            }
            ++links;
            const auto& to = g.channels[e->to];
            const auto far = perimeter ? std::optional<RouterIndex>{ch.router} : router.peer;
            if (to.kind != ChannelKind::Receiver || to.vc != (perimeter ? 2 : ch.vc) || !far || to.router != *far) {
                problems.push_back(where(c) + ": link edge does not reach the far receiver on its VC");
            }
        }
        if (links != 1) {
            problems.push_back(where(c) + ": expected one link edge, found " + std::to_string(links));
        }
    }

    for (const auto& e : g.edges) {
        if (e.transport != Transport::Noc) {
            continue;
        }
        const auto& from = g.channels[e.from];
        const auto& to = g.channels[e.to];
        const auto& up = g.routers[from.router].port;
        const auto& down = g.routers[to.router].port;
        const std::string what = where(e.from) + " -> " + where(e.to);
        if (from.kind == ChannelKind::Sender && up.relay_plane) {
            // a relay: from a down perimeter router into the relay sender of its side's router on its relay plane
            if (down.capability != EdgeCapability::INTERDOMAIN || up.node != down.node ||
                up.direction != down.direction || down.plane != *up.relay_plane || to.kind != ChannelKind::Sender ||
                to.vc != 0 || to.injection || !is_relay_sender(g, to)) {
                problems.push_back(
                    what + ": relay edge that does not reach its side's relay sender on its relay plane");
            }
            continue;
        }
        if (from.kind != ChannelKind::Receiver || to.kind != ChannelKind::Sender) {
            problems.push_back(what + ": noc edge must run receiver -> sender");
        } else if (to.injection || to.from != up.direction) {
            // a sender slot has one producer: the forwarding sender for where the traffic came in
            problems.push_back(what + ": noc edge must feed the forwarding sender for its ingress direction");
        }
        if (from.router == to.router) {
            problems.push_back(what + ": U-turn into its own router");
        }
        if (up.node != down.node || up.plane != down.plane) {
            problems.push_back(what + ": noc edge leaves its chip or routing plane");
        }
        if (from.vc != to.vc) {
            problems.push_back(what + ": noc edge changes VC");
        } else if (!rules.wires(up, down, from.vc) && !is_folded_corner_turn(g, rules, from.router, to.router, from.vc)) {
            problems.push_back(what + ": the wiring rules don't wire this");
        }
    }

    for (const auto& [key, n] : counts) {
        const auto& id = g.routers[key.first].id;
        if (limits.max_sender_channels_per_vc != 0 && n.first > limits.max_sender_channels_per_vc) {
            problems.push_back(
                id + ".vc" + std::to_string(key.second) + ": " + std::to_string(n.first) + " sender channels, limit " +
                std::to_string(limits.max_sender_channels_per_vc));
        }
        if (limits.max_receiver_channels_per_vc != 0 && n.second > limits.max_receiver_channels_per_vc) {
            problems.push_back(
                id + ".vc" + std::to_string(key.second) + ": " + std::to_string(n.second) +
                " receiver channels, limit " + std::to_string(limits.max_receiver_channels_per_vc));
        }
    }
    return problems;
}

void validate(const LogicalGraph& graph, const WiringRules& rules, const ValidationLimits& limits) {
    const auto problems = find_problems(graph, rules, limits);
    if (problems.empty()) {
        return;
    }
    std::string all;
    for (const auto& p : problems) {
        all += "\n  " + p;
    }
    TT_THROW("fabric graph: {} validation problem(s):{}", problems.size(), all);
}

}  // namespace tt::tt_fabric::graph
