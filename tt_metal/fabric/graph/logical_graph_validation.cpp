// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "logical_graph_validation.hpp"

#include "logical_graph_lowering.hpp"

#include <map>
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
        size_t eth = 0;
        for (const auto* e : g.edges_from(c)) {
            if (e->transport != Transport::Eth) {
                continue;
            }
            ++eth;
            const auto& to = g.channels[e->to];
            if (to.kind != ChannelKind::Receiver || to.vc != ch.vc || !router.peer || to.router != *router.peer) {
                problems.push_back(where(c) + ": eth edge does not reach the peer's receiver on its VC");
            }
        }
        if (eth != 1) {
            problems.push_back(where(c) + ": expected one eth edge, found " + std::to_string(eth));
        }
        if (!ch.injection) {
            bool fed = false;
            for (const auto* e : g.edges_into(c)) {
                fed |= e->transport == Transport::Noc;
            }
            if (!fed) {
                problems.push_back(where(c) + ": forwarding sender with no inbound noc edge");
            }
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
        if (from.kind != ChannelKind::Receiver || to.kind != ChannelKind::Sender) {
            problems.push_back(what + ": noc edge must run receiver -> sender");
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
