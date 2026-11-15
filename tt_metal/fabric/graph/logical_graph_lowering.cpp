// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "logical_graph_lowering.hpp"

#include <algorithm>
#include <map>
#include <set>
#include <tuple>

#include <tt_stl/assert.hpp>

#include "tt_metal/fabric/builder/router_wiring_rules.hpp"

namespace tt::tt_fabric::graph {

namespace {

using RouterKey = std::tuple<FabricNodeId, PortDirection, uint32_t>;  // (node, direction, plane)

// the port on the far end of a same-mesh link faces back; an express chord is Z at both ends
PortDirection facing_back(PortDirection d) {
    return *d == RoutingDirection::Z ? d : PortDirection{get_opposite_direction(*d)};
}

class Lowering {
public:
    Lowering(const std::vector<Port>& ports, const WiringRules& rules) : ports_(ports), rules_(rules) {}

    LogicalGraph run() {
        g_.vcs = rules_.vcs();
        add_routers();
        link_peers();
        for (const auto& vc : g_.vcs) {
            add_channels(vc.id);
        }
        for (const auto& vc : g_.vcs) {
            add_turns(vc.id);
        }
        fold_1d_corners();
        for (const auto& vc : g_.vcs) {
            add_eth_edges(vc.id);
        }
        add_rings();
        return std::move(g_);
    }

private:
    ChannelIndex add_channel(LogicalChannel ch) {
        const auto idx = static_cast<ChannelIndex>(g_.channels.size());
        g_.routers[ch.router].channels.push_back(idx);
        g_.channels.push_back(ch);
        return idx;
    }

    void add_routers() {
        for (const auto& p : ports_) {
            const auto r = static_cast<RouterIndex>(g_.routers.size());
            g_.routers.push_back(LogicalRouter{.id = router_id(p.node, p.direction, p.plane), .port = p});
            by_key_.emplace(RouterKey{p.node, p.direction, p.plane}, r);
            on_chip_plane_[{p.node, p.plane}].push_back(r);
        }
    }

    void link_peers() {
        for (auto& router : g_.routers) {
            const auto& p = router.port;
            std::optional<RouterIndex> match;
            if (p.capability != EdgeCapability::INTERMESH) {
                const auto it = by_key_.find(RouterKey{p.peer, facing_back(p.direction), p.plane});
                if (it != by_key_.end() && g_.routers[it->second].port.peer == p.node) {
                    match = it->second;
                }
            } else {
                for (RouterIndex c : on_chip_plane_[{p.peer, p.plane}]) {
                    const auto& cp = g_.routers[c].port;
                    if (cp.capability == EdgeCapability::INTERMESH && cp.peer == p.node) {
                        TT_FATAL(!match, "fabric graph: router {} has more than one link peer", router.id);
                        match = c;
                    }
                }
            }
            TT_FATAL(match.has_value(), "fabric graph: router {} has no peer router at the far end", router.id);
            router.peer = match;
        }
    }

    void add_channels(uint32_t vc) {
        for (RouterIndex r = 0; r < g_.routers.size(); ++r) {
            const auto& port = g_.routers[r].port;
            if (!rules_.carries(port, vc)) {
                continue;
            }
            add_channel(LogicalChannel{.router = r, .vc = vc, .kind = ChannelKind::Receiver});
            if (rules_.has_worker(port, vc)) {
                add_channel(LogicalChannel{.router = r, .vc = vc, .kind = ChannelKind::Sender, .injection = true});
            }
        }
    }

    // the receiver on `up` feeds a forwarding sender on `down`, one sender per ingress direction
    void add_turn(RouterIndex up, RouterIndex down, uint32_t vc) {
        const PortDirection from = g_.routers[up].port.direction;
        auto [it, inserted] = forwarding_.try_emplace({down, vc, from}, 0);
        if (inserted) {
            it->second = add_channel(LogicalChannel{.router = down, .vc = vc, .kind = ChannelKind::Sender, .from = from});
        }
        g_.edges.push_back(LogicalEdge{.from = *g_.receiver(up, vc), .to = it->second, .transport = Transport::Noc});
    }

    // the stock turn sets
    void add_turns(uint32_t vc) {
        for (RouterIndex down = 0; down < g_.routers.size(); ++down) {
            const auto& d = g_.routers[down].port;
            for (RouterIndex up : on_chip_plane_[{d.node, d.plane}]) {
                const auto& u = g_.routers[up].port;
                if (up != down && rules_.carries(u, vc) && rules_.carries(d, vc) && rules_.wires(u, d, vc)) {
                    add_turn(up, down, vc);
                }
            }
        }
    }

    // 1D on a wrap-around mesh: a corner chip (two non-opposite directions) turns VC0 between them
    void fold_1d_corners() {
        for (const auto& [chip_plane, routers] : on_chip_plane_) {
            if (routers.size() == 2 && is_folded_corner_turn(g_, rules_, routers[0], routers[1], 0)) {
                add_turn(routers[0], routers[1], 0);
                add_turn(routers[1], routers[0], 0);
            }
        }
    }

    void add_eth_edges(uint32_t vc) {
        for (RouterIndex r = 0; r < g_.routers.size(); ++r) {
            const auto rx = g_.receiver(*g_.routers[r].peer, vc);
            if (!rx) {
                continue;
            }
            for (ChannelIndex c : g_.routers[r].channels) {
                const auto& ch = g_.channels[c];
                if (ch.kind == ChannelKind::Sender && ch.vc == vc) {
                    g_.edges.push_back(LogicalEdge{.from = c, .to = *rx, .transport = Transport::Eth});
                }
            }
        }
    }

    // walk E (or N) routers chip to chip; getting back to the start means it's a ring
    void add_rings() {
        std::set<RouterIndex> seen;
        for (RouterIndex start = 0; start < g_.routers.size(); ++start) {
            const auto& s = g_.routers[start].port;
            const bool walks = *s.direction == RoutingDirection::E || *s.direction == RoutingDirection::N;
            if (s.capability != EdgeCapability::INTRAMESH_CARDINAL || !walks || seen.contains(start)) {
                continue;
            }
            std::vector<RouterIndex> members{start};
            RouterIndex at = start;
            bool closed = false;
            while (members.size() <= g_.routers.size()) {
                const auto& far = g_.routers[*g_.routers[at].peer].port;
                const auto next = by_key_.find(RouterKey{far.node, s.direction, s.plane});
                if (next == by_key_.end()) {
                    break;
                }
                if (next->second == start) {
                    closed = true;
                    break;
                }
                members.push_back(next->second);
                at = next->second;
            }
            seen.insert(members.begin(), members.end());
            if (!closed) {
                continue;
            }
            for (const auto& vc : g_.vcs) {
                g_.rings.push_back(Ring{
                    .id = "ring." + g_.routers[start].id + ".vc" + std::to_string(vc.id),
                    .vc = vc.id,
                    .plane = s.plane,
                    .members = members});
            }
        }
    }

    const std::vector<Port>& ports_;
    const WiringRules& rules_;
    LogicalGraph g_;
    std::map<RouterKey, RouterIndex> by_key_;
    std::map<std::pair<FabricNodeId, uint32_t>, std::vector<RouterIndex>> on_chip_plane_;
    std::map<std::tuple<RouterIndex, uint32_t, PortDirection>, ChannelIndex> forwarding_;  // (router, vc, from)
};

}  // namespace

bool is_folded_corner_turn(
    const LogicalGraph& g, const WiringRules& rules, RouterIndex up, RouterIndex down, uint32_t vc) {
    const auto& u = g.routers[up].port;
    const auto& d = g.routers[down].port;
    if (vc != 0 || u.node != d.node || u.plane != d.plane || !rules.folds_corners(u.node.mesh_id) ||
        u.direction == facing_back(d.direction)) {
        return false;
    }
    return std::count_if(g.routers.begin(), g.routers.end(), [&](const LogicalRouter& r) {
               return r.port.node == u.node && r.port.plane == u.plane;
           }) == 2;
}

LogicalGraph lower_to_logical(const std::vector<Port>& ports, const WiringRules& rules) {
    return Lowering(ports, rules).run();
}

}  // namespace tt::tt_fabric::graph
