// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "physical_graph_lowering.hpp"

#include <algorithm>
#include <map>
#include <unordered_map>
#include <utility>

#include <tt_stl/assert.hpp>

#include "tt_metal/fabric/erisc_datamover_builder.hpp"

namespace tt::tt_fabric::graph {

PhysicalGraph lower_to_physical(ChannelGraph graph, const ChannelAllocator& allocate) {
    PhysicalGraph pg;
    static_cast<ChannelGraph&>(pg) = std::move(graph);
    const auto& g = pg.logical;
    pg.addresses.resize(g.channels.size());
    for (RouterIndex r = 0; r < g.routers.size(); ++r) {
        std::array<size_t, builder_config::MAX_NUM_VCS> counts{};
        std::copy(pg.sender_counts[r].begin(), pg.sender_counts[r].end(), counts.begin());
        const RouterChannelLayout& layout = pg.layouts.emplace_back(allocate(counts));
        for (uint32_t vc = 0; vc < pg.sender_counts[r].size(); ++vc) {
            TT_FATAL(
                layout.senders[vc].size() == pg.sender_counts[r][vc],
                "fabric graph: the allocator gave {} {} vc{} senders, not {}",
                g.routers[r].id,
                layout.senders[vc].size(),
                vc,
                pg.sender_counts[r][vc]);
        }
        for (ChannelIndex c : g.routers[r].channels) {
            const auto& ch = g.channels[c];
            const bool sender = ch.kind == ChannelKind::Sender;
            const auto& built = sender ? layout.senders[ch.vc] : layout.receivers[ch.vc];
            const uint32_t i = sender ? pg.index_in_vc[c] : 0;
            TT_FATAL(
                i < built.size(),
                "fabric graph: the allocator gave {} no vc{} {}",
                g.routers[r].id,
                ch.vc,
                channel_name(ch));
            pg.addresses[c] = ChannelAddress{.base = built[i].base_address, .slots = built[i].num_slots};
        }
    }
    return pg;
}

std::vector<chan_id_t> bind_eth_channels(const LogicalGraph& g, const LiveEthChannels& live) {
    std::vector<chan_id_t> chans;
    for (const auto& r : g.routers) {
        const auto planes = live(r.port.node, *r.port.direction);
        TT_FATAL(
            r.port.plane < planes.size(),
            "fabric graph: router {} has no live eth channel ({} live that way)",
            r.id,
            planes.size());
        chans.push_back(planes[r.port.plane]);
    }
    return chans;
}

std::map<FabricNodeId, RouterIndex> legacy_handshake_masters(
    const LogicalGraph& g, const std::vector<chan_id_t>& eth_chans) {
    // FabricBuilder::discover_channels, then create_routers: directions in FabricContext::routing_directions
    // order into one hash map, then each direction's channels by plane into another; the master is its first.
    std::map<FabricNodeId, std::map<RoutingDirection, std::vector<RouterIndex>>> chips;
    for (RouterIndex r = 0; r < g.routers.size(); ++r) {
        const auto& p = g.routers[r].port;
        auto& planes = chips[p.node][*p.direction];
        planes.resize(std::max<size_t>(planes.size(), p.plane + 1));
        planes[p.plane] = r;
    }
    std::map<FabricNodeId, RouterIndex> masters;
    for (const auto& [node, dirs] : chips) {
        std::unordered_map<RoutingDirection, std::vector<RouterIndex>> by_direction;
        for (const auto d : {RoutingDirection::N, RoutingDirection::S, RoutingDirection::E, RoutingDirection::W, RoutingDirection::Z}) {
            if (dirs.contains(d)) {
                by_direction[d] = dirs.at(d);
            }
        }
        std::unordered_map<chan_id_t, RouterIndex> routers;
        for (const auto& [d, planes] : by_direction) {
            for (RouterIndex r : planes) {
                routers.insert({eth_chans.at(r), r});
            }
        }
        masters[node] = routers.begin()->second;
    }
    return masters;
}

std::vector<LinkPairing> legacy_link_pairings(
    const LogicalGraph& g, Topology topology, const std::set<MeshId>& wrap_around_meshes) {
    using enum RoutingDirection;
    if (topology == Topology::NeighborExchange) {
        return {};
    }
    // per chip, each direction's routers by plane
    std::map<FabricNodeId, std::map<RoutingDirection, std::vector<RouterIndex>>> chips;
    for (RouterIndex r = 0; r < g.routers.size(); ++r) {
        const auto& p = g.routers[r].port;
        auto& planes = chips[p.node][*p.direction];
        planes.resize(std::max<size_t>(planes.size(), p.plane + 1));
        planes[p.plane] = r;
    }

    std::vector<LinkPairing> links;
    for (const auto& [node, dirs] : chips) {
        const auto pair = [&](RoutingDirection d1, RoutingDirection d2) {
            if (!dirs.contains(d1) || !dirs.contains(d2)) {
                return;
            }
            const auto& a = dirs.at(d1);
            const auto& b = dirs.at(d2);
            const auto num_links = static_cast<uint32_t>(std::min(a.size(), b.size()));
            for (uint32_t link = 0; link < num_links; ++link) {
                links.push_back(LinkPairing{
                    .a = a[link],
                    .b = b[link],
                    .link = link,
                    .num_links = num_links,
                    .noc_vc = static_cast<uint32_t>(
                        FabricEriscDatamoverConfig::DEFAULT_NOC_VC + (link % FabricEriscDatamoverConfig::NUM_EDM_NOC_VCS))});
            }
        };
        // same pairs, in the same order, as FabricBuilder::get_router_connection_pairs
        if (is_2D_topology(topology)) {
            for (const auto& [d1, d2] : {std::pair{N, S}, {E, W}, {N, E}, {N, W}, {S, E}, {S, W}, {Z, N}, {Z, S}, {Z, E}, {Z, W}}) {
                pair(d1, d2);
            }
        } else if (wrap_around_meshes.contains(node.mesh_id) && dirs.size() == 2) {
            pair(dirs.begin()->first, std::next(dirs.begin())->first);
        } else {
            pair(N, S);
            pair(E, W);
        }
    }
    return links;
}

}  // namespace tt::tt_fabric::graph
