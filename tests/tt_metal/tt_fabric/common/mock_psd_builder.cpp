// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mock_psd_builder.hpp"

#include <map>

#include "protobuf/physical_system_descriptor.pb.h"
#include "tt_metal/fabric/serialization/physical_system_descriptor_serialization.hpp"

namespace tt::tt_fabric::test {

namespace proto = ::tt::fabric::proto;

tt::tt_metal::PhysicalSystemDescriptor build_mock_psd(
    const std::vector<std::string>& host_of_asic,
    const std::vector<MockLink>& links,
    const std::vector<std::pair<uint32_t, uint32_t>>& positions,
    uint64_t base_asic_id) {
    const int n = static_cast<int>(host_of_asic.size());
    auto asic_id = [&](int i) { return base_asic_id + static_cast<uint64_t>(i); };

    // Symmetric adjacency (each undirected link listed both ways, in insertion order).
    // Channel IDs are unique per ASIC so get_connected_asic_and_channel cannot alias peers.
    struct DirectedLink {
        int nb = 0;
        uint32_t channels = 0;
        uint32_t src_chan0 = 0;
    };
    std::vector<std::vector<DirectedLink>> adj(n);
    std::vector<uint32_t> next_chan(n, 0);
    for (const auto& link : links) {
        adj[link.a].push_back({link.b, link.channels, next_chan[link.a]});
        next_chan[link.a] += link.channels;
        adj[link.b].push_back({link.a, link.channels, next_chan[link.b]});
        next_chan[link.b] += link.channels;
    }
    // Hosts ranked in first-seen order.
    std::map<std::string, uint32_t> host_rank;
    std::vector<std::string> host_order;
    for (const auto& h : host_of_asic) {
        if (host_rank.emplace(h, static_cast<uint32_t>(host_order.size())).second) {
            host_order.push_back(h);
        }
    }

    auto add_eth = [&](proto::EthConnection* ec, uint32_t chan, bool is_local) {
        ec->set_src_chan(chan);
        ec->set_dst_chan(chan);
        ec->set_is_local(is_local);
    };

    proto::PhysicalSystemDescriptor p;
    p.set_target_device_type(0);
    auto* fw = p.mutable_ethernet_firmware_version();
    fw->set_major(0);
    fw->set_minor(0);
    fw->set_patch(0);

    // Per-ASIC descriptor (asic_location is the ASIC's index; single tray, one board type).
    for (int i = 0; i < n; ++i) {
        auto* m = p.add_asic_descriptors();
        m->set_asic_id(asic_id(i));
        auto* d = m->mutable_asic_descriptor();
        const bool have_pos = static_cast<size_t>(i) < positions.size();
        d->set_tray_id(have_pos ? positions[i].first : 0u);
        d->set_asic_location(have_pos ? positions[i].second : static_cast<uint32_t>(i));
        d->set_board_type(1);
        d->set_host_name(host_of_asic[i]);
    }
    for (const auto& h : host_order) {
        auto* hr = p.add_host_to_rank();
        hr->set_host_name(h);
        hr->set_rank(host_rank[h]);
    }

    // system_graph.asic_connectivity_graph: one HostAsicConnectivity per host, in host order.
    auto* sg = p.mutable_system_graph();
    std::map<std::string, proto::HostAsicConnectivity*> host_conn;
    for (const auto& h : host_order) {
        auto* hc = sg->add_asic_connectivity_graph();
        hc->set_host_name(h);
        host_conn[h] = hc;
    }
    for (int i = 0; i < n; ++i) {
        auto* g = host_conn[host_of_asic[i]]->add_asic_topologies();
        g->set_asic_id(asic_id(i));
        auto* topo = g->mutable_topology();
        for (const auto& link : adj[i]) {
            auto* e = topo->add_asic_connections();
            e->set_dst_asic_id(asic_id(link.nb));
            const bool is_local = host_of_asic[i] == host_of_asic[link.nb];
            for (uint32_t c = 0; c < link.channels; ++c) {
                add_eth(e->add_eth_connections(), link.src_chan0 + c, is_local);
            }
        }
    }

    // Cross-host edges also populate host_connectivity_graph and the per-host exit-node table.
    std::map<std::string, proto::HostConnections*> src_host;
    std::map<std::string, std::map<std::string, proto::HostConnectionEdge*>> host_edge;
    std::map<std::string, proto::ExitNodeConnectionTable*> exit_tbl;
    auto get_edge = [&](const std::string& s, const std::string& d) -> proto::HostConnectionEdge* {
        auto& bydst = host_edge[s];
        auto it = bydst.find(d);
        if (it != bydst.end()) {
            return it->second;
        }
        auto* hc = src_host[s];
        if (hc == nullptr) {
            hc = sg->add_host_connectivity_graph();
            hc->set_src_host_name(s);
            src_host[s] = hc;
        }
        auto* e = hc->add_host_connections();
        e->set_dst_host_name(d);
        bydst[d] = e;
        return e;
    };
    auto get_tbl = [&](const std::string& h) -> proto::ExitNodeConnectionTable* {
        auto it = exit_tbl.find(h);
        if (it != exit_tbl.end()) {
            return it->second;
        }
        auto* t = p.add_exit_node_connection_table();
        t->set_host_name(h);
        exit_tbl[h] = t;
        return t;
    };
    for (int i = 0; i < n; ++i) {
        for (const auto& link : adj[i]) {
            if (host_of_asic[i] == host_of_asic[link.nb]) {
                continue;
            }
            auto* he = get_edge(host_of_asic[i], host_of_asic[link.nb]);
            auto* tbl = get_tbl(host_of_asic[i]);
            for (uint32_t c = 0; c < link.channels; ++c) {
                auto* xc = he->add_exit_node_connections();
                xc->set_src_exit_node(asic_id(i));
                xc->set_dst_exit_node(asic_id(link.nb));
                add_eth(xc->mutable_eth_conn(), link.src_chan0 + c, false);
                auto* tc = tbl->add_exit_connections();
                tc->set_src_exit_node(asic_id(i));
                tc->set_dst_exit_node(asic_id(link.nb));
                add_eth(tc->mutable_eth_conn(), link.src_chan0 + c, false);
            }
        }
    }

    return tt::tt_metal::deserialize_physical_system_descriptor_from_proto(p);
}

tt::tt_metal::PhysicalSystemDescriptor build_mock_psd(
    const std::vector<std::string>& host_of_asic,
    const std::vector<std::pair<int, int>>& edges,
    uint32_t channels,
    uint64_t base_asic_id) {
    std::vector<MockLink> links;
    links.reserve(edges.size());
    for (const auto& [a, b] : edges) {
        links.push_back(MockLink{a, b, channels});
    }
    return build_mock_psd(host_of_asic, links, {}, base_asic_id);
}

tt::tt_metal::PhysicalSystemDescriptor build_grid_mock_psd(
    int rows,
    int cols,
    const std::vector<std::string>& host_of_asic,
    uint32_t channels,
    const std::vector<std::pair<int, int>>& extra_links,
    uint64_t base_asic_id) {
    std::vector<MockLink> links;
    for (const auto& [a, b] : grid_edges(rows, cols)) {
        links.push_back(MockLink{a, b, channels});
    }
    for (const auto& [a, b] : extra_links) {
        links.push_back(MockLink{a, b, channels});
    }
    // Grid position convention (matches the hand-written grid PSDs): tray_id = row + 1, asic_location = col + 1.
    std::vector<std::pair<uint32_t, uint32_t>> positions;
    positions.reserve(static_cast<size_t>(rows) * cols);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            positions.emplace_back(static_cast<uint32_t>(r + 1), static_cast<uint32_t>(c + 1));
        }
    }
    return build_mock_psd(host_of_asic, links, positions, base_asic_id);
}

std::vector<std::pair<int, int>> line_edges(int num_asics) {
    std::vector<std::pair<int, int>> e;
    for (int i = 0; i + 1 < num_asics; ++i) {
        e.emplace_back(i, i + 1);
    }
    return e;
}

std::vector<std::pair<int, int>> ring_edges(int num_asics) {
    auto e = line_edges(num_asics);
    if (num_asics > 2) {
        e.emplace_back(num_asics - 1, 0);
    }
    return e;
}

std::vector<std::pair<int, int>> grid_edges(int rows, int cols) {
    std::vector<std::pair<int, int>> e;
    auto id = [&](int r, int c) { return r * cols + c; };
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (c + 1 < cols) {
                e.emplace_back(id(r, c), id(r, c + 1));
            }
            if (r + 1 < rows) {
                e.emplace_back(id(r, c), id(r + 1, c));
            }
        }
    }
    return e;
}

std::vector<std::pair<int, int>> star_edges(int num_leaves) {
    std::vector<std::pair<int, int>> e;
    for (int i = 1; i <= num_leaves; ++i) {
        e.emplace_back(0, i);
    }
    return e;
}

}  // namespace tt::tt_fabric::test
