// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "fabric_graph_inputs.hpp"

#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <tuple>

#include <enchantum/enchantum.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/experimental/fabric/physical_descriptor_builder.hpp>
#include <tt-metalium/experimental/fabric/physical_node_id.hpp>
#include <tt-metalium/experimental/fabric/physical_system_descriptor.hpp>
#include <tt-metalium/experimental/fabric/topology_mapper.hpp>
#include <tt_stl/assert.hpp>

#include "llrt/tt_cluster.hpp"

namespace tt::tt_fabric::graph {

namespace {

PortDirection port_direction(RoutingDirection direction) {
    TT_FATAL(
        direction != RoutingDirection::C && direction != RoutingDirection::NONE,
        "fabric graph: {} is not a port direction",
        static_cast<int>(direction));
    return PortDirection{direction};
}

// first mesh-edge side (N, E, S, W), then Z, that the chip isn't using yet
std::optional<RoutingDirection> free_direction(
    const FabricNodeId& node, const MeshGraph& mesh_graph, const std::set<RoutingDirection>& taken) {
    const auto c = mesh_graph.chip_to_coordinate(node.mesh_id, node.chip_id);
    const auto shape = mesh_graph.get_mesh_shape(node.mesh_id);
    const std::pair<RoutingDirection, bool> candidates[] = {
        {RoutingDirection::N, c[0] == 0},
        {RoutingDirection::E, c[1] + 1 == shape[1]},
        {RoutingDirection::S, c[0] + 1 == shape[0]},
        {RoutingDirection::W, c[1] == 0},
        {RoutingDirection::Z, mesh_graph.get_chip_spec().num_z_ports > 0}};
    for (const auto& [d, ok] : candidates) {
        if (ok && !taken.contains(d)) {
            return d;
        }
    }
    return std::nullopt;
}

}  // namespace

std::vector<Port> ports_from_mesh_graph(const MeshGraph& mesh_graph, const std::map<MeshId, ExpressAxis>& express_axis) {
    std::vector<Port> ports;
    const auto& intra = mesh_graph.get_intra_mesh_connectivity();
    const auto& inter = mesh_graph.get_inter_mesh_connectivity();
    for (MeshId mesh : mesh_graph.get_mesh_ids()) {
        const bool express = express_axis.contains(mesh);
        const auto num_chips = static_cast<uint32_t>(intra.at(*mesh).size());
        for (uint32_t chip = 0; chip < num_chips; ++chip) {
            const FabricNodeId node(mesh, chip);
            for (const auto& [neighbour, edge] : intra.at(*mesh).at(chip)) {
                const PortDirection direction = port_direction(edge.port_direction);
                const FabricNodeId peer(mesh, neighbour);
                const auto capability = classify_fabric_edge(node, peer, *direction, express);
                for (uint32_t plane = 0; plane < edge.connected_chip_ids.size(); ++plane) {
                    ports.push_back(
                        Port{.node = node, .direction = direction, .plane = plane, .peer = peer, .capability = capability});
                }
            }
            if (*mesh < inter.size() && chip < inter.at(*mesh).size()) {
                for (const auto& [peer_mesh, edge] : inter.at(*mesh).at(chip)) {
                    for (uint32_t plane = 0; plane < edge.connected_chip_ids.size(); ++plane) {
                        ports.push_back(Port{
                            .node = node,
                            .direction = port_direction(edge.port_direction),
                            .plane = plane,
                            .peer = FabricNodeId(peer_mesh, edge.connected_chip_ids[plane]),
                            .capability = EdgeCapability::INTERMESH});
                    }
                }
            }
        }
    }
    std::sort(ports.begin(), ports.end(), [](const Port& a, const Port& b) {
        return std::tie(a.node, a.direction, a.plane) < std::tie(b.node, b.direction, b.plane);
    });
    for (size_t i = 1; i < ports.size(); ++i) {
        const auto& a = ports[i - 1];
        const auto& b = ports[i];
        TT_FATAL(
            !(a.node == b.node && a.direction == b.direction && a.plane == b.plane),
            "fabric graph: two ports at {}",
            router_id(a.node, a.direction, a.plane));
    }
    return ports;
}

std::vector<Port> with_live_planes(const std::vector<Port>& ports, const ControlPlane& control_plane) {
    std::vector<Port> live;
    for (const auto& p : ports) {
        if (p.plane != 0) {
            continue;
        }
        const auto planes =
            control_plane.get_active_fabric_eth_routing_planes_in_direction(p.node, *p.direction).size();
        for (uint32_t plane = 0; plane < planes; ++plane) {
            Port copy = p;
            copy.plane = plane;
            live.push_back(copy);
        }
    }
    return live;
}

std::vector<UnusedLink> unused_links(
    const ControlPlane& control_plane, const tt::Cluster& cluster, const std::string& fsd_path) {
    using tt::tt_metal::experimental::PhysicalNodeId;
    const auto& psd = control_plane.get_physical_system_descriptor();
    const auto& mapper = control_plane.get_topology_mapper();
    auto node_of = [&](uint64_t asic) -> std::optional<FabricNodeId> {
        const auto id = psd.find_physical_node_id(tt::tt_metal::AsicID{asic});
        return id ? mapper.find_fabric_node_id_from_physical_node_id(*id) : std::nullopt;
    };

    // UMD only lists trained links, so a cable that never came up is only in the FSD. The whole FSD, not
    // the job's slice: perimeter cables mostly go to hosts outside the job.
    std::optional<tt::tt_metal::PhysicalSystemDescriptor> fsd;
    if (!fsd_path.empty()) {
        fsd.emplace(tt::tt_metal::experimental::tt_fabric::build_physical_descriptor_from_file(fsd_path));
    }

    // this host's chips only: each host builds its own routers, so it only needs its own ports
    std::vector<UnusedLink> links;
    std::vector<UnusedLink> missing;
    const auto& asics = cluster.get_unique_chip_ids();
    const auto& local = cluster.get_ethernet_connections();
    const auto& remote = cluster.get_ethernet_connections_to_remote_devices();
    for (const auto& [chip, asic] : asics) {
        const auto node = node_of(asic);
        if (!node) {
            continue;  // not a fabric chip
        }
        std::set<chan_id_t> used;
        for (const auto& [chan, direction] : control_plane.get_active_fabric_eth_channels(*node)) {
            used.insert(chan);
        }
        std::map<chan_id_t, uint64_t> live;  // every trained chan -> peer chip
        if (auto it = local.find(chip); it != local.end()) {
            for (const auto& [chan, peer] : it->second) {
                live[chan] = asics.at(std::get<0>(peer));
            }
        }
        if (auto it = remote.find(chip); it != remote.end()) {
            for (const auto& [chan, peer] : it->second) {
                live[chan] = std::get<0>(peer);
            }
        }
        for (const auto& [chan, peer_asic] : live) {
            if (used.contains(chan)) {
                continue;
            }
            const auto core = cluster.get_soc_desc(chip).get_eth_core_for_channel(chan, CoordSystem::LOGICAL);
            links.push_back(UnusedLink{
                .node = *node,
                .chan = chan,
                .peer_asic = peer_asic,
                .peer_in_domain = node_of(peer_asic).has_value(),
                .trained = cluster.is_ethernet_link_up(chip, core)});
        }

        const auto address = psd.find_physical_node_id(tt::tt_metal::AsicID{asic});
        const auto fsd_asic = fsd && address ? fsd->find_asic_id(*address) : std::nullopt;
        if (!fsd_asic) {
            continue;
        }
        for (const auto& peer : fsd->get_asic_neighbors(*fsd_asic)) {
            const auto peer_address = fsd->get_physical_node_id(peer);
            const auto cables = fsd->get_eth_connections(*fsd_asic, peer);
            // the peer chip as live knows it (a trained cable to it, or discovery), else by address
            const auto peer_id = psd.find_asic_id(peer_address);
            uint64_t key = peer_id ? **peer_id : std::hash<PhysicalNodeId>{}(peer_address);
            for (const auto& c : cables) {
                if (auto it = live.find(c.src_chan); it != live.end()) {
                    key = it->second;
                }
            }
            for (const auto& c : cables) {
                if (!live.contains(c.src_chan)) {
                    missing.push_back(UnusedLink{
                        .node = *node,
                        .chan = c.src_chan,
                        .peer_asic = key,
                        .peer_in_domain = mapper.find_fabric_node_id_from_physical_node_id(peer_address).has_value(),
                        .trained = false});
                }
            }
        }
    }
    return with_missing_links(std::move(links), missing);
}

std::vector<UnusedLink> with_missing_links(std::vector<UnusedLink> links, const std::vector<UnusedLink>& missing) {
    std::set<std::pair<FabricNodeId, chan_id_t>> listed;
    for (const auto& l : links) {
        listed.insert({l.node, l.chan});
    }
    for (const auto& m : missing) {
        if (listed.insert({m.node, m.chan}).second) {
            links.push_back(m);
        }
    }
    std::sort(links.begin(), links.end(), [](const UnusedLink& a, const UnusedLink& b) {
        return std::tie(a.node, a.chan) < std::tie(b.node, b.chan);
    });
    return links;
}

std::vector<PerimeterPort> perimeter_ports(
    const std::vector<UnusedLink>& links,
    const std::vector<Port>& ports,
    const MeshGraph& mesh_graph,
    Topology topology) {
    TT_FATAL(
        is_2D_topology(topology),
        "fabric graph: inter-domain needs a 2D fabric, got {}",
        enchantum::to_string(topology));
    std::map<FabricNodeId, std::set<RoutingDirection>> taken;
    for (const auto& p : ports) {
        taken[p.node].insert(*p.direction);
    }
    auto sorted = links;
    std::sort(sorted.begin(), sorted.end(), [](const UnusedLink& a, const UnusedLink& b) {
        return std::tie(a.node, a.chan) < std::tie(b.node, b.chan);
    });

    // one direction per peer chip, like an intermesh link
    std::map<std::pair<FabricNodeId, uint64_t>, std::optional<RoutingDirection>> direction_of;
    std::map<std::pair<FabricNodeId, RoutingDirection>, uint32_t> planes;
    std::vector<PerimeterPort> out;
    for (const auto& l : sorted) {
        if (l.peer_in_domain) {
            continue;  // folds back into the domain
        }
        auto [it, fresh] = direction_of.try_emplace({l.node, l.peer_asic});
        if (fresh) {
            it->second = free_direction(l.node, mesh_graph, taken[l.node]);
            if (it->second) {
                taken[l.node].insert(*it->second);
            }
        }
        if (!it->second) {
            log_warning(tt::LogFabric, "fabric graph: no free direction for perimeter link {} chan {}", l.node, l.chan);
            continue;
        }
        if (!l.trained) {
            log_warning(tt::LogFabric, "fabric graph: perimeter link {} chan {} is down", l.node, l.chan);
        }
        out.push_back(PerimeterPort{
            .node = l.node,
            .eth_chan = l.chan,
            .direction = PortDirection{*it->second},
            .plane = planes[{l.node, *it->second}]++,
            .link_trained = l.trained});
    }
    std::sort(out.begin(), out.end(), [](const PerimeterPort& a, const PerimeterPort& b) {
        return std::tie(a.node, a.direction, a.plane) < std::tie(b.node, b.direction, b.plane);
    });
    return out;
}

}  // namespace tt::tt_fabric::graph
