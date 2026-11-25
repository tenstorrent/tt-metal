// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "fabric_graph_inputs.hpp"

#include <algorithm>
#include <tuple>

#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt_stl/assert.hpp>

namespace tt::tt_fabric::graph {

namespace {

PortDirection port_direction(RoutingDirection direction) {
    TT_FATAL(
        direction != RoutingDirection::C && direction != RoutingDirection::NONE,
        "fabric graph: {} is not a port direction",
        static_cast<int>(direction));
    return PortDirection{direction};
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

}  // namespace tt::tt_fabric::graph
