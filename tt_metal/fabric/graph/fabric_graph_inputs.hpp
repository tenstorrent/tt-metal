// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <map>
#include <string>
#include <tuple>
#include <vector>

#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>

#include "fabric_graph_types.hpp"
#include "tt_metal/fabric/inter_domain_mesh.hpp"

namespace tt {
class Cluster;
}  // namespace tt

namespace tt::tt_fabric {
class ControlPlane;
}  // namespace tt::tt_fabric

namespace tt::tt_fabric::graph {

// Every port in the system from MeshGraph, sorted by (node, direction, plane). k links to a
// neighbour give planes 0..k-1. A same-mesh Z edge is an express chord on a mesh with an express axis.
std::vector<Port> ports_from_mesh_graph(const MeshGraph& mesh_graph, const std::map<MeshId, ExpressAxis>& express_axis);

// Same ports with each (node, direction) given the control plane's live plane count. MeshGraph
// doesn't count express Z channels, and live planes can be trimmed.
std::vector<Port> with_live_planes(const std::vector<Port>& ports, const ControlPlane& control_plane);

struct UnusedLink {
    FabricNodeId node{MeshId{0}, 0};
    chan_id_t chan = 0;
    uint64_t peer_asic = 0;
    chan_id_t peer_chan = 0;
    bool peer_in_domain = false;
    bool trained = true;
};

std::vector<UnusedLink> unused_links(
    const ControlPlane& control_plane, const tt::Cluster& cluster, const std::string& fsd_path = "");

std::vector<PerimeterPort> perimeter_ports(
    const std::vector<UnusedLink>& links,
    const std::vector<Port>& ports,
    const MeshGraph& mesh_graph,
    Topology topology);

// Each down perimeter link's relay plane, (node, direction, plane) -> plane: on each side, the down planes in order
// relay into the live ones in order, and once each live one takes a relay, into the down ones that relay before them
// (a chain that ends at a live link). A side with no live link has none.
std::map<std::tuple<FabricNodeId, RoutingDirection, uint32_t>, uint32_t> relay_planes(
    const std::vector<PerimeterPort>& perimeter);

// A port for each perimeter link, live or relaying (Z excluded).
std::vector<Port> ports_from_perimeter(const std::vector<PerimeterPort>& perimeter);

// A port the control plane does not know, and its eth channel.
struct ChannelPort {
    Port port;
    chan_id_t eth_chan = 0;
};

// Both ends of each of the domain's VC2-only links, each on a direction free of `ports`, never Z.
std::vector<ChannelPort> ports_from_domain_links(
    const std::vector<DomainLink>& links, const std::vector<Port>& ports, const MeshGraph& mesh_graph);

// Each port on one of the domain's edges, by its direction in the domain: where VC2 forwards.
std::map<std::pair<FabricNodeId, RoutingDirection>, RoutingDirection> vc2_directions(
    const std::vector<Port>& ports, const InterDomainMesh& domain);

}  // namespace tt::tt_fabric::graph
