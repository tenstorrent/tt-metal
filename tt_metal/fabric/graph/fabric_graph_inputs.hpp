// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <map>
#include <vector>

#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>

#include "fabric_graph_types.hpp"

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

// A live eth link on a fabric chip that the fabric doesn't use.
struct UnusedLink {
    FabricNodeId node{MeshId{0}, 0};
    chan_id_t chan = 0;
    uint64_t peer_asic = 0;
    bool peer_in_domain = false;  // the peer chip is in this fabric's meshes
    bool trained = true;
};

// Unused live links on this host's fabric chips. The domain is this fabric's MGD meshes.
std::vector<UnusedLink> unused_links(const ControlPlane& control_plane, const tt::Cluster& cluster);

// Unused links that leave the domain, as perimeter ports. Like intermesh ports, each peer chip gets a
// free mesh-edge direction (then Z), planes in channel order. Untrained links warn and stay. 2D only.
std::vector<PerimeterPort> perimeter_ports(
    const std::vector<UnusedLink>& links,
    const std::vector<Port>& ports,
    const MeshGraph& mesh_graph,
    Topology topology);

}  // namespace tt::tt_fabric::graph
