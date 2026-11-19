// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <map>
#include <vector>

#include "fabric_graph_types.hpp"

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

}  // namespace tt::tt_fabric::graph
