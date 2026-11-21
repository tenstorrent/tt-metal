// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_model.hpp"

namespace tt {
class Cluster;
}

namespace tt::tt_fabric {
class ControlPlane;
class FabricBuilderContext;
}  // namespace tt::tt_fabric

namespace tt::tt_fabric::manifest {

// Describes every mesh in the mesh graph and its chips, resolving each local chip's published routers against what
// only ControlPlane, the cluster, and the chip's other routers know.
std::vector<Mesh> describe_meshes(
    const ControlPlane& control_plane, const tt::Cluster& cluster, const FabricBuilderContext& builder_context);

}  // namespace tt::tt_fabric::manifest
