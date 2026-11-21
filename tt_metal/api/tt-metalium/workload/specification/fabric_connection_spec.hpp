// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <variant>
#include <vector>

#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include <tt-metalium/vocabulary/node_coord.hpp>
#include <tt-metalium/workload/utils/table.hpp>
#include <tt_stl/strong_type.hpp>

namespace tt::tt_metal {

// ============================================================================
//  FabricConnectionSpec API
// ============================================================================
//
// A fabric connection is one kernel instance's use of one fabric router of its device.
// The kernel instance opens the connection, sends packets into the router, and closes it.
//
// A FabricConnectionSpec describes a matrix of fabric connections: one connection for each
// pair of a node in `links` and a target in `targets`. Each node uses its link for every target.
//
// A FabricConnectionSpec is a Program-scope resource. It is immutable: it has no run args.
// The runtime allocates the L1 cells that each connection needs.
//
// FABRIC MODE: Fabric1DTargets require a 1D fabric; Fabric2DTargets require a 2D fabric.
//
// PLACEMENT: The runtime resolves the spec on each device that the Program runs on.
//   - A routing direction target is used as is.
//   - A peer fabric node target resolves to the direction of the first hop toward the peer.
//   On each node, the connection of a target uses the router of the node's link in the resolved
//   direction. If the device has no neighbor in a resolved direction, that target's connection
//   is not connected on that device.
//   Only the MeshWorkload constructors accept a ProgramSpec that has fabric connections:
//   the runtime needs the device of each Program to resolve its connections.
//
// KERNEL ACCESS: A data-movement kernel binds the spec via KernelSpec::fabric_connection_bindings.
//   The binding token holds one connection slot for each target, in the order of `targets`.
//   The kernel builds a fabric sender or a connection manager from the binding token:
//     WorkerToFabricEdmSender::build_from_binding(fabric::<accessor_name>)
//     FabricConnectionManager::build_from_binding(fabric::<accessor_name>)
//     RoutingPlaneConnectionManager::build_from_binding(fabric::<accessor_name>)
//
// ============================================================================

// A name identifying a FabricConnectionSpec within a ProgramSpec.
using FabricConnectionSpecName = ttsl::StrongType<std::string, struct FabricConnectionSpecNameTag>;

// Fabric 1D targets: routing directions.
using Fabric1DTargets = std::vector<tt::tt_fabric::RoutingDirection>;

// Fabric 2D targets: peer fabric nodes.
using Fabric2DTargets = std::vector<tt::tt_fabric::FabricNodeId>;

struct FabricConnectionSpec {
    // FabricConnectionSpec identifier: used to reference this spec within the ProgramSpec
    FabricConnectionSpecName unique_id;

    // The targets of the connections, in the order of the binding token's slots.
    std::variant<Fabric1DTargets, Fabric2DTargets> targets;

    // For each node whose kernel instance opens these connections:
    // the link that the node uses in every resolved direction.
    // A node without an entry has no connection for this spec.
    Table<NodeCoord, uint32_t> links;
};

}  // namespace tt::tt_metal
