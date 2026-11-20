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

namespace tt::tt_metal::experimental {

// ============================================================================
//  FabricConnectionsSpec API
// ============================================================================
//
// A fabric connection is one kernel instance's use of one fabric router of its device.
// The kernel instance opens the connection, sends packets into the router, and closes it.
//
// A FabricConnectionsSpec describes a matrix of fabric connections: one connection for each
// pair of a node in `links` and a target in `targets`. Each node uses its link for every target.
//
// A FabricConnectionsSpec is a Program-scope resource, declared in ProgramAdvancedOptions::fabric_connections.
// It is immutable: it has no run args.
// The runtime allocates the L1 cells that each connection needs.
//
// FABRIC MODE: Fabric1DTargets require a 1D fabric; Fabric2DTargets require a 2D fabric.
//
// PLACEMENT: When the Program is built, the runtime resolves the spec on every device that the
// Program runs on: the device range of the ProgramSpec in a MeshWorkload, or every device of the
// MeshDevice for Program(mesh_device, spec).
//   - A routing direction target resolves to the device's fabric neighbor in that direction.
//     It fails if the device has no neighbor in that direction.
//   - A peer fabric node target resolves to the direction of the first hop toward the peer.
//     It fails on the device that is the peer.
//   On each node, the connection of a target uses the router of the node's link in the resolved
//   direction. Directions resolve against the fabric, not the edge of the MeshDevice: the device
//   at the end of a submesh can connect to a chip outside the submesh.
//   The op handles the edges of the fabric: a device at an edge gets its own ProgramSpec, in its
//   own device range, that lists only the targets it has, or that does not bind the spec.
//   Each device that the Program runs on gets its own connection args. Where they differ from
//   the first device's, they are per-device runtime args, so the Program's range must support
//   them (see ArePerDeviceRuntimeArgsSupported).
//
// KERNEL ACCESS: A data-movement kernel binds the spec via KernelAdvancedOptions::fabric_connection_bindings.
//   The binding token holds one connection slot for each target, in the order of `targets`.
//   Each node of a kernel that binds the spec must be in `links`: a kernel whose nodes do not all
//   connect is split into one KernelSpec that binds the spec and one that does not. A kernel
//   source that is used with and without the binding looks it up with
//   fabric::get_token_if_present<"<accessor_name>">(), which is nullptr without the binding.
//   The kernel builds a fabric sender or a connection manager from the binding token:
//     WorkerToFabricEdmSender::build_from_binding(fabric::<accessor_name>)
//     FabricConnectionManager::build_from_binding(fabric::<accessor_name>)
//     RoutingPlaneConnectionManager::build_from_binding(fabric::<accessor_name>)
//
// ============================================================================

// A name identifying a FabricConnectionsSpec within a ProgramSpec.
using FabricConnectionsSpecName = ttsl::StrongType<std::string, struct FabricConnectionsSpecNameTag>;

struct FabricConnectionsSpec {
    // Fabric 1D targets: routing directions.
    using Fabric1DTargets = std::vector<tt::tt_fabric::RoutingDirection>;

    // Fabric 2D targets: peer fabric nodes.
    using Fabric2DTargets = std::vector<tt::tt_fabric::FabricNodeId>;

    // The targets of a FabricConnectionsSpec: routing directions on a 1D fabric, or peer fabric nodes on a 2D fabric.
    using FabricTargets = std::variant<Fabric1DTargets, Fabric2DTargets>;

    // FabricConnectionsSpec identifier: used to reference this spec within the ProgramSpec
    FabricConnectionsSpecName unique_id;

    // The targets of the connections, in the order of the binding token's slots.
    FabricTargets targets;

    // For each node whose kernel instance opens these connections:
    // the link that the node uses in every resolved direction.
    Table<NodeCoord, uint32_t> links;
};

using Fabric1DTargets = FabricConnectionsSpec::Fabric1DTargets;
using Fabric2DTargets = FabricConnectionsSpec::Fabric2DTargets;
using FabricTargets = FabricConnectionsSpec::FabricTargets;

}  // namespace tt::tt_metal::experimental
