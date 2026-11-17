// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>

#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/vocabulary/node_coord.hpp>
#include <tt-metalium/workload/utils/group.hpp>
#include <tt-metalium/workload/utils/table.hpp>
#include <tt_stl/strong_type.hpp>

namespace tt::tt_metal::experimental {

// ============================================================================
//  FabricConnectionSpec API
// ============================================================================
//
// A fabric connection is the exclusive right of a kernel instance to put packets into one
// fabric router of its device. A FabricConnectionSpec declares, for each node, the connections
// that the node's kernel instance opens.
//
// A fabric connection is a Program-scope resource. It is immutable: it has no run args.
// The runtime allocates and owns the L1 cells that each connection needs.
//
// The kernel builds a fabric sender from the binding token:
//   auto sender = tt::tt_fabric::WorkerToFabricEdmSender::build_from_binding(fabric::<accessor_name>);
//
// ============================================================================

// A name identifying a FabricConnectionSpec within a ProgramSpec.
using FabricConnectionSpecName = ttsl::StrongType<std::string, struct FabricConnectionSpecNameTag>;

// One connection: the router that sends toward the next hop.
struct FabricConnectionTarget {
    // The fabric node of the next hop.
    // Invariant: a neighbor of the source fabric node.
    tt::tt_fabric::FabricNodeId dst_fabric_node_id;

    // Which link toward dst_fabric_node_id (and so which router) the connection uses.
    // Invariant: a forwarding link from the source fabric node to dst_fabric_node_id.
    uint32_t link_idx = 0;
};

struct FabricConnectionSpec {
    // Identifier: used to reference this FabricConnectionSpec within the ProgramSpec
    FabricConnectionSpecName unique_id;

    // The fabric node of the device that the Program runs on.
    // NOTE: Declared here for now; the runtime may derive it from the Program's device later.
    tt::tt_fabric::FabricNodeId src_fabric_node_id;

    // For each node, the connections that the node's kernel instance opens.
    // A node with no entry (or an empty group) has no connection.
    // Invariant: each node has at most one target.
    Table<NodeCoord, Group<FabricConnectionTarget>> targets;
};

}  // namespace tt::tt_metal::experimental
