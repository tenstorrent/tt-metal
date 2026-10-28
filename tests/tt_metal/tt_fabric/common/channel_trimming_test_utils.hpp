// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

#include <tt-metalium/device_types.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include <hostdevcommon/fabric_common.h>

namespace tt::tt_fabric::fabric_router_tests {

// ============================================================================
// Helper: Result of running unicast traffic
// ============================================================================
struct UnicastTrafficResult {
    ChipId src_physical_device_id;
    ChipId dst_physical_device_id;
    FabricNodeId src_fabric_node_id;
    FabricNodeId dst_fabric_node_id;
    chan_id_t edm_port;
    bool skipped;
};

// ============================================================================
// Helper: Run unicast traffic between two fabric nodes using the connection API
// Follows the run_unicast_test_bw_chips pattern from test_basic_1d_fabric.cpp
// ============================================================================
UnicastTrafficResult run_unicast_traffic_bw_nodes(
    FabricNodeId src_fabric_node_id, FabricNodeId dst_fabric_node_id, uint32_t num_hops, uint32_t num_packets = 10);

// ============================================================================
// Helper: Walk the mesh to find a neighbor pair with the specified hop count.
// Scans all mesh positions and all directions (E/W/N/S) to find a source with
// at least `num_hops` neighbors. Returns the chain of nodes along the path.
// ============================================================================
struct MeshPathResult {
    FabricNodeId src_node;
    FabricNodeId dst_node;
    std::vector<FabricNodeId> path;  // src, hop1, hop2, ..., dst
    RoutingDirection direction;
    bool found;
};

MeshPathResult find_1d_path_with_hops(uint32_t num_hops);

MeshPathResult find_2d_path_with_transit(bool require_deadlock_avoidance_last_hop = false);

}  // namespace tt::tt_fabric::fabric_router_tests
