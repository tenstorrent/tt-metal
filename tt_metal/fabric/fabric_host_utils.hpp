// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <tt-metalium/fabric_edm_types.hpp>
#include <tt-metalium/fabric_types.hpp>
#include <tt-metalium/mesh_graph.hpp>                   // FabricType
#include <umd/device/types/cluster_descriptor_types.hpp>  // ChipId
#include <llrt/tt_cluster.hpp>
#include "erisc_datamover_builder.hpp"
#include <set>
#include <vector>
#include <unordered_map>
#include <queue>
#include <functional>
#include <unordered_set>
#include <optional>
namespace tt::tt_fabric {

class FabricNodeId;
bool is_tt_fabric_config(tt::tt_fabric::FabricConfig fabric_config);

void set_routing_mode(uint16_t routing_mode);
void set_routing_mode(Topology topology, tt::tt_fabric::FabricConfig fabric_config, uint32_t dimension = 1);

FabricType get_fabric_type(tt::tt_fabric::FabricConfig fabric_config);

std::vector<uint32_t> get_forwarding_link_indices_in_direction(
    const FabricNodeId& src_fabric_node_id, const FabricNodeId& dst_fabric_node_id, RoutingDirection direction);

// Helper: Build adjacency map and discover corners/edges using BFS
using AdjacencyMap = std::unordered_map<ChipId, std::vector<ChipId>>;
struct IntraMeshAdjacencyMap {
    AdjacencyMap adjacency_map;
    std::vector<ChipId> corners;  // Should always be size 2 for 1D meshes, 4 for 2D meshes, populated in order of
                                  // closest to chip 0 by default
    std::vector<ChipId> edges;  // Should always be size 2 for 1D meshes, 4 for 2D meshes, populated in order of closest
                                // to chip 0 by default
    std::uint32_t ns_size{};         // North-South size (rows)
    std::uint32_t ew_size{};         // East-West size (columns)
};

}  // namespace tt::tt_fabric
