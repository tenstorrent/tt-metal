// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>  // FabricType
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
#include <filesystem>

namespace tt::tt_fabric {

class TopologyMapper;

class FabricNodeId;
bool is_tt_fabric_config(tt::tt_fabric::FabricConfig fabric_config);

FabricType get_fabric_type(tt::tt_fabric::FabricConfig fabric_config);

// Helper to validate that requested FabricType doesn't require more connectivity than available FabricType provides
// Returns true if requested_type requires more connections than available_type provides
// mesh_shape: [rows, cols] - used to detect edge cases where 2-row/2-col torus is equivalent to mesh
bool requires_more_connectivity(FabricType requested_type, FabricType available_type, const MeshShape& mesh_shape);

// Compute maximum 1D hops across all meshes in topology
// Returns max(rows-1, cols-1) across all meshes, representing longest linear path
// Returns 0 for empty input or single-chip meshes
uint32_t compute_max_1d_hops(const std::vector<MeshShape>& mesh_shapes);

// Compute maximum 2D hops across all meshes in topology
// Returns (rows-1) + (cols-1) for largest mesh, representing Manhattan distance corner-to-corner
// Returns 0 for empty input or single-chip meshes
uint32_t compute_max_2d_hops(const std::vector<MeshShape>& mesh_shapes);

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

// Serialize chip IDs to mesh coordinates mapping to a YAML file
// Uses TopologyMapper to get the mapping between logical and physical chip IDs
void serialize_mesh_coordinates_to_file(
    const TopologyMapper& topology_mapper, const std::filesystem::path& output_file_path);

}  // namespace tt::tt_fabric
