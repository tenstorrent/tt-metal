// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <tt-metalium/fabric_edm_types.hpp>
#include <tt-metalium/fabric_types.hpp>
#include <tt-metalium/mesh_graph.hpp>                   // FabricType
#include <umd/device/types/cluster_descriptor_types.h>  // chip_id_t
#include <llrt/tt_cluster.hpp>
#include <tt-metalium/erisc_datamover_builder.hpp>
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
bool is_2d_fabric_config(tt::tt_fabric::FabricConfig fabric_config);

uint32_t get_sender_channel_count(tt::tt_fabric::Topology topology);
uint32_t get_downstream_edm_count(tt::tt_fabric::Topology topology);

void set_routing_mode(uint16_t routing_mode);
void set_routing_mode(Topology topology, tt::tt_fabric::FabricConfig fabric_config, uint32_t dimension = 1);

FabricType get_fabric_type(tt::tt_fabric::FabricConfig fabric_config, tt::ClusterType cluster_type);

std::vector<uint32_t> get_forwarding_link_indices_in_direction(
    const FabricNodeId& src_fabric_node_id, const FabricNodeId& dst_fabric_node_id, RoutingDirection direction);

void get_optimal_noc_for_edm(
    FabricEriscDatamoverBuilder& edm_builder1,
    FabricEriscDatamoverBuilder& edm_builder2,
    uint32_t num_links,
    Topology topology);

// Helper: BFS distance map from a start chip to all reachable chips using the
// provided adjacency map. Returned distances are expressed in hop count.
std::unordered_map<chip_id_t, std::uint32_t> compute_distances(
    chip_id_t start_chip, const std::unordered_map<chip_id_t, std::vector<chip_id_t>>& adjacency_map);

// Helper: Build adjacency map and discover corners/edges using BFS
struct IntraMeshAdjacencyMap {
    std::unordered_map<chip_id_t, std::vector<chip_id_t>> adjacency_map;
    std::vector<chip_id_t> corners;  // Should always be size 2 for 1D meshes, 4 for 2D meshes, populated in order of closest to chip 0 by default
    std::vector<chip_id_t> edges;    // Should always be size 2 for 1D meshes, 4 for 2D meshes, populated in order of closest to chip 0 by default
    std::uint32_t ns_size;  // North-South size (rows)
    std::uint32_t ew_size;  // East-West size (columns)
};

IntraMeshAdjacencyMap build_mesh_adjacency_map(
    const std::set<chip_id_t>& user_chip_ids,
    const tt::tt_metal::distributed::MeshShape& mesh_shape,
    std::function<std::vector<chip_id_t>(chip_id_t)> get_adjacent_chips_func,
    std::optional<chip_id_t> start_chip_id = std::nullopt);

// Helper: Convert 1D mesh adjacency map to row-major vector representation
std::vector<chip_id_t> convert_1d_mesh_adjacency_to_row_major_vector(const IntraMeshAdjacencyMap& topology_info);

// Helper: Convert 2D mesh adjacency map to row-major vector representation
std::vector<chip_id_t> convert_2d_mesh_adjacency_to_row_major_vector(
    const IntraMeshAdjacencyMap& topology_info, std::optional<chip_id_t> nw_corner_chip_id = std::nullopt);

}  // namespace tt::tt_fabric
