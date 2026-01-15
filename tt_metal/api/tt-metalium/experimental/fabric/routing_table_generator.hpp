// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>

namespace tt::tt_fabric {
class TopologyMapper;

using RoutingTable =
    std::vector<std::vector<std::vector<RoutingDirection>>>;  // [mesh_id][chip_id][target_chip_or_mesh_id]

class RoutingTableGenerator {
public:
    explicit RoutingTableGenerator(const TopologyMapper& topology_mapper);
    ~RoutingTableGenerator() = default;

    void dump_to_yaml();
    void load_from_yaml();

    RoutingTable get_intra_mesh_table() const { return this->intra_mesh_table_; }
    RoutingTable get_inter_mesh_table() const { return this->inter_mesh_table_; }

    void print_routing_tables() const;
    // Return a list of all exit nodes, across all meshes that are connected to the requested
    // MeshID.
    const std::vector<FabricNodeId>& get_exit_nodes_routing_to_mesh(MeshId mesh_id) const;
    // Return the single exit node (chip in src_mesh_id) for a given src chip and dst mesh
    FabricNodeId get_exit_node_from_mesh_to_mesh(MeshId src_mesh_id, ChipId src_chip_id, MeshId dst_mesh_id) const;

    // Load Inter-Mesh Connectivity into the Routing Table Generator
    void load_intermesh_connections(const AnnotatedIntermeshConnections& intermesh_connections);

private:
    const TopologyMapper& topology_mapper_;
    // configurable in future architectures
    const uint32_t max_nodes_in_mesh_ = 1024;
    const uint32_t max_num_meshes_ = 1024;

    RoutingTable intra_mesh_table_;
    RoutingTable inter_mesh_table_;
    std::unordered_map<MeshId, std::vector<FabricNodeId>> mesh_to_exit_nodes_;
    // Direct lookup table: [src_mesh][src_chip][dst_mesh] -> exit chip_id in src_mesh
    std::vector<std::vector<std::vector<ChipId>>> exit_node_lut_;

    std::vector<std::vector<std::vector<std::pair<ChipId, MeshId>>>> get_paths_to_all_meshes(
        MeshId src, const InterMeshConnectivity& inter_mesh_connectivity) const;
    void generate_intramesh_routing_table(const IntraMeshConnectivity& intra_mesh_connectivity);
    // when generating intermesh routing table, we use the intramesh connectivity table to find the shortest path to
    // the exit chip
    void generate_intermesh_routing_table(
        const InterMeshConnectivity& inter_mesh_connectivity, const IntraMeshConnectivity& intra_mesh_connectivity);
};

}  // namespace tt::tt_fabric
