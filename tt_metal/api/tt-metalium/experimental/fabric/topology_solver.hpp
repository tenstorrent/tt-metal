// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <map>
#include <vector>

#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/experimental/fabric/routing_table_generator.hpp>

namespace tt::tt_metal {
class PhysicalSystemDescriptor;
}  // namespace tt::tt_metal

namespace tt::tt_fabric {

/**
 * @brief Generic graph representation with minimal query interface
 *
 * AdjacencyGraph provides a generic graph representation that works with any node type.
 * It provides a minimal interface for querying graph structure: getting all nodes and
 * getting neighbors of a specific node.
 *
 * @tparam NodeId The type used to identify nodes in the graph
 */
template <typename NodeId>
class AdjacencyGraph {
public:
    using NodeType = NodeId;
    using AdjacencyMap = std::map<NodeId, std::vector<NodeId>>;

    /**
     * @brief Construct empty adjacency graph
     */
    AdjacencyGraph() = default;

    /**
     * @brief Construct adjacency graph from a MeshGraph
     *
     * @param mesh_graph The mesh graph to construct the adjacency graph from
     */
    explicit AdjacencyGraph(const AdjacencyMap& adjacency_map);

    /**
     * @brief Get all nodes in the graph
     *
     * @return const std::vector<NodeId>& Vector of all node IDs in the graph
     */
    const std::vector<NodeId>& get_nodes() const;

    /**
     * @brief Get neighbors of a specific node
     *
     * @param node The node ID to get neighbors for
     * @return const std::vector<NodeId>& Vector of neighbor node IDs
     */
    const std::vector<NodeId>& get_neighbors(NodeId node) const;

private:
    AdjacencyMap adj_map_;
    std::vector<NodeId> nodes_cache_;
};

std::map<MeshId, AdjacencyGraph<FabricNodeId>> build_adjacency_map_logical(const MeshGraph& mesh_graph);

std::map<MeshId, AdjacencyGraph<tt::tt_metal::AsicID>> build_adjacency_map_physical(
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    const std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>>& asic_id_to_mesh_rank);

}  // namespace tt::tt_fabric
