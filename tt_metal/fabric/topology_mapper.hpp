// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <queue>

#include <tt-metalium/mesh_graph.hpp>
#include <tt-metalium/fabric_types.hpp>
#include <tt-metalium/routing_table_generator.hpp>

namespace tt::tt_metal {

class PhysicalSystemDescriptor;
}  // namespace tt::tt_metal

using HostName = std::string;
using HostRank = std::vector<HostName>;  // HostName is the hostname, the index is the host rank

namespace tt::tt_fabric {

struct LocalMeshBinding;

/**
 * @brief TopologyMapper creates and manages mappings between fabric node IDs and physical ASIC IDs
 *
 * This class takes a mesh graph object and a physical system descriptor to create bidirectional
 * mappings between:
 * - FabricNodeId (mesh_id + chip_id from mesh graph)
 * - AsicID (physical ASIC IDs from physical system descriptor)
 *
 * The mapping is based on the mesh IDs and fabric chip IDs from the mesh_container and maps
 * them to the ASIC IDs of the physical descriptor.
 */
class TopologyMapper {
public:
    /**
     * @brief Construct a new TopologyMapper object
     *
     * @param mesh_graph Reference to the mesh graph object containing fabric topology
     * @param physical_system_descriptor Reference to the physical system descriptor containing ASIC topology
     * @param local_mesh_binding Reference to the local mesh binding object containing mesh binding information
     */
    TopologyMapper(
        const MeshGraph& mesh_graph,
        const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
        const LocalMeshBinding& local_mesh_binding);

    const MeshGraph& get_mesh_graph() const { return mesh_graph_; }

    const tt::tt_metal::PhysicalSystemDescriptor& get_physical_system_descriptor() const {
        return physical_system_descriptor_;
    }

    const LocalMeshBinding& get_local_mesh_binding() const { return local_mesh_binding_; }

private:
    /**
     * @brief Build the mapping between fabric node IDs and physical ASIC IDs
     *
     * This method iterates through all meshes in the mesh graph and creates mappings
     * based on the mesh IDs and fabric chip IDs from the mesh_container, mapping them
     * to the ASIC IDs of the physical descriptor.
     */
    void build_mapping();

    /**
     * @brief Build the mapping between mesh IDs and host ranks
     *
     * This method iterates through all meshes in the mesh graph and creates mappings
     * based on the mesh IDs and fabric chip IDs from the mesh_container, mapping them
     * to the ASIC IDs of the physical descriptor.
     */
    std::unordered_map<MeshId, HostRank> build_host_mesh_mappings();

    /**
     * @brief Discover hosts using DFS
     *
     * This method performs a depth-first search starting from the current host to discover
     * all hosts in the system and creates mappings between mesh IDs and host rank vectors.

     @return true if when solution is found, false otherwise
     */
    bool discover_hosts_dfs(
        const MeshId mesh_id,
        const HostName& hostname,
        std::unordered_map<MeshId, HostRank>& mesh_id_to_host_rank,
        std::unordered_set<HostName>& visited_hosts);

    const MeshGraph& mesh_graph_;
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor_;
    const LocalMeshBinding& local_mesh_binding_;

    // Bidirectional mapping between FabricNodeId and AsicID
    std::unordered_map<FabricNodeId, tt::tt_metal::AsicID> fabric_node_id_to_asic_id_;
    std::unordered_map<tt::tt_metal::AsicID, FabricNodeId> asic_id_to_fabric_node_id_;
};

}  // namespace tt::tt_fabric
