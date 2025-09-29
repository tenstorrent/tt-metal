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
     * @brief Use BFS to find all hosts in the system and map them to mesh IDs and host ranks
     *
     * This method performs a breadth-first search starting from the current host to discover
     * all hosts in the system and creates mappings between mesh IDs and host rank vectors.
     * Each host rank vector contains hostnames indexed by host rank.
     *
     * @return A map from mesh IDs to host rank vectors (vector of hostnames)
     */
    std::unordered_map<MeshId, HostRank> build_host_mesh_mappings();
    void discover_hosts_dfs(const HostName& hostname, std::unordered_map<MeshId, HostRank>& mesh_id_to_host_rank);

    /**
     * @brief Get all mesh IDs for a given host
     *
     * @param hostname The hostname to get mesh information for
     * @return A vector of mesh IDs that the given host is associated with
     */
    std::vector<MeshId> get_mesh_info_for_host(const HostName& hostname) const;

    const MeshGraph& mesh_graph_;
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor_;
    const LocalMeshBinding& local_mesh_binding_;

    // Bidirectional mapping between FabricNodeId and AsicID
    std::unordered_map<FabricNodeId, tt::tt_metal::AsicID> fabric_node_id_to_asic_id_;
    std::unordered_map<tt::tt_metal::AsicID, FabricNodeId> asic_id_to_fabric_node_id_;
};

}  // namespace tt::tt_fabric
