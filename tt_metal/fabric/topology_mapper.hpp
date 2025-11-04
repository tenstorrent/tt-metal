// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <unordered_map>
#include <unordered_set>
#include <vector>

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

using HostMeshMapping = std::unordered_map<MeshId, std::unordered_set<HostName>>;
using LogicalAdjacencyMap = std::unordered_map<tt::tt_fabric::FabricNodeId, std::vector<tt::tt_fabric::FabricNodeId>>;
using PhysicalAdjacencyMap = std::unordered_map<tt::tt_metal::AsicID, std::vector<tt::tt_metal::AsicID>>;
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

    /**
     * @brief Get logical mesh graph connectivity
     *
     * @return const MeshGraph&
     */
    const MeshGraph& get_mesh_graph() const { return mesh_graph_; }

    /**
     * @brief Get physical system descriptor
     *
     * @return const tt::tt_metal::PhysicalSystemDescriptor&
     */
    const tt::tt_metal::PhysicalSystemDescriptor& get_physical_system_descriptor() const {
        return physical_system_descriptor_;
    }

    /**
     * @brief Get local mesh ID to rank binding
     *
     * @return const LocalMeshBinding&
     */
    const LocalMeshBinding& get_local_mesh_binding() const { return local_mesh_binding_; }

    /**
     * @brief Get fabric node ID from ASIC ID mapped by the topology mapper
     *
     * @param asic_id
     * @return FabricNodeId
     */
    FabricNodeId get_fabric_node_id_from_asic_id(tt::tt_metal::AsicID asic_id) const;

    /**
     * @brief Get fabric node ID from physical chip ID mapped by the topology mapper
     *
     * @param physical_chip_id
     * @return FabricNodeId
     */
    FabricNodeId get_fabric_node_id_from_physical_chip_id(ChipId physical_chip_id) const;

    /**
     * @brief Get physical chip ID from fabric node ID mapped by the topology mapper
     *
     * @param fabric_node_id
     * @return chip_id_t
     */
    ChipId get_physical_chip_id_from_fabric_node_id(const FabricNodeId& fabric_node_id) const;

    /**
     * @brief Get ASIC ID from fabric node ID mapped by the topology mapper
     *
     * @param fabric_node_id
     * @return tt::tt_metal::AsicID
     */
    tt::tt_metal::AsicID get_asic_id_from_fabric_node_id(const FabricNodeId& fabric_node_id) const;

    /**
     * @brief Get physical chip ID from ASIC ID mapped by the topology mapper
     *
     * @param asic_id
     * @return chip_id_t
     */
    ChipId get_physical_chip_id_from_asic_id(tt::tt_metal::AsicID asic_id) const;

    /**
     * @brief Get local logical mesh chip ID to physical chip ID mapping
     *
     * @return std::map<FabricNodeId, chip_id_t>
     */
    std::map<FabricNodeId, ChipId> get_local_logical_mesh_chip_id_to_physical_chip_id_mapping() const;

    /**
     * @brief Return the host-rank layout for a mesh (independent of MeshGraph storage)
     *
     * Mirrors MeshGraph's host rank API but is derived from TopologyMapper's
     * fabric-node-to-ASIC mapping. The returned container enumerates host
     * ranks in a row-major grid describing how hosts tile the logical mesh.
     */
    const MeshContainer<MeshHostRankId>& get_host_ranks(MeshId mesh_id) const;

    /**
     * @brief Get the logical mesh shape or the per-host submesh shape
     *
     * When host_rank is not provided, returns the global mesh shape for mesh_id.
     * When host_rank is provided, returns the shape of the submesh owned by that host.
     * Shapes are derived from TopologyMapper's host-rank coordinate ranges.
     */
    MeshShape get_mesh_shape(MeshId mesh_id, std::optional<MeshHostRankId> host_rank = std::nullopt) const;

    /**
     * @brief Get hostname for a switch
     *
     * Maps switch_id to mesh_id internally and retrieves the hostname from the mesh mapping.
     *
     * @param switch_id The switch ID to get hostname for
     * @return HostName The hostname of the switch
     */
    HostName get_hostname_for_switch(SwitchId switch_id) const;

    /**
     * @brief Get hostname for a mesh
     *
     * @param mesh_id The mesh ID to get hostname for
     * @return HostName The hostname of the mesh
     */
    HostName get_hostname_for_mesh(MeshId mesh_id) const;

    /**
     * @brief Get hostname for a fabric node id
     *
     * @param fabric_node_id The fabric node id to get hostname for
     * @return HostName The hostname of the fabric node id
     */
    HostName get_hostname_for_fabric_node_id(FabricNodeId fabric_node_id) const;

    /**
     * @brief Get the coordinate range for the global mesh or a host's submesh
     *
     * When host_rank is not provided, returns the full logical mesh coordinate range (0..N-1, 0..M-1).
     * When host_rank is provided, returns the coordinate range owned by that host rank.
     * Ranges are constructed from the fabric-node-to-ASIC mapping.
     */
    MeshCoordinateRange get_coord_range(MeshId mesh_id, std::optional<MeshHostRankId> host_rank = std::nullopt) const;

    /**
     * @brief Get the host rank that owns a logical chip in a mesh
     *
     * The chip_id parameter is the Fabric Node (logical) chip id for mesh_id.
     * The returned rank is derived from TopologyMapper's host-rank coordinate ranges.
     */
    std::optional<MeshHostRankId> get_host_rank_for_chip(MeshId mesh_id, ChipId chip_id) const;

    /**
     * @brief Get the logical chip ids for a mesh or a host submesh
     *
     * When host_rank is not provided, returns a container of all logical chip
     * ids in the mesh in row-major order. When host_rank is provided, returns
     * the logical chip ids contained within that host's coordinate range,
     * preserving row-major order within the subrange. Logical chip ids are
     * derived from the global mesh shape via (i * width + j).
     */
    MeshContainer<ChipId> get_chip_ids(MeshId mesh_id, std::optional<MeshHostRankId> host_rank = std::nullopt) const;

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
     * @brief Build bidirectional mappings between ASIC IDs and physical chip IDs
     */
    void build_asic_physical_chip_id_mappings();

    /**
     * @brief Build the mapping between mesh IDs and host ranks
     *
     * This method iterates through all meshes in the mesh graph and creates mappings
     * based on the mesh IDs and fabric chip IDs from the mesh_container, mapping them
     * to the ASIC IDs of the physical descriptor. Uses MPI through distributed context
     * to gather the mappings from all ranks.
     */
    HostMeshMapping build_host_mesh_mapping();

    /**
     * @brief Validate that all meshes in the mesh graph descriptor have rank bindings
     *
     * Compares the mesh IDs from the mesh graph descriptor with the mesh IDs that have
     * hosts bound to them in rank_bindings.yaml. Throws an error if any meshes are missing
     * bindings, listing all unbound mesh IDs.
     *
     * @param mesh_id_host_names Mapping of mesh IDs to the set of host names participating in each mesh
     */
    void validate_mesh_id_host_names(const HostMeshMapping& mesh_id_host_names) const;

    /**
     * @brief Build logical adjacency maps from mesh graph connectivity
     *
     * Creates adjacency maps for each mesh based on the logical connectivity defined in the mesh graph.
     * For each fabric node in a mesh, this function identifies its logical neighbors by examining
     * the intra-mesh connectivity from the mesh graph and creates a mapping of FabricNodeId to
     * its vector of adjacent FabricNodeIds.
     *
     * @param mesh_id_to_host_names Mapping of mesh IDs to the set of host names participating in each mesh
     * @return std::unordered_map<MeshId, LogicalAdjacencyMap> Map from mesh ID to logical adjacency map
     */
    std::unordered_map<MeshId, LogicalAdjacencyMap> build_adjacency_map_logical(
        HostMeshMapping& mesh_id_to_host_names) const;

    /**
     * @brief Build physical adjacency maps from system descriptor connectivity
     *
     * Creates adjacency maps for each mesh based on the physical connectivity defined in the physical system
     * descriptor. For each ASIC in a mesh, this function identifies its physical neighbors by examining the ASIC
     * neighbors from the physical system descriptor and filters them to only include neighbors that are also part of
     * the same mesh. The resulting map contains ASIC IDs mapped to their vectors of adjacent ASIC IDs within the mesh.
     *
     * @param mesh_id_to_host_names Mapping of mesh IDs to the set of host names participating in each mesh
     * @return std::unordered_map<MeshId, PhysicalAdjacencyMap> Map from mesh ID to physical adjacency map
     */
    std::unordered_map<MeshId, PhysicalAdjacencyMap> build_adjacency_map_physical(
        HostMeshMapping& mesh_id_to_host_names) const;

    /**
     * @brief Create bidirectional mappings between logical fabric nodes and physical ASIC IDs
     *
     * This function performs the core topology mapping by creating bidirectional mappings between logical fabric nodes
     * (from the mesh graph) and physical ASIC IDs (from the physical system descriptor). It uses a constraint
     * satisfaction algorithm to find valid mappings that preserve the logical connectivity structure in the physical
     * topology.
     *
     * The algorithm:
     * 1. Validates that the logical graph can fit within the physical topology
     * 2. Uses degree-based pruning and forward checking to efficiently search for valid mappings
     * 3. Maintains consistency by ensuring logical edges are present in the physical topology
     * 4. Creates bidirectional mappings in both fabric_node_id_to_asic_id_ and asic_id_to_fabric_node_id_
     *
     * @param adjacency_map_physical Physical adjacency maps for each mesh
     * @param adjacency_map_logical Logical adjacency maps for each mesh
     */
    void populate_fabric_node_id_to_asic_id_mappings(
        const std::unordered_map<MeshId, PhysicalAdjacencyMap>& adjacency_map_physical,
        const std::unordered_map<MeshId, LogicalAdjacencyMap>& adjacency_map_logical);

    /**
     * @brief Broadcast the mapping to all hosts
     *
     * Breaks the mapping of fabric node ids into a pairs of physical chip ids to
     * logical chip ids and sends pairs to all hosts from the controller rank.
     */
    void broadcast_mapping_to_all_hosts();

    /**
     * @brief Receive the mapping from the 1st host
     */
    void receive_mapping_from_host(int rank);

    const MeshGraph& mesh_graph_;
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor_;
    const LocalMeshBinding& local_mesh_binding_;

    // Bidirectional mapping between FabricNodeId and AsicID
    std::unordered_map<FabricNodeId, tt::tt_metal::AsicID> fabric_node_id_to_asic_id_;
    std::unordered_map<tt::tt_metal::AsicID, FabricNodeId> asic_id_to_fabric_node_id_;

    // Bidirectional mapping between AsicID and physical chip id for fast lookups
    std::unordered_map<tt::tt_metal::AsicID, ChipId> asic_id_to_physical_chip_id_;
    std::unordered_map<ChipId, tt::tt_metal::AsicID> physical_chip_id_to_asic_id_;

    // Host-rank metadata for fabric-node-based queries (independent of MeshGraph's storage)
    std::vector<MeshContainer<MeshHostRankId>> mesh_host_ranks_;
    std::unordered_map<std::pair<MeshId, MeshHostRankId>, MeshCoordinateRange, hash_pair> mesh_host_rank_coord_ranges_;

    // Rebuild host-rank containers purely from fabric_node_id_to_asic_id_ mapping
    void rebuild_host_rank_structs_from_mapping();
};

}  // namespace tt::tt_fabric
