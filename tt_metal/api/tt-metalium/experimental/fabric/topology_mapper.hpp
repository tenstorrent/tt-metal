// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <map>
#include <unordered_set>
#include <vector>

#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/experimental/fabric/routing_table_generator.hpp>

namespace tt::tt_metal {

class PhysicalSystemDescriptor;
}  // namespace tt::tt_metal

using HostName = std::string;

namespace tt::tt_fabric {

struct LocalMeshBinding;

// Use ASICPosition from tt::tt_metal namespace
using AsicPosition = tt::tt_metal::ASICPosition;

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

using HostMeshMapping = std::map<MeshId, std::unordered_set<HostName>>;
using LogicalAdjacencyMap = std::map<tt::tt_fabric::FabricNodeId, std::vector<tt::tt_fabric::FabricNodeId>>;
using PhysicalAdjacencyMap = std::map<tt::tt_metal::AsicID, std::vector<tt::tt_metal::AsicID>>;

/**
 * @brief Centralized representation of chip topology information
 *
 * This struct contains all information about a chip's topology mapping.
 * Fields are filled incrementally during mapping construction.
 * Uninitialized fields remain in their default state until filled.
 */
struct MappedChipInfo {
    // Core mapping information
    FabricNodeId fabric_node_id{MeshId{0}, 0};
    tt::tt_metal::AsicID asic_id{0};
    ChipId physical_chip_id = 0;

    // Mesh coordinate information
    MeshCoordinate mesh_coord{0, 0};

    // Host information
    MeshHostRankId mesh_host_rank{0};
    HostName hostname;

    // Flag to track if this entry has been mapped (fabric_node_id is valid)
    bool is_mapped = false;
};
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

    // Construct a TopologyMapper with fixed ASIC-position pinnings (tray, location).
    // These pins must reference devices on the current host; if infeasible, construction will throw with details.
    TopologyMapper(
        const MeshGraph& mesh_graph,
        const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
        const LocalMeshBinding& local_mesh_binding,
        const std::vector<std::pair<AsicPosition, FabricNodeId>>& fixed_asic_position_pinnings);

    // Construct a TopologyMapper from a pre-provided logical mesh chip to physical chip mapping.
    // Skips discovery and builds fabric node id to asic id mapping directly from the provided mapping.
    TopologyMapper(
        const MeshGraph& mesh_graph,
        const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
        const LocalMeshBinding& local_mesh_binding,
        const std::map<FabricNodeId, ChipId>& logical_mesh_chip_id_to_physical_chip_id_mapping);

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
     * @brief Get MPI rank for a mesh_id and host_rank pair
     *
     * Uses the topology mapper's fabric node to ASIC mapping to determine which hostname
     * owns the given (mesh_id, host_rank) pair, then returns the MPI rank for that hostname
     * from the physical system descriptor.
     *
     * @param mesh_id The mesh ID
     * @param host_rank The mesh host rank
     * @return int The MPI rank associated with this (mesh_id, host_rank) pair
     */
    int get_mpi_rank_for_mesh_host_rank(MeshId mesh_id, MeshHostRankId host_rank) const;

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
     * @brief Get the host rank that owns a logical chip in a mesh
     *
     * The chip_id parameter is the Fabric Node (logical) chip id for mesh_id.
     * The returned rank is derived from TopologyMapper's host-rank coordinate ranges.
     */
    std::optional<MeshHostRankId> get_host_rank_for_coord(MeshId mesh_id, const MeshCoordinate& coord) const;

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

    IntraMeshConnectivity get_intra_mesh_connectivity(MeshId mesh_id) const;

    InterMeshConnectivity get_inter_mesh_connectivity(MeshId mesh_id) const;

    /**
     * @brief Generate a mesh graph from a physical system descriptor
     *
     * This static function creates a mesh graph by trying different mesh shapes and finding
     * one that matches the physical topology described by the physical system descriptor.
     *
     * @param physical_system_descriptor The physical system descriptor containing ASIC topology
     * @param fabric_config The fabric configuration
     * @return MeshGraph A mesh graph that matches the physical topology
     */
    static MeshGraph generate_mesh_graph_from_physical_system_descriptor(
        const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor, FabricConfig fabric_config);

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
     * @brief Initialize chip_topology_mapping_ map with all ASICs from physical system descriptor
     *
     * Creates MappedChipInfo entries for all ASICs in the system, indexed by ASIC ID.
     * Fills in available information (asic_id, hostname, physical_chip_id for local ASICs).
     * Other fields are left empty and filled incrementally during mapping.
     */
    void initialize_chip_topology_mapping_map();

    /**
     * @brief Build the mapping between ASIC IDs and mesh host ranks
     *
     * This method iterates through all hosts in the physical system descriptor and creates mappings
     * based on the host names and fabric chip IDs from the mesh_container, mapping them
     * to the ASIC IDs of the physical descriptor. Uses MPI through distributed context
     * to gather the mappings from all ranks. The mesh host ranks come directly from the gathered
     * local bindings (TT_MESH_HOST_RANK environment variable).
     *
     * @return std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>> Map from mesh ID to
     * ASIC ID to mesh host rank (ordered for deterministic iteration)
     */
    std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>> build_asic_id_to_mesh_rank_mapping();

    /**
     * @brief Build the mapping between fabric node IDs and host ranks
     *
     * This method iterates through all fabric node IDs in the mesh graph and creates mappings
     * based on the fabric node IDs and host ranks from the local mesh binding, mapping them
     * to the host ranks of the physical descriptor.
     *
     * @return std::map<MeshId, std::map<FabricNodeId, MeshHostRankId>> Map from mesh ID to
     * fabric node ID to mesh host rank (ordered for deterministic iteration)
     */
    std::map<MeshId, std::map<FabricNodeId, MeshHostRankId>> build_fabric_node_id_to_mesh_rank_mapping() const;

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
     * 4. Updates chip_topology_mapping_ entries with mapping information (fabric_node_id, mesh_coord, etc.)
     *
     * @param mesh_id Mesh ID
     * @param adjacency_map_physical Physical adjacency maps for each mesh
     * @param adjacency_map_logical Logical adjacency maps for each mesh
     * @param host_name_to_mesh_rank Mapping of host names to mesh ranks
     */
    void populate_fabric_node_id_to_asic_id_mappings(
        MeshId mesh_id,
        const PhysicalAdjacencyMap& adjacency_map_physical,
        const LogicalAdjacencyMap& adjacency_map_logical,
        const std::map<tt::tt_metal::AsicID, MeshHostRankId>& asic_id_to_mesh_rank,
        const std::map<FabricNodeId, MeshHostRankId>& fabric_node_id_to_mesh_rank);

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
    const std::vector<std::pair<AsicPosition, FabricNodeId>> fixed_asic_position_pinnings_;
    bool generate_mapping_locally_ = false;

    // Host-rank metadata for fabric-node-based queries (independent of MeshGraph's storage)
    std::vector<MeshContainer<MeshHostRankId>> mesh_host_ranks_;
    std::map<std::pair<MeshId, MeshHostRankId>, MeshCoordinateRange> mesh_host_rank_coord_ranges_;

    // Mapping from (mesh_id, host_rank) to MPI rank for lookups when fabric node isn't in local mapping
    std::map<std::pair<MeshId, MeshHostRankId>, int> mesh_host_rank_to_mpi_rank_;

    /**
     * @brief Centralized container for chip topology information
     *
     * Contains all MappedChipInfo entries, populated incrementally during mapping construction.
     */
    std::vector<MappedChipInfo> chip_topology_mapping_;

    /**
     * @brief Lookup maps with references/pointers to chip_topology_mapping_ for fast access
     */
    std::unordered_map<FabricNodeId, MappedChipInfo*> fabric_node_id_to_mapping_;
    std::unordered_map<tt::tt_metal::AsicID, MappedChipInfo*> asic_id_to_mapping_;
    std::unordered_map<ChipId, MappedChipInfo*> physical_chip_id_to_mapping_;

    /**
     * @brief Build lookup maps from chip_topology_mapping_ container
     */
    void rebuild_lookup_maps();

    // Rebuild host-rank containers purely from chip_topology_mapping_ container
    // Uses asic_id_to_mesh_rank parameter for compatibility with algorithm improvements
    void rebuild_host_rank_structs_from_mapping(
        const std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>>& asic_id_to_mesh_rank);

    void print_logical_adjacency_map(const std::map<MeshId, LogicalAdjacencyMap>& adj_map) const;

    void print_physical_adjacency_map(const std::map<MeshId, PhysicalAdjacencyMap>& adj_map) const;
};

}  // namespace tt::tt_fabric
