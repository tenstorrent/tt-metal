// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <unordered_set>

#include <tt_stl/span.hpp>
#include <tt-metalium/experimental/fabric/routing_table_generator.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <hostdevcommon/fabric_common.h>
#include <tt-metalium/distributed_context.hpp>

#include <map>
#include <unordered_map>
#include <memory>
#include <vector>

namespace tt::tt_metal {

class PhysicalSystemDescriptor;

}  // namespace tt::tt_metal

namespace tt::umd {

class Cluster;

}  // namespace tt::umd

namespace tt::tt_fabric {

class TopologyMapper;

// TODO: remove this once UMD provides API for UBB ID and bus ID
struct UbbId {
    std::uint32_t tray_id;
    std::uint32_t asic_id;
};

uint16_t get_bus_id(tt::umd::Cluster& cluster, ChipId chip_id);
UbbId get_ubb_id(tt::umd::Cluster& cluster, ChipId chip_id);

class FabricContext;

// This struct provides information for how a process binds to a particular
// mesh and local mesh rank (MeshHostRankId rename - #24178) in the mesh graph
// descriptor.
struct LocalMeshBinding {
    // Can bind multiple meshes to a single host. Most use-cases
    // only require a 1:1 host to mesh mapping. At least one mesh_id
    // is guaranteed to be present in this vector.
    std::vector<MeshId> mesh_ids;
    MeshHostRankId host_rank;
};

// In multi-host context, APIs parameterized with MeshScope, can return
// results for local mesh or global mesh.
enum class MeshScope {
    LOCAL,
    GLOBAL,
};

struct PortDescriptor {
    port_id_t port_id = {RoutingDirection::NONE, 0};
    std::size_t connection_hash = 0;
};

// Stores the logical ports (routing direction, logical channel id and connection hash) between the src mesh and its
// neighbor meshes
using PortDescriptorTable = std::unordered_map<MeshId, std::unordered_map<MeshId, std::vector<PortDescriptor>>>;

class ControlPlane {
public:
    ControlPlane();
    explicit ControlPlane(const std::string& mesh_graph_desc_file);
    explicit ControlPlane(
        const std::string& mesh_graph_desc_file,
        const std::map<FabricNodeId, ChipId>& logical_mesh_chip_id_to_physical_chip_id_mapping);

    ~ControlPlane();

    // Printing functions
    void print_routing_tables() const;
    void print_ethernet_channels() const;
    void print_active_ethernet_connections() const;
    void print_all_ethernet_connections() const;

    // Converts chip level routing tables to per ethernet channel
    void configure_routing_tables_for_fabric_ethernet_channels(
        tt::tt_fabric::FabricConfig fabric_config, tt_fabric::FabricReliabilityMode reliability_mode);
    void write_routing_tables_to_all_chips() const;
    void write_fabric_telemetry_to_all_chips(const FabricNodeId& fabric_node_id) const;

    // Return mesh_id, chip_id from physical chip id
    FabricNodeId get_fabric_node_id_from_physical_chip_id(ChipId physical_chip_id) const;
    // Return physical chip id from fabric node id
    ChipId get_physical_chip_id_from_fabric_node_id(const FabricNodeId& fabric_node_id) const;
    // Return fabric node id from ASIC id
    FabricNodeId get_fabric_node_id_from_asic_id(uint64_t asic_id) const;
    // Return user physical mesh ids
    std::vector<MeshId> get_user_physical_mesh_ids() const;

    // Queries for the MeshId(s) and MeshHostRankId owned by the local rank. A vector of MeshIds allows
    // a single host to bind to multiple meshes.
    std::vector<MeshId> get_local_mesh_id_bindings() const;
    MeshHostRankId get_local_host_rank_id_binding() const;
    MeshCoordinate get_local_mesh_offset() const;

    // Queries that are MeshScope-aware (i.e. return results for local mesh or global mesh)
    MeshShape get_physical_mesh_shape(MeshId mesh_id, MeshScope scope = MeshScope::GLOBAL) const;
    MeshCoordinateRange get_coord_range(MeshId mesh_id, MeshScope scope = MeshScope::GLOBAL) const;

    // Initializes distributed contexts for each mesh; for host-to-host communication.
    void initialize_distributed_contexts();

    // Returns distributed context for `mesh_id`.
    // Throws if `mesh_id` is unknown.
    const std::shared_ptr<tt::tt_metal::distributed::multihost::DistributedContext>& get_distributed_context(
        MeshId mesh_id) const;

    // Returns the distributed context with only one host.
    const std::shared_ptr<tt::tt_metal::distributed::multihost::DistributedContext>& get_host_local_context() const;

    // Return valid ethernet channels on the specificed routing plane
    std::vector<chan_id_t> get_valid_eth_chans_on_routing_plane(
        FabricNodeId fabric_node_id, routing_plane_id_t routing_plane_id) const;

    // Return path from device to device in the fabric.
    // Constraints:
    // - src_fabric_node_id must be local to the host on which this ControlPlane is running.
    // - If dst_fabric_node_id is not local to the current host, the path will end at a local
    // fabric node routing to the remote cluster.
    std::vector<std::pair<FabricNodeId, chan_id_t>> get_fabric_route(
        FabricNodeId src_fabric_node_id, FabricNodeId dst_fabric_node_id, chan_id_t src_chan_id) const;

    // Returns the direction in which the data should be forwarded from the src to reach the dest
    std::optional<RoutingDirection> get_forwarding_direction(
        FabricNodeId src_fabric_node_id, FabricNodeId dst_fabric_node_id) const;

    // Return eth channels that can forward the data from src to dest.
    // This will be a subset of the active routers in a given direction since some channels could be
    // reserved along the way for tunneling etc.
    std::vector<chan_id_t> get_forwarding_eth_chans_to_chip(
        FabricNodeId src_fabric_node_id, FabricNodeId dst_fabric_node_id) const;
    std::vector<chan_id_t> get_forwarding_eth_chans_to_chip(
        FabricNodeId src_fabric_node_id, FabricNodeId dst_fabric_node_id, RoutingDirection forwarding_direction) const;

    stl::Span<const ChipId> get_intra_chip_neighbors(
        FabricNodeId src_fabric_node_id, RoutingDirection routing_direction) const;
    std::unordered_map<MeshId, std::vector<ChipId>> get_chip_neighbors(
        FabricNodeId src_fabric_node_id, RoutingDirection routing_direction) const;

    routing_plane_id_t get_routing_plane_id(FabricNodeId fabric_node_id, chan_id_t eth_chan_id) const;

    size_t get_num_active_fabric_routers(FabricNodeId fabric_node_id) const;

    // Number of active ethernet channels, but these may not be participating in active routing planes
    // (in other words, they are not participating in fabric traffic)
    std::vector<chan_id_t> get_active_fabric_eth_channels_in_direction(
        FabricNodeId fabric_node_id, RoutingDirection routing_direction) const;
    // Return the active routing planes in a given direction.
    std::vector<chan_id_t> get_active_fabric_eth_routing_planes_in_direction(
        FabricNodeId fabric_node_id, RoutingDirection routing_direction) const;

    size_t get_num_available_routing_planes_in_direction(
        FabricNodeId fabric_node_id, RoutingDirection routing_direction) const;

    std::set<std::pair<chan_id_t, eth_chan_directions>> get_active_fabric_eth_channels(
        FabricNodeId fabric_node_id) const;

    eth_chan_directions get_eth_chan_direction(FabricNodeId fabric_node_id, int chan) const;
    // TODO: remove this converter, we should consolidate the directions here
    eth_chan_directions routing_direction_to_eth_direction(RoutingDirection direction) const;

    // Return ethernet channels on a chip that face external meshes (inter-mesh exit nodes)
    std::vector<chan_id_t> get_intermesh_facing_eth_chans(FabricNodeId fabric_node_id) const;
    // Return ethernet channels on a chip that face other chips within the same mesh (intra-mesh)
    std::vector<chan_id_t> get_intramesh_facing_eth_chans(FabricNodeId fabric_node_id) const;

    void initialize_fabric_context(tt_fabric::FabricConfig fabric_config, const FabricRouterConfig& router_config);

    FabricContext& get_fabric_context() const;

    // Get all fabric defines for kernel compilation (used by tt_metal.cpp)
    std::map<std::string, std::string> get_fabric_kernel_defines() const;

    void clear_fabric_context();

    // Initialize fabric tensix config (call after routing tables are configured)
    void initialize_fabric_tensix_datamover_config();

    // Check if the provided chip and channel is a cross-host eth link
    bool is_cross_host_eth_link(ChipId chip_id, chan_id_t chan_id) const;

    // Returns set of logical active ethernet coordinates on chip
    // If skip_reserved_cores is true, will return cores that dispatch is not using,
    // intended for users to grab available eth cores for testing
    // `skip_reserved_cores` is ignored on BH because there are no ethernet cores used for Fast Dispatch
    // tunneling
    std::unordered_set<CoreCoord> get_active_ethernet_cores(ChipId chip_id, bool skip_reserved_cores = false) const;
    std::unordered_set<CoreCoord> get_inactive_ethernet_cores(ChipId chip_id) const;

    // Collect router port directions map from all hosts via MPI and merge into local map
    void collect_and_merge_router_port_directions_from_all_hosts();

    // Get the mesh graph from the control plane
    const MeshGraph& get_mesh_graph() const;

    // Get the logical node id to mesh id and mesh host rank id mapping
    const std::unordered_map<tt_metal::distributed::multihost::Rank, std::pair<MeshId, MeshHostRankId>>&
    get_global_logical_bindings() const;

    // Check if the physical system supports the specified fabric configuration
    // Returns true if valid, false otherwise.
    bool is_fabric_config_valid(tt::tt_fabric::FabricConfig fabric_config) const;

    // Returns true if any of the local mesh bindings correspond to a switch mesh
    bool is_local_host_on_switch_mesh() const;

    // Returns physical chip IDs for all devices belonging to switch meshes on this host
    // Returns empty vector if no switch meshes are on this host
    std::vector<ChipId> get_switch_mesh_device_ids() const;

    tt::tt_metal::AsicID get_asic_id_from_fabric_node_id(const FabricNodeId& fabric_node_id) const;

private:
    // Check if the provided mesh is local to this host
    bool is_local_mesh(MeshId mesh_id) const;

    void init_control_plane(
        const std::string& mesh_graph_desc_file,
        std::optional<std::reference_wrapper<const std::map<FabricNodeId, ChipId>>>
            logical_mesh_chip_id_to_physical_chip_id_mapping = std::nullopt);

    void init_control_plane_auto_discovery();

    // TODO: remove this from local node control plane. Can get it from the global control plane
    std::unique_ptr<tt::tt_metal::PhysicalSystemDescriptor> physical_system_descriptor_;
    std::unique_ptr<tt::tt_fabric::TopologyMapper> topology_mapper_;
    std::unique_ptr<RoutingTableGenerator> routing_table_generator_;
    std::unique_ptr<MeshGraph> mesh_graph_;

    std::map<FabricNodeId, ChipId> logical_mesh_chip_id_to_physical_chip_id_mapping_;

    // map[mesh_fabric_id][direction] has a vector of ethernet channels in that direction
    std::map<FabricNodeId, std::unordered_map<RoutingDirection, std::vector<chan_id_t>>>
        router_port_directions_to_physical_eth_chan_map_;

    // map[mesh_fabric_id][direction] has the number of live routing planes in that direction
    std::map<FabricNodeId, std::unordered_map<RoutingDirection, size_t>>
        router_port_directions_to_num_routing_planes_map_;

    // tables[mesh_fabric_id][eth_chan]
    std::map<FabricNodeId, std::vector<std::vector<chan_id_t>>>
        intra_mesh_routing_tables_;  // table that will be written to each ethernet core
    std::map<FabricNodeId, std::vector<std::vector<chan_id_t>>>
        inter_mesh_routing_tables_;  // table that will be written to each ethernet core
    // Store the logical direction assigned to each exit node (an exit node is fully specified by
    // a FabricNodeId and logical channel id)
    std::map<FabricNodeId, std::unordered_map<chan_id_t, RoutingDirection>> exit_node_directions_;
    // For each FabricNode, store a mapping of the logical port (direction and logical channel id)
    // to the physical channel id
    std::map<FabricNodeId, std::unordered_map<port_id_t, chan_id_t>> logical_port_to_eth_chan_;
    // Mapping from MeshId, MeshHostRankId to MPI rank
    std::unordered_map<MeshId, std::unordered_map<MeshHostRankId, tt_metal::distributed::multihost::Rank>> mpi_ranks_;
    std::unordered_map<tt_metal::distributed::multihost::Rank, std::pair<MeshId, MeshHostRankId>>
        global_logical_bindings_;

    // custom logic to order eth channels
    void order_ethernet_channels();

    routing_plane_id_t get_routing_plane_id(
        chan_id_t eth_chan_id, const std::vector<chan_id_t>& eth_chans_in_direction) const;

    // Tries to get a valid downstream channel from the candidate_target_chans
    // First along same routing plane, but if not available, take round robin from candidates
    chan_id_t get_downstream_eth_chan_id(
        chan_id_t src_routing_plane_id, const std::vector<chan_id_t>& candidate_target_chans) const;

    ChipId get_physical_chip_id_from_eth_coord(const EthCoord& eth_coord) const;

    void load_physical_chip_mapping(
        const std::map<FabricNodeId, ChipId>& logical_mesh_chip_id_to_physical_chip_id_mapping);
    size_t get_num_live_routing_planes(FabricNodeId fabric_node_id, RoutingDirection routing_direction) const;
    void initialize_dynamic_routing_plane_counts(
        const IntraMeshConnectivity& intra_mesh_connectivity,
        tt_fabric::FabricConfig fabric_config,
        tt_fabric::FabricReliabilityMode reliability_mode);
    void trim_ethernet_channels_not_mapped_to_live_routing_planes();

    void validate_mesh_connections(MeshId mesh_id) const;
    void validate_mesh_connections() const;

    std::pair<FabricNodeId, chan_id_t> get_connected_mesh_chip_chan_ids(
        FabricNodeId fabric_node_id, chan_id_t chan_id) const;

    // Takes RoutingTableGenerator table and converts to routing tables for each ethernet port
    void convert_fabric_routing_table_to_chip_routing_table();

    void write_routing_tables_to_eth_cores(MeshId mesh_id, ChipId chip_id) const;
    void write_routing_info_to_devices(MeshId mesh_id, ChipId chip_id) const;
    void write_fabric_connections_to_tensix_cores(MeshId mesh_id, ChipId chip_id) const;
    // Helper functions to compute and embed routing path tables
    void compute_and_embed_1d_routing_path_table(MeshId mesh_id, routing_l1_info_t& routing_info) const;
    void compute_and_embed_2d_routing_path_table(MeshId mesh_id, ChipId chip_id, routing_l1_info_t& routing_info) const;

    // Helper to populate fabric connection info for both router and mux configurations
    void populate_fabric_connection_info(
        tt::tt_fabric::fabric_connection_info_t& worker_connection_info,
        tt::tt_fabric::fabric_connection_info_t& dispatcher_connection_info,
        tt::tt_fabric::fabric_connection_info_t& tensix_connection_info,
        ChipId physical_chip_id,
        chan_id_t eth_channel_id) const;

    // UDM-specific helper to write per-worker connection info to each worker core's L1
    void write_udm_fabric_connections_to_tensix_cores(
        ChipId physical_chip_id,
        const tt::tt_fabric::tensix_fabric_connections_l1_info_t& fabric_mux_connections,
        const tt::tt_fabric::tensix_fabric_connections_l1_info_t& fabric_dispatcher_connections) const;

    void assign_direction_to_fabric_eth_chan(
        const FabricNodeId& fabric_node_id, chan_id_t chan_id, RoutingDirection direction);

    void assign_direction_to_fabric_eth_core(
        const FabricNodeId& fabric_node_id, const CoreCoord& eth_core, RoutingDirection direction);

    // Initialize the local mesh binding from the environment variables
    // Returns std::nullopt if not in multi-host context
    LocalMeshBinding initialize_local_mesh_binding();

    template <uint8_t dim, bool compressed>
    void write_all_to_all_routing_fields(MeshId mesh_id) const;

    // Top level function responsible for generating intermesh connectivity
    // based on the MGD and Physical State of the system.
    // This function will annotate physical connections between physical meshes (exit nodes)
    // with a logical direction and channel id.
    // This function requires the Physical System Descriptor and Intra-Mesh Connectivity in
    // the Mesh Graph to be initialized.
    // Once this function returns, Inter Mesh Connectivity will be specified in the Mesh Graph
    // and Routing Table Generator.
    void generate_intermesh_connectivity();

    // Multi-Host Intermesh Connectivity Helper Function:
    // Assign a logical direction and channel id to each local exit node.
    std::vector<PortDescriptor> assign_logical_ports_to_exit_nodes(
        const std::string& my_host,
        const std::string& neighbor_host,
        bool strict_binding,
        const std::unordered_set<FabricNodeId>& requested_exit_nodes,
        std::unordered_set<port_id_t>& assigned_port_ids);

    // Multi-Host Intermesh Connectivity Helper Function:
    // Fully annotate local physical exit nodes in logical space (src/dst mesh id, direction, channel id)
    PortDescriptorTable generate_port_descriptors_for_exit_nodes();

    // Multi-Host Intermesh Connectivity Helper Function:
    // If the user has specified the logical devices to connect between meshes, this function will return
    // the FabricNodeIds of the requested logical exit nodes.
    std::unordered_set<FabricNodeId> get_requested_exit_nodes(
        MeshId my_mesh_id,
        MeshId neighbor_mesh_id,
        const RequestedIntermeshPorts& requested_intermesh_ports,
        const std::vector<uint64_t>& src_exit_node_chips) const;

    // Multi-Host Intermesh Connectivity Helper Function:
    // Have each host send their port descriptors to the controller host, for intermesh connectivity generation.
    void forward_descriptors_to_controller(
        PortDescriptorTable& port_descriptors, uint32_t my_rank, const std::string& my_host);

    // Multi-Host Intermesh Connectivity Helper Function:
    // Have the controller host send the generated intermesh connections to each host.
    void forward_intermesh_connections_from_controller(AnnotatedIntermeshConnections& intermesh_connections);

    // Multi-Host Intermesh Connectivity Helper Function:
    // Runs on the controller host and pairs logical port descriptors to generate intermesh connections.
    // Pairing is based on the physical connections between exit nodes.
    AnnotatedIntermeshConnections pair_logical_intermesh_ports(const PortDescriptorTable& port_descriptors);

    // Multi-Host Intermesh Connectivity Helper Function:
    // Runs on all hosts: Given the local port descriptors, this function will return the full list of intermesh
    // connections.
    AnnotatedIntermeshConnections convert_port_desciptors_to_intermesh_connections(
        PortDescriptorTable& port_descriptors);

    // Single-Host Intermesh Connectivity Helper Function:
    // Generate intermesh connections for the local host.
    AnnotatedIntermeshConnections generate_intermesh_connections_on_local_host();

    // Validate that the intermesh connections requested in the MGD can be mapped to physical links.
    void validate_requested_intermesh_connections(
        const RequestedIntermeshConnections& requested_intermesh_connections,
        const PortDescriptorTable& port_descriptors);

    std::unique_ptr<FabricContext> fabric_context_;
    LocalMeshBinding local_mesh_binding_;

    // Distributed contexts for each multi-host mesh, that this host is part of - this is typically a single mesh.
    std::unordered_map<MeshId, std::shared_ptr<tt::tt_metal::distributed::multihost::DistributedContext>>
        distributed_contexts_;

    std::shared_ptr<tt::tt_metal::distributed::multihost::DistributedContext> host_local_context_;

    // Performance caches for frequently accessed data
    // Cache for faster asic_id to fabric_node_id lookup
    // Valid for the lifetime of the physical_system_descriptor_; cleared when fabric context is reset
    mutable std::unordered_map<uint64_t, FabricNodeId> asic_id_to_fabric_node_cache_;

    // This method performs validation through assertions and exceptions.
    void validate_torus_setup(tt::tt_fabric::FabricConfig fabric_config) const;
    std::string get_galaxy_cabling_descriptor_path(tt::tt_fabric::FabricConfig fabric_config) const;
};

}  // namespace tt::tt_fabric
