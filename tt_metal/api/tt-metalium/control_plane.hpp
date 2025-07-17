// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <unordered_set>

#include <tt_stl/span.hpp>
#include <tt-metalium/routing_table_generator.hpp>
#include <tt-metalium/fabric_host_interface.h>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/fabric_types.hpp>
#include <tt-metalium/multi_mesh_types.hpp>

#include <map>
#include <unordered_map>
#include <memory>
#include <vector>

namespace tt::tt_fabric {

class FabricContext;

// This struct provides information for how a process binds to a particular
// mesh and local mesh rank (HostRankId rename - #24178) in the mesh graph
// descriptor.
struct LocalMeshBinding {
    // Can bind multiple meshes to a single host. Most use-cases
    // only require a 1:1 host to mesh mapping. At least one mesh_id
    // is guaranteed to be present in this vector.
    std::vector<MeshId> mesh_ids;
    HostRankId host_rank;
};

// In multi-host context, APIs parameterized with MeshScope, can return
// results for local mesh or global mesh.
enum class MeshScope {
    LOCAL,
    GLOBAL,
};

class ControlPlane {
public:
    explicit ControlPlane(const std::string& mesh_graph_desc_yaml_file);
    explicit ControlPlane(
        const std::string& mesh_graph_desc_yaml_file,
        const std::map<FabricNodeId, chip_id_t>& logical_mesh_chip_id_to_physical_chip_id_mapping);

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

    // Return mesh_id, chip_id from physical chip id
    FabricNodeId get_fabric_node_id_from_physical_chip_id(chip_id_t physical_chip_id) const;
    chip_id_t get_physical_chip_id_from_fabric_node_id(const FabricNodeId& fabric_node_id) const;

    std::vector<MeshId> get_user_physical_mesh_ids() const;

    // Queries for the MeshId(s) and HostRankId owned by the local rank. A vector of MeshIds allows
    // a single host to bind to multiple meshes.
    std::vector<MeshId> get_local_mesh_id_bindings() const;
    HostRankId get_local_host_rank_id_binding() const;
    MeshCoordinate get_local_mesh_offset() const;

    // Queries that are MeshScope-aware (i.e. return results for local mesh or global mesh)
    MeshShape get_physical_mesh_shape(MeshId mesh_id, MeshScope scope = MeshScope::GLOBAL) const;
    MeshCoordinateRange get_coord_range(MeshId mesh_id, MeshScope scope = MeshScope::GLOBAL) const;

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

    stl::Span<const chip_id_t> get_intra_chip_neighbors(
        FabricNodeId src_fabric_node_id, RoutingDirection routing_direction) const;
    std::unordered_map<MeshId, std::vector<chip_id_t>> get_chip_neighbors(
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

    // The following apis should probably be private, and exposed only to some Metal runtime objects
    void set_routing_mode(uint16_t mode);
    uint16_t get_routing_mode() const;

    void initialize_fabric_context(tt_fabric::FabricConfig fabric_config);

    FabricContext& get_fabric_context() const;

    void clear_fabric_context();

    // Check if ANY managed chip supports intermesh links
    bool system_has_intermesh_links() const;

    // Check if a specific chip has intermesh links configured
    bool has_intermesh_links(chip_id_t chip_id) const;

    // Get intermesh ethernet links for a specific chip
    // Returns: vector of (eth_core, channel)
    const std::vector<std::pair<CoreCoord, chan_id_t>>& get_intermesh_eth_links(chip_id_t chip_id) const;

    // Get all intermesh ethernet links in the system
    // Returns: map of chip_id -> vector of (eth_core, channel)
    const std::unordered_map<chip_id_t, std::vector<std::pair<CoreCoord, chan_id_t>>>& get_all_intermesh_eth_links()
        const;

    // Check if a specific ethernet core is an intermesh link
    bool is_intermesh_eth_link(chip_id_t chip_id, CoreCoord eth_core) const;

    // If the ethernet core is an intermesh link, probe to see if it is trained
    bool is_intermesh_eth_link_trained(chip_id_t chip_id, CoreCoord eth_core) const;

    // Returns set of logical active ethernet coordinates on chip
    // If skip_reserved_cores is true, will return cores that dispatch is not using,
    // intended for users to grab available eth cores for testing
    // `skip_reserved_cores` is ignored on BH because there are no ethernet cores used for Fast Dispatch
    // tunneling
    std::unordered_set<CoreCoord> get_active_ethernet_cores(chip_id_t chip_id, bool skip_reserved_cores = false) const;
    std::unordered_set<CoreCoord> get_inactive_ethernet_cores(chip_id_t chip_id) const;

    // Query the local intermesh link table containing the local to remote link mapping
    const IntermeshLinkTable& get_local_intermesh_link_table() const;

    // Get the ASIC ID for a chip (the ASIC ID is unique per chip, even in multi-host systems and is programmed
    // by SPI-ROM firmware)
    uint64_t get_asic_id(chip_id_t chip_id) const;

private:
    uint16_t routing_mode_ = 0;  // ROUTING_MODE_UNDEFINED
    // TODO: remove this from local node control plane. Can get it from the global control plane
    std::unique_ptr<RoutingTableGenerator> routing_table_generator_;

    std::map<FabricNodeId, chip_id_t> logical_mesh_chip_id_to_physical_chip_id_mapping_;
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
    // map[phys_chip_id] has a vector of (eth_core, channel) pairs used for intermesh routing
    // TODO: remove once UMD can provide all intermesh links
    std::unordered_map<chip_id_t, std::vector<std::pair<CoreCoord, chan_id_t>>> intermesh_eth_links_;
    // Stores a table of all local intermesh links (board_id, chan_id) and the corresponding remote intermesh links
    IntermeshLinkTable intermesh_link_table_;

    std::unordered_map<MeshId, std::unordered_map<HostRankId, std::map<EthChanDescriptor, EthChanDescriptor>>>
        peer_intermesh_link_tables_;

    // TODO: remove once UMD can provide all intermesh links
    std::unordered_map<chip_id_t, uint64_t> chip_id_to_asic_id_;

    // custom logic to order eth channels
    void order_ethernet_channels();

    routing_plane_id_t get_routing_plane_id(
        chan_id_t eth_chan_id, const std::vector<chan_id_t>& eth_chans_in_direction) const;

    std::vector<chip_id_t> get_mesh_physical_chip_ids(
        const tt::tt_metal::distributed::MeshContainer<chip_id_t>& mesh_container,
        std::optional<chip_id_t> nw_corner_chip_id = std::nullopt) const;

    std::map<FabricNodeId, chip_id_t> get_logical_chip_to_physical_chip_mapping(
        const std::string& mesh_graph_desc_file);

    // Tries to get a valid downstream channel from the candidate_target_chans
    // First along same routing plane, but if not available, take round robin from candidates
    chan_id_t get_downstream_eth_chan_id(
        chan_id_t src_chan_id, const std::vector<chan_id_t>& candidate_target_chans) const;

    chip_id_t get_physical_chip_id_from_eth_coord(const eth_coord_t& eth_coord) const;

    void load_physical_chip_mapping(
        const std::map<FabricNodeId, chip_id_t>& logical_mesh_chip_id_to_physical_chip_id_mapping);
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

    void write_routing_tables_to_eth_cores(MeshId mesh_id, chip_id_t chip_id) const;
    void write_routing_tables_to_tensix_cores(MeshId mesh_id, chip_id_t chip_id) const;
    void write_fabric_connections_to_tensix_cores(MeshId mesh_id, chip_id_t chip_id) const;

    // TODO: remove once UMD can provide all intermesh links
    // Populate the local intermesh link to remote intermesh link table
    void generate_local_intermesh_link_table();

    // All to All exchange of intermesh link tables between all hosts in the system
    void exchange_intermesh_link_tables();

    // TODO: remove once UMD can provide all intermesh links
    // Initialize internal map of physical chip_id to intermesh ethernet links
    void initialize_intermesh_eth_links();

    // TODO: remove once UMD can provide all intermesh links
    // Check if intermesh links are available by reading SPI ROM config from first chip
    bool is_intermesh_enabled() const;

    // Check if the provided mesh is local to this host
    bool is_local_mesh(MeshId mesh_id) const;

    void assign_direction_to_fabric_eth_core(
        const FabricNodeId& fabric_node_id, const CoreCoord& eth_core, RoutingDirection direction);

    void assign_intermesh_link_directions_to_local_host(const FabricNodeId& fabric_node_id);

    void assign_intermesh_link_directions_to_remote_host(const FabricNodeId& fabric_node_id);

    // Initialize the local mesh binding from the environment variables
    // Returns std::nullopt if not in multi-host context
    LocalMeshBinding initialize_local_mesh_binding();

    std::unique_ptr<FabricContext> fabric_context_;
    LocalMeshBinding local_mesh_binding_;
};

class GlobalControlPlane {
public:
    explicit GlobalControlPlane(const std::string& mesh_graph_desc_yaml_file);
    explicit GlobalControlPlane(
        const std::string& mesh_graph_desc_yaml_file,
        const std::map<FabricNodeId, chip_id_t>& logical_mesh_chip_id_to_physical_chip_id_mapping);
    ~GlobalControlPlane();

    tt::tt_fabric::ControlPlane& get_local_node_control_plane() { return *control_plane_; }

private:
    std::unique_ptr<RoutingTableGenerator> routing_table_generator_;
    std::unique_ptr<tt::tt_fabric::ControlPlane> control_plane_;

    std::string mesh_graph_desc_file_;
};

}  // namespace tt::tt_fabric
