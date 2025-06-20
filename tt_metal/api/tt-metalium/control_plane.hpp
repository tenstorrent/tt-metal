// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt_stl/span.hpp>
#include <tt-metalium/routing_table_generator.hpp>
#include <tt-metalium/fabric_host_interface.h>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/fabric_types.hpp>

#include <map>
#include <unordered_map>
#include <memory>
#include <vector>

namespace tt::tt_fabric {

class FabricContext;

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
    void configure_routing_tables_for_fabric_ethernet_channels(tt_metal::FabricReliabilityMode reliability_mode);
    void write_routing_tables_to_all_chips() const;

    // Return mesh_id, chip_id from physical chip id
    FabricNodeId get_fabric_node_id_from_physical_chip_id(chip_id_t physical_chip_id) const;
    chip_id_t get_physical_chip_id_from_fabric_node_id(const FabricNodeId& fabric_node_id) const;

    std::vector<MeshId> get_user_physical_mesh_ids() const;
    MeshShape get_physical_mesh_shape(MeshId mesh_id) const;

    // Return valid ethernet channels on the specificed routing plane
    std::vector<chan_id_t> get_valid_eth_chans_on_routing_plane(
        FabricNodeId fabric_node_id, routing_plane_id_t routing_plane_id) const;

    // Return path from device to device in the fabric
    std::vector<std::pair<chip_id_t, chan_id_t>> get_fabric_route(
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

    void initialize_fabric_context(tt_metal::FabricConfig fabric_config, tt_metal::FabricReliabilityMode reliability_mode);

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
    std::unordered_map<chip_id_t, std::vector<std::pair<CoreCoord, chan_id_t>>> intermesh_eth_links_;

    // custom logic to order eth channels
    void order_ethernet_channels();

    routing_plane_id_t get_routing_plane_id(
        chan_id_t eth_chan_id, const std::vector<chan_id_t>& eth_chans_in_direction) const;

    std::map<FabricNodeId, chip_id_t> get_physical_chip_mapping_from_mesh_graph_desc_file(
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
        const IntraMeshConnectivity& intra_mesh_connectivity, tt_metal::FabricConfig fabric_config, tt_metal::FabricReliabilityMode reliability_mode);
    void trim_ethernet_channels_not_mapped_to_live_routing_planes();

    void validate_mesh_connections(MeshId mesh_id) const;
    void validate_mesh_connections() const;

    std::vector<chip_id_t> get_mesh_physical_chip_ids(
        std::uint32_t mesh_ns_size, std::uint32_t mesh_ew_size, chip_id_t nw_chip_physical_chip_id) const;

    std::pair<FabricNodeId, chan_id_t> get_connected_mesh_chip_chan_ids(
        FabricNodeId fabric_node_id, chan_id_t chan_id) const;

    // Takes RoutingTableGenerator table and converts to routing tables for each ethernet port
    void convert_fabric_routing_table_to_chip_routing_table();

    void write_routing_tables_to_chip(MeshId mesh_id, chip_id_t chip_id) const;

    // Initialize internal map of physical chip_id to intermesh ethernet links
    void initialize_intermesh_eth_links();

    // Check if intermesh links are available by reading SPI ROM config from first chip
    bool is_intermesh_enabled() const;

    std::unique_ptr<FabricContext> fabric_context_;
};

class GlobalControlPlane {
public:
    explicit GlobalControlPlane(const std::string& mesh_graph_desc_yaml_file);
    explicit GlobalControlPlane(
        const std::string& mesh_graph_desc_yaml_file,
        const std::map<FabricNodeId, chip_id_t>& logical_mesh_chip_id_to_physical_chip_id_mapping);
    ~GlobalControlPlane();

    void initialize_host_mapping();

    tt::tt_fabric::ControlPlane& get_local_node_control_plane() { return *control_plane_; }

private:
    std::unique_ptr<RoutingTableGenerator> routing_table_generator_;
    // Host rank to sub mesh shape
    std::unordered_map<HostRankId, std::vector<MeshCoordinate>> host_rank_to_sub_mesh_shape_;
    std::unique_ptr<tt::tt_fabric::ControlPlane> control_plane_;

    std::string mesh_graph_desc_file_;
};

}  // namespace tt::tt_fabric
