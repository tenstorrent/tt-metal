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

namespace tt::tt_fabric {

class FabricContext;

class ControlPlane {
public:
    explicit ControlPlane(const std::string& mesh_graph_desc_yaml_file);
    explicit ControlPlane(
        const std::string& mesh_graph_desc_yaml_file,
        const std::vector<std::vector<chip_id_t>>& logical_mesh_chip_id_to_physical_chip_id_mapping);
    ~ControlPlane();

    void write_routing_tables_to_chip(mesh_id_t mesh_id, chip_id_t chip_id) const;
    void write_routing_tables_to_all_chips() const;

    // Printing functions
    void print_routing_tables() const;
    void print_ethernet_channels() const;

    // Converts chip level routing tables to per ethernet channel
    void configure_routing_tables_for_fabric_ethernet_channels();

    // Return mesh_id, chip_id from physical chip id
    std::pair<mesh_id_t, chip_id_t> get_mesh_chip_id_from_physical_chip_id(chip_id_t physical_chip_id) const;
    chip_id_t get_physical_chip_id_from_mesh_chip_id(const std::pair<mesh_id_t, chip_id_t>& mesh_chip_id) const;

    std::vector<mesh_id_t> get_user_physical_mesh_ids() const;
    tt::tt_metal::distributed::MeshShape get_physical_mesh_shape(mesh_id_t mesh_id) const;

    // Return valid ethernet channels on the specificed routing plane
    std::vector<chan_id_t> get_valid_eth_chans_on_routing_plane(
        mesh_id_t mesh_id, chip_id_t chip_id, routing_plane_id_t routing_plane_id) const;

    // Return path from device to device in the fabric
    std::vector<std::pair<chip_id_t, chan_id_t>> get_fabric_route(
        mesh_id_t src_mesh_id,
        chip_id_t src_chip_id,
        mesh_id_t dst_mesh_id,
        chip_id_t dst_chip_id,
        chan_id_t src_chan_id) const;

    // Returns the direction in which the data should be forwarded from the src to reach the dest
    std::optional<RoutingDirection> get_forwarding_direction(
        mesh_id_t src_mesh_id, chip_id_t src_chip_id, mesh_id_t dst_mesh_id, chip_id_t dst_chip_id) const;

    // Return eth channels that can forward the data from src to dest.
    // This will be a subset of the active routers in a given direction since some channels could be
    // reserved along the way for tunneling etc.
    std::vector<chan_id_t> get_forwarding_eth_chans_to_chip(
        mesh_id_t src_mesh_id, chip_id_t src_chip_id, mesh_id_t dst_mesh_id, chip_id_t dst_chip_id) const;
    std::vector<chan_id_t> get_forwarding_eth_chans_to_chip(
        mesh_id_t src_mesh_id,
        chip_id_t src_chip_id,
        mesh_id_t dst_mesh_id,
        chip_id_t dst_chip_id,
        RoutingDirection forwarding_direction) const;

    stl::Span<const chip_id_t> get_intra_chip_neighbors(
        mesh_id_t src_mesh_id, chip_id_t src_chip_id, RoutingDirection routing_direction) const;
    std::unordered_map<mesh_id_t, std::vector<chip_id_t>> get_chip_neighbors(
        mesh_id_t src_mesh_id, chip_id_t src_chip_id, RoutingDirection routing_direction) const;

    routing_plane_id_t get_routing_plane_id(mesh_id_t mesh_id, chip_id_t chip_id, chan_id_t eth_chan_id) const;

    size_t get_num_active_fabric_routers(mesh_id_t mesh_id, chip_id_t chip_id) const;

    std::vector<chan_id_t> get_active_fabric_eth_channels_in_direction(
        mesh_id_t mesh_id, chip_id_t chip_id, RoutingDirection routing_direction) const;

    std::set<std::pair<chan_id_t, eth_chan_directions>> get_active_fabric_eth_channels(
        mesh_id_t mesh_id, chip_id_t chip_id) const;
    eth_chan_directions get_eth_chan_direction(mesh_id_t mesh_id, chip_id_t chip_id, int chan) const;
    // TODO: remove this converter, we should consolidate the directions here
    eth_chan_directions routing_direction_to_eth_direction(RoutingDirection direction) const;

    void set_routing_mode(uint16_t mode);
    uint16_t get_routing_mode() const;

    void initialize_fabric_context(tt_metal::FabricConfig fabric_config);

    FabricContext& get_fabric_context() const;

    void clear_fabric_context();

private:
    uint16_t routing_mode_ = 0;  // ROUTING_MODE_UNDEFINED
    std::unique_ptr<RoutingTableGenerator> routing_table_generator_;
    std::vector<std::vector<chip_id_t>> logical_mesh_chip_id_to_physical_chip_id_mapping_;
    // map[mesh_id][chip_id][direction] has a list of ethernet channels in that direction
    std::vector<std::vector<std::unordered_map<RoutingDirection, std::vector<chan_id_t>>>>
        router_port_directions_to_physical_eth_chan_map_;
    // tables[mesh_id][chip_id][eth_chan]
    std::vector<std::vector<std::vector<std::vector<chan_id_t>>>>
        intra_mesh_routing_tables_;  // table that will be written to each ethernet core
    std::vector<std::vector<std::vector<std::vector<chan_id_t>>>>
        inter_mesh_routing_tables_;  // table that will be written to each ethernet core

    // custom logic to order eth channels
    void order_ethernet_channels();

    routing_plane_id_t get_routing_plane_id(
        chan_id_t eth_chan_id, const std::vector<chan_id_t>& eth_chans_in_direction) const;

    std::vector<std::vector<chip_id_t>> get_physical_chip_mapping_from_mesh_graph_desc_file(
        const std::string& mesh_graph_desc_file);

    // Tries to get a valid downstream channel from the candidate_target_chans
    // First along same routing plane, but if not available, take round robin from candidates
    chan_id_t get_downstream_eth_chan_id(
        chan_id_t src_chan_id, const std::vector<chan_id_t>& candidate_target_chans) const;

    chip_id_t get_physical_chip_id_from_eth_coord(const eth_coord_t& eth_coord) const;

    void load_physical_chip_mapping(
        const std::vector<std::vector<chip_id_t>>& logical_mesh_chip_id_to_physical_chip_id_mapping);

    void validate_mesh_connections(mesh_id_t mesh_id) const;
    void validate_mesh_connections() const;

    std::vector<chip_id_t> get_mesh_physical_chip_ids(
        std::uint32_t mesh_ns_size, std::uint32_t mesh_ew_size, chip_id_t nw_chip_physical_chip_id) const;

    std::tuple<mesh_id_t, chip_id_t, chan_id_t> get_connected_mesh_chip_chan_ids(
        mesh_id_t mesh_id, chip_id_t chip_id, chan_id_t chan_id) const;

    // Takes RoutingTableGenerator table and converts to routing tables for each ethernet port
    void convert_fabric_routing_table_to_chip_routing_table();

    std::unique_ptr<FabricContext> fabric_context_;
};
}  // namespace tt::tt_fabric
