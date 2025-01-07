// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_fabric/routing_table_generator.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/llrt/rtoptions.hpp"
#include "tt_fabric/hw/inc/routing_table.h"

namespace tt::tt_fabric {

class ControlPlane {
   public:
       explicit ControlPlane(const std::string& mesh_graph_desc_yaml_file);
       ~ControlPlane() = default;
       void initialize_from_mesh_graph_desc_file(const std::string& mesh_graph_desc_file);

       // Takes RoutingTableGenerator table and converts to routing tables for each ethernet port
       void convert_fabric_routing_table_to_chip_routing_table();

       void write_routing_tables_to_chip(mesh_id_t mesh_id, chip_id_t chip_id) const;
       void configure_routing_tables() const;

       // Printing functions
       void print_routing_tables() const;
       void print_ethernet_channels() const;

       // Return mesh_id, chip_id from physical chip id
       std::pair<mesh_id_t, chip_id_t> get_mesh_chip_id_from_physical_chip_id(chip_id_t physical_chip_id) const;
       chip_id_t get_physical_chip_id_from_mesh_chip_id(const std::pair<mesh_id_t, chip_id_t>& mesh_chip_id) const;

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

   private:
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

       // Tries to get a valid downstream channel from the candidate_target_chans
       // First along same routing plane, but if not available, take round robin from candidates
       chan_id_t get_downstream_eth_chan_id(
           chan_id_t src_chan_id, const std::vector<chan_id_t>& candidate_target_chans) const;

       std::vector<chip_id_t> get_mesh_physical_chip_ids(
           std::uint32_t mesh_ns_size,
           std::uint32_t mesh_ew_size,
           std::uint32_t num_ports_per_side,
           std::uint32_t nw_chip_physical_chip_id);

       std::tuple<mesh_id_t, chip_id_t, chan_id_t> get_connected_mesh_chip_chan_ids(
           mesh_id_t mesh_id, chip_id_t chip_id, chan_id_t chan_id) const;

       routing_plane_id_t get_routing_plane_id(chan_id_t eth_chan_id) const;
};

}  // namespace tt::tt_fabric
