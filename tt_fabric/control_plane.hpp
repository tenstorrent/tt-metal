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
    ControlPlane(const std::string& mesh_graph_desc_yaml_file);
    ~ControlPlane() = default;
    void initialize_from_mesh_graph_desc_file(const std::string& mesh_graph_desc_file);

    // Takes RoutingTableGenerator table and converts to routing tables for each ethernet port
    void convert_fabric_routing_table_to_chip_routing_table();

    void write_routing_tables_to_chip(mesh_id_t mesh_id, chip_id_t chip_id) const;
    void configure_routing_tables() const;

    // Printing functions
    void print_routing_tables() const;
    void print_ethernet_channels() const;

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

    std::vector<chip_id_t> get_mesh_physical_chip_ids(
        std::uint32_t mesh_ns_size,
        std::uint32_t mesh_ew_size,
        std::uint32_t num_ports_per_side,
        std::uint32_t nw_chip_physical_chip_id);

    chan_id_t get_eth_chan_id(chan_id_t src_chan_id, const std::vector<chan_id_t>& candidate_target_chans) const {
        std::uint32_t num_eth_ports_per_direction =
            routing_table_generator_->get_chip_spec().num_eth_ports_per_direction;
        // Explicitly map router plane channels based on mod
        //   - chan 0,4,8,12 talk to each other
        //   - chan 1,5,9,13 talk to each other
        //   - chan 2,6,10,14 talk to each other
        //   - chan 3,7,11,15 talk to each other
        std::uint32_t src_chan_mod = src_chan_id % num_eth_ports_per_direction;
        for (const auto& target_chan_id : candidate_target_chans) {
            if (src_chan_mod == target_chan_id % num_eth_ports_per_direction) {
                return target_chan_id;
            }
        }
        // If no match found, return a channel from candidate_target_chans
        while (src_chan_mod >= candidate_target_chans.size()) {
            src_chan_mod = src_chan_mod % candidate_target_chans.size();
        }
        return candidate_target_chans[src_chan_mod];
    };
};

}  // namespace tt::tt_fabric
