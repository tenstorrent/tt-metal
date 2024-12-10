// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_fabric/control_plane.hpp"
#include <queue>

namespace tt::tt_fabric {

// Get the physical chip ids for a mesh
std::unordered_map<chip_id_t, std::vector<CoreCoord>> get_ethernet_cores_grouped_by_connected_chips(chip_id_t chip_id) {
    return tt::Cluster::instance().get_ethernet_cores_grouped_by_connected_chips(chip_id);
}

bool is_chip_on_edge_of_mesh(
    chip_id_t physical_chip_id,
    int chips_per_side,
    const std::unordered_map<chip_id_t, std::vector<CoreCoord>>& ethernet_cores_grouped_by_connected_chips) {
    // TODO: check if syseng chip routing info has this info
    // Chip is on edge if it does not have full connections to four sides
    int i = 0;
    for (const auto& [connected_chip_id, eth_ports] : ethernet_cores_grouped_by_connected_chips) {
        if (eth_ports.size() == chips_per_side) {
            i++;
        }
    }
    return (i == 3);
}

bool is_chip_on_corner_of_mesh(
    chip_id_t physical_chip_id,
    int chips_per_side,
    const std::unordered_map<chip_id_t, std::vector<CoreCoord>>& ethernet_cores_grouped_by_connected_chips) {
    // Chip is a corner if it has exactly 2 fully connected sides
    int i = 0;
    for (const auto& [connected_chip_id, eth_ports] : ethernet_cores_grouped_by_connected_chips) {
        if (eth_ports.size() == chips_per_side) {
            i++;
        }
    }
    return (i < 3);
}

ControlPlane::ControlPlane(const std::string& mesh_graph_desc_file) {
    this->routing_table_generator_ = std::make_unique<RoutingTableGenerator>(mesh_graph_desc_file);
    // Printing, only enabled with log_debug
    this->routing_table_generator_->print_connectivity();
    // Printing, only enabled with log_debug
    this->routing_table_generator_->print_routing_tables();

    this->print_ethernet_channels();
    this->initialize_from_mesh_graph_desc_file(mesh_graph_desc_file);
    this->configure_routing_tables();

    // Printing, only enabled with log_debug
    this->print_ethernet_channels();
    // Printing, only enabled with log_debug
    this->print_routing_tables();
}

std::vector<chip_id_t> ControlPlane::get_mesh_physical_chip_ids(
    std::uint32_t mesh_ns_size,
    std::uint32_t mesh_ew_size,
    std::uint32_t num_ports_per_side,
    std::uint32_t nw_chip_physical_chip_id) {
    // Get shortest paths to to all chips on edges
    std::unordered_map<chip_id_t, std::vector<std::vector<chip_id_t>>> paths;
    paths.insert({nw_chip_physical_chip_id, {{nw_chip_physical_chip_id}}});

    std::unordered_map<chip_id_t, std::uint32_t> dist;
    dist.insert({nw_chip_physical_chip_id, 0});

    std::unordered_set<chip_id_t> visited_physical_chips;

    std::queue<chip_id_t> q;
    q.push(nw_chip_physical_chip_id);
    visited_physical_chips.insert(nw_chip_physical_chip_id);
    while (!q.empty()) {
        chip_id_t current_chip_id = q.front();
        q.pop();

        auto eth_links = get_ethernet_cores_grouped_by_connected_chips(current_chip_id);
        for (const auto& [connected_chip_id, eth_ports] : eth_links) {
            bool is_edge = is_chip_on_edge_of_mesh(
                connected_chip_id,
                num_ports_per_side,
                get_ethernet_cores_grouped_by_connected_chips(connected_chip_id));
            if (eth_ports.size() == num_ports_per_side) {
                if (visited_physical_chips.find(connected_chip_id) == visited_physical_chips.end()) {
                    q.push(connected_chip_id);
                    visited_physical_chips.insert(connected_chip_id);
                    dist.insert({connected_chip_id, std::numeric_limits<std::uint32_t>::max()});
                    paths.insert({connected_chip_id, {}});
                }
                if (dist.at(connected_chip_id) > dist.at(current_chip_id) + 1) {
                    dist.at(connected_chip_id) = dist.at(current_chip_id) + 1;
                    paths.at(connected_chip_id) = paths.at(current_chip_id);
                    for (auto& path : paths.at(connected_chip_id)) {
                        path.push_back(connected_chip_id);
                    }
                } else if (dist.at(connected_chip_id) == dist.at(current_chip_id) + 1) {
                    // another possible path discovered
                    for (auto& path : paths.at(current_chip_id)) {
                        paths.at(connected_chip_id).push_back(path);
                    }

                    paths.at(connected_chip_id)[paths.at(connected_chip_id).size() - 1].push_back(connected_chip_id);
                }
            }
        }
    }

    std::vector<chip_id_t> physical_chip_ids;
    // TODO: if square mesh, we might need to pin another corner chip
    for (const auto& [dest_id, equal_dist_paths] : paths) {
        if (equal_dist_paths.size() == 1) {
            auto dest_chip_id = equal_dist_paths[0][equal_dist_paths[0].size() - 1];
            bool is_corner = is_chip_on_corner_of_mesh(
                dest_chip_id, num_ports_per_side, get_ethernet_cores_grouped_by_connected_chips(dest_chip_id));
            if (is_corner and equal_dist_paths[0].size() == mesh_ew_size) {
                physical_chip_ids = equal_dist_paths[0];
                break;
            }
        }
    }

    TT_ASSERT(
        physical_chip_ids.size() == mesh_ew_size,
        "Did not find edge with expected number of East-West chips {}",
        mesh_ew_size);

    // Loop over edge and populate entire mesh with physical chip ids
    // reset and reuse the visited set of physical chip ids
    visited_physical_chips = {};
    visited_physical_chips.insert(physical_chip_ids.begin(), physical_chip_ids.end());
    physical_chip_ids.resize(mesh_ns_size * mesh_ew_size);

    for (int i = 1; i < mesh_ns_size; i++) {
        for (int j = 0; j < mesh_ew_size; j++) {
            chip_id_t physical_chip_id_from_north = physical_chip_ids[(i - 1) * mesh_ew_size + j];
            auto eth_links_grouped_by_connected_chips =
                get_ethernet_cores_grouped_by_connected_chips(physical_chip_id_from_north);
            for (const auto& [connected_chip_id, eth_ports] : eth_links_grouped_by_connected_chips) {
                if (visited_physical_chips.find(connected_chip_id) == visited_physical_chips.end() and
                    eth_ports.size() == num_ports_per_side) {
                    physical_chip_ids[i * mesh_ew_size + j] = connected_chip_id;
                    visited_physical_chips.insert(connected_chip_id);
                    break;
                }
            }
        }
    }

    std::stringstream ss;
    for (int i = 0; i < mesh_ns_size * mesh_ew_size; i++) {
        if (i % mesh_ew_size == 0) {
            ss << std::endl;
        }
        ss << " " << std::setfill('0') << std::setw(2) << physical_chip_ids[i];
    }
    log_debug(tt::LogFabric, "Control Plane: NW {} Physical Device Ids: {}", nw_chip_physical_chip_id, ss.str());
    return physical_chip_ids;
}

void ControlPlane::initialize_from_mesh_graph_desc_file(const std::string& mesh_graph_desc_file) {
    // TODO: temp testing code, probably will remove
    std::filesystem::path cluster_desc_file_path;
    eth_coord_t nw_chip_eth_coord;
    std::uint32_t mesh_ns_size, mesh_ew_size;
    if (mesh_graph_desc_file.find("tg_mesh_graph_descriptor.yaml") != std::string::npos) {
        cluster_desc_file_path = std::filesystem::path(tt::llrt::OptionsG.get_root_dir()) /
                                 "tests/tt_metal/tt_fabric/common/tg_cluster_desc.yaml";

        // Add the N150 MMIO devices
        this->logical_mesh_chip_id_to_physical_chip_id_mapping_.push_back(this->get_mesh_physical_chip_ids(1, 1, 4, 0));
        this->logical_mesh_chip_id_to_physical_chip_id_mapping_.push_back(this->get_mesh_physical_chip_ids(1, 1, 4, 1));
        this->logical_mesh_chip_id_to_physical_chip_id_mapping_.push_back(this->get_mesh_physical_chip_ids(1, 1, 4, 2));
        this->logical_mesh_chip_id_to_physical_chip_id_mapping_.push_back(this->get_mesh_physical_chip_ids(1, 1, 4, 3));

        nw_chip_eth_coord = {0, 3, 7, 0, 1};
        mesh_ns_size = routing_table_generator_->get_mesh_ns_size(/*mesh_id=*/4);
        mesh_ew_size = routing_table_generator_->get_mesh_ew_size(/*mesh_id=*/4);
    } else if (mesh_graph_desc_file.find("t3k_mesh_graph_descriptor.yaml") != std::string::npos) {
        cluster_desc_file_path = std::filesystem::path(tt::llrt::OptionsG.get_root_dir()) /
                                 "tests/tt_metal/tt_fabric/common/t3k_cluster_desc.yaml";
        nw_chip_eth_coord = {0, 0, 1, 0, 0};
        mesh_ns_size = routing_table_generator_->get_mesh_ns_size(/*mesh_id=*/0);
        mesh_ew_size = routing_table_generator_->get_mesh_ew_size(/*mesh_id=*/0);
    } else {
        TT_FATAL(false, "Unsupported mesh graph descriptor file {}", mesh_graph_desc_file);
    }
    std::unique_ptr<tt_ClusterDescriptor> cluster_desc =
        tt_ClusterDescriptor::create_from_yaml(cluster_desc_file_path.string());

    // Must pin a NW chip
    // TODO: need to rework how this is determined
    chip_id_t nw_chip_physical_chip_id;
    for (const auto& [physical_chip_id, eth_coord] : cluster_desc->get_chip_locations()) {
        if (eth_coord == nw_chip_eth_coord) {
            nw_chip_physical_chip_id = physical_chip_id;
            break;
        }
    }
    const std::uint32_t num_ports_per_side = routing_table_generator_->get_chip_spec().num_eth_ports_per_direction;
    this->logical_mesh_chip_id_to_physical_chip_id_mapping_.push_back(
        this->get_mesh_physical_chip_ids(mesh_ns_size, mesh_ew_size, num_ports_per_side, nw_chip_physical_chip_id));

    // From here, should be production init
    const auto& intra_mesh_connectivity = this->routing_table_generator_->get_intra_mesh_connectivity();
    const auto& inter_mesh_connectivity = this->routing_table_generator_->get_inter_mesh_connectivity();

    // Initialize the bookkeeping for mapping from mesh/chip/direction to physical ethernet channels
    this->router_port_directions_to_physical_eth_chan_map_.resize(intra_mesh_connectivity.size());
    for (mesh_id_t mesh_id = 0; mesh_id < intra_mesh_connectivity.size(); mesh_id++) {
        this->router_port_directions_to_physical_eth_chan_map_[mesh_id].resize(intra_mesh_connectivity[mesh_id].size());
        for (chip_id_t chip_id = 0; chip_id < intra_mesh_connectivity[mesh_id].size(); chip_id++) {
            for (const auto& [logical_connected_chip_id, edge] : intra_mesh_connectivity[mesh_id][chip_id]) {
                const auto& physical_chip_id =
                    this->logical_mesh_chip_id_to_physical_chip_id_mapping_[mesh_id][chip_id];
                const auto& physical_connected_chip_id =
                    this->logical_mesh_chip_id_to_physical_chip_id_mapping_[mesh_id][logical_connected_chip_id];
                const auto& connected_eth_channels =
                    cluster_desc->get_directly_connected_ethernet_channels_between_chips(
                        physical_chip_id, physical_connected_chip_id);
                TT_ASSERT(
                    connected_eth_channels.size() == edge.connected_chip_ids.size(),
                    "Expected {} eth links from physical chip {} to physical chip {}",
                    edge.connected_chip_ids.size(),
                    physical_chip_id,
                    physical_connected_chip_id);

                for (const auto& eth_chan : connected_eth_channels) {
                    // There could be an optimization here to create entry for both chips here, assuming links are
                    // bidirectional
                    this->router_port_directions_to_physical_eth_chan_map_[mesh_id][chip_id][edge.port_direction]
                        .push_back(std::get<0>(eth_chan));
                }
            }
        }
    }
    for (mesh_id_t mesh_id = 0; mesh_id < inter_mesh_connectivity.size(); mesh_id++) {
        for (chip_id_t chip_id = 0; chip_id < inter_mesh_connectivity[mesh_id].size(); chip_id++) {
            for (const auto& [connected_mesh_id, edge] : inter_mesh_connectivity[mesh_id][chip_id]) {
                // Loop over edges connected chip ids, they could connect to different chips for intermesh traffic
                for (const auto& logical_connected_chip_id : edge.connected_chip_ids) {
                    auto physical_chip_id = this->logical_mesh_chip_id_to_physical_chip_id_mapping_[mesh_id][chip_id];
                    const auto& connected_eth_channels =
                        cluster_desc->get_directly_connected_ethernet_channels_between_chips(
                            physical_chip_id,
                            this->logical_mesh_chip_id_to_physical_chip_id_mapping_[connected_mesh_id]
                                                                                   [logical_connected_chip_id]);
                    TT_ASSERT(
                        connected_eth_channels.size() == 1,
                        "Expect exactly one connected eth channel");  // Since we are looping over the connected chip
                                                                      // ids

                    for (const auto& eth_chan : connected_eth_channels) {
                        this->router_port_directions_to_physical_eth_chan_map_[mesh_id][chip_id][edge.port_direction]
                            .push_back(std::get<0>(eth_chan));
                    }
                }
            }
        }
    }

    this->convert_fabric_routing_table_to_chip_routing_table();
}

routing_plane_id_t ControlPlane::get_routing_plane_id(chan_id_t eth_chan_id) const {
    // Assumes that ethernet channels are incrementing by one in the same direction
    // Same mapping for all variants of active eth cores
    std::uint32_t num_eth_ports_per_direction = routing_table_generator_->get_chip_spec().num_eth_ports_per_direction;
    return eth_chan_id % num_eth_ports_per_direction;
}

chan_id_t ControlPlane::get_downstream_eth_chan_id(
    chan_id_t src_chan_id, const std::vector<chan_id_t>& candidate_target_chans) const {
    // Explicitly map router plane channels based on mod
    //   - chan 0,4,8,12 talk to each other
    //   - chan 1,5,9,13 talk to each other
    //   - chan 2,6,10,14 talk to each other
    //   - chan 3,7,11,15 talk to each other
    std::uint32_t src_routing_plane_id = this->get_routing_plane_id(src_chan_id);
    for (const auto& target_chan_id : candidate_target_chans) {
        if (src_routing_plane_id == this->get_routing_plane_id(target_chan_id)) {
            return target_chan_id;
        }
    }
    // If no match found, return a channel from candidate_target_chans
    while (src_routing_plane_id >= candidate_target_chans.size()) {
        src_routing_plane_id = src_routing_plane_id % candidate_target_chans.size();
    }
    return candidate_target_chans[src_routing_plane_id];
};

void ControlPlane::convert_fabric_routing_table_to_chip_routing_table() {
    // Routing tables contain direction from chip to chip
    // Convert it to be unique per ethernet channel
    const auto& router_intra_mesh_routing_table = this->routing_table_generator_->get_intra_mesh_table();
    this->intra_mesh_routing_tables_.resize(router_intra_mesh_routing_table.size());
    for (mesh_id_t mesh_id = 0; mesh_id < router_intra_mesh_routing_table.size(); mesh_id++) {
        this->intra_mesh_routing_tables_[mesh_id].resize(router_intra_mesh_routing_table[mesh_id].size());
        for (chip_id_t src_chip_id = 0; src_chip_id < router_intra_mesh_routing_table[mesh_id].size(); src_chip_id++) {
            const auto& physical_chip_id =
                this->logical_mesh_chip_id_to_physical_chip_id_mapping_[mesh_id][src_chip_id];
            std::uint32_t num_ports_per_chip =
                tt::Cluster::instance().get_soc_desc(physical_chip_id).ethernet_cores.size();
            this->intra_mesh_routing_tables_[mesh_id][src_chip_id].resize(
                num_ports_per_chip);  // contains more entries than needed, this size is for all eth channels on chip
            for (int i = 0; i < num_ports_per_chip; i++) {
                // Size the routing table to the number of chips in the mesh
                this->intra_mesh_routing_tables_[mesh_id][src_chip_id][i].resize(
                    router_intra_mesh_routing_table[mesh_id][src_chip_id].size());
            }
            for (chip_id_t dst_chip_id = 0; dst_chip_id < router_intra_mesh_routing_table[mesh_id][src_chip_id].size();
                 dst_chip_id++) {
                // Target direction is the direction to the destination chip for all ethernet channesl
                const auto& target_direction = router_intra_mesh_routing_table[mesh_id][src_chip_id][dst_chip_id];
                // We view ethernet channels on one side of the chip as parallel planes. So N[0] talks to S[0], E[0],
                // W[0] and so on For all live ethernet channels on this chip, set the routing table entry to the
                // destination chip as the ethernet channel on the same plane
                for (const auto& [direction, eth_chans_on_side] :
                     this->router_port_directions_to_physical_eth_chan_map_[mesh_id][src_chip_id]) {
                    for (const auto& src_chan_id : eth_chans_on_side) {
                        if (src_chip_id == dst_chip_id) {
                            TT_ASSERT(
                                (target_direction == RoutingDirection::C),
                                "Expecting same direction for intra mesh routing");
                            // This entry represents chip to itself, should not be used by FW
                            this->intra_mesh_routing_tables_[mesh_id][src_chip_id][src_chan_id][dst_chip_id] =
                                src_chan_id;
                        } else if (target_direction == direction) {
                            // This entry represents an outgoing eth channel
                            this->intra_mesh_routing_tables_[mesh_id][src_chip_id][src_chan_id][dst_chip_id] =
                                src_chan_id;
                        } else {
                            const auto& eth_chans_in_target_direction =
                                this->router_port_directions_to_physical_eth_chan_map_[mesh_id][src_chip_id].at(
                                    target_direction);
                            this->intra_mesh_routing_tables_[mesh_id][src_chip_id][src_chan_id][dst_chip_id] =
                                this->get_downstream_eth_chan_id(src_chan_id, eth_chans_in_target_direction);
                        }
                    }
                }
            }
        }
    }

    const auto& router_inter_mesh_routing_table = this->routing_table_generator_->get_inter_mesh_table();
    this->inter_mesh_routing_tables_.resize(router_inter_mesh_routing_table.size());
    for (mesh_id_t src_mesh_id = 0; src_mesh_id < router_inter_mesh_routing_table.size(); src_mesh_id++) {
        this->inter_mesh_routing_tables_[src_mesh_id].resize(router_inter_mesh_routing_table[src_mesh_id].size());
        for (chip_id_t src_chip_id = 0; src_chip_id < router_inter_mesh_routing_table[src_mesh_id].size();
             src_chip_id++) {
            const auto& physical_chip_id =
                this->logical_mesh_chip_id_to_physical_chip_id_mapping_[src_mesh_id][src_chip_id];
            std::uint32_t num_ports_per_chip =
                tt::Cluster::instance().get_soc_desc(physical_chip_id).ethernet_cores.size();
            this->inter_mesh_routing_tables_[src_mesh_id][src_chip_id].resize(
                num_ports_per_chip);  // contains more entries than needed
            for (int i = 0; i < num_ports_per_chip; i++) {
                // Size the routing table to the number of meshes
                this->inter_mesh_routing_tables_[src_mesh_id][src_chip_id][i].resize(
                    router_inter_mesh_routing_table[src_mesh_id][src_chip_id].size());
            }
            for (chip_id_t dst_mesh_id = 0;
                 dst_mesh_id < router_inter_mesh_routing_table[src_mesh_id][src_chip_id].size();
                 dst_mesh_id++) {
                // Target direction is the direction to the destination mesh for all ethernet channesl
                const auto& target_direction = router_inter_mesh_routing_table[src_mesh_id][src_chip_id][dst_mesh_id];

                // We view ethernet channels on one side of the chip as parallel planes. So N[0] talks to S[0], E[0],
                // W[0] and so on For all live ethernet channels on this chip, set the routing table entry to the
                // destination mesh as the ethernet channel on the same plane
                for (const auto& [direction, eth_chans_on_side] :
                     this->router_port_directions_to_physical_eth_chan_map_[src_mesh_id][src_chip_id]) {
                    for (const auto& src_chan_id : eth_chans_on_side) {
                        if (src_mesh_id == dst_mesh_id) {
                            TT_ASSERT(
                                (target_direction == RoutingDirection::C),
                                "ControlPlane: Expecting same direction for inter mesh routing");
                            // This entry represents mesh to itself, should not be used by FW
                            this->inter_mesh_routing_tables_[src_mesh_id][src_chip_id][src_chan_id][dst_mesh_id] =
                                src_chan_id;
                        } else if (target_direction == direction) {
                            // This entry represents an outgoing eth channel
                            this->inter_mesh_routing_tables_[src_mesh_id][src_chip_id][src_chan_id][dst_mesh_id] =
                                src_chan_id;
                        } else {
                            const auto& eth_chans_in_target_direction =
                                this->router_port_directions_to_physical_eth_chan_map_[src_mesh_id][src_chip_id].at(
                                    target_direction);
                            this->inter_mesh_routing_tables_[src_mesh_id][src_chip_id][src_chan_id][dst_mesh_id] =
                                this->get_downstream_eth_chan_id(src_chan_id, eth_chans_in_target_direction);
                        }
                    }
                }
            }
        }
    }
}

void ControlPlane::write_routing_tables_to_chip(mesh_id_t mesh_id, chip_id_t chip_id) const {
    // TODO: remove this
    const auto& chip_intra_mesh_routing_tables = this->intra_mesh_routing_tables_[mesh_id][chip_id];
    const auto& chip_inter_mesh_routing_tables = this->inter_mesh_routing_tables_[mesh_id][chip_id];
    const auto& physical_chip_id = this->logical_mesh_chip_id_to_physical_chip_id_mapping_[mesh_id][chip_id];
    // Loop over ethernet channels to only write to cores with ethernet links
    // Looping over chip_intra/inter_mesh_routing_tables will write to all cores, even if they don't have ethernet links
    const auto& chip_eth_chans_map = this->router_port_directions_to_physical_eth_chan_map_[mesh_id][chip_id];
    for (const auto& [direction, eth_chans] : chip_eth_chans_map) {
        for (const auto& eth_chan : eth_chans) {
            // eth_chans are the active ethernet channels on this chip
            const auto& eth_chan_intra_mesh_routing_table = chip_intra_mesh_routing_tables[eth_chan];
            const auto& eth_chan_inter_mesh_routing_table = chip_inter_mesh_routing_tables[eth_chan];
            tt::tt_fabric::fabric_router_l1_config_t fabric_router_config;
            std::fill_n(
                fabric_router_config.intra_mesh_table.dest_entry,
                tt::tt_fabric::MAX_MESH_SIZE,
                eth_chan_magic_values::INVALID_ROUTING_TABLE_ENTRY);
            std::fill_n(
                fabric_router_config.inter_mesh_table.dest_entry,
                tt::tt_fabric::MAX_NUM_MESHES,
                eth_chan_magic_values::INVALID_ROUTING_TABLE_ENTRY);
            for (uint32_t i = 0; i < eth_chan_intra_mesh_routing_table.size(); i++) {
                fabric_router_config.intra_mesh_table.dest_entry[i] = eth_chan_intra_mesh_routing_table[i];
            }
            for (uint32_t i = 0; i < eth_chan_inter_mesh_routing_table.size(); i++) {
                fabric_router_config.inter_mesh_table.dest_entry[i] = eth_chan_inter_mesh_routing_table[i];
            }

            if (chip_eth_chans_map.find(RoutingDirection::N) != chip_eth_chans_map.end()) {
                fabric_router_config.port_direction.north =
                    this->get_downstream_eth_chan_id(eth_chan, chip_eth_chans_map.at(RoutingDirection::N));
            } else {
                fabric_router_config.port_direction.north = eth_chan_magic_values::INVALID_DIRECTION;
            }
            if (chip_eth_chans_map.find(RoutingDirection::S) != chip_eth_chans_map.end()) {
                fabric_router_config.port_direction.south =
                    this->get_downstream_eth_chan_id(eth_chan, chip_eth_chans_map.at(RoutingDirection::S));
            } else {
                fabric_router_config.port_direction.south = eth_chan_magic_values::INVALID_DIRECTION;
            }
            if (chip_eth_chans_map.find(RoutingDirection::E) != chip_eth_chans_map.end()) {
                fabric_router_config.port_direction.east =
                    this->get_downstream_eth_chan_id(eth_chan, chip_eth_chans_map.at(RoutingDirection::E));
            } else {
                fabric_router_config.port_direction.east = eth_chan_magic_values::INVALID_DIRECTION;
            }
            if (chip_eth_chans_map.find(RoutingDirection::W) != chip_eth_chans_map.end()) {
                fabric_router_config.port_direction.west =
                    this->get_downstream_eth_chan_id(eth_chan, chip_eth_chans_map.at(RoutingDirection::W));
            } else {
                fabric_router_config.port_direction.west = eth_chan_magic_values::INVALID_DIRECTION;
            }

            fabric_router_config.my_mesh_id = mesh_id;
            fabric_router_config.my_device_id = chip_id;

            // Write data to physical eth core
            tt_cxy_pair physical_eth_core(
                physical_chip_id,
                tt::Cluster::instance().get_soc_desc(physical_chip_id).physical_ethernet_cores[eth_chan]);

            tt::Cluster::instance().write_core(
                (void*)&fabric_router_config,
                sizeof(tt::tt_fabric::fabric_router_l1_config_t),
                physical_eth_core,
                eth_l1_mem::address_map::FABRIC_ROUTER_CONFIG_BASE,
                false);
        }
    }
}

std::pair<mesh_id_t, chip_id_t> ControlPlane::get_mesh_chip_id_from_physical_chip_id(chip_id_t physical_chip_id) const {
    for (mesh_id_t mesh_id = 0; mesh_id < logical_mesh_chip_id_to_physical_chip_id_mapping_.size(); ++mesh_id) {
        for (chip_id_t logical_mesh_chip_id = 0;
             logical_mesh_chip_id < logical_mesh_chip_id_to_physical_chip_id_mapping_[mesh_id].size();
             ++logical_mesh_chip_id) {
            if (logical_mesh_chip_id_to_physical_chip_id_mapping_[mesh_id][logical_mesh_chip_id] == physical_chip_id) {
                return {mesh_id, logical_mesh_chip_id};
            }
        }
    }
    TT_FATAL(false, "Physical chip id not found in logical mesh chip id mapping");
    return {};
}

chip_id_t ControlPlane::get_physical_chip_id_from_mesh_chip_id(
    const std::pair<mesh_id_t, chip_id_t>& mesh_chip_id) const {
    return logical_mesh_chip_id_to_physical_chip_id_mapping_[mesh_chip_id.first][mesh_chip_id.second];
}

std::tuple<mesh_id_t, chip_id_t, chan_id_t> ControlPlane::get_connected_mesh_chip_chan_ids(
    mesh_id_t mesh_id, chip_id_t chip_id, chan_id_t chan_id) const {
    // TODO: simplify this and maybe have this functionality in ControlPlane
    auto physical_chip_id = logical_mesh_chip_id_to_physical_chip_id_mapping_[mesh_id][chip_id];
    auto eth_core = tt::Cluster::instance().get_soc_desc(physical_chip_id).chan_to_logical_eth_core_map.at(chan_id);
    auto [connected_physical_chip_id, connected_eth_core] =
        tt::Cluster::instance().get_connected_ethernet_core(std::make_tuple(physical_chip_id, eth_core));

    auto [connected_mesh_id, connected_chip_id] =
        this->get_mesh_chip_id_from_physical_chip_id(connected_physical_chip_id);
    auto connected_chan_id = tt::Cluster::instance()
                                 .get_soc_desc(connected_physical_chip_id)
                                 .logical_eth_core_to_chan_map.at(connected_eth_core);
    return std::make_tuple(connected_mesh_id, connected_chip_id, connected_chan_id);
}

std::vector<chan_id_t> ControlPlane::get_valid_eth_chans_on_routing_plane(
    mesh_id_t mesh_id, chip_id_t chip_id, routing_plane_id_t routing_plane_id) const {
    std::vector<chan_id_t> valid_eth_chans;
    for (const auto& [direction, eth_chans] :
         this->router_port_directions_to_physical_eth_chan_map_[mesh_id][chip_id]) {
        for (const auto& eth_chan : eth_chans) {
            if (this->get_routing_plane_id(eth_chan) == routing_plane_id) {
                valid_eth_chans.push_back(eth_chan);
            }
        }
    }
    return valid_eth_chans;
}

std::vector<std::pair<chip_id_t, chan_id_t>> ControlPlane::get_fabric_route(
    mesh_id_t src_mesh_id,
    chip_id_t src_chip_id,
    mesh_id_t dst_mesh_id,
    chip_id_t dst_chip_id,
    chan_id_t src_chan_id) const {
    std::vector<std::pair<chip_id_t, chan_id_t>> route;
    int i = 0;
    // Find any eth chan on the plane id
    while (src_mesh_id != dst_mesh_id or src_chip_id != dst_chip_id) {
        i++;
        if (i >= tt::tt_fabric::MAX_MESH_SIZE * tt::tt_fabric::MAX_NUM_MESHES) {
            TT_THROW(
                "Control Plane could not find route from M{}D{} to M{}D{}",
                src_mesh_id,
                src_chip_id,
                dst_mesh_id,
                dst_chip_id);
        }
        auto physical_chip_id = logical_mesh_chip_id_to_physical_chip_id_mapping_[src_mesh_id][src_chip_id];
        if (src_mesh_id != dst_mesh_id) {
            // Inter-mesh routing
            chan_id_t next_chan_id =
                this->inter_mesh_routing_tables_[src_mesh_id][src_chip_id][src_chan_id][dst_mesh_id];
            if (src_chan_id != next_chan_id) {
                // Chan to chan within chip
                route.push_back({physical_chip_id, next_chan_id});
            }
            std::tie(src_mesh_id, src_chip_id, src_chan_id) =
                this->get_connected_mesh_chip_chan_ids(src_mesh_id, src_chip_id, next_chan_id);
            auto connected_physical_chip_id =
                logical_mesh_chip_id_to_physical_chip_id_mapping_[src_mesh_id][src_chip_id];
            route.push_back({connected_physical_chip_id, src_chan_id});
        } else if (src_chip_id != dst_chip_id) {
            // Intra-mesh routing
            chan_id_t next_chan_id =
                this->intra_mesh_routing_tables_[src_mesh_id][src_chip_id][src_chan_id][dst_chip_id];
            if (src_chan_id != next_chan_id) {
                // Chan to chan within chip
                route.push_back({physical_chip_id, next_chan_id});
            }
            std::tie(src_mesh_id, src_chip_id, src_chan_id) =
                this->get_connected_mesh_chip_chan_ids(src_mesh_id, src_chip_id, next_chan_id);
            auto connected_physical_chip_id =
                logical_mesh_chip_id_to_physical_chip_id_mapping_[src_mesh_id][src_chip_id];
            route.push_back({connected_physical_chip_id, src_chan_id});
        }
    }

    return route;
}

void ControlPlane::configure_routing_tables() const {
    // Configure the routing tables on the chips
    for (mesh_id_t mesh_id = 0; mesh_id < this->intra_mesh_routing_tables_.size(); mesh_id++) {
        for (chip_id_t chip_id = 0; chip_id < this->intra_mesh_routing_tables_[mesh_id].size(); chip_id++) {
            this->write_routing_tables_to_chip(mesh_id, chip_id);
        }
    }
    for (mesh_id_t mesh_id = 0; mesh_id < this->inter_mesh_routing_tables_.size(); mesh_id++) {
        for (chip_id_t chip_id = 0; chip_id < this->inter_mesh_routing_tables_[mesh_id].size(); chip_id++) {
            this->write_routing_tables_to_chip(mesh_id, chip_id);
        }
    }
}

void ControlPlane::print_routing_tables() const {
    std::stringstream ss;
    ss << "Control Plane: IntraMesh Routing Tables" << std::endl;
    for (int mesh_id = 0; mesh_id < this->intra_mesh_routing_tables_.size(); mesh_id++) {
        ss << "M" << mesh_id << ":" << std::endl;
        for (int chip_id = 0; chip_id < this->intra_mesh_routing_tables_[mesh_id].size(); chip_id++) {
            const auto& chip_routing_table = this->intra_mesh_routing_tables_[mesh_id][chip_id];
            for (int eth_chan = 0; eth_chan < chip_routing_table.size(); eth_chan++) {
                ss << "   D" << chip_id << " Eth Chan " << eth_chan << ": ";
                for (const auto& dst_chan_id : chip_routing_table[eth_chan]) {
                    ss << (std::uint16_t)dst_chan_id << " ";
                }
                ss << std::endl;
            }
        }
    }

    log_debug(tt::LogFabric, "{}", ss.str());
    ss.str(std::string());
    ss << "Control Plane: InterMesh Routing Tables" << std::endl;

    for (int mesh_id = 0; mesh_id < this->inter_mesh_routing_tables_.size(); mesh_id++) {
        ss << "M" << mesh_id << ":" << std::endl;
        for (int chip_id = 0; chip_id < this->inter_mesh_routing_tables_[mesh_id].size(); chip_id++) {
            const auto& chip_routing_table = this->inter_mesh_routing_tables_[mesh_id][chip_id];
            for (int eth_chan = 0; eth_chan < chip_routing_table.size(); eth_chan++) {
                ss << "   D" << chip_id << " Eth Chan " << eth_chan << ": ";
                for (const auto& dst_chan_id : chip_routing_table[eth_chan]) {
                    ss << (std::uint16_t)dst_chan_id << " ";
                }
                ss << std::endl;
            }
        }
    }
    log_debug(tt::LogFabric, "{}", ss.str());
}

void ControlPlane::print_ethernet_channels() const {
    std::stringstream ss;
    ss << "Control Plane: Physical eth channels in each direction" << std::endl;
    for (uint32_t mesh_id = 0; mesh_id < this->router_port_directions_to_physical_eth_chan_map_.size(); mesh_id++) {
        for (uint32_t chip_id = 0; chip_id < this->router_port_directions_to_physical_eth_chan_map_[mesh_id].size();
             chip_id++) {
            ss << "M" << mesh_id << "D" << chip_id << ": " << std::endl;
            for (const auto& [direction, eth_chans] :
                 this->router_port_directions_to_physical_eth_chan_map_[mesh_id][chip_id]) {
                ss << "   " << magic_enum::enum_name(direction) << ":";
                for (const auto& eth_chan : eth_chans) {
                    ss << " " << (std::uint16_t)eth_chan;
                }
                ss << std::endl;
            }
        }
    }
    log_debug(tt::LogFabric, "{}", ss.str());
}

}  // namespace tt::tt_fabric
