// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_fabric/control_plane.hpp"

namespace tt::tt_fabric {

// TODO: store these functions somewhere, likely in the Mesh Descriptor
std::unordered_map<chip_id_t, std::vector<CoreCoord>> get_ethernet_cores_grouped_by_connected_chips(chip_id_t chip_id) {
    return tt::Cluster::instance().get_ethernet_cores_grouped_by_connected_chips(chip_id);
}

bool is_chip_on_edge_of_mesh(
    chip_id_t physical_chip_id,
    int chips_per_side,
    const std::unordered_map<chip_id_t, std::vector<CoreCoord>>& ethernet_cores_grouped_by_connected_chips) {
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

ControlPlane::ControlPlane(const std::string& mesh_graph_desc_yaml_file) {
    this->routing_table_generator_ = std::make_unique<RoutingTableGenerator>(mesh_graph_desc_yaml_file);
    if (mesh_graph_desc_yaml_file.find("tg_mesh_graph_descriptor.yaml") != std::string::npos) {
        const std::filesystem::path tg_cluster_desc_path = std::filesystem::path(tt::llrt::OptionsG.get_root_dir()) /
                                                           "tests/tt_metal/tt_fabric/common/tg_cluster_desc.yaml";
        this->initialize_from_cluster_desc_yaml_file(tg_cluster_desc_path.string());
        this->configure_routing_tables();
    } else {
        TT_FATAL(false, "Unsupported mesh graph descriptor file {}", mesh_graph_desc_yaml_file);
    }
    // Only enabled with log_debug
    this->routing_table_generator_->print_connectivity();
    this->routing_table_generator_->print_routing_tables();
    this->print_ethernet_channels();
    this->print_routing_tables();
}

// Get the physical chip ids for a mesh
std::vector<chip_id_t> ControlPlane::get_mesh_physical_chip_ids(
    uint32_t mesh_ns_size, uint32_t mesh_ew_size, uint32_t num_ports_per_side, uint32_t nw_chip_physical_chip_id) {
    std::vector<std::vector<chip_id_t>> mesh_edges{{nw_chip_physical_chip_id}, {nw_chip_physical_chip_id}};
    std::unordered_set<chip_id_t> visited_physical_chips{nw_chip_physical_chip_id};

    // Fill in the mesh_medges with the full edge
    for (int i = 0; i < mesh_edges.size(); i++) {
        chip_id_t current_physical_chip_id = mesh_edges[i][0];
        while (true) {
            // TODO: change to a for loop
            auto eth_links = get_ethernet_cores_grouped_by_connected_chips(current_physical_chip_id);
            for (const auto& [connected_chip_id, eth_ports] : eth_links) {
                if (eth_ports.size() == num_ports_per_side and
                    (is_chip_on_edge_of_mesh(
                         connected_chip_id,
                         num_ports_per_side,
                         get_ethernet_cores_grouped_by_connected_chips(connected_chip_id)) or
                     is_chip_on_corner_of_mesh(
                         connected_chip_id,
                         num_ports_per_side,
                         get_ethernet_cores_grouped_by_connected_chips(connected_chip_id)))) {
                    if (visited_physical_chips.find(connected_chip_id) == visited_physical_chips.end()) {
                        mesh_edges[i].push_back(connected_chip_id);
                        current_physical_chip_id = connected_chip_id;
                        visited_physical_chips.insert(connected_chip_id);
                        break;
                    }
                }
            }
            if (is_chip_on_corner_of_mesh(
                    current_physical_chip_id,
                    num_ports_per_side,
                    get_ethernet_cores_grouped_by_connected_chips(current_physical_chip_id))) {
                break;
            }
        }
    }

    // Now we have two full edges from the NW chip, loop through the EW edge and
    // fill in the rest of the mesh
    std::vector<chip_id_t> physical_chip_ids;
    visited_physical_chips = {};

    if (mesh_edges[0].size() == mesh_ew_size) {
        physical_chip_ids.insert(physical_chip_ids.end(), mesh_edges[0].begin(), mesh_edges[0].end());
        visited_physical_chips.insert(mesh_edges[0].begin(), mesh_edges[0].end());
    } else if (mesh_edges[1].size() == mesh_ew_size) {
        physical_chip_ids.insert(physical_chip_ids.end(), mesh_edges[1].begin(), mesh_edges[1].end());
        visited_physical_chips.insert(mesh_edges[1].begin(), mesh_edges[1].end());
    } else {
        TT_ASSERT(false, "Did not find edge with expected number of East-West chips");
    }
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

void ControlPlane::initialize_from_cluster_desc_yaml_file(const std::string& cluster_desc_yaml_file) {
    // TODO: temp testing code, probably will remove
    // Initialize the control plane from the cluster descriptor
    std::unique_ptr<tt_ClusterDescriptor> tg_cluster_desc =
        tt_ClusterDescriptor::create_from_yaml(cluster_desc_yaml_file);
    for (const auto& [physical_chip_id, connections] : tg_cluster_desc->get_ethernet_connections()) {
        for (const auto& [port_id, connection] : connections) {
        }
    }
    // TODO: get all of this from mesh descriptor
    const int galaxy_mesh_ns_size = 4;
    const int galaxy_mesh_ew_size = 8;
    const int num_ports_per_side = 4;
    const int mesh_sides = 4;
    // Must pin a NW chip
    // TODO: need to rework how this is determined
    eth_coord_t nw_chip_eth_coord = {0, 3, 7, 0, 1};
    chip_id_t nw_chip_physical_chip_id;
    for (const auto& [physical_chip_id, eth_coord] : tg_cluster_desc->get_chip_locations()) {
        if (eth_coord == nw_chip_eth_coord) {
            nw_chip_physical_chip_id = physical_chip_id;
            break;
        }
    }
    this->logical_mesh_chip_id_to_physical_chip_id_mapping_.push_back(this->get_mesh_physical_chip_ids(1, 1, 4, 0));
    this->logical_mesh_chip_id_to_physical_chip_id_mapping_.push_back(this->get_mesh_physical_chip_ids(1, 1, 4, 1));
    this->logical_mesh_chip_id_to_physical_chip_id_mapping_.push_back(this->get_mesh_physical_chip_ids(1, 1, 4, 2));
    this->logical_mesh_chip_id_to_physical_chip_id_mapping_.push_back(this->get_mesh_physical_chip_ids(1, 1, 4, 3));
    this->logical_mesh_chip_id_to_physical_chip_id_mapping_.push_back(this->get_mesh_physical_chip_ids(
        galaxy_mesh_ns_size, galaxy_mesh_ew_size, num_ports_per_side, nw_chip_physical_chip_id));

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
                    tg_cluster_desc->get_directly_connected_ethernet_channels_between_chips(
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
                        tg_cluster_desc->get_directly_connected_ethernet_channels_between_chips(
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

    this->convert_fabric_routing_table_to_chip_routing_table(4 * num_ports_per_side);
}

void ControlPlane::convert_fabric_routing_table_to_chip_routing_table(std::uint32_t num_ports_per_chip) {
    // Routing tables contain direction from chip to chip
    // Convert it to be unique per ethernet channel
    const auto& router_intra_mesh_routing_table = this->routing_table_generator_->get_intra_mesh_table();
    this->intra_mesh_routing_tables_.resize(router_intra_mesh_routing_table.size());
    for (mesh_id_t mesh_id = 0; mesh_id < router_intra_mesh_routing_table.size(); mesh_id++) {
        this->intra_mesh_routing_tables_[mesh_id].resize(router_intra_mesh_routing_table[mesh_id].size());
        for (chip_id_t src_chip_id = 0; src_chip_id < router_intra_mesh_routing_table[mesh_id].size(); src_chip_id++) {
            this->intra_mesh_routing_tables_[mesh_id][src_chip_id].resize(
                num_ports_per_chip);  // contains more entries than needed
            for (int i = 0; i < num_ports_per_chip; i++) {
                // Size the routing table to the number of chips in the mesh
                this->intra_mesh_routing_tables_[mesh_id][src_chip_id][i].resize(
                    router_intra_mesh_routing_table[mesh_id][src_chip_id].size());
            }
            for (chip_id_t dst_chip_id = 0; dst_chip_id < router_intra_mesh_routing_table[mesh_id][src_chip_id].size();
                 dst_chip_id++) {
                // Target direction is the direction to the destination chip for all ethernet channesl
                const auto& target_direction = router_intra_mesh_routing_table[mesh_id][src_chip_id][dst_chip_id];
                const auto& eth_chans_in_target_direction =
                    this->router_port_directions_to_physical_eth_chan_map_[mesh_id][src_chip_id][target_direction];
                // We view ethernet channels on one side of the chip as parallel planes. So N[0] talks to S[0], E[0],
                // W[0] and so on For all live ethernet channels on this chip, set the routing table entry to the
                // destination chip as the ethernet channel on the same plane
                for (const auto& [direction, eth_chans_on_side] :
                     this->router_port_directions_to_physical_eth_chan_map_[mesh_id][src_chip_id]) {
                    for (const auto& src_chan_id : eth_chans_on_side) {
                        if (src_chip_id == dst_chip_id) {
                            // This entry represents chip to itself, should not be used by FW
                            this->intra_mesh_routing_tables_[mesh_id][src_chip_id][src_chan_id][dst_chip_id] = eth_chan_magic_values::ROUTE_TO_LOCAL_CHIP;
                        } else if (target_direction == direction) {
                            // This entry represents an outgoing eth channel
                            this->intra_mesh_routing_tables_[mesh_id][src_chip_id][src_chan_id][dst_chip_id] = eth_chan_magic_values::OUTGOING_ETH_LINK;
                        } else {
                            this->intra_mesh_routing_tables_[mesh_id][src_chip_id][src_chan_id][dst_chip_id] =
                                this->get_eth_chan_id(src_chan_id, eth_chans_in_target_direction);
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
                const auto& eth_chans_in_target_direction =
                    this->router_port_directions_to_physical_eth_chan_map_[src_mesh_id][src_chip_id][target_direction];

                // We view ethernet channels on one side of the chip as parallel planes. So N[0] talks to S[0], E[0],
                // W[0] and so on For all live ethernet channels on this chip, set the routing table entry to the
                // destination mesh as the ethernet channel on the same plane
                for (const auto& [direction, eth_chans_on_side] :
                     this->router_port_directions_to_physical_eth_chan_map_[src_mesh_id][src_chip_id]) {
                    for (const auto& src_chan_id : eth_chans_on_side) {
                        if (src_mesh_id == dst_mesh_id) {
                            // This entry represents mesh to itself, should not be used by FW
                            this->inter_mesh_routing_tables_[src_mesh_id][src_chip_id][src_chan_id][dst_mesh_id] = eth_chan_magic_values::ROUTE_TO_LOCAL_CHIP;
                        } else if (target_direction == direction) {
                            // This entry represents an outgoing eth channel
                            this->inter_mesh_routing_tables_[src_mesh_id][src_chip_id][src_chan_id][dst_mesh_id] = eth_chan_magic_values::OUTGOING_ETH_LINK;
                        } else {
                            this->inter_mesh_routing_tables_[src_mesh_id][src_chip_id][src_chan_id][dst_mesh_id] =
                                this->get_eth_chan_id(src_chan_id, eth_chans_in_target_direction);
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
            /*std::vector<std::uint32_t> intra_mesh_data, inter_mesh_data;
            intra_mesh_data.reserve(eth_chan_intra_mesh_routing_table.size() / entries_per_uint32_t);
            inter_mesh_data.reserve(eth_chan_inter_mesh_routing_table.size() / entries_per_uint32_t);
            for (std::uint32_t i = 0; i < eth_chan_intra_mesh_routing_table.size(); i += entries_per_uint32_t) {
                std::uint32_t entry = 0;
                for (std::uint32_t j = 0; j < entries_per_uint32_t; j++) {
                    if ((i + j) < eth_chan_intra_mesh_routing_table.size()) {
                        entry |= (eth_chan_intra_mesh_routing_table[i + j] & 0xFF) << (32 / entries_per_uint32_t * j);
                    }
                }
                intra_mesh_data.push_back(entry);
            }
            for (std::uint32_t i = 0; i < eth_chan_inter_mesh_routing_table.size(); i += entries_per_uint32_t) {
                std::uint32_t entry = 0;
                for (std::uint32_t j = 0; j < entries_per_uint32_t; j++) {
                    if ((i + j) < eth_chan_inter_mesh_routing_table.size()) {
                        entry |= (eth_chan_inter_mesh_routing_table[i + j] & 0xFF) << (32 / entries_per_uint32_t * j);
                    }
                }
                inter_mesh_data.push_back(entry);
            }*/

            tt::tt_fabric::fabric_router_l1_config_t fabric_router_config;
            for (uint32_t i = 0; i < eth_chan_intra_mesh_routing_table.size(); i++) {
                fabric_router_config.intra_mesh_table.dest_entry[i] = eth_chan_intra_mesh_routing_table[i];
            }
            for (uint32_t i = 0; i < eth_chan_inter_mesh_routing_table.size(); i++) {
                fabric_router_config.inter_mesh_table.dest_entry[i] = eth_chan_inter_mesh_routing_table[i];
            }

            if (chip_eth_chans_map.find(RoutingDirection::N) != chip_eth_chans_map.end()) {
                fabric_router_config.port_direction.north = this->get_eth_chan_id(eth_chan, chip_eth_chans_map.at(RoutingDirection::N));
            } else {
                fabric_router_config.port_direction.north = eth_chan_magic_values::INVALID_DIRECTION;
            }
            if (chip_eth_chans_map.find(RoutingDirection::S) != chip_eth_chans_map.end()) {
                fabric_router_config.port_direction.south = this->get_eth_chan_id(eth_chan, chip_eth_chans_map.at(RoutingDirection::S));
            } else {
                fabric_router_config.port_direction.south = eth_chan_magic_values::INVALID_DIRECTION;
            }
            if (chip_eth_chans_map.find(RoutingDirection::E) != chip_eth_chans_map.end()) {
                fabric_router_config.port_direction.east = this->get_eth_chan_id(eth_chan, chip_eth_chans_map.at(RoutingDirection::E));
            } else {
                fabric_router_config.port_direction.east = eth_chan_magic_values::INVALID_DIRECTION;
            }
            if (chip_eth_chans_map.find(RoutingDirection::W) != chip_eth_chans_map.end()) {
                fabric_router_config.port_direction.west = this->get_eth_chan_id(eth_chan, chip_eth_chans_map.at(RoutingDirection::W));
            } else {
                fabric_router_config.port_direction.west = eth_chan_magic_values::INVALID_DIRECTION;
            }

            fabric_router_config.my_mesh_id = mesh_id;
            fabric_router_config.my_device_id = chip_id;

            // Write data to physical eth core
            tt_cxy_pair physical_eth_core(
                physical_chip_id, tt::Cluster::instance().get_soc_desc(physical_chip_id).ethernet_cores[eth_chan]);

            // TODO: remove this when merging to main
            /*
            std::cout << " writing routing table entry data for eth core " << eth_chan << " " <<
            (std::uint16_t)fabric_router_config.port_direction.north << " " <<
            (std::uint16_t)fabric_router_config.port_direction.east << " " <<
            (std::uint16_t)fabric_router_config.port_direction.south << " " <<
            (std::uint16_t)fabric_router_config.port_direction.west << std::endl; std::cout << " dumping routing table
            entry data for eth core " << physical_eth_core.str() << std::endl; std::cout << "  intra mesh table: "; for
            (int i = 0; i < eth_chan_intra_mesh_routing_table.size(); i++) { std::cout << std::hex << "0x" <<
            (std::uint16_t)fabric_router_config.intra_mesh_table.dest_entry[i]
                          << " " << std::dec;
            }
            std::cout << std::endl << "  inter mesh table: ";
            for (int i = 0; i < eth_chan_inter_mesh_routing_table.size(); i++) {
                std::cout << std::hex << "0x" << (std::uint16_t)fabric_router_config.inter_mesh_table.dest_entry[i]
                          << " " << std::dec;
            }
            std::cout << std::endl;
*/
            // TODO: use eth_l1_mem::address_map::FABRIC_ROUTER_CONFIG_BASE when bug with queues is fixed, hardcoded to
            // enable sanity testing
            tt::Cluster::instance().write_core(
                (void*)&fabric_router_config,
                sizeof(tt::tt_fabric::fabric_router_l1_config_t),
                physical_eth_core,
                eth_l1_mem::address_map::FABRIC_ROUTER_CONFIG_HARDCODED_TESTING_ADDR,
                false);
        }
    }
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
                    ss << (std::uint16_t)eth_chan << " ";
                }
                ss << std::endl;
            }
        }
    }
    log_debug(tt::LogFabric, "{}", ss.str());
}

}  // namespace tt::tt_fabric
