// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <magic_enum/magic_enum.hpp>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <initializer_list>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <ostream>
#include <queue>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "assert.hpp"
#include "control_plane.hpp"
#include "core_coord.hpp"
#include "fabric_host_interface.h"
#include "hal_types.hpp"
#include "logger.hpp"
#include "mesh_coord.hpp"
#include "mesh_graph.hpp"
#include "metal_soc_descriptor.h"
#include "routing_table_generator.hpp"
#include "impl/context/metal_context.hpp"
#include <umd/device/tt_core_coordinates.h>
#include <umd/device/tt_xy_pair.h>
#include <umd/device/types/cluster_descriptor_types.h>
#include <umd/device/types/xy_pair.h>

namespace tt::tt_fabric {

// Get the physical chip ids for a mesh
std::unordered_map<chip_id_t, std::vector<CoreCoord>> get_ethernet_cores_grouped_by_connected_chips(chip_id_t chip_id) {
    return tt::tt_metal::MetalContext::instance().get_cluster().get_ethernet_cores_grouped_by_connected_chips(chip_id);
}

// Get the physical chip ids for a mesh
// TODO: get this from Cluster, once UMD unique id changes are merged
std::uint32_t get_ubb_asic_id(chip_id_t physical_chip_id) {
    std::vector<uint32_t> ubb_asic_loc_vec;
    const auto& eth_cores = tt::tt_metal::MetalContext::instance().get_cluster().get_active_ethernet_cores(physical_chip_id, false);
    auto virtual_eth_core = tt::tt_metal::MetalContext::instance().get_cluster().get_virtual_coordinate_from_logical_coordinates(
        physical_chip_id, *eth_cores.begin(), CoreType::ETH);

    std::uint32_t addr = 0x1ec0 + 65 * sizeof(uint32_t);
    tt::tt_metal::MetalContext::instance().get_cluster().read_core(
        ubb_asic_loc_vec, sizeof(uint32_t), tt_cxy_pair(physical_chip_id, virtual_eth_core), addr);
    return ((ubb_asic_loc_vec[0] >> 24) & 0xFF);
}

bool is_external_ubb_cable(chip_id_t physical_chip_id, CoreCoord eth_core) {
    auto chan_id = tt::tt_metal::MetalContext::instance()
                       .get_cluster()
                       .get_soc_desc(physical_chip_id)
                       .logical_eth_core_to_chan_map.at(eth_core);
    auto ubb_asic_id = get_ubb_asic_id(physical_chip_id);
    bool is_external_cable = false;
    if (ubb_asic_id == 1) {
        // UBB 1 has external cables on channesl 0-7
        is_external_cable = (chan_id >= 0 and chan_id <= 7);
    } else if (ubb_asic_id >= 2 and ubb_asic_id <= 4) {
        // UBB 2 to 4 has external cables on channesl 0-3
        is_external_cable = (chan_id >= 0 and chan_id <= 3);
    } else if (ubb_asic_id == 5) {
        // UBB 5 has external cables on channesl 4-7
        is_external_cable = (chan_id >= 4 and chan_id <= 7);
    }
    return is_external_cable;
}

bool is_chip_on_edge_of_mesh(
    chip_id_t physical_chip_id,
    int num_ports_per_side,
    const std::unordered_map<chip_id_t, std::vector<CoreCoord>>& ethernet_cores_grouped_by_connected_chips) {
    // Chip is on edge if it does not have full connections to four sides
    if (tt::tt_metal::MetalContext::instance().get_cluster().get_board_type(physical_chip_id) == BoardType::UBB) {
        auto ubb_asic_id = get_ubb_asic_id(physical_chip_id);
        return (ubb_asic_id >= 2) and (ubb_asic_id <= 5);
    } else {
        int i = 0;
        for (const auto& [connected_chip_id, eth_ports] : ethernet_cores_grouped_by_connected_chips) {
            if (eth_ports.size() == num_ports_per_side) {
                i++;
            }
        }
        return (i == 3);
    }
}

bool is_chip_on_corner_of_mesh(
    chip_id_t physical_chip_id,
    int num_ports_per_side,
    const std::unordered_map<chip_id_t, std::vector<CoreCoord>>& ethernet_cores_grouped_by_connected_chips) {
    if (tt::tt_metal::MetalContext::instance().get_cluster().get_board_type(physical_chip_id) == BoardType::UBB) {
        auto ubb_asic_id = get_ubb_asic_id(physical_chip_id);
        return (ubb_asic_id == 1);
    } else {
        // Chip is a corner if it has exactly 2 fully connected sides
        int i = 0;
        for (const auto& [connected_chip_id, eth_ports] : ethernet_cores_grouped_by_connected_chips) {
            if (eth_ports.size() == num_ports_per_side) {
                i++;
            }
        }
        return (i < 3);
    }
}

ControlPlane::ControlPlane(const std::string& mesh_graph_desc_file) {
    this->routing_table_generator_ = std::make_unique<RoutingTableGenerator>(mesh_graph_desc_file);
    // Printing, only enabled with log_debug
    this->routing_table_generator_->print_connectivity();
    // Printing, only enabled with log_debug
    this->routing_table_generator_->print_routing_tables();

    // Initialize the control plane routers based on mesh graph
    this->initialize_from_mesh_graph_desc_file(mesh_graph_desc_file);

    // Printing, only enabled with log_debug
    this->print_ethernet_channels();
}

chip_id_t ControlPlane::get_physical_chip_id_from_eth_coord(const eth_coord_t& eth_coord) const {
    chip_id_t nw_chip_physical_chip_id = 0;
    for (const auto& [physical_chip_id, coord] :
         tt::tt_metal::MetalContext::instance().get_cluster().get_user_chip_ethernet_coordinates()) {
        if (coord == eth_coord) {
            return physical_chip_id;
        }
    }
    TT_FATAL(false, "Physical chip id not found for eth coord");
    return 0;
}

void ControlPlane::validate_mesh_connections(mesh_id_t mesh_id) const {
    std::uint32_t mesh_ns_size = routing_table_generator_->get_mesh_ns_size(mesh_id);
    std::uint32_t mesh_ew_size = routing_table_generator_->get_mesh_ew_size(mesh_id);
    std::uint32_t num_ports_per_side = routing_table_generator_->get_chip_spec().num_eth_ports_per_direction;
    for (int i = 0; i < mesh_ns_size; i++) {
        for (int j = 0; j < mesh_ew_size - 1; j++) {
            chip_id_t logical_chip_id = i * mesh_ew_size + j;
            chip_id_t physical_chip_id = logical_mesh_chip_id_to_physical_chip_id_mapping_[mesh_id][logical_chip_id];
            chip_id_t physical_chip_id_next =
                logical_mesh_chip_id_to_physical_chip_id_mapping_[mesh_id][logical_chip_id + 1];
            chip_id_t physical_chip_id_next_row =
                logical_mesh_chip_id_to_physical_chip_id_mapping_[mesh_id][logical_chip_id + mesh_ew_size];

            const auto& eth_links = get_ethernet_cores_grouped_by_connected_chips(physical_chip_id);
            auto eth_links_to_next = eth_links.find(physical_chip_id_next);
            TT_FATAL(
                eth_links_to_next != eth_links.end(),
                "Chip {} not connected to chip {}",
                physical_chip_id,
                physical_chip_id_next);
            TT_FATAL(
                eth_links_to_next->second.size() == num_ports_per_side,
                "Chip {} to chip {} has {} links but expecting {}",
                physical_chip_id,
                physical_chip_id_next,
                eth_links.at(physical_chip_id_next).size(),
                num_ports_per_side);
            if (i != mesh_ns_size - 1) {
                auto eth_links_to_next_row = eth_links.find(physical_chip_id_next_row);
                TT_FATAL(
                    eth_links_to_next_row != eth_links.end(),
                    "Chip {} not connected to chip {}",
                    physical_chip_id,
                    physical_chip_id_next_row);
                TT_FATAL(
                    eth_links_to_next_row->second.size() == num_ports_per_side,
                    "Chip {} to chip {} has {} links but expecting {}",
                    physical_chip_id,
                    physical_chip_id_next_row,
                    eth_links.at(physical_chip_id_next_row).size(),
                    num_ports_per_side);
            }
        }
    }
}

std::vector<chip_id_t> ControlPlane::get_mesh_physical_chip_ids(
    std::uint32_t mesh_ns_size,
    std::uint32_t mesh_ew_size,
    chip_id_t nw_chip_physical_chip_id) const {
    std::uint32_t num_ports_per_side = routing_table_generator_->get_chip_spec().num_eth_ports_per_direction;

    const auto user_chips = tt::tt_metal::MetalContext::instance().get_cluster().user_exposed_chip_ids();
    std::set<chip_id_t> corner_chips;
    std::set<chip_id_t> edge_chips;
    // Check if user provided chip is on corner or edge of mesh
    // Ideally we want to always enable an auto detected chip, but to preserve current pinning
    // in MeshDevice, we pin a specific chip on the corner for TGs/T3ks
    for (const auto& physical_chip_id : user_chips) {
        const auto& connected_ethernet_cores = get_ethernet_cores_grouped_by_connected_chips(physical_chip_id);
        if (is_chip_on_corner_of_mesh(physical_chip_id, num_ports_per_side, connected_ethernet_cores)) {
            corner_chips.insert(physical_chip_id);
        } else if (is_chip_on_edge_of_mesh(nw_chip_physical_chip_id, num_ports_per_side, connected_ethernet_cores)) {
            edge_chips.insert(physical_chip_id);
        }
    }
    if (corner_chips.find(nw_chip_physical_chip_id) == corner_chips.end()) {
        log_warning(
            tt::LogFabric,
            "NW chip {} is not on corner of mesh, using detected chip {}",
            nw_chip_physical_chip_id,
            *corner_chips.begin());
        nw_chip_physical_chip_id = *corner_chips.begin();
    }

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
            // Do not include any corner to corner links on UBB
            if (tt::tt_metal::MetalContext::instance().get_cluster().get_board_type(connected_chip_id) ==
                BoardType::UBB) {
                if (is_external_ubb_cable(current_chip_id, eth_ports[0])) {
                    continue;
                }
            }
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
            } else {
                log_debug(
                    tt::LogFabric,
                    "Number of eth ports {} does not match num ports specified in Mesh graph descriptor {}",
                    eth_ports.size(),
                    num_ports_per_side);
            }
        }
    }

    std::vector<chip_id_t> physical_chip_ids;
    // TODO: if square mesh, we might need to pin another corner chip, or potentially have multiple possible orientations
    for (const auto& [dest_id, equal_dist_paths] : paths) {
        // TODO: can change this to not check for corner?
        // Look for size of equal dist paths == mesh_ew_size and num paths == 1
        if (equal_dist_paths.size() == 1) {
            auto dest_chip_id = equal_dist_paths[0].back();
            bool is_corner = is_chip_on_corner_of_mesh(
                dest_chip_id, num_ports_per_side, get_ethernet_cores_grouped_by_connected_chips(dest_chip_id));
            if (is_corner and equal_dist_paths[0].size() == mesh_ew_size) {
                physical_chip_ids = equal_dist_paths[0];
                break;
            }
        }
    }

    TT_FATAL(
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
            bool found_chip = false;
            for (const auto& [connected_chip_id, eth_ports] : eth_links_grouped_by_connected_chips) {
                if (is_external_ubb_cable(physical_chip_id_from_north, eth_ports[0])) {
                    continue;
                }
                if (visited_physical_chips.find(connected_chip_id) == visited_physical_chips.end() and
                    eth_ports.size() == num_ports_per_side) {
                    physical_chip_ids[i * mesh_ew_size + j] = connected_chip_id;
                    visited_physical_chips.insert(connected_chip_id);
                    found_chip = true;
                    break;
                }
            }
            TT_FATAL(found_chip, "Did not find chip for mesh row {} and column {}", i, j);
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
    chip_id_t nw_chip_physical_id;
    std::uint32_t mesh_ns_size, mesh_ew_size;
    std::string mesh_graph_desc_filename = std::filesystem::path(mesh_graph_desc_file).filename().string();
    if (mesh_graph_desc_filename == "tg_mesh_graph_descriptor.yaml") {
        // Add the N150 MMIO devices
        auto eth_coords_per_chip =
            tt::tt_metal::MetalContext::instance().get_cluster().get_all_chip_ethernet_coordinates();
        std::unordered_map<int, chip_id_t> eth_coord_y_for_gateway_chips = {};
        for (const auto [chip_id, eth_coord] : eth_coords_per_chip) {
            if (tt::tt_metal::MetalContext::instance().get_cluster().get_board_type(chip_id) == BoardType::N150) {
                eth_coord_y_for_gateway_chips[eth_coord.y] = chip_id;
            }
        }
        this->logical_mesh_chip_id_to_physical_chip_id_mapping_.push_back({eth_coord_y_for_gateway_chips[3]});
        this->logical_mesh_chip_id_to_physical_chip_id_mapping_.push_back({eth_coord_y_for_gateway_chips[2]});
        this->logical_mesh_chip_id_to_physical_chip_id_mapping_.push_back({eth_coord_y_for_gateway_chips[1]});
        this->logical_mesh_chip_id_to_physical_chip_id_mapping_.push_back({eth_coord_y_for_gateway_chips[0]});

        nw_chip_physical_id = this->get_physical_chip_id_from_eth_coord({0, 3, 7, 0, 1});
        mesh_ns_size = routing_table_generator_->get_mesh_ns_size(/*mesh_id=*/4);
        mesh_ew_size = routing_table_generator_->get_mesh_ew_size(/*mesh_id=*/4);
        // Main board
        this->logical_mesh_chip_id_to_physical_chip_id_mapping_.push_back(
            this->get_mesh_physical_chip_ids(mesh_ns_size, mesh_ew_size, nw_chip_physical_id));
    } else if (
        mesh_graph_desc_filename == "quanta_galaxy_mesh_graph_descriptor.yaml" ||
        mesh_graph_desc_filename == "quanta_galaxy_torus_2d_graph_descriptor.yaml" ||
        mesh_graph_desc_filename == "p100_mesh_graph_descriptor.yaml" ||
        mesh_graph_desc_filename == "p150_mesh_graph_descriptor.yaml" ||
        mesh_graph_desc_filename == "p150_x2_mesh_graph_descriptor.yaml" ||
        mesh_graph_desc_filename == "p150_x4_mesh_graph_descriptor.yaml") {
        // TODO: update to pick out chip automatically
        nw_chip_physical_id = 0;
        mesh_ns_size = routing_table_generator_->get_mesh_ns_size(/*mesh_id=*/0);
        mesh_ew_size = routing_table_generator_->get_mesh_ew_size(/*mesh_id=*/0);
        // Main board
        this->logical_mesh_chip_id_to_physical_chip_id_mapping_.push_back(
            this->get_mesh_physical_chip_ids(mesh_ns_size, mesh_ew_size, nw_chip_physical_id));
        this->validate_mesh_connections(0);
    } else if (
        mesh_graph_desc_filename == "t3k_mesh_graph_descriptor.yaml" ||
        mesh_graph_desc_filename == "n150_mesh_graph_descriptor.yaml" ||
        mesh_graph_desc_filename == "n300_mesh_graph_descriptor.yaml") {
        nw_chip_physical_id = this->get_physical_chip_id_from_eth_coord({0, 0, 0, 0, 0});
        mesh_ns_size = routing_table_generator_->get_mesh_ns_size(/*mesh_id=*/0);
        mesh_ew_size = routing_table_generator_->get_mesh_ew_size(/*mesh_id=*/0);
        // Main board
        this->logical_mesh_chip_id_to_physical_chip_id_mapping_.push_back(
            this->get_mesh_physical_chip_ids(mesh_ns_size, mesh_ew_size, nw_chip_physical_id));
    } else {
        TT_THROW("Unsupported mesh graph descriptor file {}", mesh_graph_desc_file);
    }
}

routing_plane_id_t ControlPlane::get_routing_plane_id(chan_id_t eth_chan_id) const {
    // Assumes that ethernet channels are incrementing by one in the same direction
    // Same mapping for all variants of active eth cores
    std::uint32_t num_eth_ports_per_direction = routing_table_generator_->get_chip_spec().num_eth_ports_per_direction;
    return eth_chan_id % num_eth_ports_per_direction;
}

chan_id_t ControlPlane::get_downstream_eth_chan_id(
    chan_id_t src_chan_id, const std::vector<chan_id_t>& candidate_target_chans) const {
    if (candidate_target_chans.empty()) {
        return eth_chan_magic_values::INVALID_ROUTING_TABLE_ENTRY;
    }
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
            std::uint32_t num_ports_per_chip = tt::tt_metal::MetalContext::instance()
                                                   .get_cluster()
                                                   .get_soc_desc(physical_chip_id)
                                                   .get_cores(CoreType::ETH)
                                                   .size();
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
                                this->router_port_directions_to_physical_eth_chan_map_[mesh_id][src_chip_id]
                                                                                      [target_direction];
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
            std::uint32_t num_ports_per_chip = tt::tt_metal::MetalContext::instance()
                                                   .get_cluster()
                                                   .get_soc_desc(physical_chip_id)
                                                   .get_cores(CoreType::ETH)
                                                   .size();
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
                                this->router_port_directions_to_physical_eth_chan_map_[src_mesh_id][src_chip_id]
                                                                                      [target_direction];
                            this->inter_mesh_routing_tables_[src_mesh_id][src_chip_id][src_chan_id][dst_mesh_id] =
                                this->get_downstream_eth_chan_id(src_chan_id, eth_chans_in_target_direction);
                        }
                    }
                }
            }
        }
    }
    // Printing, only enabled with log_debug
    this->print_routing_tables();
}

void ControlPlane::configure_routing_tables_for_fabric_ethernet_channels() {
    this->intra_mesh_routing_tables_.clear();
    this->inter_mesh_routing_tables_.clear();
    this->router_port_directions_to_physical_eth_chan_map_.clear();

    const auto& intra_mesh_connectivity = this->routing_table_generator_->get_intra_mesh_connectivity();
    const auto& inter_mesh_connectivity = this->routing_table_generator_->get_inter_mesh_connectivity();

    auto add_ethernet_channel_to_router_mapping = [&](mesh_id_t mesh_id,
                                                      chip_id_t chip_id,
                                                      const CoreCoord& eth_core,
                                                      RoutingDirection direction) {
        auto physical_chip_id = this->logical_mesh_chip_id_to_physical_chip_id_mapping_[mesh_id][chip_id];
        auto fabric_router_channels_on_chip = tt::tt_metal::MetalContext::instance().get_cluster().get_fabric_ethernet_channels(physical_chip_id);
        // TODO: get_fabric_ethernet_channels accounts for down links, but we should manage down links in control plane
        auto chan_id = tt::tt_metal::MetalContext::instance()
                           .get_cluster()
                           .get_soc_desc(physical_chip_id)
                           .logical_eth_core_to_chan_map.at(eth_core);
        // TODO: add logic here to disable unsed routers, e.g. Mesh on Torus system
        if (fabric_router_channels_on_chip.contains(chan_id)) {
            this->router_port_directions_to_physical_eth_chan_map_[mesh_id][chip_id][direction].push_back(chan_id);
        } else {
            log_debug(
                tt::LogFabric, "Control Plane: Disabling router on M{}D{} eth channel {}", mesh_id, chip_id, chan_id);
        }
    };

    // Initialize the bookkeeping for mapping from mesh/chip/direction to physical ethernet channels
    this->router_port_directions_to_physical_eth_chan_map_.resize(intra_mesh_connectivity.size());
    for (mesh_id_t mesh_id = 0; mesh_id < intra_mesh_connectivity.size(); mesh_id++) {
        this->router_port_directions_to_physical_eth_chan_map_[mesh_id].resize(intra_mesh_connectivity[mesh_id].size());
        for (chip_id_t chip_id = 0; chip_id < intra_mesh_connectivity[mesh_id].size(); chip_id++) {
            auto physical_chip_id = this->logical_mesh_chip_id_to_physical_chip_id_mapping_[mesh_id][chip_id];
            const auto& connected_chips_and_eth_cores =
                tt::tt_metal::MetalContext::instance().get_cluster().get_ethernet_cores_grouped_by_connected_chips(
                    physical_chip_id);
            for (const auto& [logical_connected_chip_id, edge] : intra_mesh_connectivity[mesh_id][chip_id]) {
                const auto& physical_connected_chip_id =
                    this->logical_mesh_chip_id_to_physical_chip_id_mapping_[mesh_id][logical_connected_chip_id];
                const auto& connected_eth_cores = connected_chips_and_eth_cores.at(physical_connected_chip_id);
                TT_FATAL(
                    connected_eth_cores.size() == edge.connected_chip_ids.size(),
                    "Expected {} eth links from physical chip {} to physical chip {}",
                    edge.connected_chip_ids.size(),
                    physical_chip_id,
                    physical_connected_chip_id);

                for (const auto& eth_core : connected_eth_cores) {
                    // There could be an optimization here to create entry for both chips here, assuming links are
                    // bidirectional
                    add_ethernet_channel_to_router_mapping(mesh_id, chip_id, eth_core, edge.port_direction);
                }
            }
        }
    }
    for (mesh_id_t mesh_id = 0; mesh_id < inter_mesh_connectivity.size(); mesh_id++) {
        for (chip_id_t chip_id = 0; chip_id < inter_mesh_connectivity[mesh_id].size(); chip_id++) {
            auto physical_chip_id = this->logical_mesh_chip_id_to_physical_chip_id_mapping_[mesh_id][chip_id];
            const auto& connected_chips_and_eth_cores =
                tt::tt_metal::MetalContext::instance().get_cluster().get_ethernet_cores_grouped_by_connected_chips(
                    physical_chip_id);
            for (const auto& [connected_mesh_id, edge] : inter_mesh_connectivity[mesh_id][chip_id]) {
                // Loop over edges connected chip ids, they could connect to different chips for intermesh traffic
                for (const auto& logical_connected_chip_id : edge.connected_chip_ids) {
                    const auto& physical_connected_chip_id =
                        this->logical_mesh_chip_id_to_physical_chip_id_mapping_[connected_mesh_id]
                                                                               [logical_connected_chip_id];
                    const auto& connected_eth_cores = connected_chips_and_eth_cores.at(physical_connected_chip_id);
                    for (const auto& eth_core : connected_eth_cores) {
                        add_ethernet_channel_to_router_mapping(mesh_id, chip_id, eth_core, edge.port_direction);
                    }
                }
            }
        }
    }
    this->convert_fabric_routing_table_to_chip_routing_table();
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
                fabric_router_config.port_direction.directions[eth_chan_directions::NORTH] =
                    this->get_downstream_eth_chan_id(eth_chan, chip_eth_chans_map.at(RoutingDirection::N));
            } else {
                fabric_router_config.port_direction.directions[eth_chan_directions::NORTH] =
                    eth_chan_magic_values::INVALID_DIRECTION;
            }
            if (chip_eth_chans_map.find(RoutingDirection::S) != chip_eth_chans_map.end()) {
                fabric_router_config.port_direction.directions[eth_chan_directions::SOUTH] =
                    this->get_downstream_eth_chan_id(eth_chan, chip_eth_chans_map.at(RoutingDirection::S));
            } else {
                fabric_router_config.port_direction.directions[eth_chan_directions::SOUTH] =
                    eth_chan_magic_values::INVALID_DIRECTION;
            }
            if (chip_eth_chans_map.find(RoutingDirection::E) != chip_eth_chans_map.end()) {
                fabric_router_config.port_direction.directions[eth_chan_directions::EAST] =
                    this->get_downstream_eth_chan_id(eth_chan, chip_eth_chans_map.at(RoutingDirection::E));
            } else {
                fabric_router_config.port_direction.directions[eth_chan_directions::EAST] =
                    eth_chan_magic_values::INVALID_DIRECTION;
            }
            if (chip_eth_chans_map.find(RoutingDirection::W) != chip_eth_chans_map.end()) {
                fabric_router_config.port_direction.directions[eth_chan_directions::WEST] =
                    this->get_downstream_eth_chan_id(eth_chan, chip_eth_chans_map.at(RoutingDirection::W));
            } else {
                fabric_router_config.port_direction.directions[eth_chan_directions::WEST] =
                    eth_chan_magic_values::INVALID_DIRECTION;
            }

            fabric_router_config.my_mesh_id = mesh_id;
            fabric_router_config.my_device_id = chip_id;
            fabric_router_config.north_dim = this->routing_table_generator_->get_mesh_ns_size(mesh_id);
            fabric_router_config.east_dim = this->routing_table_generator_->get_mesh_ew_size(mesh_id);

            // Write data to physical eth core
            CoreCoord virtual_eth_core =
                tt::tt_metal::MetalContext::instance().get_cluster().get_virtual_eth_core_from_channel(
                    physical_chip_id, eth_chan);

            TT_ASSERT(
                tt_metal::MetalContext::instance().hal().get_dev_size(
                    tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt_metal::HalL1MemAddrType::FABRIC_ROUTER_CONFIG) ==
                    sizeof(tt::tt_fabric::fabric_router_l1_config_t),
                "ControlPlane: Fabric router config size mismatch");
            log_debug(
                tt::LogFabric,
                "ControlPlane: Writing routing table to on M{}D{} eth channel {}",
                mesh_id,
                chip_id,
                eth_chan);
            tt::tt_metal::MetalContext::instance().get_cluster().write_core(
                (void*)&fabric_router_config,
                sizeof(tt::tt_fabric::fabric_router_l1_config_t),
                tt_cxy_pair(physical_chip_id, virtual_eth_core),
                tt_metal::MetalContext::instance().hal().get_dev_addr(
                    tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt_metal::HalL1MemAddrType::FABRIC_ROUTER_CONFIG),
                false);
        }
    }
    tt::tt_metal::MetalContext::instance().get_cluster().l1_barrier(physical_chip_id);
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
    tt::umd::CoreCoord eth_core = tt::tt_metal::MetalContext::instance()
                                      .get_cluster()
                                      .get_soc_desc(physical_chip_id)
                                      .get_eth_core_for_channel(chan_id, CoordSystem::LOGICAL);
    auto [connected_physical_chip_id, connected_eth_core] =
        tt::tt_metal::MetalContext::instance().get_cluster().get_connected_ethernet_core(
            std::make_tuple(physical_chip_id, CoreCoord{eth_core.x, eth_core.y}));

    auto [connected_mesh_id, connected_chip_id] =
        this->get_mesh_chip_id_from_physical_chip_id(connected_physical_chip_id);
    auto connected_chan_id = tt::tt_metal::MetalContext::instance()
                                 .get_cluster()
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

eth_chan_directions ControlPlane::routing_direction_to_eth_direction(RoutingDirection direction) const {
    eth_chan_directions dir;
    switch (direction) {
        case RoutingDirection::N: dir = eth_chan_directions::NORTH; break;
        case RoutingDirection::S: dir = eth_chan_directions::SOUTH; break;
        case RoutingDirection::E: dir = eth_chan_directions::EAST; break;
        case RoutingDirection::W: dir = eth_chan_directions::WEST; break;
        default: TT_FATAL(false, "Invalid Routing Direction");
    }
    return dir;
}

std::set<std::pair<chan_id_t, eth_chan_directions>> ControlPlane::get_active_fabric_eth_channels(
    mesh_id_t mesh_id, chip_id_t chip_id) const {
    std::set<std::pair<chan_id_t, eth_chan_directions>> active_fabric_eth_channels;
    for (const auto& [direction, eth_chans] :
         this->router_port_directions_to_physical_eth_chan_map_[mesh_id][chip_id]) {
        for (const auto& eth_chan : eth_chans) {
            active_fabric_eth_channels.insert({eth_chan, this->routing_direction_to_eth_direction(direction)});
        }
    }
    return active_fabric_eth_channels;
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
        chan_id_t next_chan_id = 0;
        if (src_mesh_id != dst_mesh_id) {
            // Inter-mesh routing
            next_chan_id = this->inter_mesh_routing_tables_[src_mesh_id][src_chip_id][src_chan_id][dst_mesh_id];
        } else if (src_chip_id != dst_chip_id) {
            // Intra-mesh routing
            next_chan_id = this->intra_mesh_routing_tables_[src_mesh_id][src_chip_id][src_chan_id][dst_chip_id];
        }
        if (src_chan_id != next_chan_id) {
            // Chan to chan within chip
            route.push_back({physical_chip_id, next_chan_id});
        }
        std::tie(src_mesh_id, src_chip_id, src_chan_id) =
            this->get_connected_mesh_chip_chan_ids(src_mesh_id, src_chip_id, next_chan_id);
        auto connected_physical_chip_id = logical_mesh_chip_id_to_physical_chip_id_mapping_[src_mesh_id][src_chip_id];
        route.push_back({connected_physical_chip_id, src_chan_id});
    }

    return route;
}

std::vector<std::pair<routing_plane_id_t, CoreCoord>> ControlPlane::get_routers_to_chip(
    mesh_id_t src_mesh_id, chip_id_t src_chip_id, mesh_id_t dst_mesh_id, chip_id_t dst_chip_id) const {
    std::vector<std::pair<routing_plane_id_t, CoreCoord>> routers;
    const auto& router_direction_eth_channels =
        router_port_directions_to_physical_eth_chan_map_[src_mesh_id][src_chip_id];
    for (const auto& [direction, eth_chans] : router_direction_eth_channels) {
        for (const auto& src_chan_id : eth_chans) {
            chan_id_t next_chan_id = 0;
            if (src_mesh_id != dst_mesh_id) {
                // Inter-mesh routing
                next_chan_id = this->inter_mesh_routing_tables_[src_mesh_id][src_chip_id][src_chan_id][dst_mesh_id];

            } else if (src_chip_id != dst_chip_id) {
                // Intra-mesh routing
                next_chan_id = this->intra_mesh_routing_tables_[src_mesh_id][src_chip_id][src_chan_id][dst_chip_id];
            }
            if (src_chan_id != next_chan_id) {
                continue;
            }
            const auto& physical_chip_id =
                this->logical_mesh_chip_id_to_physical_chip_id_mapping_[src_mesh_id][src_chip_id];
            routers.emplace_back(
                this->get_routing_plane_id(src_chan_id),
                tt::tt_metal::MetalContext::instance().get_cluster().get_virtual_eth_core_from_channel(
                    physical_chip_id, src_chan_id));
        }
    }
    return routers;
}

stl::Span<const chip_id_t> ControlPlane::get_intra_chip_neighbors(
    mesh_id_t src_mesh_id, chip_id_t src_chip_id, RoutingDirection routing_direction) const {
    for (const auto& [_, routing_edge] :
         this->routing_table_generator_->get_intra_mesh_connectivity()[src_mesh_id][src_chip_id]) {
        if (routing_edge.port_direction == routing_direction) {
            return routing_edge.connected_chip_ids;
        }
    }
    return {};
}

size_t ControlPlane::get_num_active_fabric_routers(mesh_id_t mesh_id, chip_id_t chip_id) const {
    // Return the number of active fabric routers on the chip
    // Not always all the available FABRIC_ROUTER cores given by Cluster, since some may be disabled
    size_t num_routers = 0;
    for (const auto& [direction, eth_chans] :
         this->router_port_directions_to_physical_eth_chan_map_[mesh_id][chip_id]) {
        num_routers += eth_chans.size();
    }
    return num_routers;
}

std::set<chan_id_t> ControlPlane::get_active_fabric_eth_channels_in_direction(
    mesh_id_t mesh_id, chip_id_t chip_id, RoutingDirection routing_direction) const {
    for (const auto& [direction, eth_chans] :
         this->router_port_directions_to_physical_eth_chan_map_[mesh_id][chip_id]) {
        if (routing_direction == direction) {
            return std::set<chan_id_t>(eth_chans.begin(), eth_chans.end());
        }
    }
    return {};
}

void ControlPlane::write_routing_tables_to_all_chips() const {
    // Configure the routing tables on the chips
    TT_ASSERT(
        this->intra_mesh_routing_tables_.size() == this->inter_mesh_routing_tables_.size(),
        "Intra mesh routing tables num_meshes mismatch with inter mesh routing tables");
    for (mesh_id_t mesh_id = 0; mesh_id < this->intra_mesh_routing_tables_.size(); mesh_id++) {
        TT_ASSERT(
            this->intra_mesh_routing_tables_[mesh_id].size() == this->inter_mesh_routing_tables_[mesh_id].size(),
            "Intra mesh routing tables num_devices in mesh {} mismatch with inter mesh routing tables",
            mesh_id);
        for (chip_id_t chip_id = 0; chip_id < this->intra_mesh_routing_tables_[mesh_id].size(); chip_id++) {
            this->write_routing_tables_to_chip(mesh_id, chip_id);
        }
    }
}

std::vector<mesh_id_t> ControlPlane::get_user_physical_mesh_ids() const {
    std::vector<mesh_id_t> physical_mesh_ids;
    const auto user_chips = tt::tt_metal::MetalContext::instance().get_cluster().user_exposed_chip_ids();
    for (int mesh_id = 0; mesh_id < this->logical_mesh_chip_id_to_physical_chip_id_mapping_.size(); mesh_id++) {
        bool add_mesh = true;
        for (int chip_id = 0; chip_id < this->logical_mesh_chip_id_to_physical_chip_id_mapping_[mesh_id].size();
             chip_id++) {
            if (user_chips.find(this->logical_mesh_chip_id_to_physical_chip_id_mapping_[mesh_id][chip_id]) ==
                user_chips.end()) {
                add_mesh = false;
                break;
            }
        }
        if (add_mesh) {
            physical_mesh_ids.push_back(mesh_id);
        }
    }

    return physical_mesh_ids;
}

tt::tt_metal::distributed::MeshShape ControlPlane::get_physical_mesh_shape(mesh_id_t mesh_id) const {
    uint32_t x = this->routing_table_generator_->get_mesh_ns_size(mesh_id);
    uint32_t y = this->routing_table_generator_->get_mesh_ew_size(mesh_id);
    return tt::tt_metal::distributed::MeshShape(x, y);
};

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

void ControlPlane::set_routing_mode(uint16_t mode) {
    if (!(this->routing_mode_ == 0 || this->routing_mode_ == mode)) {
        tt::log_warning(
            tt::LogFabric,
            "Control Plane: Routing mode already set to {}. Setting to {}",
            (uint16_t)this->routing_mode_,
            (uint16_t)mode);
    }
    this->routing_mode_ = mode;
}

uint16_t ControlPlane::get_routing_mode() const { return this->routing_mode_; }

}  // namespace tt::tt_fabric
