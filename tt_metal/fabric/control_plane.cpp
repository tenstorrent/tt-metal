// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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
#include "impl/context/metal_context.hpp"
#include <tt-logger/tt-logger.hpp>
#include "mesh_coord.hpp"
#include "mesh_graph.hpp"
#include "metal_soc_descriptor.h"
#include "routing_table_generator.hpp"
#include <umd/device/tt_core_coordinates.h>
#include <umd/device/tt_xy_pair.h>
#include <umd/device/types/cluster_descriptor_types.h>
#include <umd/device/types/xy_pair.h>
#include "tt_metal/fabric/fabric_context.hpp"

namespace tt::tt_fabric {

// Get the physical chip ids for a mesh
std::unordered_map<chip_id_t, std::vector<CoreCoord>> get_ethernet_cores_grouped_by_connected_chips(chip_id_t chip_id) {
    return tt::tt_metal::MetalContext::instance().get_cluster().get_ethernet_cores_grouped_by_connected_chips(chip_id);
}

bool is_chip_on_edge_of_mesh(
    chip_id_t physical_chip_id,
    int num_ports_per_side,
    const std::unordered_map<chip_id_t, std::vector<CoreCoord>>& ethernet_cores_grouped_by_connected_chips) {
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    // Chip is on edge if it does not have full connections to four sides
    if (cluster.get_board_type(physical_chip_id) == BoardType::UBB) {
        auto ubb_asic_id = cluster.get_ubb_asic_id(physical_chip_id);
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
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    if (cluster.get_board_type(physical_chip_id) == BoardType::UBB) {
        auto ubb_asic_id = cluster.get_ubb_asic_id(physical_chip_id);
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
    this->routing_table_generator_->mesh_graph->print_connectivity();
    // Printing, only enabled with log_debug
    this->routing_table_generator_->print_routing_tables();

    // Initialize the control plane routers based on mesh graph
    const auto& logical_mesh_chip_id_to_physical_chip_id_mapping =
        this->get_physical_chip_mapping_from_mesh_graph_desc_file(mesh_graph_desc_file);
    this->load_physical_chip_mapping(logical_mesh_chip_id_to_physical_chip_id_mapping);
}

ControlPlane::ControlPlane(
    const std::string& mesh_graph_desc_file,
    const std::map<FabricNodeId, chip_id_t>& logical_mesh_chip_id_to_physical_chip_id_mapping) {
    this->routing_table_generator_ = std::make_unique<RoutingTableGenerator>(mesh_graph_desc_file);
    // Printing, only enabled with log_debug
    this->routing_table_generator_->mesh_graph->print_connectivity();
    // Printing, only enabled with log_debug
    this->routing_table_generator_->print_routing_tables();

    // Initialize the control plane routers based on mesh graph
    this->load_physical_chip_mapping(logical_mesh_chip_id_to_physical_chip_id_mapping);
}

void ControlPlane::load_physical_chip_mapping(
    const std::map<FabricNodeId, chip_id_t>& logical_mesh_chip_id_to_physical_chip_id_mapping) {
    this->logical_mesh_chip_id_to_physical_chip_id_mapping_ = logical_mesh_chip_id_to_physical_chip_id_mapping;
    this->validate_mesh_connections();
}

void ControlPlane::validate_mesh_connections(MeshId mesh_id) const {
    MeshShape mesh_shape = routing_table_generator_->mesh_graph->get_mesh_shape(mesh_id);
    std::uint32_t mesh_ns_size = mesh_shape[0];
    std::uint32_t mesh_ew_size = mesh_shape[1];
    std::uint32_t num_ports_per_side =
        routing_table_generator_->mesh_graph->get_chip_spec().num_eth_ports_per_direction;
    for (std::uint32_t i = 0; i < mesh_ns_size; i++) {
        for (std::uint32_t j = 0; j < mesh_ew_size - 1; j++) {
            chip_id_t logical_chip_id = i * mesh_ew_size + j;
            FabricNodeId fabric_node_id{mesh_id, logical_chip_id};
            FabricNodeId fabric_node_id_next{mesh_id, logical_chip_id + 1};
            chip_id_t physical_chip_id = logical_mesh_chip_id_to_physical_chip_id_mapping_.at(fabric_node_id);
            chip_id_t physical_chip_id_next = logical_mesh_chip_id_to_physical_chip_id_mapping_.at(fabric_node_id_next);

            const auto& eth_links = get_ethernet_cores_grouped_by_connected_chips(physical_chip_id);
            auto eth_links_to_next = eth_links.find(physical_chip_id_next);
            TT_FATAL(
                eth_links_to_next != eth_links.end(),
                "Chip {} not connected to chip {}",
                physical_chip_id,
                physical_chip_id_next);
            TT_FATAL(
                eth_links_to_next->second.size() >= num_ports_per_side,
                "Chip {} to chip {} has {} links but expecting {}",
                physical_chip_id,
                physical_chip_id_next,
                eth_links.at(physical_chip_id_next).size(),
                num_ports_per_side);
            if (i != mesh_ns_size - 1) {
                FabricNodeId fabric_node_id_next_row{mesh_id, logical_chip_id + mesh_ew_size};
                chip_id_t physical_chip_id_next_row =
                    logical_mesh_chip_id_to_physical_chip_id_mapping_.at(fabric_node_id_next_row);
                auto eth_links_to_next_row = eth_links.find(physical_chip_id_next_row);
                TT_FATAL(
                    eth_links_to_next_row != eth_links.end(),
                    "Chip {} not connected to chip {}",
                    physical_chip_id,
                    physical_chip_id_next_row);
                TT_FATAL(
                    eth_links_to_next_row->second.size() >= num_ports_per_side,
                    "Chip {} to chip {} has {} links but expecting {}",
                    physical_chip_id,
                    physical_chip_id_next_row,
                    eth_links.at(physical_chip_id_next_row).size(),
                    num_ports_per_side);
            }
        }
    }
}

void ControlPlane::validate_mesh_connections() const {
    for (const auto& mesh_id : this->routing_table_generator_->mesh_graph->get_mesh_ids()) {
        this->validate_mesh_connections(mesh_id);
    }
}

std::vector<chip_id_t> ControlPlane::get_mesh_physical_chip_ids(
    std::uint32_t mesh_ns_size,
    std::uint32_t mesh_ew_size,
    chip_id_t nw_chip_physical_chip_id) const {
    std::uint32_t num_ports_per_side =
        routing_table_generator_->mesh_graph->get_chip_spec().num_eth_ports_per_direction;

    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();

    const auto user_chips = cluster.user_exposed_chip_ids();
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
        bool is_ubb = cluster.get_board_type(current_chip_id) == BoardType::UBB;
        for (const auto& [connected_chip_id, eth_ports] : eth_links) {
            // Do not include any corner to corner links on UBB
            if (is_ubb && cluster.is_external_cable(current_chip_id, eth_ports[0])) {
                continue;
            }
            if (eth_ports.size() >= num_ports_per_side) {
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
            bool is_ubb = cluster.get_board_type(physical_chip_id_from_north) == BoardType::UBB;
            for (const auto& [connected_chip_id, eth_ports] : eth_links_grouped_by_connected_chips) {
                if (is_ubb && cluster.is_external_cable(physical_chip_id_from_north, eth_ports[0])) {
                    continue;
                }
                if (visited_physical_chips.find(connected_chip_id) == visited_physical_chips.end() and
                    eth_ports.size() >= num_ports_per_side) {
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

std::map<FabricNodeId, chip_id_t> ControlPlane::get_physical_chip_mapping_from_mesh_graph_desc_file(
    const std::string& mesh_graph_desc_file) {
    chip_id_t nw_chip_physical_id;
    std::uint32_t mesh_ns_size, mesh_ew_size;
    std::string mesh_graph_desc_filename = std::filesystem::path(mesh_graph_desc_file).filename().string();
    std::map<FabricNodeId, chip_id_t> logical_mesh_chip_id_to_physical_chip_id_mapping;
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
        logical_mesh_chip_id_to_physical_chip_id_mapping.insert({FabricNodeId(MeshId{0}, 0), eth_coord_y_for_gateway_chips[3]});
        logical_mesh_chip_id_to_physical_chip_id_mapping.insert({FabricNodeId(MeshId{1}, 0), eth_coord_y_for_gateway_chips[2]});
        logical_mesh_chip_id_to_physical_chip_id_mapping.insert({FabricNodeId(MeshId{2}, 0), eth_coord_y_for_gateway_chips[1]});
        logical_mesh_chip_id_to_physical_chip_id_mapping.insert({FabricNodeId(MeshId{3}, 0), eth_coord_y_for_gateway_chips[0]});

        nw_chip_physical_id =
            tt::tt_metal::MetalContext::instance().get_cluster().get_physical_chip_id_from_eth_coord({0, 3, 7, 0, 1});
        auto mesh_shape = routing_table_generator_->mesh_graph->get_mesh_shape(MeshId{4});
        mesh_ns_size = mesh_shape[0];
        mesh_ew_size = mesh_shape[1];
        // Main board
        const auto& physical_chip_ids =
            this->get_mesh_physical_chip_ids(mesh_ns_size, mesh_ew_size, nw_chip_physical_id);
        for (std::uint32_t i = 0; i < physical_chip_ids.size(); i++) {
            logical_mesh_chip_id_to_physical_chip_id_mapping.insert({FabricNodeId(MeshId{4}, i), physical_chip_ids[i]});
        }
    } else if (
        mesh_graph_desc_filename == "quanta_galaxy_mesh_graph_descriptor.yaml" ||
        mesh_graph_desc_filename == "quanta_galaxy_torus_2d_graph_descriptor.yaml" ||
        mesh_graph_desc_filename == "dual_galaxy_mesh_graph_descriptor.yaml" ||
        mesh_graph_desc_filename == "p100_mesh_graph_descriptor.yaml" ||
        mesh_graph_desc_filename == "p150_mesh_graph_descriptor.yaml" ||
        mesh_graph_desc_filename == "p150_x2_mesh_graph_descriptor.yaml" ||
        mesh_graph_desc_filename == "p150_x4_mesh_graph_descriptor.yaml") {
        // TODO: update to pick out chip automatically
        nw_chip_physical_id = 0;
        auto mesh_shape = routing_table_generator_->mesh_graph->get_mesh_shape(MeshId{0});
        mesh_ns_size = mesh_shape[0];
        mesh_ew_size = mesh_shape[1];
        // Main board
        const auto& physical_chip_ids =
            this->get_mesh_physical_chip_ids(mesh_ns_size, mesh_ew_size, nw_chip_physical_id);
        for (std::uint32_t i = 0; i < physical_chip_ids.size(); i++) {
            logical_mesh_chip_id_to_physical_chip_id_mapping.insert({FabricNodeId(MeshId{0}, i), physical_chip_ids[i]});
        }
    } else if (
        mesh_graph_desc_filename == "t3k_mesh_graph_descriptor.yaml" ||
        mesh_graph_desc_filename == "n150_mesh_graph_descriptor.yaml" ||
        mesh_graph_desc_filename == "n300_mesh_graph_descriptor.yaml" ||
        mesh_graph_desc_filename == "multihost_t3k_mesh_graph_descriptor.yaml") {
        // Pick out the chip with the lowest ethernet coordinate (i.e. NW chip)
        const auto& chip_eth_coords =
            tt::tt_metal::MetalContext::instance().get_cluster().get_all_chip_ethernet_coordinates();
        TT_FATAL(!chip_eth_coords.empty(), "No chip ethernet coordinates found in ethernet coordinates map");

        // TODO: Support custom operator< for eth_coord_t to allow usage in std::set
        const auto min_coord =
            *std::min_element(chip_eth_coords.begin(), chip_eth_coords.end(), [](const auto& a, const auto& b) {
                const auto& [chip_a, eth_coord_a] = a;
                const auto& [chip_b, eth_coord_b] = b;

                if (eth_coord_a.cluster_id != eth_coord_b.cluster_id) {
                    return eth_coord_a.cluster_id < eth_coord_b.cluster_id;
                }
                if (eth_coord_a.x != eth_coord_b.x) {
                    return eth_coord_a.x < eth_coord_b.x;
                }
                if (eth_coord_a.y != eth_coord_b.y) {
                    return eth_coord_a.y < eth_coord_b.y;
                }
                if (eth_coord_a.rack != eth_coord_b.rack) {
                    return eth_coord_a.rack < eth_coord_b.rack;
                }
                return eth_coord_a.shelf < eth_coord_b.shelf;
            });

        nw_chip_physical_id =
            tt::tt_metal::MetalContext::instance().get_cluster().get_physical_chip_id_from_eth_coord(min_coord.second);
        auto mesh_shape = routing_table_generator_->mesh_graph->get_mesh_shape(MeshId{0});

        const auto& physical_chip_ids =
            this->get_mesh_physical_chip_ids(mesh_shape[0], mesh_shape[1], nw_chip_physical_id);
        for (std::uint32_t i = 0; i < physical_chip_ids.size(); i++) {
            logical_mesh_chip_id_to_physical_chip_id_mapping.insert({FabricNodeId(MeshId{0}, i), physical_chip_ids[i]});
        }
    } else {
        TT_THROW("Unsupported mesh graph descriptor file {}", mesh_graph_desc_file);
    }
    return logical_mesh_chip_id_to_physical_chip_id_mapping;
}

routing_plane_id_t ControlPlane::get_routing_plane_id(
    chan_id_t eth_chan_id, const std::vector<chan_id_t>& eth_chans_in_direction) const {
    auto it = std::find(eth_chans_in_direction.begin(), eth_chans_in_direction.end(), eth_chan_id);
    return std::distance(eth_chans_in_direction.begin(), it);
}

routing_plane_id_t ControlPlane::get_routing_plane_id(FabricNodeId fabric_node_id, chan_id_t eth_chan_id) const {
    TT_FATAL(
        this->router_port_directions_to_physical_eth_chan_map_.contains(fabric_node_id),
        "Mesh {} Chip {} out of bounds",
        fabric_node_id.mesh_id,
        fabric_node_id.chip_id);

    std::optional<std::vector<chan_id_t>> eth_chans_in_direction;
    const auto& chip_eth_chans_map = this->router_port_directions_to_physical_eth_chan_map_.at(fabric_node_id);
    for (const auto& [_, eth_chans] : chip_eth_chans_map) {
        if (std::find(eth_chans.begin(), eth_chans.end(), eth_chan_id) != eth_chans.end()) {
            eth_chans_in_direction = eth_chans;
            break;
        }
    }
    TT_FATAL(
        eth_chans_in_direction.has_value(),
        "Could not find Eth chan ID {} for Chip ID {}, Mesh ID {}",
        eth_chan_id,
        fabric_node_id.chip_id,
        fabric_node_id.mesh_id);

    return get_routing_plane_id(eth_chan_id, eth_chans_in_direction.value());
}

chan_id_t ControlPlane::get_downstream_eth_chan_id(
    routing_plane_id_t src_routing_plane_id, const std::vector<chan_id_t>& candidate_target_chans) const {
    if (candidate_target_chans.empty()) {
        return eth_chan_magic_values::INVALID_DIRECTION;
    }

    for (const auto& target_chan_id : candidate_target_chans) {
        if (src_routing_plane_id == this->get_routing_plane_id(target_chan_id, candidate_target_chans)) {
            return target_chan_id;
        }
    }

    /* TODO: for now disable collapsing routing planes until we add the corresponding logic for
        connecting the routers on these planes
    // If no match found, return a channel from candidate_target_chans
    while (src_routing_plane_id >= candidate_target_chans.size()) {
        src_routing_plane_id = src_routing_plane_id % candidate_target_chans.size();
    }
    return candidate_target_chans[src_routing_plane_id];
    */

    return eth_chan_magic_values::INVALID_DIRECTION;
};

void ControlPlane::convert_fabric_routing_table_to_chip_routing_table() {
    // Routing tables contain direction from chip to chip
    // Convert it to be unique per ethernet channel

    const auto& router_intra_mesh_routing_table = this->routing_table_generator_->get_intra_mesh_table();
    for (std::uint32_t mesh_id = 0; mesh_id < router_intra_mesh_routing_table.size(); mesh_id++) {
        for (std::uint32_t src_chip_id = 0; src_chip_id < router_intra_mesh_routing_table[mesh_id].size();
             src_chip_id++) {
            FabricNodeId src_fabric_node_id{MeshId{mesh_id}, src_chip_id};
            const auto& physical_chip_id =
                this->logical_mesh_chip_id_to_physical_chip_id_mapping_.at(src_fabric_node_id);
            std::uint32_t num_ports_per_chip = tt::tt_metal::MetalContext::instance()
                                                   .get_cluster()
                                                   .get_soc_desc(physical_chip_id)
                                                   .get_cores(CoreType::ETH)
                                                   .size();
            this->intra_mesh_routing_tables_[src_fabric_node_id].resize(
                num_ports_per_chip);  // contains more entries than needed, this size is for all eth channels on chip
            for (int i = 0; i < num_ports_per_chip; i++) {
                // Size the routing table to the number of chips in the mesh
                this->intra_mesh_routing_tables_[src_fabric_node_id][i].resize(
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
                     this->router_port_directions_to_physical_eth_chan_map_.at(src_fabric_node_id)) {
                    for (const auto& src_chan_id : eth_chans_on_side) {
                        if (src_chip_id == dst_chip_id) {
                            TT_ASSERT(
                                (target_direction == RoutingDirection::C),
                                "Expecting same direction for intra mesh routing");
                            // This entry represents chip to itself, should not be used by FW
                            this->intra_mesh_routing_tables_.at(src_fabric_node_id)[src_chan_id][dst_chip_id] =
                                src_chan_id;
                        } else if (target_direction == direction) {
                            // This entry represents an outgoing eth channel
                            this->intra_mesh_routing_tables_.at(src_fabric_node_id)[src_chan_id][dst_chip_id] =
                                src_chan_id;
                        } else {
                            const auto& eth_chans_in_target_direction =
                                this->router_port_directions_to_physical_eth_chan_map_.at(
                                    src_fabric_node_id)[target_direction];
                            const auto src_routing_plane_id =
                                this->get_routing_plane_id(src_chan_id, eth_chans_on_side);
                            this->intra_mesh_routing_tables_.at(src_fabric_node_id)[src_chan_id][dst_chip_id] =
                                this->get_downstream_eth_chan_id(src_routing_plane_id, eth_chans_in_target_direction);
                        }
                    }
                }
            }
        }
    }

    const auto& router_inter_mesh_routing_table = this->routing_table_generator_->get_inter_mesh_table();
    for (std::uint32_t src_mesh_id = 0; src_mesh_id < router_inter_mesh_routing_table.size(); src_mesh_id++) {
        for (std::uint32_t src_chip_id = 0; src_chip_id < router_inter_mesh_routing_table[src_mesh_id].size();
             src_chip_id++) {
            FabricNodeId src_fabric_node_id{MeshId{src_mesh_id}, src_chip_id};
            const auto& physical_chip_id =
                this->logical_mesh_chip_id_to_physical_chip_id_mapping_.at(src_fabric_node_id);
            std::uint32_t num_ports_per_chip = tt::tt_metal::MetalContext::instance()
                                                   .get_cluster()
                                                   .get_soc_desc(physical_chip_id)
                                                   .get_cores(CoreType::ETH)
                                                   .size();
            this->inter_mesh_routing_tables_[src_fabric_node_id].resize(
                num_ports_per_chip);  // contains more entries than needed
            for (int i = 0; i < num_ports_per_chip; i++) {
                // Size the routing table to the number of meshes
                this->inter_mesh_routing_tables_[src_fabric_node_id][i].resize(
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
                     this->router_port_directions_to_physical_eth_chan_map_.at(src_fabric_node_id)) {
                    for (const auto& src_chan_id : eth_chans_on_side) {
                        if (src_mesh_id == dst_mesh_id) {
                            TT_ASSERT(
                                (target_direction == RoutingDirection::C),
                                "ControlPlane: Expecting same direction for inter mesh routing");
                            // This entry represents mesh to itself, should not be used by FW
                            this->inter_mesh_routing_tables_.at(src_fabric_node_id)[src_chan_id][dst_mesh_id] =
                                src_chan_id;
                        } else if (target_direction == RoutingDirection::NONE) {
                            // This entry represents a mesh to mesh connection that is not reachable
                            // Set to an invalid channel id
                            this->inter_mesh_routing_tables_.at(src_fabric_node_id)[src_chan_id][dst_mesh_id] =
                                eth_chan_magic_values::INVALID_DIRECTION;
                        } else if (target_direction == direction) {
                            // This entry represents an outgoing eth channel
                            this->inter_mesh_routing_tables_.at(src_fabric_node_id)[src_chan_id][dst_mesh_id] =
                                src_chan_id;
                        } else {
                            const auto& eth_chans_in_target_direction =
                                this->router_port_directions_to_physical_eth_chan_map_.at(
                                    src_fabric_node_id)[target_direction];
                            const auto src_routing_plane_id =
                                this->get_routing_plane_id(src_chan_id, eth_chans_on_side);
                            this->inter_mesh_routing_tables_.at(src_fabric_node_id)[src_chan_id][dst_mesh_id] =
                                this->get_downstream_eth_chan_id(src_routing_plane_id, eth_chans_in_target_direction);
                        }
                    }
                }
            }
        }
    }
    // Printing, only enabled with log_debug
    this->print_routing_tables();
}

// order ethernet channels using virtual coordinates
void ControlPlane::order_ethernet_channels() {
    for (auto& [fabric_node_id, eth_chans_by_dir] : this->router_port_directions_to_physical_eth_chan_map_) {
        for (auto& [_, eth_chans] : eth_chans_by_dir) {
            auto phys_chip_id = this->get_physical_chip_id_from_fabric_node_id(fabric_node_id);
            const auto& soc_desc = tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(phys_chip_id);
            std::sort(eth_chans.begin(), eth_chans.end(), [&soc_desc](const auto& a, const auto& b) {
                auto virt_coords_a = soc_desc.get_eth_core_for_channel(a, CoordSystem::VIRTUAL);
                auto virt_coords_b = soc_desc.get_eth_core_for_channel(b, CoordSystem::VIRTUAL);
                return virt_coords_a.x < virt_coords_b.x;
            });
        }
    }
}

void ControlPlane::configure_routing_tables_for_fabric_ethernet_channels() {
    this->intra_mesh_routing_tables_.clear();
    this->inter_mesh_routing_tables_.clear();
    this->router_port_directions_to_physical_eth_chan_map_.clear();

    const auto& intra_mesh_connectivity = this->routing_table_generator_->mesh_graph->get_intra_mesh_connectivity();
    const auto& inter_mesh_connectivity = this->routing_table_generator_->mesh_graph->get_inter_mesh_connectivity();

    auto add_ethernet_channel_to_router_mapping = [&](MeshId mesh_id,
                                                      chip_id_t chip_id,
                                                      const CoreCoord& eth_core,
                                                      RoutingDirection direction) {
        FabricNodeId fabric_node_id{mesh_id, chip_id};
        auto physical_chip_id = this->logical_mesh_chip_id_to_physical_chip_id_mapping_.at(fabric_node_id);
        auto fabric_router_channels_on_chip =
            tt::tt_metal::MetalContext::instance().get_cluster().get_fabric_ethernet_channels(physical_chip_id);
        // TODO: get_fabric_ethernet_channels accounts for down links, but we should manage down links in control plane
        auto chan_id = tt::tt_metal::MetalContext::instance()
                           .get_cluster()
                           .get_soc_desc(physical_chip_id)
                           .logical_eth_core_to_chan_map.at(eth_core);
        // TODO: add logic here to disable unsed routers, e.g. Mesh on Torus system
        if (fabric_router_channels_on_chip.contains(chan_id)) {
            this->router_port_directions_to_physical_eth_chan_map_.at(fabric_node_id)[direction].push_back(chan_id);
        } else {
            log_debug(
                tt::LogFabric, "Control Plane: Disabling router on M{}D{} eth channel {}", mesh_id, chip_id, chan_id);
        }
    };

    // Initialize the bookkeeping for mapping from mesh/chip/direction to physical ethernet channels
    for (const auto& [fabric_node_id, _] : this->logical_mesh_chip_id_to_physical_chip_id_mapping_) {
        if (!this->router_port_directions_to_physical_eth_chan_map_.contains(fabric_node_id)) {
            this->router_port_directions_to_physical_eth_chan_map_[fabric_node_id] = {};
        }
    }

    for (std::uint32_t mesh_id = 0; mesh_id < intra_mesh_connectivity.size(); mesh_id++) {
        for (std::uint32_t chip_id = 0; chip_id < intra_mesh_connectivity[mesh_id].size(); chip_id++) {
            auto physical_chip_id =
                this->logical_mesh_chip_id_to_physical_chip_id_mapping_.at(FabricNodeId(MeshId{mesh_id}, chip_id));
            const auto& connected_chips_and_eth_cores =
                tt::tt_metal::MetalContext::instance().get_cluster().get_ethernet_cores_grouped_by_connected_chips(
                    physical_chip_id);
            for (const auto& [logical_connected_chip_id, edge] : intra_mesh_connectivity[mesh_id][chip_id]) {
                const auto& physical_connected_chip_id = this->logical_mesh_chip_id_to_physical_chip_id_mapping_.at(
                    FabricNodeId(MeshId{mesh_id}, logical_connected_chip_id));
                const auto& connected_eth_cores = connected_chips_and_eth_cores.at(physical_connected_chip_id);
                TT_FATAL(
                    connected_eth_cores.size() >= edge.connected_chip_ids.size(),
                    "Expected {} eth links from physical chip {} to physical chip {}",
                    edge.connected_chip_ids.size(),
                    physical_chip_id,
                    physical_connected_chip_id);

                for (const auto& eth_core : connected_eth_cores) {
                    // There could be an optimization here to create entry for both chips here, assuming links are
                    // bidirectional
                    add_ethernet_channel_to_router_mapping(MeshId{mesh_id}, chip_id, eth_core, edge.port_direction);
                }
            }
        }
    }
    for (std::uint32_t mesh_id = 0; mesh_id < inter_mesh_connectivity.size(); mesh_id++) {
        for (std::uint32_t chip_id = 0; chip_id < inter_mesh_connectivity[mesh_id].size(); chip_id++) {
            auto physical_chip_id =
                this->logical_mesh_chip_id_to_physical_chip_id_mapping_.at(FabricNodeId(MeshId{mesh_id}, chip_id));
            const auto& connected_chips_and_eth_cores =
                tt::tt_metal::MetalContext::instance().get_cluster().get_ethernet_cores_grouped_by_connected_chips(
                    physical_chip_id);
            for (const auto& [connected_mesh_id, edge] : inter_mesh_connectivity[mesh_id][chip_id]) {
                // Loop over edges connected chip ids, they could connect to different chips for intermesh traffic
                // edge.connected_chip_ids is a vector of chip ids, that is populated per port. Since we push all
                // connected ports into the map when we visit a chip id, we should skip if we have already visited this
                // chip id
                std::unordered_set<chip_id_t> visited_chip_ids;
                for (const auto& logical_connected_chip_id : edge.connected_chip_ids) {
                    if (visited_chip_ids.count(logical_connected_chip_id)) {
                        continue;
                    }
                    visited_chip_ids.insert(logical_connected_chip_id);
                    const auto& physical_connected_chip_id = this->logical_mesh_chip_id_to_physical_chip_id_mapping_.at(
                        FabricNodeId(connected_mesh_id, logical_connected_chip_id));
                    const auto& connected_eth_cores = connected_chips_and_eth_cores.at(physical_connected_chip_id);
                    for (const auto& eth_core : connected_eth_cores) {
                        add_ethernet_channel_to_router_mapping(MeshId{mesh_id}, chip_id, eth_core, edge.port_direction);
                    }
                }
            }
        }
    }

    this->order_ethernet_channels();

    this->convert_fabric_routing_table_to_chip_routing_table();
}

void ControlPlane::write_routing_tables_to_chip(MeshId mesh_id, chip_id_t chip_id) const {
    FabricNodeId fabric_node_id{mesh_id, chip_id};
    const auto& chip_intra_mesh_routing_tables = this->intra_mesh_routing_tables_.at(fabric_node_id);
    const auto& chip_inter_mesh_routing_tables = this->inter_mesh_routing_tables_.at(fabric_node_id);
    auto physical_chip_id = this->logical_mesh_chip_id_to_physical_chip_id_mapping_.at(fabric_node_id);
    // Loop over ethernet channels to only write to cores with ethernet links
    // Looping over chip_intra/inter_mesh_routing_tables will write to all cores, even if they don't have ethernet links
    const auto& chip_eth_chans_map = this->router_port_directions_to_physical_eth_chan_map_.at(fabric_node_id);
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

            const auto src_routing_plane_id = this->get_routing_plane_id(eth_chan, eth_chans);
            if (chip_eth_chans_map.find(RoutingDirection::N) != chip_eth_chans_map.end()) {
                fabric_router_config.port_direction.directions[eth_chan_directions::NORTH] =
                    this->get_downstream_eth_chan_id(src_routing_plane_id, chip_eth_chans_map.at(RoutingDirection::N));
            } else {
                fabric_router_config.port_direction.directions[eth_chan_directions::NORTH] =
                    eth_chan_magic_values::INVALID_DIRECTION;
            }
            if (chip_eth_chans_map.find(RoutingDirection::S) != chip_eth_chans_map.end()) {
                fabric_router_config.port_direction.directions[eth_chan_directions::SOUTH] =
                    this->get_downstream_eth_chan_id(src_routing_plane_id, chip_eth_chans_map.at(RoutingDirection::S));
            } else {
                fabric_router_config.port_direction.directions[eth_chan_directions::SOUTH] =
                    eth_chan_magic_values::INVALID_DIRECTION;
            }
            if (chip_eth_chans_map.find(RoutingDirection::E) != chip_eth_chans_map.end()) {
                fabric_router_config.port_direction.directions[eth_chan_directions::EAST] =
                    this->get_downstream_eth_chan_id(src_routing_plane_id, chip_eth_chans_map.at(RoutingDirection::E));
            } else {
                fabric_router_config.port_direction.directions[eth_chan_directions::EAST] =
                    eth_chan_magic_values::INVALID_DIRECTION;
            }
            if (chip_eth_chans_map.find(RoutingDirection::W) != chip_eth_chans_map.end()) {
                fabric_router_config.port_direction.directions[eth_chan_directions::WEST] =
                    this->get_downstream_eth_chan_id(src_routing_plane_id, chip_eth_chans_map.at(RoutingDirection::W));
            } else {
                fabric_router_config.port_direction.directions[eth_chan_directions::WEST] =
                    eth_chan_magic_values::INVALID_DIRECTION;
            }

            fabric_router_config.my_mesh_id = *mesh_id;
            fabric_router_config.my_device_id = chip_id;
            MeshShape fabric_mesh_shape = this->routing_table_generator_->mesh_graph->get_mesh_shape(mesh_id);
            fabric_router_config.north_dim = fabric_mesh_shape[0];
            fabric_router_config.east_dim = fabric_mesh_shape[1];

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
                    tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt_metal::HalL1MemAddrType::FABRIC_ROUTER_CONFIG));
        }
    }
    tt::tt_metal::MetalContext::instance().get_cluster().l1_barrier(physical_chip_id);
}

FabricNodeId ControlPlane::get_fabric_node_id_from_physical_chip_id(chip_id_t physical_chip_id) const {
    for (const auto& [fabric_node_id, mapped_physical_chip_id] :
         this->logical_mesh_chip_id_to_physical_chip_id_mapping_) {
        if (mapped_physical_chip_id == physical_chip_id) {
            return fabric_node_id;
        }
    }
    TT_FATAL(false, "Physical chip id not found in logical mesh chip id mapping");
    return FabricNodeId(MeshId{0}, 0);
}

chip_id_t ControlPlane::get_physical_chip_id_from_fabric_node_id(const FabricNodeId& fabric_node_id) const {
    TT_ASSERT(logical_mesh_chip_id_to_physical_chip_id_mapping_.contains(fabric_node_id));
    return logical_mesh_chip_id_to_physical_chip_id_mapping_.at(fabric_node_id);
}

std::pair<FabricNodeId, chan_id_t> ControlPlane::get_connected_mesh_chip_chan_ids(
    FabricNodeId fabric_node_id, chan_id_t chan_id) const {
    // TODO: simplify this and maybe have this functionality in ControlPlane
    auto physical_chip_id = logical_mesh_chip_id_to_physical_chip_id_mapping_.at(fabric_node_id);
    tt::umd::CoreCoord eth_core = tt::tt_metal::MetalContext::instance()
                                      .get_cluster()
                                      .get_soc_desc(physical_chip_id)
                                      .get_eth_core_for_channel(chan_id, CoordSystem::LOGICAL);
    auto [connected_physical_chip_id, connected_eth_core] =
        tt::tt_metal::MetalContext::instance().get_cluster().get_connected_ethernet_core(
            std::make_tuple(physical_chip_id, CoreCoord{eth_core.x, eth_core.y}));

    auto connected_fabric_node_id = this->get_fabric_node_id_from_physical_chip_id(connected_physical_chip_id);
    auto connected_chan_id = tt::tt_metal::MetalContext::instance()
                                 .get_cluster()
                                 .get_soc_desc(connected_physical_chip_id)
                                 .logical_eth_core_to_chan_map.at(connected_eth_core);
    return std::make_pair(connected_fabric_node_id, connected_chan_id);
}

std::vector<chan_id_t> ControlPlane::get_valid_eth_chans_on_routing_plane(
    FabricNodeId fabric_node_id, routing_plane_id_t routing_plane_id) const {
    std::vector<chan_id_t> valid_eth_chans;
    for (const auto& [direction, eth_chans] :
         this->router_port_directions_to_physical_eth_chan_map_.at(fabric_node_id)) {
        for (const auto& eth_chan : eth_chans) {
            if (this->get_routing_plane_id(eth_chan, eth_chans) == routing_plane_id) {
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
    FabricNodeId fabric_node_id) const {
    std::set<std::pair<chan_id_t, eth_chan_directions>> active_fabric_eth_channels;
    for (const auto& [direction, eth_chans] :
         this->router_port_directions_to_physical_eth_chan_map_.at(fabric_node_id)) {
        for (const auto& eth_chan : eth_chans) {
            active_fabric_eth_channels.insert({eth_chan, this->routing_direction_to_eth_direction(direction)});
        }
    }
    return active_fabric_eth_channels;
}

eth_chan_directions ControlPlane::get_eth_chan_direction(FabricNodeId fabric_node_id, int chan) const {
    for (const auto& [direction, eth_chans] :
         this->router_port_directions_to_physical_eth_chan_map_.at(fabric_node_id)) {
        for (const auto& eth_chan : eth_chans) {
            if (chan == eth_chan) {
                return this->routing_direction_to_eth_direction(direction);
            }
        }
    }
    TT_THROW("Cannot Find Ethernet Channel Direction");
}

std::vector<std::pair<chip_id_t, chan_id_t>> ControlPlane::get_fabric_route(
    FabricNodeId src_fabric_node_id, FabricNodeId dst_fabric_node_id, chan_id_t src_chan_id) const {
    std::vector<std::pair<chip_id_t, chan_id_t>> route;
    int i = 0;
    // Find any eth chan on the plane id
    while (src_fabric_node_id != dst_fabric_node_id) {
        i++;
        auto src_mesh_id = src_fabric_node_id.mesh_id;
        auto src_chip_id = src_fabric_node_id.chip_id;
        auto dst_mesh_id = dst_fabric_node_id.mesh_id;
        auto dst_chip_id = dst_fabric_node_id.chip_id;
        if (i >= tt::tt_fabric::MAX_MESH_SIZE * tt::tt_fabric::MAX_NUM_MESHES) {
            return {};
        }
        auto physical_chip_id = logical_mesh_chip_id_to_physical_chip_id_mapping_.at(src_fabric_node_id);
        chan_id_t next_chan_id = 0;
        if (src_mesh_id != dst_mesh_id) {
            // Inter-mesh routing
            next_chan_id = this->inter_mesh_routing_tables_.at(src_fabric_node_id)[src_chan_id][*dst_mesh_id];
        } else if (src_chip_id != dst_chip_id) {
            // Intra-mesh routing
            next_chan_id = this->intra_mesh_routing_tables_.at(src_fabric_node_id)[src_chan_id][dst_chip_id];
        }
        if (next_chan_id == eth_chan_magic_values::INVALID_DIRECTION) {
            // The complete route b/w src and dst not found, probably some eth cores are reserved along the path
            return {};
        }
        if (src_chan_id != next_chan_id) {
            // Chan to chan within chip
            route.push_back({physical_chip_id, next_chan_id});
        }

        std::tie(src_fabric_node_id, src_chan_id) =
            this->get_connected_mesh_chip_chan_ids(src_fabric_node_id, next_chan_id);
        auto connected_physical_chip_id =
            this->logical_mesh_chip_id_to_physical_chip_id_mapping_.at(src_fabric_node_id);
        route.push_back({connected_physical_chip_id, src_chan_id});
    }

    return route;
}

std::optional<RoutingDirection> ControlPlane::get_forwarding_direction(
    FabricNodeId src_fabric_node_id, FabricNodeId dst_fabric_node_id) const {
    const auto& router_direction_eth_channels =
        this->router_port_directions_to_physical_eth_chan_map_.at(src_fabric_node_id);
    auto src_mesh_id = src_fabric_node_id.mesh_id;
    auto src_chip_id = src_fabric_node_id.chip_id;
    auto dst_mesh_id = dst_fabric_node_id.mesh_id;
    auto dst_chip_id = dst_fabric_node_id.chip_id;
    for (const auto& [direction, eth_chans] : router_direction_eth_channels) {
        for (const auto& src_chan_id : eth_chans) {
            chan_id_t next_chan_id = 0;
            if (src_mesh_id != dst_mesh_id) {
                // Inter-mesh routing
                next_chan_id = this->inter_mesh_routing_tables_.at(src_fabric_node_id)[src_chan_id][*dst_mesh_id];
            } else if (src_chip_id != dst_chip_id) {
                // Intra-mesh routing
                next_chan_id = this->intra_mesh_routing_tables_.at(src_fabric_node_id)[src_chan_id][dst_chip_id];
            }
            if (src_chan_id != next_chan_id) {
                continue;
            }

            // dimension-order routing: only 1 direction should give the desired shortest path from src to dst
            return direction;
        }
    }

    return std::nullopt;
}

std::vector<chan_id_t> ControlPlane::get_forwarding_eth_chans_to_chip(
    FabricNodeId src_fabric_node_id, FabricNodeId dst_fabric_node_id) const {
    const auto& forwarding_direction = get_forwarding_direction(src_fabric_node_id, dst_fabric_node_id);
    if (!forwarding_direction.has_value()) {
        return {};
    }

    return this->get_forwarding_eth_chans_to_chip(src_fabric_node_id, dst_fabric_node_id, *forwarding_direction);
}

std::vector<chan_id_t> ControlPlane::get_forwarding_eth_chans_to_chip(
    FabricNodeId src_fabric_node_id, FabricNodeId dst_fabric_node_id, RoutingDirection forwarding_direction) const {
    std::vector<chan_id_t> forwarding_channels;
    const auto& active_channels =
        this->get_active_fabric_eth_channels_in_direction(src_fabric_node_id, forwarding_direction);
    for (const auto& src_chan_id : active_channels) {
        // check for end-to-end route before accepting this channel
        if (this->get_fabric_route(src_fabric_node_id, dst_fabric_node_id, src_chan_id).empty()) {
            continue;
        }
        forwarding_channels.push_back(src_chan_id);
    }

    return forwarding_channels;
}

stl::Span<const chip_id_t> ControlPlane::get_intra_chip_neighbors(
    FabricNodeId src_fabric_node_id, RoutingDirection routing_direction) const {
    for (const auto& [_, routing_edge] :
         this->routing_table_generator_->mesh_graph
             ->get_intra_mesh_connectivity()[*src_fabric_node_id.mesh_id][src_fabric_node_id.chip_id]) {
        if (routing_edge.port_direction == routing_direction) {
            return routing_edge.connected_chip_ids;
        }
    }
    return {};
}

std::unordered_map<MeshId, std::vector<chip_id_t>> ControlPlane::get_chip_neighbors(
    FabricNodeId src_fabric_node_id, RoutingDirection routing_direction) const {
    std::unordered_map<MeshId, std::vector<chip_id_t>> neighbors;
    auto intra_neighbors = this->get_intra_chip_neighbors(src_fabric_node_id, routing_direction);
    auto src_mesh_id = src_fabric_node_id.mesh_id;
    auto src_chip_id = src_fabric_node_id.chip_id;
    if (!intra_neighbors.empty()) {
        neighbors[src_mesh_id].insert(neighbors[src_mesh_id].end(), intra_neighbors.begin(), intra_neighbors.end());
    }
    for (const auto& [mesh_id, routing_edge] :
         this->routing_table_generator_->mesh_graph->get_inter_mesh_connectivity()[*src_mesh_id][src_chip_id]) {
        if (routing_edge.port_direction == routing_direction) {
            neighbors[mesh_id] = routing_edge.connected_chip_ids;
        }
    }
    return neighbors;
}

size_t ControlPlane::get_num_active_fabric_routers(FabricNodeId fabric_node_id) const {
    // Return the number of active fabric routers on the chip
    // Not always all the available FABRIC_ROUTER cores given by Cluster, since some may be disabled
    size_t num_routers = 0;
    for (const auto& [direction, eth_chans] :
         this->router_port_directions_to_physical_eth_chan_map_.at(fabric_node_id)) {
        num_routers += eth_chans.size();
    }
    return num_routers;
}

std::vector<chan_id_t> ControlPlane::get_active_fabric_eth_channels_in_direction(
    FabricNodeId fabric_node_id, RoutingDirection routing_direction) const {
    for (const auto& [direction, eth_chans] :
         this->router_port_directions_to_physical_eth_chan_map_.at(fabric_node_id)) {
        if (routing_direction == direction) {
            return eth_chans;
        }
    }
    return {};
}

void ControlPlane::write_routing_tables_to_all_chips() const {
    // Configure the routing tables on the chips
    TT_ASSERT(
        this->intra_mesh_routing_tables_.size() == this->inter_mesh_routing_tables_.size(),
        "Intra mesh routing tables size mismatch with inter mesh routing tables");
    for (const auto& [fabric_node_id, _] : this->intra_mesh_routing_tables_) {
        TT_ASSERT(
            this->inter_mesh_routing_tables_.contains(fabric_node_id),
            "Intra mesh routing tables keys mismatch with inter mesh routing tables");
        this->write_routing_tables_to_chip(fabric_node_id.mesh_id, fabric_node_id.chip_id);
    }
}

// TODO: remove this after TG is deprecated
std::vector<MeshId> ControlPlane::get_user_physical_mesh_ids() const {
    std::vector<MeshId> physical_mesh_ids;
    const auto user_chips = tt::tt_metal::MetalContext::instance().get_cluster().user_exposed_chip_ids();
    for (const auto& [fabric_node_id, physical_chip_id] : this->logical_mesh_chip_id_to_physical_chip_id_mapping_) {
        if (user_chips.find(physical_chip_id) != user_chips.end() and
            std::find(physical_mesh_ids.begin(), physical_mesh_ids.end(), fabric_node_id.mesh_id) ==
                physical_mesh_ids.end()) {
            physical_mesh_ids.push_back(fabric_node_id.mesh_id);
        }
    }
    return physical_mesh_ids;
}

MeshShape ControlPlane::get_physical_mesh_shape(MeshId mesh_id) const {
    return this->routing_table_generator_->mesh_graph->get_mesh_shape(mesh_id);
};

void ControlPlane::print_routing_tables() const {
    this->print_ethernet_channels();

    std::stringstream ss;
    ss << "Control Plane: IntraMesh Routing Tables" << std::endl;
    for (const auto& [fabric_node_id, chip_routing_table] : this->intra_mesh_routing_tables_) {
        ss << fabric_node_id << ":" << std::endl;
        for (int eth_chan = 0; eth_chan < chip_routing_table.size(); eth_chan++) {
            ss << "   Eth Chan " << eth_chan << ": ";
            for (const auto& dst_chan_id : chip_routing_table[eth_chan]) {
                ss << (std::uint16_t)dst_chan_id << " ";
            }
            ss << std::endl;
        }
    }

    log_debug(tt::LogFabric, "{}", ss.str());
    ss.str(std::string());
    ss << "Control Plane: InterMesh Routing Tables" << std::endl;

    for (const auto& [fabric_node_id, chip_routing_table] : this->inter_mesh_routing_tables_) {
        ss << fabric_node_id << ":" << std::endl;
        for (int eth_chan = 0; eth_chan < chip_routing_table.size(); eth_chan++) {
            ss << "   Eth Chan " << eth_chan << ": ";
            for (const auto& dst_chan_id : chip_routing_table[eth_chan]) {
                ss << (std::uint16_t)dst_chan_id << " ";
            }
            ss << std::endl;
        }
    }
    log_debug(tt::LogFabric, "{}", ss.str());
}

void ControlPlane::print_ethernet_channels() const {
    std::stringstream ss;
    ss << "Control Plane: Physical eth channels in each direction" << std::endl;
    for (const auto& [fabric_node_id, fabric_eth_channels] : this->router_port_directions_to_physical_eth_chan_map_) {
        ss << fabric_node_id << ": " << std::endl;
        for (const auto& [direction, eth_chans] : fabric_eth_channels) {
            ss << "   " << magic_enum::enum_name(direction) << ":";
            for (const auto& eth_chan : eth_chans) {
                ss << " " << (std::uint16_t)eth_chan;
            }
            ss << std::endl;
        }
    }
    log_debug(tt::LogFabric, "{}", ss.str());
}

void ControlPlane::set_routing_mode(uint16_t mode) {
    if (!(this->routing_mode_ == 0 || this->routing_mode_ == mode)) {
        log_warning(
            tt::LogFabric,
            "Control Plane: Routing mode already set to {}. Setting to {}",
            (uint16_t)this->routing_mode_,
            (uint16_t)mode);
    }
    this->routing_mode_ = mode;
}

uint16_t ControlPlane::get_routing_mode() const { return this->routing_mode_; }

void ControlPlane::initialize_fabric_context(tt_metal::FabricConfig fabric_config) {
    TT_FATAL(this->fabric_context_ == nullptr, "Trying to re-initialize fabric context");
    this->fabric_context_ = std::make_unique<FabricContext>(fabric_config);
}

FabricContext& ControlPlane::get_fabric_context() const {
    TT_FATAL(this->fabric_context_ != nullptr, "Trying to get un-initialized fabric context");
    return *this->fabric_context_.get();
}

void ControlPlane::clear_fabric_context() { this->fabric_context_.reset(nullptr); }

ControlPlane::~ControlPlane() = default;

GlobalControlPlane::GlobalControlPlane(const std::string& mesh_graph_desc_file) {
    mesh_graph_desc_file_ = mesh_graph_desc_file;
    // Initialize host mappings
    this->initialize_host_mapping();
    control_plane_ = std::make_unique<ControlPlane>(mesh_graph_desc_file);
}

GlobalControlPlane::GlobalControlPlane(
    const std::string& mesh_graph_desc_file,
    const std::map<FabricNodeId, chip_id_t>& logical_mesh_chip_id_to_physical_chip_id_mapping) {
    mesh_graph_desc_file_ = mesh_graph_desc_file;
    this->initialize_host_mapping();
    control_plane_ =
        std::make_unique<ControlPlane>(mesh_graph_desc_file, logical_mesh_chip_id_to_physical_chip_id_mapping);
}

void GlobalControlPlane::initialize_host_mapping() {
    this->routing_table_generator_ = std::make_unique<RoutingTableGenerator>(mesh_graph_desc_file_);
    // Grab available hosts in the system and map to physical chip ids
    // ping for all hosts in cluster, grab mapping of all physical chip ids/physical hosts
    const auto& mesh_ids = this->routing_table_generator_->mesh_graph->get_mesh_ids();

    for (const auto& mesh_id : mesh_ids) {
        MeshShape mesh_shape = this->routing_table_generator_->mesh_graph->get_mesh_shape(mesh_id);
        const auto& host_ranks = this->routing_table_generator_->mesh_graph->get_host_ranks(mesh_id);
        for (const auto& [coord, rank] : host_ranks) {
            this->host_rank_to_sub_mesh_shape_[rank].push_back(coord);
        }
    }
}

GlobalControlPlane::~GlobalControlPlane() = default;

}  // namespace tt::tt_fabric
