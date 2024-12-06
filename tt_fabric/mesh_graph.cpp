// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_fabric/mesh_graph.hpp"

#include <fstream>
#include <iostream>

#include "yaml-cpp/yaml.h"

namespace tt::tt_fabric {

MeshGraph::MeshGraph(const std::string& mesh_graph_desc_file_path) {
    this->initialize_from_yaml(mesh_graph_desc_file_path);
}

void MeshGraph::add_to_connectivity(
    mesh_id_t src_mesh_id,
    chip_id_t src_chip_id,
    chip_id_t dest_mesh_id,
    chip_id_t dest_chip_id,
    RoutingDirection port_direction) {
    TT_ASSERT(
        src_mesh_id < this->intra_mesh_connectivity_.size(),
        "MeshGraph: Invalid src_mesh_id: {} or unsized intramesh map",
        src_mesh_id);
    TT_ASSERT(
        dest_mesh_id < this->intra_mesh_connectivity_.size(),
        "MeshGraph: Invalid dest_mesh_id: {} or unsized intramesh map",
        dest_mesh_id);
    TT_ASSERT(
        src_chip_id < this->intra_mesh_connectivity_[src_mesh_id].size(),
        "MeshGraph: Invalid src_chip_id: {} or unsized intramesh map",
        src_chip_id);
    TT_ASSERT(
        dest_chip_id < this->intra_mesh_connectivity_[dest_mesh_id].size(),
        "MeshGraph: Invalid dest_chip_id: {} or unsized intramesh map",
        dest_chip_id);

    TT_ASSERT(
        src_mesh_id < this->inter_mesh_connectivity_.size(),
        "MeshGraph: Invalid src_mesh_id: {} or unsized intermesh map",
        src_mesh_id);
    TT_ASSERT(
        dest_mesh_id < this->inter_mesh_connectivity_.size(),
        "MeshGraph: Invalid dest_mesh_id: {} or unsized intermesh map",
        dest_mesh_id);
    TT_ASSERT(
        src_chip_id < this->inter_mesh_connectivity_[src_mesh_id].size(),
        "MeshGraph: Invalid src_chip_id: {} or unsized intermesh map",
        src_chip_id);
    TT_ASSERT(
        dest_chip_id < this->inter_mesh_connectivity_[dest_mesh_id].size(),
        "MeshGraph: Invalid dest_chip_id: {} or unsized intermesh map",
        dest_chip_id);

    if (src_mesh_id != dest_mesh_id) {
        // Intermesh Connection
        auto& edge = this->inter_mesh_connectivity_[src_mesh_id][src_chip_id];
        if (edge.find(dest_mesh_id) == edge.end()) {
            edge.insert({dest_mesh_id, RouterEdge{.port_direction = port_direction, {dest_chip_id}, .weight = 0}});
        } else {
            edge[dest_mesh_id].connected_chip_ids.push_back(dest_chip_id);
        }
    } else {
        // Intramesh Connection
        auto& edge = this->intra_mesh_connectivity_[src_mesh_id][src_chip_id];
        if (edge.find(dest_chip_id) == edge.end()) {
            edge.insert({dest_chip_id, RouterEdge{.port_direction = port_direction, {dest_chip_id}, .weight = 0}});
        } else {
            edge[dest_mesh_id].connected_chip_ids.push_back(dest_chip_id);
        }
    }
}
std::unordered_map<chip_id_t, RouterEdge> MeshGraph::get_valid_connections(
    chip_id_t src_chip_id, std::uint32_t row_size, std::uint32_t num_chips_in_board, FabricType fabric_type) const {
    std::unordered_map<chip_id_t, RouterEdge> valid_connections;
    chip_id_t N = src_chip_id - row_size;
    chip_id_t E = src_chip_id + 1;
    chip_id_t S = src_chip_id + row_size;
    chip_id_t W = src_chip_id - 1;
    if (fabric_type == FabricType::MESH) {
        if (N >= 0) {
            valid_connections.insert(
                {N,
                 RouterEdge{
                     .port_direction = RoutingDirection::N,
                     std::vector<chip_id_t>(this->chip_spec_.num_eth_ports_per_direction, N),
                     .weight = 0}});
        }
        if (E < num_chips_in_board && (E / row_size == src_chip_id / row_size)) {
            valid_connections.insert(
                {E,
                 RouterEdge{
                     .port_direction = RoutingDirection::E,
                     std::vector<chip_id_t>(this->chip_spec_.num_eth_ports_per_direction, E),
                     .weight = 0}});
        }
        if (S < num_chips_in_board) {
            valid_connections.insert(
                {S,
                 RouterEdge{
                     .port_direction = RoutingDirection::S,
                     std::vector<chip_id_t>(this->chip_spec_.num_eth_ports_per_direction, S),
                     .weight = 0}});
        }
        if (W >= 0 && (W / row_size == src_chip_id / row_size)) {
            valid_connections.insert(
                {W,
                 RouterEdge{
                     .port_direction = RoutingDirection::W,
                     std::vector<chip_id_t>(this->chip_spec_.num_eth_ports_per_direction, W),
                     .weight = 0}});
        }
    } else if (fabric_type == FabricType::TORUS) {
        // TODO: add support
    }
    return valid_connections;
}

void MeshGraph::initialize_from_yaml(const std::string& mesh_graph_desc_file_path) {
    std::ifstream fdesc(mesh_graph_desc_file_path);
    TT_FATAL(not fdesc.fail(), "Failed to open file: {}", mesh_graph_desc_file_path);

    YAML::Node yaml = YAML::LoadFile(mesh_graph_desc_file_path);

    TT_FATAL(yaml["ChipSpec"].IsMap(), "MeshGraph: Expecting yaml to define a ChipSpec as a Map");
    TT_FATAL(yaml["Board"].IsSequence(), "MeshGraph: Expecting yaml to define Board as a Sequence");
    TT_FATAL(yaml["Mesh"].IsSequence(), "MeshGraph: Expecting yaml to define Mesh as a Sequence");
    TT_FATAL(yaml["Graph"].IsSequence(), "MeshGraph: Expecting yaml to define Graph as a Sequence");

    // Parse Chip
    const auto& chip = yaml["ChipSpec"];
    auto arch = magic_enum::enum_cast<tt::ARCH>(chip["arch"].as<std::string>(), magic_enum::case_insensitive);
    TT_FATAL(arch.has_value(), "MeshGraph: Invalid yaml chip arch: {}", chip["arch"].as<std::string>());

    std::uint32_t num_eth_ports_per_direction = chip["ethernet_ports"]["N"].as<std::uint32_t>();
    if (num_eth_ports_per_direction != chip["ethernet_ports"]["E"].as<std::uint32_t>() ||
        num_eth_ports_per_direction != chip["ethernet_ports"]["S"].as<std::uint32_t>() ||
        num_eth_ports_per_direction != chip["ethernet_ports"]["W"].as<std::uint32_t>()) {
        TT_FATAL(true, "MeshGraph: Expecting the same number of ethernet ports in each direction");
    }
    this->chip_spec_ = ChipSpec{
        .arch = arch.value(),
        .num_eth_ports_per_direction = num_eth_ports_per_direction,
        .num_z_ports = chip["ethernet_ports"]["Z"].IsDefined() ? chip["ethernet_ports"]["Z"].as<std::uint32_t>() : 0};

    std::unordered_map<std::string, std::vector<std::unordered_map<chip_id_t, RouterEdge>>> board_to_mesh_connectivity;
    std::unordered_map<std::string, std::array<std::uint32_t, 2>> board_to_topology;
    for (const auto& board : yaml["Board"]) {
        std::string board_name = board["name"].as<std::string>();
        TT_FATAL(
            board_to_mesh_connectivity.find(board_name) == board_to_mesh_connectivity.end(),
            "MeshGraph: Duplicate board name: {}",
            board_name);
        auto fabric_type =
            magic_enum::enum_cast<FabricType>(board["type"].as<std::string>(), magic_enum::case_insensitive);
        TT_FATAL(
            fabric_type.has_value(), "MeshGraph: Invalid yaml fabric board type: {}", board["type"].as<std::string>());
        TT_FATAL(board["topology"].IsDefined(), "MeshGraph: Expecting yaml board to define topology");
        // Topology
        // e.g. [4, 8]:
        //    4x8 mesh or torus, 4 rows (NS direction), 8 columns (EW direction)
        //    chip 0 is NW corner, chip 1 is E of chip 0
        std::uint32_t row_size = board["topology"][1].as<std::uint32_t>();
        std::uint32_t col_size = board["topology"][0].as<std::uint32_t>();
        std::uint32_t num_chips_in_board = row_size * col_size;
        board_to_topology[board_name] = {col_size, row_size};

        // Fill in connectivity for Board
        board_to_mesh_connectivity[board_name].resize(num_chips_in_board);
        for (std::uint32_t i = 0; i < num_chips_in_board; i++) {
            board_to_mesh_connectivity[board_name][i] =
                this->get_valid_connections(i, row_size, num_chips_in_board, fabric_type.value());
        }
    }
    // Loop over Meshes, populate intra mesh
    std::vector<std::unordered_map<port_id_t, chip_id_t, hash_pair>> mesh_edge_ports_to_chip_id;
    for (const auto& mesh : yaml["Mesh"]) {
        // TODO: handle host mapping and topology
        std::string mesh_board = mesh["board"].as<std::string>();
        mesh_id_t mesh_id = mesh["id"].as<std::uint32_t>();
        if (this->intra_mesh_connectivity_.size() <= mesh_id) {
            // Resize all variables that loop over mesh_ids
            this->intra_mesh_connectivity_.resize(mesh_id + 1);
            this->inter_mesh_connectivity_.resize(mesh_id + 1);
            this->mesh_shapes_.resize(mesh_id + 1);
            mesh_edge_ports_to_chip_id.resize(mesh_id + 1);
        }
        TT_FATAL(
            board_to_mesh_connectivity.find(mesh_board) != board_to_mesh_connectivity.end(),
            "MeshGraph: Board not found: {}",
            mesh["board"].as<std::string>());

        TT_FATAL(
            (mesh["topology"][0].as<std::uint32_t>() == 1) and (mesh["topology"][1].as<std::uint32_t>() == 1),
            "Add support for non 1x1 mesh");
        // Find board in board_to_mesh_connectivity and populate, need to add support for topology
        this->intra_mesh_connectivity_[mesh_id] = board_to_mesh_connectivity[mesh["board"].as<std::string>()];
        this->inter_mesh_connectivity_[mesh_id].resize(this->intra_mesh_connectivity_[mesh_id].size());

        std::uint32_t mesh_ew_size = mesh["topology"][1].as<std::uint32_t>() * board_to_topology[mesh_board][1];
        std::uint32_t mesh_ns_size = mesh["topology"][0].as<std::uint32_t>() * board_to_topology[mesh_board][0];
        std::uint32_t mesh_size = mesh_ew_size * mesh_ns_size;
        this->mesh_shapes_[mesh_id] = {mesh_ns_size, mesh_ew_size};

        // Print Mesh
        std::stringstream ss;
        for (int i = 0; i < mesh_size; i++) {
            if (i % mesh_ew_size == 0) {
                ss << std::endl;
            }
            ss << " " << std::setfill('0') << std::setw(2) << i;
        }
        log_debug(tt::LogFabric, "Mesh Graph: Mesh {} Logical Device Ids {}", mesh_id, ss.str());

        // Get the edge ports of each mesh
        // North, start from NW corner
        std::uint32_t chan_id = 0;
        for (std::uint32_t chip_id = 0; chip_id < mesh_ew_size; chip_id++) {
            for (std::uint32_t i = 0; i < this->chip_spec_.num_eth_ports_per_direction; i++) {
                mesh_edge_ports_to_chip_id[mesh_id][{RoutingDirection::N, chan_id++}] = chip_id;
            }
        }
        // South, start from SW corner
        chan_id = 0;
        for (std::uint32_t chip_id = (mesh_size - mesh_ew_size); chip_id < mesh_size; chip_id++) {
            for (std::uint32_t i = 0; i < this->chip_spec_.num_eth_ports_per_direction; i++) {
                mesh_edge_ports_to_chip_id[mesh_id][{RoutingDirection::S, chan_id++}] = chip_id;
            }
        }
        // East, start from NE corner
        chan_id = 0;
        for (std::uint32_t chip_id = (mesh_ew_size - 1); chip_id < mesh_size; chip_id += mesh_ew_size) {
            for (std::uint32_t i = 0; i < this->chip_spec_.num_eth_ports_per_direction; i++) {
                mesh_edge_ports_to_chip_id[mesh_id][{RoutingDirection::E, chan_id++}] = chip_id;
            }
        }
        // WEST, start from SW corner
        chan_id = 0;
        for (std::uint32_t chip_id = 0; chip_id < (mesh_size - mesh_ew_size); chip_id += mesh_ew_size) {
            for (std::uint32_t i = 0; i < this->chip_spec_.num_eth_ports_per_direction; i++) {
                mesh_edge_ports_to_chip_id[mesh_id][{RoutingDirection::W, chan_id++}] = chip_id;
            }
        }
    }
    // Loop over Graph, populate inter mesh
    auto convert_yaml_to_port_id = [](const YAML::Node& node) -> std::pair<mesh_id_t, port_id_t> {
        mesh_id_t mesh_id = node[0].as<std::uint32_t>();
        std::string port_string = node[1].as<std::string>();
        RoutingDirection port_direction =
            magic_enum::enum_cast<RoutingDirection>(port_string.substr(0, 1), magic_enum::case_insensitive).value();
        std::uint32_t chan_id = static_cast<uint32_t>(std::stoul(port_string.substr(1, port_string.size() - 1)));
        return {mesh_id, {port_direction, chan_id}};
    };
    for (const auto& mesh_connection : yaml["Graph"]) {
        TT_FATAL(mesh_connection.size() == 2, "MeshGraph: Expecting 2 elements in each Graph connection");
        const auto& [src_mesh_id, src_port_id] = convert_yaml_to_port_id(mesh_connection[0]);
        const auto& [dst_mesh_id, dst_port_id] = convert_yaml_to_port_id(mesh_connection[1]);
        const auto& src_chip_id = mesh_edge_ports_to_chip_id[src_mesh_id][src_port_id];
        const auto& dst_chip_id = mesh_edge_ports_to_chip_id[dst_mesh_id][dst_port_id];
        this->add_to_connectivity(src_mesh_id, src_chip_id, dst_mesh_id, dst_chip_id, src_port_id.first);
    }
}

void MeshGraph::print_connectivity() const {
    // Print Connectivity

    std::stringstream ss;
    ss << " Mesh Graph:  Intra Mesh Connectivity: " << std::endl;
    for (uint32_t mesh_id = 0; mesh_id < this->intra_mesh_connectivity_.size(); mesh_id++) {
        ss << "M" << mesh_id << ":" << std::endl;
        for (uint32_t chip_id = 0; chip_id < this->intra_mesh_connectivity_[mesh_id].size(); chip_id++) {
            ss << "   D" << chip_id << ": ";
            for (auto [connected_chip_id, edge] : this->intra_mesh_connectivity_[mesh_id][chip_id]) {
                for (int i = 0; i < edge.connected_chip_ids.size(); i++) {
                    ss << edge.connected_chip_ids[i] << "(" << magic_enum::enum_name(edge.port_direction) << ", "
                       << edge.weight << ") ";
                }
            }
            ss << std::endl;
        }
    }
    log_debug(tt::LogFabric, "{}", ss.str());
    ss.str(std::string());
    ss << " Mesh Graph:  Inter Mesh Connectivity: " << std::endl;
    for (uint32_t mesh_id = 0; mesh_id < this->inter_mesh_connectivity_.size(); mesh_id++) {
        ss << "M" << mesh_id << ":" << std::endl;
        for (uint32_t chip_id = 0; chip_id < this->inter_mesh_connectivity_[mesh_id].size(); chip_id++) {
            ss << "   D" << chip_id << ": ";
            for (auto [connected_mesh_id, edge] : this->inter_mesh_connectivity_[mesh_id][chip_id]) {
                for (int i = 0; i < edge.connected_chip_ids.size(); i++) {
                    ss << "M" << connected_mesh_id << "D" << edge.connected_chip_ids[i] << "("
                       << magic_enum::enum_name(edge.port_direction) << ", " << edge.weight << ") ";
                }
            }
            ss << std::endl;
        }
    }
    log_debug(tt::LogFabric, "{}", ss.str());
}

}  // namespace tt::tt_fabric
