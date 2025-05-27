// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mesh_graph.hpp"

#include <magic_enum/magic_enum.hpp>
#include <yaml-cpp/yaml.h>
#include <array>
#include <fstream>
#include <iomanip>
#include <optional>

#include "assert.hpp"
#include "logger.hpp"
#include <umd/device/types/cluster_descriptor_types.h>

namespace tt {
enum class ARCH;
}  // namespace tt

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
        auto [it, is_inserted] = edge.insert(
            {dest_mesh_id,
             RouterEdge{.port_direction = port_direction, .connected_chip_ids = {dest_chip_id}, .weight = 0}});
        if (!is_inserted) {
            it->second.connected_chip_ids.push_back(dest_chip_id);
        }
    } else {
        // Intramesh Connection
        auto& edge = this->intra_mesh_connectivity_[src_mesh_id][src_chip_id];
        auto [it, is_inserted] = edge.insert(
            {dest_chip_id,
             RouterEdge{.port_direction = port_direction, .connected_chip_ids = {dest_chip_id}, .weight = 0}});
        if (!is_inserted) {
            it->second.connected_chip_ids.push_back(dest_chip_id);
        }
    }
}
std::unordered_map<chip_id_t, RouterEdge> MeshGraph::get_valid_connections(
    chip_id_t src_chip_id, std::uint32_t row_size, std::uint32_t num_chips_in_board, FabricType fabric_type) const {
    std::unordered_map<chip_id_t, RouterEdge> valid_connections;
    if (fabric_type == FabricType::MESH) {
        chip_id_t N = src_chip_id - row_size;
        chip_id_t E = src_chip_id + 1;
        chip_id_t S = src_chip_id + row_size;
        chip_id_t W = src_chip_id - 1;
        if (N >= 0) {
            valid_connections.insert(
                {N,
                 RouterEdge{
                     .port_direction = RoutingDirection::N,
                     .connected_chip_ids = std::vector<chip_id_t>(this->chip_spec_.num_eth_ports_per_direction, N),
                     .weight = 0}});
        }
        if (E < num_chips_in_board && (E / row_size == src_chip_id / row_size)) {
            valid_connections.insert(
                {E,
                 RouterEdge{
                     .port_direction = RoutingDirection::E,
                     .connected_chip_ids = std::vector<chip_id_t>(this->chip_spec_.num_eth_ports_per_direction, E),
                     .weight = 0}});
        }
        if (S < num_chips_in_board) {
            valid_connections.insert(
                {S,
                 RouterEdge{
                     .port_direction = RoutingDirection::S,
                     .connected_chip_ids = std::vector<chip_id_t>(this->chip_spec_.num_eth_ports_per_direction, S),
                     .weight = 0}});
        }
        if (W >= 0 && (W / row_size == src_chip_id / row_size)) {
            valid_connections.insert(
                {W,
                 RouterEdge{
                     .port_direction = RoutingDirection::W,
                     .connected_chip_ids = std::vector<chip_id_t>(this->chip_spec_.num_eth_ports_per_direction, W),
                     .weight = 0}});
        }
    } else if (fabric_type == FabricType::TORUS_1D) {
        // TODO: add support
    } else if (fabric_type == FabricType::TORUS_2D) {
        auto row = src_chip_id / row_size;
        auto col = src_chip_id % row_size;
        chip_id_t N = (src_chip_id - row_size + num_chips_in_board) % num_chips_in_board;
        chip_id_t E = row * row_size + (col + 1) % row_size;
        chip_id_t S = (src_chip_id + row_size) % num_chips_in_board;
        chip_id_t W = row * row_size + (col - 1 + row_size) % row_size;
        valid_connections.insert(
            {N,
             RouterEdge{
                 .port_direction = RoutingDirection::N,
                 .connected_chip_ids = std::vector<chip_id_t>(this->chip_spec_.num_eth_ports_per_direction, N),
                 .weight = 0}});
        valid_connections.insert(
            {E,
             RouterEdge{
                 .port_direction = RoutingDirection::E,
                 .connected_chip_ids = std::vector<chip_id_t>(this->chip_spec_.num_eth_ports_per_direction, E),
                 .weight = 0}});
        valid_connections.insert(
            {S,
             RouterEdge{
                 .port_direction = RoutingDirection::S,
                 .connected_chip_ids = std::vector<chip_id_t>(this->chip_spec_.num_eth_ports_per_direction, S),
                 .weight = 0}});
        valid_connections.insert(
            {W,
             RouterEdge{
                 .port_direction = RoutingDirection::W,
                 .connected_chip_ids = std::vector<chip_id_t>(this->chip_spec_.num_eth_ports_per_direction, W),
                 .weight = 0}});
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

    std::unordered_map<std::string, std::array<std::uint32_t, 2>> board_name_to_topology;
    std::unordered_map<std::string, FabricType> board_name_to_fabric_type;
    for (const auto& board : yaml["Board"]) {
        std::string board_name = board["name"].as<std::string>();
        TT_FATAL(
            board_name_to_topology.find(board_name) == board_name_to_topology.end(),
            "MeshGraph: Duplicate board name: {}",
            board_name);
        auto fabric_type =
            magic_enum::enum_cast<FabricType>(board["type"].as<std::string>(), magic_enum::case_insensitive);
        TT_FATAL(
            fabric_type.has_value(), "MeshGraph: Invalid yaml fabric board type: {}", board["type"].as<std::string>());

        board_name_to_fabric_type[board_name] = fabric_type.value();
        TT_FATAL(board["topology"].IsDefined(), "MeshGraph: Expecting yaml board to define topology");
        // Topology
        // e.g. [4, 8]:
        //    4x8 mesh or torus, 4 rows (NS direction), 8 columns (EW direction)
        //    chip 0 is NW corner, chip 1 is E of chip 0
        std::uint32_t row_size = board["topology"][1].as<std::uint32_t>();
        std::uint32_t col_size = board["topology"][0].as<std::uint32_t>();
        board_name_to_topology[board_name] = {col_size, row_size};
    }
    // Loop over Meshes, populate intra mesh
    std::vector<std::unordered_map<port_id_t, chip_id_t, hash_pair>> mesh_edge_ports_to_chip_id;
    for (const auto& mesh : yaml["Mesh"]) {
        std::string mesh_board = mesh["board"].as<std::string>();
        mesh_id_t mesh_id = mesh["id"].as<std::uint32_t>();
        if (this->intra_mesh_connectivity_.size() <= mesh_id) {
            // Resize all variables that loop over mesh_ids
            this->intra_mesh_connectivity_.resize(mesh_id + 1);
            this->inter_mesh_connectivity_.resize(mesh_id + 1);
            this->mesh_shapes_.resize(mesh_id + 1);
            this->mesh_host_ranks_.resize(mesh_id + 1, MeshContainer<std::uint32_t>({}, {}));
            this->host_rank_coord_ranges_.resize(mesh_id + 1);
            mesh_edge_ports_to_chip_id.resize(mesh_id + 1);
            this->mesh_ids_.push_back(mesh_id);
        }
        TT_FATAL(
            board_name_to_topology.find(mesh_board) != board_name_to_topology.end(),
            "MeshGraph: Board not found: {}",
            mesh["board"].as<std::string>());

        std::uint32_t mesh_board_ew_size = mesh["topology"][1].as<std::uint32_t>();
        std::uint32_t mesh_board_ns_size = mesh["topology"][0].as<std::uint32_t>();
        std::uint32_t mesh_ew_size = mesh_board_ew_size * board_name_to_topology[mesh_board][1];
        std::uint32_t mesh_ns_size = mesh_board_ns_size * board_name_to_topology[mesh_board][0];

        std::uint32_t mesh_size = mesh_ew_size * mesh_ns_size;
        this->mesh_shapes_[mesh_id] = {mesh_ns_size, mesh_ew_size};

        // Fill in host ranks for Mesh
        TT_FATAL(
            mesh["host_ranks"].IsSequence() and mesh["host_ranks"].size() == mesh_board_ns_size,
            "MeshGraph: Expecting host_ranks to define a 2D array that matches topology");

        std::vector<std::uint32_t> mesh_host_ranks_values;
        mesh_host_ranks_values.reserve(mesh_board_ns_size * mesh_board_ew_size);

        // Track the start and end coordinates of each host rank
        std::unordered_map<std::uint32_t, std::pair<MeshCoordinate, MeshCoordinate>> host_rank_submesh_start_end_coords;
        for (std::uint32_t i = 0; i < mesh_board_ns_size; i++) {
            TT_FATAL(
                mesh["host_ranks"][i].IsSequence() and mesh["host_ranks"][i].size() == mesh_board_ew_size,
                "MeshGraph: Expecting host_ranks to define a 2D array that matches topology");
            for (std::uint32_t j = 0; j < mesh_board_ew_size; j++) {
                std::uint32_t host_rank = mesh["host_ranks"][i][j].as<std::uint32_t>();
                if (host_rank_submesh_start_end_coords.find(host_rank) == host_rank_submesh_start_end_coords.end()) {
                    host_rank_submesh_start_end_coords.insert(
                        {host_rank, std::make_pair(MeshCoordinate(i, j), MeshCoordinate(i, j))});
                } else {
                    host_rank_submesh_start_end_coords.at(host_rank).second = MeshCoordinate(i, j);
                }
                mesh_host_ranks_values.push_back(host_rank);
            }
        }
        // Fill in all host rank coordinate ranges
        this->host_rank_coord_ranges_[mesh_id].resize(mesh_host_ranks_values.size(), MeshCoordinateRange({}));
        for (const auto& [host_rank, coords] : host_rank_submesh_start_end_coords) {
            this->host_rank_coord_ranges_[mesh_id][host_rank] = MeshCoordinateRange(
                MeshCoordinate(
                    coords.first[0] * board_name_to_topology[mesh_board][0],
                    coords.first[1] * board_name_to_topology[mesh_board][1]),
                MeshCoordinate(
                    (coords.second[0] + 1) * board_name_to_topology[mesh_board][0] - 1,
                    (coords.second[1] + 1) * board_name_to_topology[mesh_board][1] - 1));
        }
        this->mesh_host_ranks_[mesh_id] =
            MeshContainer<std::uint32_t>(MeshShape(mesh_board_ns_size, mesh_board_ew_size), mesh_host_ranks_values);

        // Fill in connectivity for Mesh
        this->intra_mesh_connectivity_[mesh_id].resize(mesh_size);
        for (std::uint32_t i = 0; i < mesh_size; i++) {
            this->intra_mesh_connectivity_[mesh_id][i] =
                this->get_valid_connections(i, mesh_ew_size, mesh_size, board_name_to_fabric_type[mesh_board]);
        }

        this->inter_mesh_connectivity_[mesh_id].resize(this->intra_mesh_connectivity_[mesh_id].size());

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
        // WEST, start from NW corner
        chan_id = 0;
        for (std::uint32_t chip_id = 0; chip_id < mesh_size; chip_id += mesh_ew_size) {
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
        const auto& src_chip_id = mesh_edge_ports_to_chip_id[src_mesh_id].at(src_port_id);
        const auto& dst_chip_id = mesh_edge_ports_to_chip_id[dst_mesh_id].at(dst_port_id);
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
