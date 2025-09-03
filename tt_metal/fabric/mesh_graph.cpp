// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mesh_graph.hpp"

#include <algorithm>
#include <enchantum/enchantum.hpp>
#include <yaml-cpp/yaml.h>
#include <array>
#include <fstream>
#include <iomanip>
#include <optional>

#include "assert.hpp"
#include <tt-logger/tt-logger.hpp>
#include <llrt/tt_cluster.hpp>
#include <umd/device/types/cluster_descriptor_types.h>
#include <tt_stl/indestructible.hpp>
#include <tt_stl/caseless_comparison.hpp>
#include "tt_metal/fabric/physical_system_descriptor.hpp"
#include "tt_metal/impl/context/metal_context.hpp"
#include <tt-metalium/multi_mesh_types.hpp>
#include "tt_metal/fabric/serialization/logical_port_to_eth_chan.hpp"
#include "tt_metal/fabric/serialization/connections_table.hpp"
#include "tt-metalium/control_plane.hpp"
#include "tt_metal/fabric/serialization/port_id_table.hpp"

namespace tt {
enum class ARCH;
}  // namespace tt

namespace tt::tt_fabric {
FabricType operator|(FabricType lhs, FabricType rhs) {
    return static_cast<FabricType>(static_cast<uint32_t>(lhs) | static_cast<uint32_t>(rhs));
}

FabricType operator&(FabricType lhs, FabricType rhs) {
    return static_cast<FabricType>(static_cast<uint32_t>(lhs) & static_cast<uint32_t>(rhs));
}

namespace {
constexpr const char* MESH_GRAPH_DESCRIPTOR_DIR = "tt_metal/fabric/mesh_graph_descriptors";
}

const tt::stl::Indestructible<std::unordered_map<tt::tt_metal::ClusterType, std::string_view>>&
    MeshGraph::cluster_type_to_mesh_graph_descriptor =
        *new tt::stl::Indestructible<std::unordered_map<tt::tt_metal::ClusterType, std::string_view>>(
            std::unordered_map<tt::tt_metal::ClusterType, std::string_view>{
                {tt::tt_metal::ClusterType::N150, "n150_mesh_graph_descriptor.yaml"},
                {tt::tt_metal::ClusterType::N300, "n300_mesh_graph_descriptor.yaml"},
                {tt::tt_metal::ClusterType::T3K, "t3k_mesh_graph_descriptor.yaml"},
                {tt::tt_metal::ClusterType::GALAXY, "single_galaxy_mesh_graph_descriptor.yaml"},
                {tt::tt_metal::ClusterType::TG, "tg_mesh_graph_descriptor.yaml"},
                {tt::tt_metal::ClusterType::P100, "p100_mesh_graph_descriptor.yaml"},
                {tt::tt_metal::ClusterType::P150, "p150_mesh_graph_descriptor.yaml"},
                {tt::tt_metal::ClusterType::P150_X2, "p150_x2_mesh_graph_descriptor.yaml"},
                {tt::tt_metal::ClusterType::P150_X4, "p150_x4_mesh_graph_descriptor.yaml"},
                {tt::tt_metal::ClusterType::P150_X8, "p150_x8_mesh_graph_descriptor.yaml"},
                {tt::tt_metal::ClusterType::SIMULATOR_WORMHOLE_B0, "n150_mesh_graph_descriptor.yaml"},
                {tt::tt_metal::ClusterType::SIMULATOR_BLACKHOLE, "p150_mesh_graph_descriptor.yaml"},
                {tt::tt_metal::ClusterType::SIMULATOR_QUASAR,
                 "p150_mesh_graph_descriptor.yaml"},  // TODO use quasar mesh
                {tt::tt_metal::ClusterType::N300_2x2, "n300_2x2_mesh_graph_descriptor.yaml"},
                {tt::tt_metal::ClusterType::P300, "p300_mesh_graph_descriptor.yaml"},
            });

bool has_flag(FabricType flags, FabricType test) { return (flags & test) == test; }

MeshGraph::MeshGraph(
    const std::string& mesh_graph_desc_file_path,
    std::optional<std::map<FabricNodeId, chip_id_t>> logical_mesh_chip_id_to_physical_chip_id_mapping,
    std::shared_ptr<tt_metal::PhysicalSystemDescriptor> physical_system_descriptor) {
    std::cout << "Mesh Graph: Creating Mesh Graph" << std::endl;
    this->initialize_from_yaml(
        mesh_graph_desc_file_path, logical_mesh_chip_id_to_physical_chip_id_mapping, physical_system_descriptor);
    std::cout << "Mesh Graph: Mesh Graph Created" << std::endl;
}

void MeshGraph::add_to_connectivity(
    MeshId src_mesh_id,
    chip_id_t src_chip_id,
    MeshId dest_mesh_id,
    chip_id_t dest_chip_id,
    RoutingDirection port_direction) {
    TT_ASSERT(
        *src_mesh_id < this->intra_mesh_connectivity_.size(),
        "MeshGraph: Invalid src_mesh_id: {} or unsized intramesh map",
        *src_mesh_id);
    TT_ASSERT(
        *dest_mesh_id < this->intra_mesh_connectivity_.size(),
        "MeshGraph: Invalid dest_mesh_id: {} or unsized intramesh map",
        *dest_mesh_id);
    TT_ASSERT(
        src_chip_id < this->intra_mesh_connectivity_[*src_mesh_id].size(),
        "MeshGraph: Invalid src_chip_id: {} or unsized intramesh map",
        src_chip_id);
    TT_ASSERT(
        dest_chip_id < this->intra_mesh_connectivity_[*dest_mesh_id].size(),
        "MeshGraph: Invalid dest_chip_id: {} or unsized intramesh map",
        dest_chip_id);

    TT_ASSERT(
        *src_mesh_id < this->inter_mesh_connectivity_.size(),
        "MeshGraph: Invalid src_mesh_id: {} or unsized intermesh map",
        *src_mesh_id);
    TT_ASSERT(
        *dest_mesh_id < this->inter_mesh_connectivity_.size(),
        "MeshGraph: Invalid dest_mesh_id: {} or unsized intermesh map",
        *dest_mesh_id);
    TT_ASSERT(
        src_chip_id < this->inter_mesh_connectivity_[*src_mesh_id].size(),
        "MeshGraph: Invalid src_chip_id: {} or unsized intermesh map",
        src_chip_id);
    TT_ASSERT(
        dest_chip_id < this->inter_mesh_connectivity_[*dest_mesh_id].size(),
        "MeshGraph: Invalid dest_chip_id: {} or unsized intermesh map",
        dest_chip_id);

    if (src_mesh_id != dest_mesh_id) {
        // Intermesh Connection
        auto& edge = this->inter_mesh_connectivity_[*src_mesh_id][src_chip_id];
        auto [it, is_inserted] = edge.insert(
            {dest_mesh_id,
             RouterEdge{.port_direction = port_direction, .connected_chip_ids = {dest_chip_id}, .weight = 0}});
        if (!is_inserted) {
            it->second.connected_chip_ids.push_back(dest_chip_id);
        }
    } else {
        // Intramesh Connection
        auto& edge = this->intra_mesh_connectivity_[*src_mesh_id][src_chip_id];
        auto [it, is_inserted] = edge.insert(
            {dest_chip_id,
             RouterEdge{.port_direction = port_direction, .connected_chip_ids = {dest_chip_id}, .weight = 0}});
        if (!is_inserted) {
            it->second.connected_chip_ids.push_back(dest_chip_id);
        }
    }
}

std::unordered_map<chip_id_t, RouterEdge> MeshGraph::get_valid_connections(
    const MeshCoordinate& src_mesh_coord, const MeshCoordinateRange& mesh_coord_range, FabricType fabric_type) const {
    std::unordered_map<chip_id_t, RouterEdge> valid_connections;

    MeshShape mesh_shape = mesh_coord_range.shape();
    MeshCoordinate N(src_mesh_coord[0] - 1, src_mesh_coord[1]);
    MeshCoordinate E(src_mesh_coord[0], src_mesh_coord[1] + 1);
    MeshCoordinate S(src_mesh_coord[0] + 1, src_mesh_coord[1]);
    MeshCoordinate W(src_mesh_coord[0], src_mesh_coord[1] - 1);

    if (has_flag(fabric_type, FabricType::TORUS_X)) {
        E = MeshCoordinate(src_mesh_coord[0], (src_mesh_coord[1] + 1) % mesh_shape[1]);
        W = MeshCoordinate(src_mesh_coord[0], (src_mesh_coord[1] - 1 + mesh_shape[1]) % mesh_shape[1]);
    }
    if (has_flag(fabric_type, FabricType::TORUS_Y)) {
        N = MeshCoordinate((src_mesh_coord[0] - 1 + mesh_shape[0]) % mesh_shape[0], src_mesh_coord[1]);
        S = MeshCoordinate((src_mesh_coord[0] + 1) % mesh_shape[0], src_mesh_coord[1]);
    }
    for (auto& [coord, direction] :
         {std::pair{N, RoutingDirection::N},
          std::pair{E, RoutingDirection::E},
          std::pair{S, RoutingDirection::S},
          std::pair{W, RoutingDirection::W}}) {
        if (mesh_coord_range.contains(coord)) {
            chip_id_t fabric_chip_id = coord[0] * mesh_shape[1] + coord[1];
            valid_connections.insert(
                {fabric_chip_id,
                 RouterEdge{
                     .port_direction = direction,
                     .connected_chip_ids =
                         std::vector<chip_id_t>(this->chip_spec_.num_eth_ports_per_direction, fabric_chip_id),
                     .weight = 0}});
        }
    }

    return valid_connections;
}

void MeshGraph::initialize_intermesh_mapping(
    std::optional<std::map<FabricNodeId, chip_id_t>> logical_mesh_chip_id_to_physical_chip_id_mapping,
    std::shared_ptr<tt_metal::PhysicalSystemDescriptor> physical_system_descriptor,
    const std::vector<std::unordered_map<port_id_t, chip_id_t, hash_pair>>& mesh_edge_ports_to_chip_id) {
    using namespace tt::tt_metal::distributed::multihost;
    const auto& my_host = physical_system_descriptor->my_host_name();
    const auto my_rank = physical_system_descriptor->get_rank_for_hostname(my_host);
    const auto my_mesh_id =
        tt::tt_metal::MetalContext::instance().get_control_plane().get_logical_node_ids().at(Rank{my_rank}).first;

    PortIdTable port_id_table;
    std::unordered_set<std::string> assigned_port_tags;
    port_id_table[my_mesh_id] = {};

    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    auto chip_unique_ids = cluster.get_unique_chip_ids();

    for (const auto& neighbor_host : physical_system_descriptor->get_host_neighbors(my_host)) {
        auto neighbor_rank = physical_system_descriptor->get_rank_for_hostname(neighbor_host);
        auto neighbor_mesh_id = tt::tt_metal::MetalContext::instance()
                                    .get_control_plane()
                                    .get_logical_node_ids()
                                    .at(Rank{neighbor_rank})
                                    .first;
        // Get all exit nodes between my host and neighbor host
        const auto& exit_nodes = physical_system_descriptor->get_connecting_exit_nodes(my_host, neighbor_host);
        // Determine a valid Direction and Port for each exit node
        // Step 1: Determine which Fabric Node ID each exit node belongs to (assert if exit node is not on edge)
        // Step 2: Compute associative connection hash for the exit node
        // Step 3: Assign an unused direction and logical chan id on that exit node
        // Step 4: Add logical mapping and hash to the port_id_table for the neighbor mesh
        // Step 5: Serialize and forward port id table to controller host
        // Step 6: Pair logical ports across meshes on controller host
        for (const auto& exit_node : exit_nodes) {
            // Step 1
            FabricNodeId exit_node_fabric_node_id(MeshId{0}, 0);
            for (const auto& [physical_chip_id, unique_id] : chip_unique_ids) {
                // TODO: We can maintain a map of unique_id to physical_chip_id for faster lookup
                if (unique_id == *(exit_node.src_exit_node)) {
                    // TODO: This is a double lookup, we can maintain a map of unique_id to fabric_node_id for faster
                    // lookup
                    for (const auto& [fabric_node_id, chip] :
                         logical_mesh_chip_id_to_physical_chip_id_mapping.value()) {
                        if (chip == physical_chip_id) {
                            exit_node_fabric_node_id = fabric_node_id;
                            break;
                        }
                    }
                }
            }
            // Step 2
            // Associative connection hash
            auto assoc_connection_hash = std::hash<tt::tt_metal::ExitNodeConnection>{}(exit_node);
            TT_FATAL(exit_node_fabric_node_id.mesh_id == my_mesh_id, "MeshGraph: Exit node is not on my mesh");
            auto exit_node_chip = exit_node_fabric_node_id.chip_id;
            for (const auto& [port_id, chip_id] : mesh_edge_ports_to_chip_id[*my_mesh_id]) {
                if (chip_id == exit_node_chip) {
                    auto port_direction = port_id.first;
                    auto logical_chan_id = port_id.second;
                    auto port_tag = std::string(enchantum::to_string(port_direction)) + std::to_string(logical_chan_id);
                    // Assign this tag to the exit node if it is not already assigned
                    if (assigned_port_tags.find(port_tag) == assigned_port_tags.end()) {
                        // Step 3
                        assigned_port_tags.insert(port_tag);
                        // Step 4
                        port_id_table[my_mesh_id][neighbor_mesh_id].push_back(
                            PortIdentifier{port_tag, assoc_connection_hash});
                        break;
                    }
                }
            }
        }
    }

    // Step 5
    auto& distributed_context = tt::tt_metal::MetalContext::instance().global_distributed_context();
    std::size_t serialized_table_size = 0;
    std::vector<uint8_t> serialized_table;
    if (my_rank != 0) {
        serialized_table = serialize_to_bytes(port_id_table);
        serialized_table_size = serialized_table.size();
        distributed_context.send(
            tt::stl::Span<std::byte>(
                reinterpret_cast<std::byte*>(&serialized_table_size), sizeof(serialized_table_size)),
            Rank{0},
            Tag{0});
        distributed_context.send(
            tt::stl::as_writable_bytes(tt::stl::Span<uint8_t>(serialized_table.data(), serialized_table.size())),
            Rank{0},
            Tag{0});
    } else {
        for (const auto& hostname : physical_system_descriptor->get_all_hostnames()) {
            if (hostname == my_host) {
                continue;
            }
            auto peer_rank = physical_system_descriptor->get_rank_for_hostname(hostname);
            distributed_context.recv(
                tt::stl::Span<std::byte>(
                    reinterpret_cast<std::byte*>(&serialized_table_size), sizeof(serialized_table_size)),
                Rank{peer_rank},
                Tag{0});
            serialized_table.resize(serialized_table_size);
            distributed_context.recv(
                tt::stl::as_writable_bytes(tt::stl::Span<uint8_t>(serialized_table.data(), serialized_table.size())),
                Rank{peer_rank},
                Tag{0});
            auto peer_port_id_table = deserialize_port_id_table_from_bytes(serialized_table);
            TT_FATAL(
                peer_port_id_table.size() == 1, "MeshGraph: Expecting peer port id table to have exactly one mesh");
            port_id_table[peer_port_id_table.begin()->first] = std::move(peer_port_id_table.begin()->second);
        }
    }
    distributed_context.barrier();
    // Step 6
    std::vector<uint8_t> serialized_connections;
    std::vector<std::tuple<std::pair<uint32_t, std::string>, std::pair<uint32_t, std::string>>> resolved_connections;
    if (my_rank == 0) {
        for (const auto& [src_mesh, port_identifiers] : port_id_table) {
            for (const auto& [dest_mesh, src_ports] : port_identifiers) {
                const auto& dest_ports = port_id_table[dest_mesh][src_mesh];
                // Iterate over src ports. For each src port, determine which dst port it connects to
                for (const auto& src_port : src_ports) {
                    const auto& connection_hash = src_port.connection_hash;
                    for (const auto& dest_port : dest_ports) {
                        if (dest_port.connection_hash == connection_hash) {
                            std::cout << "Connecting: " << *src_mesh << " " << src_port.port_tag << " and "
                                      << *dest_mesh << " " << dest_port.port_tag << std::endl;
                            resolved_connections.push_back(
                                {{*src_mesh, src_port.port_tag}, {*dest_mesh, dest_port.port_tag}});
                            break;
                        }
                    }
                }
            }
        }
        for (const auto& hostname : physical_system_descriptor->get_all_hostnames()) {
            if (hostname == my_host) {
                continue;
            }
            auto peer_rank = physical_system_descriptor->get_rank_for_hostname(hostname);
            serialized_connections = serialize_connections_table_to_bytes(resolved_connections);
            serialized_table_size = serialized_connections.size();
            distributed_context.send(
                tt::stl::Span<std::byte>(
                    reinterpret_cast<std::byte*>(&serialized_table_size), sizeof(serialized_table_size)),
                Rank{peer_rank},
                Tag{0});
            distributed_context.send(
                tt::stl::as_writable_bytes(
                    tt::stl::Span<uint8_t>(serialized_connections.data(), serialized_connections.size())),
                Rank{peer_rank},
                Tag{0});
        }
    } else {
        distributed_context.recv(
            tt::stl::Span<std::byte>(
                reinterpret_cast<std::byte*>(&serialized_table_size), sizeof(serialized_table_size)),
            Rank{0},
            Tag{0});
        serialized_connections.resize(serialized_table_size);
        distributed_context.recv(
            tt::stl::as_writable_bytes(
                tt::stl::Span<uint8_t>(serialized_connections.data(), serialized_connections.size())),
            Rank{0},
            Tag{0});
        resolved_connections = deserialize_connections_table_from_bytes(serialized_connections);
    }
    distributed_context.barrier();
    for (const auto& connection : resolved_connections) {
        auto src_mesh = std::get<0>(connection).first;
        auto dst_mesh = std::get<1>(connection).first;
        auto src_port = std::get<0>(connection).second;
        auto dst_port = std::get<1>(connection).second;
        auto src_chan = static_cast<uint32_t>(std::stoul(src_port.substr(1, src_port.size() - 1)));
        auto dst_chan = static_cast<uint32_t>(std::stoul(dst_port.substr(1, dst_port.size() - 1)));
        auto src_port_dir = enchantum::cast<RoutingDirection>(src_port.substr(0, 1), ttsl::ascii_caseless_comp).value();
        auto dst_port_dir = enchantum::cast<RoutingDirection>(dst_port.substr(0, 1), ttsl::ascii_caseless_comp).value();
        port_id_t src_port_id = {src_port_dir, src_chan};
        port_id_t dst_port_id = {dst_port_dir, dst_chan};
        auto src_chip = mesh_edge_ports_to_chip_id[src_mesh].at(src_port_id);
        auto dst_chip = mesh_edge_ports_to_chip_id[dst_mesh].at(dst_port_id);
        this->add_to_connectivity(MeshId{src_mesh}, src_chip, MeshId{dst_mesh}, dst_chip, src_port_dir);
    }
}

void MeshGraph::initialize_from_yaml(
    const std::string& mesh_graph_desc_file_path,
    std::optional<std::map<FabricNodeId, chip_id_t>> logical_mesh_chip_id_to_physical_chip_id_mapping,
    std::shared_ptr<tt_metal::PhysicalSystemDescriptor> physical_system_descriptor) {
    using namespace tt::tt_metal::distributed::multihost;
    std::ifstream fdesc(mesh_graph_desc_file_path);
    TT_FATAL(not fdesc.fail(), "Failed to open file: {}", mesh_graph_desc_file_path);

    YAML::Node yaml = YAML::LoadFile(mesh_graph_desc_file_path);

    TT_FATAL(yaml["ChipSpec"].IsMap(), "MeshGraph: Expecting yaml to define a ChipSpec as a Map");
    TT_FATAL(yaml["Board"].IsSequence(), "MeshGraph: Expecting yaml to define Board as a Sequence");
    TT_FATAL(yaml["Mesh"].IsSequence(), "MeshGraph: Expecting yaml to define Mesh as a Sequence");
    TT_FATAL(yaml["Graph"].IsSequence(), "MeshGraph: Expecting yaml to define Graph as a Sequence");

    // Parse Chip
    const auto& chip = yaml["ChipSpec"];
    auto arch = enchantum::cast<tt::ARCH>(chip["arch"].as<std::string>(), ttsl::ascii_caseless_comp);
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
        auto fabric_type = enchantum::cast<FabricType>(board["type"].as<std::string>(), ttsl::ascii_caseless_comp);
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
        MeshId mesh_id{mesh["id"].as<std::uint32_t>()};
        if (this->intra_mesh_connectivity_.size() <= *mesh_id) {
            // Resize all variables that loop over mesh_ids
            this->intra_mesh_connectivity_.resize(*mesh_id + 1);
            this->inter_mesh_connectivity_.resize(*mesh_id + 1);
            // Resize mesh_host_ranks_ by adding empty containers
            while (this->mesh_host_ranks_.size() <= *mesh_id) {
                this->mesh_host_ranks_.emplace_back(MeshShape{1, 1}, MeshHostRankId{0});
            }
            mesh_edge_ports_to_chip_id.resize(*mesh_id + 1);
        }
        TT_FATAL(
            board_name_to_topology.find(mesh_board) != board_name_to_topology.end(),
            "MeshGraph: Board not found: {}",
            mesh["board"].as<std::string>());

        // Parse device topology (actual number of chips used)
        TT_FATAL(mesh["device_topology"].IsDefined(), "MeshGraph: Expecting yaml mesh to define device_topology");
        std::uint32_t mesh_ns_size = mesh["device_topology"][0].as<std::uint32_t>();
        std::uint32_t mesh_ew_size = mesh["device_topology"][1].as<std::uint32_t>();

        // Parse host topology (number of boards)
        TT_FATAL(mesh["host_topology"].IsDefined(), "MeshGraph: Expecting yaml mesh to define host_topology");
        std::uint32_t mesh_board_ns_size = mesh["host_topology"][0].as<std::uint32_t>();
        std::uint32_t mesh_board_ew_size = mesh["host_topology"][1].as<std::uint32_t>();

        std::uint32_t board_ns_size = board_name_to_topology[mesh_board][0];
        std::uint32_t board_ew_size = board_name_to_topology[mesh_board][1];

        // Assert that device topology is divisible by board topology
        TT_FATAL(
            mesh_ns_size % board_ns_size == 0 and mesh_ew_size % board_ew_size == 0,
            "MeshGraph: Device topology size {}x{} must be divisible by board topology size {}x{}",
            mesh_ns_size,
            mesh_ew_size,
            board_ns_size,
            board_ew_size);

        // Assert that device topology aligns with host topology and board topology
        TT_FATAL(
            mesh_ns_size == mesh_board_ns_size * board_ns_size and mesh_ew_size == mesh_board_ew_size * board_ew_size,
            "MeshGraph: Device topology size {}x{} must equal host topology size {}x{} * board topology size {}x{}",
            mesh_ns_size,
            mesh_ew_size,
            mesh_board_ns_size,
            mesh_board_ew_size,
            board_ns_size,
            board_ew_size);

        std::uint32_t mesh_size = mesh_ns_size * mesh_ew_size;
        MeshShape mesh_shape(mesh_ns_size, mesh_ew_size);
        std::vector<chip_id_t> chip_ids(mesh_size);
        std::iota(chip_ids.begin(), chip_ids.end(), 0);
        this->mesh_to_chip_ids_.emplace(*mesh_id, MeshContainer<chip_id_t>(mesh_shape, chip_ids));

        // Assign ranks in row-major order based on host topology.
        std::vector<MeshHostRankId> mesh_host_ranks_values;
        uint32_t next_rank = 0;
        for (const auto& host_coord : MeshCoordinateRange(MeshShape(mesh_board_ns_size, mesh_board_ew_size))) {
            mesh_host_ranks_values.push_back(MeshHostRankId{next_rank++});
            mesh_host_rank_coord_ranges_.emplace(
                std::make_pair(*mesh_id, mesh_host_ranks_values.back()),
                MeshCoordinateRange(
                    MeshCoordinate(host_coord[0] * board_ns_size, host_coord[1] * board_ew_size),
                    MeshCoordinate((host_coord[0] + 1) * board_ns_size - 1, (host_coord[1] + 1) * board_ew_size - 1)));
        }

        this->mesh_host_ranks_[*mesh_id] =
            MeshContainer<MeshHostRankId>(MeshShape(mesh_board_ns_size, mesh_board_ew_size), mesh_host_ranks_values);

        // Fill in connectivity for Mesh
        MeshCoordinateRange mesh_coord_range(mesh_shape);
        this->intra_mesh_connectivity_[*mesh_id].resize(mesh_size);
        for (const auto& src_mesh_coord : mesh_coord_range) {
            // Get the chip id for the current mesh coordinate
            chip_id_t src_chip_id = src_mesh_coord[0] * mesh_shape[1] + src_mesh_coord[1];
            // Get the valid connections for the current chip
            this->intra_mesh_connectivity_[*mesh_id][src_chip_id] =
                this->get_valid_connections(src_mesh_coord, mesh_coord_range, board_name_to_fabric_type[mesh_board]);
        }

        this->inter_mesh_connectivity_[*mesh_id].resize(this->intra_mesh_connectivity_[*mesh_id].size());

        // Print Mesh
        std::stringstream ss;
        for (int i = 0; i < mesh_size; i++) {
            if (i % mesh_ew_size == 0) {
                ss << std::endl;
            }
            ss << " " << std::setfill('0') << std::setw(2) << i;
        }
        log_debug(tt::LogFabric, "Mesh Graph: Mesh {} Logical Device Ids {}", *mesh_id, ss.str());
        for (const auto& [key, coords] : this->mesh_host_rank_coord_ranges_) {
            const auto& [current_mesh_id, host_rank] = key;
            if (current_mesh_id == mesh_id) {
                log_debug(
                    tt::LogFabric,
                    "Mesh Graph: Mesh {} Host Rank {} Start: {}, End: {}",
                    *mesh_id,
                    *host_rank,
                    coords.start_coord(),
                    coords.end_coord());
                for (auto it = coords.begin(); it != coords.end(); ++it) {
                    log_debug(
                        tt::LogFabric, "\t{} -> Chip: {}", *it, this->mesh_to_chip_ids_.at(current_mesh_id).at(*it));
                }
            }
        }

        // Get the edge ports of each mesh
        // North, start from NW corner
        std::uint32_t chan_id = 0;
        for (std::uint32_t chip_id = 0; chip_id < mesh_ew_size; chip_id++) {
            for (std::uint32_t i = 0; i < this->chip_spec_.num_eth_ports_per_direction; i++) {
                mesh_edge_ports_to_chip_id[*mesh_id][{RoutingDirection::N, chan_id++}] = chip_id;
            }
        }
        // South, start from SW corner
        chan_id = 0;
        for (std::uint32_t chip_id = (mesh_size - mesh_ew_size); chip_id < mesh_size; chip_id++) {
            for (std::uint32_t i = 0; i < this->chip_spec_.num_eth_ports_per_direction; i++) {
                mesh_edge_ports_to_chip_id[*mesh_id][{RoutingDirection::S, chan_id++}] = chip_id;
            }
        }
        // East, start from NE corner
        chan_id = 0;
        for (std::uint32_t chip_id = (mesh_ew_size - 1); chip_id < mesh_size; chip_id += mesh_ew_size) {
            for (std::uint32_t i = 0; i < this->chip_spec_.num_eth_ports_per_direction; i++) {
                mesh_edge_ports_to_chip_id[*mesh_id][{RoutingDirection::E, chan_id++}] = chip_id;
            }
        }
        // WEST, start from NW corner
        chan_id = 0;
        for (std::uint32_t chip_id = 0; chip_id < mesh_size; chip_id += mesh_ew_size) {
            for (std::uint32_t i = 0; i < this->chip_spec_.num_eth_ports_per_direction; i++) {
                mesh_edge_ports_to_chip_id[*mesh_id][{RoutingDirection::W, chan_id++}] = chip_id;
            }
        }
    }
    std::vector<std::tuple<std::pair<uint32_t, std::string>, std::pair<uint32_t, std::string>>> connections;
    if (logical_mesh_chip_id_to_physical_chip_id_mapping.has_value()) {
        initialize_intermesh_mapping(
            logical_mesh_chip_id_to_physical_chip_id_mapping, physical_system_descriptor, mesh_edge_ports_to_chip_id);
        return;
    }
    // Loop over Graph, populate inter mesh
    auto convert_yaml_to_port_id = [](const YAML::Node& node) -> std::pair<MeshId, port_id_t> {
        MeshId mesh_id{node[0].as<std::uint32_t>()};
        std::string port_string = node[1].as<std::string>();
        RoutingDirection port_direction =
            enchantum::cast<RoutingDirection>(port_string.substr(0, 1), ttsl::ascii_caseless_comp).value();
        std::uint32_t chan_id = static_cast<uint32_t>(std::stoul(port_string.substr(1, port_string.size() - 1)));
        return {mesh_id, {port_direction, chan_id}};
    };
    for (const auto& mesh_connection : yaml["Graph"]) {
        TT_FATAL(mesh_connection.size() == 2, "MeshGraph: Expecting 2 elements in each Graph connection");
        const auto& [src_mesh_id, src_port_id] = convert_yaml_to_port_id(mesh_connection[0]);
        const auto& [dst_mesh_id, dst_port_id] = convert_yaml_to_port_id(mesh_connection[1]);
        const auto& src_chip_id = mesh_edge_ports_to_chip_id[*src_mesh_id].at(src_port_id);
        const auto& dst_chip_id = mesh_edge_ports_to_chip_id[*dst_mesh_id].at(dst_port_id);
        this->add_to_connectivity(src_mesh_id, src_chip_id, dst_mesh_id, dst_chip_id, src_port_id.first);
    }
}

void MeshGraph::print_connectivity() const {
    std::stringstream ss;
    ss << " Mesh Graph:  Intra Mesh Connectivity: " << std::endl;
    for (uint32_t mesh_id_val = 0; mesh_id_val < this->intra_mesh_connectivity_.size(); mesh_id_val++) {
        ss << "M" << mesh_id_val << ":" << std::endl;
        for (uint32_t chip_id = 0; chip_id < this->intra_mesh_connectivity_[mesh_id_val].size(); chip_id++) {
            ss << "   D" << chip_id << ": ";
            for (auto [connected_chip_id, edge] : this->intra_mesh_connectivity_[mesh_id_val][chip_id]) {
                for (int i = 0; i < edge.connected_chip_ids.size(); i++) {
                    ss << edge.connected_chip_ids[i] << "(" << enchantum::to_string(edge.port_direction) << ", "
                       << edge.weight << ") ";
                }
            }
            ss << std::endl;
        }
    }
    log_debug(tt::LogFabric, "{}", ss.str());
    ss.str(std::string());
    ss << " Mesh Graph:  Inter Mesh Connectivity: " << std::endl;
    for (uint32_t mesh_id_val = 0; mesh_id_val < this->inter_mesh_connectivity_.size(); mesh_id_val++) {
        ss << "M" << mesh_id_val << ":" << std::endl;
        for (uint32_t chip_id = 0; chip_id < this->inter_mesh_connectivity_[mesh_id_val].size(); chip_id++) {
            ss << "   D" << chip_id << ": ";
            for (auto [connected_mesh_id, edge] : this->inter_mesh_connectivity_[mesh_id_val][chip_id]) {
                for (int i = 0; i < edge.connected_chip_ids.size(); i++) {
                    ss << "M" << *connected_mesh_id << "D" << edge.connected_chip_ids[i] << "("
                       << enchantum::to_string(edge.port_direction) << ", " << edge.weight << ") ";
                }
            }
            ss << std::endl;
        }
    }
    log_debug(tt::LogFabric, "{}", ss.str());
}

void MeshGraph::validate_mesh_id(MeshId mesh_id) const {
    TT_FATAL(
        this->mesh_to_chip_ids_.find(mesh_id) != this->mesh_to_chip_ids_.end(),
        "MeshGraph: mesh_id {} not found",
        mesh_id);
}

MeshShape MeshGraph::get_mesh_shape(MeshId mesh_id, std::optional<MeshHostRankId> host_rank) const {
    this->validate_mesh_id(mesh_id);

    if (host_rank.has_value()) {
        return this->mesh_host_rank_coord_ranges_.at(std::make_pair(mesh_id, *host_rank)).shape();
    }

    return this->mesh_to_chip_ids_.at(mesh_id).shape();
}

MeshCoordinateRange MeshGraph::get_coord_range(MeshId mesh_id, std::optional<MeshHostRankId> host_rank) const {
    this->validate_mesh_id(mesh_id);

    if (host_rank.has_value()) {
        auto it = this->mesh_host_rank_coord_ranges_.find(std::make_pair(mesh_id, *host_rank));
        TT_FATAL(
            it != this->mesh_host_rank_coord_ranges_.end(),
            "MeshGraph: host_rank {} not found for mesh {}",
            *host_rank,
            *mesh_id);
        return it->second;
    }
    auto mesh_shape = this->mesh_to_chip_ids_.at(mesh_id).shape();
    return MeshCoordinateRange(mesh_shape);
}

const IntraMeshConnectivity& MeshGraph::get_intra_mesh_connectivity() const { return intra_mesh_connectivity_; }
const InterMeshConnectivity& MeshGraph::get_inter_mesh_connectivity() const { return inter_mesh_connectivity_; }

std::vector<MeshId> MeshGraph::get_mesh_ids() const {
    std::vector<MeshId> mesh_ids;
    mesh_ids.reserve(this->mesh_to_chip_ids_.size());
    for (const auto& [mesh_id, _] : this->mesh_to_chip_ids_) {
        mesh_ids.push_back(mesh_id);
    }
    return mesh_ids;
}

MeshContainer<chip_id_t> MeshGraph::get_chip_ids(MeshId mesh_id, std::optional<MeshHostRankId> host_rank) const {
    auto it = mesh_to_chip_ids_.find(mesh_id);
    TT_FATAL(it != mesh_to_chip_ids_.end(), "MeshGraph: mesh_id {} not found", mesh_id);

    if (!host_rank.has_value()) {
        // Return the entire mesh
        return it->second;
    }

    // Return submesh for the specific host rank
    MeshCoordinateRange coord_range = get_coord_range(mesh_id, host_rank);
    MeshShape submesh_shape = coord_range.shape();

    std::vector<chip_id_t> submesh_chip_ids;
    submesh_chip_ids.reserve(submesh_shape.mesh_size());

    for (const auto& coord : coord_range) {
        submesh_chip_ids.push_back(it->second.at(coord));
    }

    return MeshContainer<chip_id_t>(submesh_shape, submesh_chip_ids);
}

MeshCoordinate MeshGraph::chip_to_coordinate(MeshId mesh_id, chip_id_t chip_id) const {
    const auto& mesh_shape = this->mesh_to_chip_ids_.at(mesh_id).shape();
    int ns = chip_id / mesh_shape[1];
    int ew = chip_id % mesh_shape[1];
    return MeshCoordinate(ns, ew);
}

chip_id_t MeshGraph::coordinate_to_chip(MeshId mesh_id, MeshCoordinate coordinate) const {
    const auto& mesh_shape = this->mesh_to_chip_ids_.at(mesh_id).shape();
    return coordinate[0] * mesh_shape[1] + coordinate[1];
}

std::optional<MeshHostRankId> MeshGraph::get_host_rank_for_chip(MeshId mesh_id, chip_id_t chip_id) const {
    auto it = mesh_to_chip_ids_.find(mesh_id);
    if (it == mesh_to_chip_ids_.end()) {
        return std::nullopt;
    }

    // Convert chip_id to mesh coordinates
    MeshCoordinate chip_coord = this->chip_to_coordinate(mesh_id, chip_id);

    // Find which host rank owns this coordinate
    for (const auto& [mesh_id_host_rank_pair, coord_range] : mesh_host_rank_coord_ranges_) {
        if (mesh_id_host_rank_pair.first == mesh_id && chip_coord[0] >= coord_range.start_coord()[0] &&
            chip_coord[0] <= coord_range.end_coord()[0] && chip_coord[1] >= coord_range.start_coord()[1] &&
            chip_coord[1] <= coord_range.end_coord()[1]) {
            return mesh_id_host_rank_pair.second;
        }
    }

    return std::nullopt;
}

const MeshContainer<MeshHostRankId>& MeshGraph::get_host_ranks(MeshId mesh_id) const {
    return mesh_host_ranks_[*mesh_id];
}

std::filesystem::path MeshGraph::get_mesh_graph_descriptor_path_for_cluster_type(
    tt::tt_metal::ClusterType cluster_type, const std::string& root_dir) {
    auto it = cluster_type_to_mesh_graph_descriptor.get().find(cluster_type);
    if (it != cluster_type_to_mesh_graph_descriptor.get().end()) {
        return std::filesystem::path(root_dir) / MESH_GRAPH_DESCRIPTOR_DIR / it->second;
    }
    TT_THROW("Cannot find mesh graph descriptor for cluster type {}", cluster_type);
}

}  // namespace tt::tt_fabric
