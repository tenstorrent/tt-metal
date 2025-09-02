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
#include <impl/context/metal_context.hpp>
#include "tt_metal/fabric/physical_system_descriptor.hpp"

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
std::string get_ubb_edge_label(int tray_id, int asic_location, int chan_id, MeshShape topology) {
    std::map<std::tuple<int, int, int>, std::string> label_map;

    if (topology == MeshShape{8, 4}) {
        // Tray 1 mappings
        label_map[{1, 1, 4}] = "N0";
        label_map[{1, 1, 5}] = "N1";
        label_map[{1, 1, 6}] = "N2";
        label_map[{1, 1, 7}] = "N3";
        label_map[{1, 5, 4}] = "N4";
        label_map[{1, 5, 5}] = "N5";
        label_map[{1, 5, 6}] = "N6";
        label_map[{1, 5, 7}] = "N7";
        label_map[{1, 1, 0}] = "W0";
        label_map[{1, 1, 1}] = "W1";
        label_map[{1, 1, 2}] = "W2";
        label_map[{1, 1, 3}] = "W3";
        label_map[{1, 2, 0}] = "W4";
        label_map[{1, 2, 1}] = "W5";
        label_map[{1, 2, 2}] = "W6";
        label_map[{1, 2, 3}] = "W7";
        label_map[{1, 3, 0}] = "W8";
        label_map[{1, 3, 1}] = "W9";
        label_map[{1, 3, 2}] = "W10";
        label_map[{1, 3, 3}] = "W11";
        label_map[{1, 4, 0}] = "W12";
        label_map[{1, 4, 1}] = "W13";
        label_map[{1, 4, 2}] = "W14";
        label_map[{1, 4, 3}] = "W15";

        // Tray 3 mappings
        label_map[{3, 1, 4}] = "S0";
        label_map[{3, 1, 5}] = "S1";
        label_map[{3, 1, 6}] = "S2";
        label_map[{3, 1, 7}] = "S3";
        label_map[{3, 5, 4}] = "S4";
        label_map[{3, 5, 5}] = "S5";
        label_map[{3, 5, 6}] = "S6";
        label_map[{3, 5, 7}] = "S7";
        label_map[{3, 4, 0}] = "W16";
        label_map[{3, 4, 1}] = "W17";
        label_map[{3, 4, 2}] = "W18";
        label_map[{3, 4, 3}] = "W19";
        label_map[{3, 3, 0}] = "W20";
        label_map[{3, 3, 1}] = "W21";
        label_map[{3, 3, 2}] = "W22";
        label_map[{3, 3, 3}] = "W23";
        label_map[{3, 2, 0}] = "W24";
        label_map[{3, 2, 1}] = "W25";
        label_map[{3, 2, 2}] = "W26";
        label_map[{3, 2, 3}] = "W27";
        label_map[{3, 1, 0}] = "W28";
        label_map[{3, 1, 1}] = "W29";
        label_map[{3, 1, 2}] = "W30";
        label_map[{3, 1, 3}] = "W31";

        // Tray 2 mappings
        label_map[{2, 5, 4}] = "N8";
        label_map[{2, 5, 5}] = "N9";
        label_map[{2, 5, 6}] = "N10";
        label_map[{2, 5, 7}] = "N11";
        label_map[{2, 1, 4}] = "N12";
        label_map[{2, 1, 5}] = "N13";
        label_map[{2, 1, 6}] = "N14";
        label_map[{2, 1, 7}] = "N15";
        label_map[{2, 1, 0}] = "E0";
        label_map[{2, 1, 1}] = "E1";
        label_map[{2, 1, 2}] = "E2";
        label_map[{2, 1, 3}] = "E3";
        label_map[{2, 2, 0}] = "E4";
        label_map[{2, 2, 1}] = "E5";
        label_map[{2, 2, 2}] = "E6";
        label_map[{2, 2, 3}] = "E7";
        label_map[{2, 3, 0}] = "E8";
        label_map[{2, 3, 1}] = "E9";
        label_map[{2, 3, 2}] = "E10";
        label_map[{2, 3, 3}] = "E11";
        label_map[{2, 4, 0}] = "E12";
        label_map[{2, 4, 1}] = "E13";
        label_map[{2, 4, 2}] = "E14";
        label_map[{2, 4, 3}] = "E15";

        // Tray 4 mappings
        label_map[{4, 5, 4}] = "S8";
        label_map[{4, 5, 5}] = "S9";
        label_map[{4, 5, 6}] = "S10";
        label_map[{4, 5, 7}] = "S11";
        label_map[{4, 1, 4}] = "S12";
        label_map[{4, 1, 5}] = "S13";
        label_map[{4, 1, 6}] = "S14";
        label_map[{4, 1, 7}] = "S15";
        label_map[{4, 4, 0}] = "E16";
        label_map[{4, 4, 1}] = "E17";
        label_map[{4, 4, 2}] = "E18";
        label_map[{4, 4, 3}] = "E19";
        label_map[{4, 3, 0}] = "E20";
        label_map[{4, 3, 1}] = "E21";
        label_map[{4, 3, 2}] = "E22";
        label_map[{4, 3, 3}] = "E23";
        label_map[{4, 2, 0}] = "E24";
        label_map[{4, 2, 1}] = "E25";
        label_map[{4, 2, 2}] = "E26";
        label_map[{4, 2, 3}] = "E27";
        label_map[{4, 1, 0}] = "E28";
        label_map[{4, 1, 1}] = "E29";
        label_map[{4, 1, 2}] = "E30";
        label_map[{4, 1, 3}] = "E31";
    } else if (topology == MeshShape{1, 32}) {
        // only contains mapping for channels 0, 1, 2, 3
        // Assumes the 1x32 will use mapping of
        // T1 N1  T1 N2  T1 N3  T1 N4  T1 N8  T1 N7  T1 N6  T1 N5
        // T2 N5  T2 N1  T2 N2  T2 N3  T2 N4  T4 N4 T4 N3  T4 N2
        // T4 N1  T4 N5  T3 N5  T3 N1  T3 N2  T3 N3  T3 N4  T3 N8
        // T3 N7  T3 N6  T4 N6  T4 N7  T4 N8  T2 N8  T2 N7  T2 N6
        label_map[{1, 1, 0}] = "N0";
        label_map[{1, 1, 1}] = "N1";
        label_map[{1, 1, 2}] = "N2";
        label_map[{1, 1, 3}] = "N3";
        label_map[{1, 2, 0}] = "N4";
        label_map[{1, 2, 1}] = "N5";
        label_map[{1, 2, 2}] = "N6";
        label_map[{1, 2, 3}] = "N7";
        label_map[{1, 3, 0}] = "N8";
        label_map[{1, 3, 1}] = "N9";
        label_map[{1, 3, 2}] = "N10";
        label_map[{1, 3, 3}] = "N11";
        label_map[{1, 4, 0}] = "N12";
        label_map[{1, 4, 1}] = "N13";
        label_map[{1, 4, 2}] = "N14";
        label_map[{1, 4, 3}] = "N15";

        // Tray 2 mappings
        label_map[{2, 1, 0}] = "N36";
        label_map[{2, 1, 1}] = "N37";
        label_map[{2, 1, 2}] = "N38";
        label_map[{2, 1, 3}] = "N39";
        label_map[{2, 2, 0}] = "N40";
        label_map[{2, 2, 1}] = "N41";
        label_map[{2, 2, 2}] = "N42";
        label_map[{2, 2, 3}] = "N43";
        label_map[{2, 3, 0}] = "N44";
        label_map[{2, 3, 1}] = "N45";
        label_map[{2, 3, 2}] = "N46";
        label_map[{2, 3, 3}] = "N47";
        label_map[{2, 4, 0}] = "N48";
        label_map[{2, 4, 1}] = "N49";
        label_map[{2, 4, 2}] = "N50";
        label_map[{2, 4, 3}] = "N51";

        // Tray 4 mappings
        label_map[{4, 4, 0}] = "N52";
        label_map[{4, 4, 1}] = "N53";
        label_map[{4, 4, 2}] = "N54";
        label_map[{4, 4, 3}] = "N55";
        label_map[{4, 3, 0}] = "N56";
        label_map[{4, 3, 1}] = "N57";
        label_map[{4, 3, 2}] = "N58";
        label_map[{4, 3, 3}] = "N59";
        label_map[{4, 2, 0}] = "N60";
        label_map[{4, 2, 1}] = "N61";
        label_map[{4, 2, 2}] = "N62";
        label_map[{4, 2, 3}] = "N63";
        label_map[{4, 1, 0}] = "N64";
        label_map[{4, 1, 1}] = "N65";
        label_map[{4, 1, 2}] = "N66";
        label_map[{4, 1, 3}] = "N67";

        // Tray 3 mappings
        label_map[{3, 1, 0}] = "N76";
        label_map[{3, 1, 1}] = "N77";
        label_map[{3, 1, 2}] = "N78";
        label_map[{3, 1, 3}] = "N79";
        label_map[{3, 2, 0}] = "N80";
        label_map[{3, 2, 1}] = "N81";
        label_map[{3, 2, 2}] = "N82";
        label_map[{3, 2, 3}] = "N83";
        label_map[{3, 3, 0}] = "N84";
        label_map[{3, 3, 1}] = "N85";
        label_map[{3, 3, 2}] = "N86";
        label_map[{3, 3, 3}] = "N87";
        label_map[{3, 4, 0}] = "N88";
        label_map[{3, 4, 1}] = "N89";
        label_map[{3, 4, 2}] = "N90";
        label_map[{3, 4, 3}] = "N91";
    }

    auto key = std::make_tuple(tray_id, asic_location, chan_id);
    auto it = label_map.find(key);
    if (it == label_map.end()) {
        throw std::runtime_error(
            "No label found for tray_id: " + std::to_string(tray_id) +
            ", asic_location: " + std::to_string(asic_location) + ", chan_id: " + std::to_string(chan_id));
    }
    return it->second;
}

YAML::Node get_mgd_graph_from_physical_system_descriptor(
    tt::tt_metal::PhysicalSystemDescriptor& physical_system_desc,
    MeshShape topology,
    std::uint32_t num_eth_ports_per_direction) {
    YAML::Node mgd_graph;
    const auto& host_to_rank = physical_system_desc.get_host_to_rank_map();
    const auto& system_graph = physical_system_desc.get_system_graph();
    const auto& asic_descriptors = physical_system_desc.get_asic_descriptors();
    auto hostnames = physical_system_desc.get_all_hostnames();
    const auto& distributed_context = tt_metal::distributed::multihost::DistributedContext::get_current_world();

    for (const auto& src_host : hostnames) {
        for (const auto& dst_host : hostnames) {
            if (src_host == dst_host) {
                continue;
            }
            auto exit_nodes = physical_system_desc.get_connecting_exit_nodes(src_host, dst_host);
            // Group exit nodes by src_asic_desc.asic_location, currently do not support intramesh to intramesh
            // connections mismatching
            std::map<tt_metal::TrayID, std::map<tt_metal::ASICLocation, std::vector<tt_metal::ExitNodeConnection>>>
                exit_nodes_by_src_asic_location;
            for (const auto& exit_node : exit_nodes) {
                auto src_asic = exit_node.src_exit_node;
                auto src_asic_desc = asic_descriptors.at(src_asic);
                exit_nodes_by_src_asic_location[src_asic_desc.tray_id][src_asic_desc.asic_location].push_back(
                    exit_node);
            }
            if (*(distributed_context->rank()) == 0) {
                log_info(tt::LogFabric, "Detected {} exit nodes from {} to {}", exit_nodes.size(), src_host, dst_host);
            }
            for (const auto& [src_tray_id, src_asic_location_exit_nodes] : exit_nodes_by_src_asic_location) {
                for (const auto& [src_asic_location, exit_nodes] : src_asic_location_exit_nodes) {
                    for (const auto& exit_node : exit_nodes) {
                        if (exit_nodes.size() == num_eth_ports_per_direction) {
                            auto src_asic = exit_node.src_exit_node;
                            auto src_chan = exit_node.eth_conn.src_chan;
                            auto dst_asic = exit_node.dst_exit_node;
                            auto dst_chan = exit_node.eth_conn.dst_chan;
                            auto src_asic_desc = asic_descriptors.at(src_asic);
                            auto dst_asic_desc = asic_descriptors.at(dst_asic);

                            YAML::Node port_0 = YAML::Node(YAML::NodeType::Sequence);
                            port_0.SetStyle(YAML::EmitterStyle::Flow);
                            port_0.push_back(host_to_rank.at(src_asic_desc.host_name));
                            port_0.push_back(get_ubb_edge_label(
                                *src_asic_desc.tray_id, *src_asic_desc.asic_location, src_chan, topology));
                            YAML::Node port_1 = YAML::Node(YAML::NodeType::Sequence);
                            port_1.SetStyle(YAML::EmitterStyle::Flow);
                            port_1.push_back(host_to_rank.at(dst_asic_desc.host_name));
                            port_1.push_back(get_ubb_edge_label(
                                *dst_asic_desc.tray_id, *dst_asic_desc.asic_location, dst_chan, topology));

                            YAML::Node connection_pair_fwd = YAML::Node(YAML::NodeType::Sequence);
                            connection_pair_fwd.SetStyle(YAML::EmitterStyle::Flow);
                            connection_pair_fwd.push_back(port_0);
                            connection_pair_fwd.push_back(port_1);

                            mgd_graph.push_back(connection_pair_fwd);
                        }
                    }
                }
            }
        }
    }

    std::ofstream fout("mgd_graph.yaml");
    fout << mgd_graph;
    fout.close();
    return mgd_graph;
}
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

MeshGraph::MeshGraph(const std::string& mesh_graph_desc_file_path) {
    this->initialize_from_yaml(mesh_graph_desc_file_path);
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

void MeshGraph::load_intermesh_connections(const AnnotatedIntermeshConnections& intermesh_connections) {
    for (const auto& connection : intermesh_connections) {
        auto src_mesh = std::get<0>(connection).first;
        auto dst_mesh = std::get<1>(connection).first;
        auto src_port = std::get<0>(connection).second;
        auto dst_port = std::get<1>(connection).second;
        auto src_port_dir = src_port.first;
        auto src_chip = mesh_edge_ports_to_chip_id_[src_mesh].at(src_port);
        auto dst_chip = mesh_edge_ports_to_chip_id_[dst_mesh].at(dst_port);

        this->add_to_connectivity(MeshId{src_mesh}, src_chip, MeshId{dst_mesh}, dst_chip, src_port_dir);
    }
}

const std::unordered_map<uint32_t, std::unordered_map<uint32_t, uint32_t>>&
MeshGraph::get_requested_intermesh_connections() const {
    return requested_intermesh_connections_;
}

const std::unordered_map<uint32_t, std::unordered_map<uint32_t, std::vector<std::tuple<uint32_t, uint32_t, uint32_t>>>>&
MeshGraph::get_requested_intermesh_ports() const {
    return requested_intermesh_ports_;
}

const std::vector<std::unordered_map<port_id_t, chip_id_t, hash_pair>>& MeshGraph::get_mesh_edge_ports_to_chip_id()
    const {
    return mesh_edge_ports_to_chip_id_;
}

void MeshGraph::initialize_from_yaml(const std::string& mesh_graph_desc_file_path) {
    legacy_mode_ = true;
    std::ifstream fdesc(mesh_graph_desc_file_path);
    TT_FATAL(not fdesc.fail(), "Failed to open file: {}", mesh_graph_desc_file_path);

    YAML::Node yaml = YAML::LoadFile(mesh_graph_desc_file_path);

    TT_FATAL(yaml["ChipSpec"].IsMap(), "MeshGraph: Expecting yaml to define a ChipSpec as a Map");
    TT_FATAL(yaml["Board"].IsSequence(), "MeshGraph: Expecting yaml to define Board as a Sequence");
    TT_FATAL(yaml["Mesh"].IsSequence(), "MeshGraph: Expecting yaml to define Mesh as a Sequence");
    TT_FATAL(
        yaml["RelaxedGraph"].IsSequence() || yaml["Graph"].IsSequence(),
        "MeshGraph: Expecting yaml to define RelaxedGraph or Graph as a Sequence");

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
            mesh_edge_ports_to_chip_id_.resize(*mesh_id + 1);
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
            for (std::uint32_t i = 0; i < chip_spec_.num_eth_ports_per_direction; i++) {
                mesh_edge_ports_to_chip_id_[*mesh_id][{RoutingDirection::N, chan_id++}] = chip_id;
            }
        }
        // South, start from SW corner
        chan_id = 0;
        for (std::uint32_t chip_id = (mesh_size - mesh_ew_size); chip_id < mesh_size; chip_id++) {
            for (std::uint32_t i = 0; i < chip_spec_.num_eth_ports_per_direction; i++) {
                mesh_edge_ports_to_chip_id_[*mesh_id][{RoutingDirection::S, chan_id++}] = chip_id;
            }
        }
        // East, start from NE corner
        chan_id = 0;
        for (std::uint32_t chip_id = (mesh_ew_size - 1); chip_id < mesh_size; chip_id += mesh_ew_size) {
            for (std::uint32_t i = 0; i < chip_spec_.num_eth_ports_per_direction; i++) {
                mesh_edge_ports_to_chip_id_[*mesh_id][{RoutingDirection::E, chan_id++}] = chip_id;
            }
        }
        // WEST, start from NW corner
        chan_id = 0;
        for (std::uint32_t chip_id = 0; chip_id < mesh_size; chip_id += mesh_ew_size) {
            for (std::uint32_t i = 0; i < chip_spec_.num_eth_ports_per_direction; i++) {
                mesh_edge_ports_to_chip_id_[*mesh_id][{RoutingDirection::W, chan_id++}] = chip_id;
            }
        }
    }
    std::vector<std::tuple<std::pair<uint32_t, std::string>, std::pair<uint32_t, std::string>>> connections;

    TT_FATAL(
        !(yaml["RelaxedGraph"] && yaml["Graph"]),
        "Mesh Graph Descriptor cannot specify both RelaxedGraph and Graph connections.");
    if (yaml["RelaxedGraph"]) {
        // Specify the number of active channels between meshes
        for (const auto& connection : yaml["RelaxedGraph"]) {
            auto src_mesh_str = connection[0].as<std::string>();
            auto dst_mesh_str = connection[1].as<std::string>();
            auto num_chans = connection[2].as<uint32_t>();
            auto src_mesh = static_cast<uint32_t>(std::stoul(src_mesh_str.substr(1, src_mesh_str.size() - 1)));
            auto dst_mesh = static_cast<uint32_t>(std::stoul(dst_mesh_str.substr(1, dst_mesh_str.size() - 1)));
            requested_intermesh_connections_[src_mesh][dst_mesh] = num_chans;
            requested_intermesh_connections_[dst_mesh][src_mesh] = num_chans;
        }
    } else {
        // Specify the number of active channels between specific logical devices across meshes
        TT_FATAL(yaml["Graph"], "Mesh Graph Descriptor must specify either RelaxedGraph or Graph connections.");
        for (const auto& connection : yaml["Graph"]) {
            auto src_mesh_str = connection[0][0].as<std::string>();
            auto dst_mesh_str = connection[1][0].as<std::string>();
            auto src_device_str = connection[0][1].as<std::string>();
            auto dst_device_str = connection[1][1].as<std::string>();
            auto num_chans = connection[2].as<uint32_t>();

            auto src_mesh = static_cast<uint32_t>(std::stoul(src_mesh_str.substr(1, src_mesh_str.size() - 1)));
            auto dst_mesh = static_cast<uint32_t>(std::stoul(dst_mesh_str.substr(1, dst_mesh_str.size() - 1)));
            auto src_device = static_cast<uint32_t>(std::stoul(src_device_str.substr(1, src_device_str.size() - 1)));
            auto dst_device = static_cast<uint32_t>(std::stoul(dst_device_str.substr(1, dst_device_str.size() - 1)));

            requested_intermesh_ports_[src_mesh][dst_mesh].push_back({src_device, dst_device, num_chans});
            requested_intermesh_ports_[dst_mesh][src_mesh].push_back({dst_device, src_device, num_chans});
        }
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
