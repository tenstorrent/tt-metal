// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mesh_graph.hpp"
#include "fabric_host_utils.hpp"

#include <enchantum/enchantum.hpp>
#include <yaml-cpp/yaml.h>
#include <array>
#include <fstream>
#include <iomanip>
#include <optional>

#include <tt_stl/assert.hpp>
#include <tt-logger/tt-logger.hpp>
#include <llrt/tt_cluster.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>
#include <tt_stl/indestructible.hpp>
#include <tt_stl/caseless_comparison.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_graph_descriptor.hpp>
#include <protobuf/mesh_graph_descriptor.pb.h>

// Implementation of hash function for port_id_t
std::size_t std::hash<tt::tt_fabric::port_id_t>::operator()(const tt::tt_fabric::port_id_t& p) const {
    return tt::stl::hash::hash_objects_with_default_seed(p.first, p.second);
}

namespace tt::tt_fabric {

constexpr const char* MESH_GRAPH_DESCRIPTOR_DIR = "tt_metal/fabric/mesh_graph_descriptors";

RoutingDirection routing_direction_to_port_direction(const proto::RoutingDirection& routing_direction) {
    switch (routing_direction) {
        case proto::RoutingDirection::N: return RoutingDirection::N;
        case proto::RoutingDirection::E: return RoutingDirection::E;
        case proto::RoutingDirection::S: return RoutingDirection::S;
        case proto::RoutingDirection::W: return RoutingDirection::W;
        case proto::RoutingDirection::C: return RoutingDirection::C;
        case proto::RoutingDirection::NONE: return RoutingDirection::NONE;
        default: TT_THROW("Invalid routing direction: {}", routing_direction);
    }
}


using ClusterToDescriptorMap = std::unordered_map<tt::tt_metal::ClusterType, std::string_view>;
using FabricToClusterDescriptorMap = std::unordered_map<tt::tt_fabric::FabricType, ClusterToDescriptorMap>;

const tt::stl::Indestructible<FabricToClusterDescriptorMap>& cluster_type_to_mesh_graph_descriptor =
    tt::stl::Indestructible<FabricToClusterDescriptorMap>(FabricToClusterDescriptorMap{
        {tt::tt_fabric::FabricType::MESH,
         ClusterToDescriptorMap{
             {tt::tt_metal::ClusterType::N150, "n150_mesh_graph_descriptor.textproto"},
             {tt::tt_metal::ClusterType::N300, "n300_mesh_graph_descriptor.textproto"},
             {tt::tt_metal::ClusterType::T3K, "t3k_mesh_graph_descriptor.textproto"},
             {tt::tt_metal::ClusterType::GALAXY, "single_galaxy_mesh_graph_descriptor.textproto"},
             {tt::tt_metal::ClusterType::TG, "tg_mesh_graph_descriptor.textproto"},
             {tt::tt_metal::ClusterType::P100, "p100_mesh_graph_descriptor.textproto"},
             {tt::tt_metal::ClusterType::P150, "p150_mesh_graph_descriptor.textproto"},
             {tt::tt_metal::ClusterType::P150_X2, "p150_x2_mesh_graph_descriptor.textproto"},
             {tt::tt_metal::ClusterType::P150_X4, "p150_x4_mesh_graph_descriptor.textproto"},
             {tt::tt_metal::ClusterType::P150_X8, "p150_x8_mesh_graph_descriptor.textproto"},
             {tt::tt_metal::ClusterType::SIMULATOR_WORMHOLE_B0, "n150_mesh_graph_descriptor.textproto"},
             {tt::tt_metal::ClusterType::SIMULATOR_BLACKHOLE, "p150_mesh_graph_descriptor.textproto"},
             {tt::tt_metal::ClusterType::SIMULATOR_QUASAR, "p150_mesh_graph_descriptor.textproto"},
             {tt::tt_metal::ClusterType::N300_2x2, "n300_2x2_mesh_graph_descriptor.textproto"},
             {tt::tt_metal::ClusterType::P300, "p300_mesh_graph_descriptor.textproto"},
             {tt::tt_metal::ClusterType::BLACKHOLE_GALAXY, "single_bh_galaxy_mesh_graph_descriptor.textproto"},
             {tt::tt_metal::ClusterType::P300_X2, "p300_x2_mesh_graph_descriptor.textproto"},
         }},
        {tt::tt_fabric::FabricType::TORUS_X,
         ClusterToDescriptorMap{
             {tt::tt_metal::ClusterType::GALAXY, "single_galaxy_torus_x_graph_descriptor.textproto"},
         }},
        {tt::tt_fabric::FabricType::TORUS_Y,
         ClusterToDescriptorMap{
             {tt::tt_metal::ClusterType::GALAXY, "single_galaxy_torus_y_graph_descriptor.textproto"},
         }},
        {tt::tt_fabric::FabricType::TORUS_XY,
         ClusterToDescriptorMap{
             {tt::tt_metal::ClusterType::GALAXY, "single_galaxy_torus_xy_graph_descriptor.textproto"},
         }}});

MeshGraph::MeshGraph(const std::string& mesh_graph_desc_file_path, std::optional<FabricConfig> fabric_config) {
    if (mesh_graph_desc_file_path.ends_with(".textproto")) {
        auto filepath = std::filesystem::path(mesh_graph_desc_file_path);
        MeshGraphDescriptor mgd(filepath, true);
        this->initialize_from_mgd(mgd, fabric_config);
    } else {
        TT_THROW(
            "Mesh graph descriptor file must use the .textproto format. "
            "The YAML format (MGD 1.0) has been deprecated.\n\n"
            "Please convert your YAML mesh graph descriptor to textproto format.\n\n"
            "For examples of textproto format, see:\n"
            "  - tests/tt_metal/tt_fabric/custom_mesh_descriptors/mgd2_syntax_check_mesh_graph_descriptor.textproto\n"
            "  - tests/tt_metal/tt_fabric/custom_mesh_descriptors/t3k_2x2_mesh_graph_descriptor.textproto\n\n"
            "For conversion instructions from MGD 1.0 (YAML) to MGD (textproto), see:\n"
            "  - tt_metal/fabric/MGD_README.md (section: Converting from MGD 1.0 to MGD)\n\n"
            "File provided: {}",
            mesh_graph_desc_file_path);
    }
}

void MeshGraph::add_to_connectivity(
    MeshId src_mesh_id, ChipId src_chip_id, MeshId dest_mesh_id, ChipId dest_chip_id, RoutingDirection port_direction) {
    TT_ASSERT(
        *src_mesh_id < intra_mesh_connectivity_.size(),
        "MeshGraph: Invalid src_mesh_id: {} or unsized intramesh map",
        *src_mesh_id);
    TT_ASSERT(
        *dest_mesh_id < intra_mesh_connectivity_.size(),
        "MeshGraph: Invalid dest_mesh_id: {} or unsized intramesh map",
        *dest_mesh_id);
    TT_ASSERT(
        src_chip_id < intra_mesh_connectivity_[*src_mesh_id].size(),
        "MeshGraph: Invalid src_chip_id: {} or unsized intramesh map",
        src_chip_id);
    TT_ASSERT(
        dest_chip_id < intra_mesh_connectivity_[*dest_mesh_id].size(),
        "MeshGraph: Invalid dest_chip_id: {} or unsized intramesh map",
        dest_chip_id);

    TT_ASSERT(
        *src_mesh_id < inter_mesh_connectivity_.size(),
        "MeshGraph: Invalid src_mesh_id: {} or unsized intermesh map",
        *src_mesh_id);
    TT_ASSERT(
        *dest_mesh_id < inter_mesh_connectivity_.size(),
        "MeshGraph: Invalid dest_mesh_id: {} or unsized intermesh map",
        *dest_mesh_id);
    TT_ASSERT(
        src_chip_id < inter_mesh_connectivity_[*src_mesh_id].size(),
        "MeshGraph: Invalid src_chip_id: {} or unsized intermesh map",
        src_chip_id);
    TT_ASSERT(
        dest_chip_id < inter_mesh_connectivity_[*dest_mesh_id].size(),
        "MeshGraph: Invalid dest_chip_id: {} or unsized intermesh map",
        dest_chip_id);

    if (src_mesh_id != dest_mesh_id) {
        // Intermesh Connection
        auto& edge = inter_mesh_connectivity_[*src_mesh_id][src_chip_id];
        auto [it, is_inserted] = edge.insert(
            {dest_mesh_id,
             RouterEdge{.port_direction = port_direction, .connected_chip_ids = {dest_chip_id}, .weight = 0}});
        if (!is_inserted) {
            it->second.connected_chip_ids.push_back(dest_chip_id);
        }
    } else {
        // Intramesh Connection
        auto& edge = intra_mesh_connectivity_[*src_mesh_id][src_chip_id];
        auto [it, is_inserted] = edge.insert(
            {dest_chip_id,
             RouterEdge{.port_direction = port_direction, .connected_chip_ids = {dest_chip_id}, .weight = 0}});
        if (!is_inserted) {
            it->second.connected_chip_ids.push_back(dest_chip_id);
        }
    }
}

std::unordered_map<ChipId, RouterEdge> MeshGraph::get_valid_connections(
    const MeshCoordinate& src_mesh_coord, const MeshCoordinateRange& mesh_coord_range, FabricType fabric_type) const {
    std::unordered_map<ChipId, RouterEdge> valid_connections;

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
            ChipId fabric_chip_id = (coord[0] * mesh_shape[1]) + coord[1];
            valid_connections.insert(
                {fabric_chip_id,
                 RouterEdge{
                     .port_direction = direction,
                     .connected_chip_ids = std::vector<ChipId>(chip_spec_.num_eth_ports_per_direction, fabric_chip_id),
                     .weight = 0}});
        }
    }

    return valid_connections;
}

void MeshGraph::initialize_from_mgd(const MeshGraphDescriptor& mgd, std::optional<FabricConfig> fabric_config) {
    static const std::unordered_map<const proto::Architecture, tt::ARCH> proto_arch_to_arch = {
        {proto::Architecture::WORMHOLE_B0, tt::ARCH::WORMHOLE_B0},
        {proto::Architecture::BLACKHOLE, tt::ARCH::BLACKHOLE},
    };

    // TODO: need to fix
    chip_spec_ = ChipSpec{
        .arch = proto_arch_to_arch.at(mgd.get_arch()),
        .num_eth_ports_per_direction = mgd.get_num_eth_ports_per_direction(),
        .num_z_ports = (mgd.get_arch() == proto::Architecture::BLACKHOLE)
                           ? mgd.get_num_eth_ports_per_direction()
                           : 0,  // Z set to the same number as xy if in black hole
    };

    // Make intramesh connectivity
    // NOTE: Building connectivity based on FabricConfig override (if provided) or MGD's fabric type
    this->intra_mesh_connectivity_.resize(mgd.all_meshes().size());

    this->inter_mesh_connectivity_.resize(mgd.all_meshes().size());

    // This is to make sure emtpy elements are filled
    for (const auto& mesh : mgd.all_meshes()) {
        const auto& mesh_instance = mgd.get_instance(mesh);
        this->inter_mesh_connectivity_[mesh_instance.local_id].resize(mesh_instance.sub_instances.size());
    }

    for (const auto& connection : mgd.connections_by_type("FABRIC")) {
        const auto& connection_data = mgd.get_connection(connection);

        const auto& src_instance = mgd.get_instance(connection_data.nodes[0]);
        const auto& dst_instance = mgd.get_instance(connection_data.nodes[1]);

        bool is_device_level = (src_instance.kind == NodeKind::Device) && (dst_instance.kind == NodeKind::Device);

        if (is_device_level) {
            const auto& src_mesh_instance = mgd.get_instance(src_instance.hierarchy.back());
            const auto& dst_mesh_instance = mgd.get_instance(dst_instance.hierarchy.back());

            const MeshId src_mesh_id = MeshId(src_mesh_instance.local_id);
            const MeshId dst_mesh_id = MeshId(dst_mesh_instance.local_id);

            const ChipId src_chip_id = src_instance.local_id;
            const ChipId dst_chip_id = dst_instance.local_id;

            requested_intermesh_ports_[*src_mesh_id][*dst_mesh_id].push_back(
                {src_chip_id, dst_chip_id, connection_data.count});
        } else {
            const MeshId src_mesh_id = MeshId(src_instance.local_id);
            const MeshId dst_mesh_id = MeshId(dst_instance.local_id);

            requested_intermesh_connections_[*src_mesh_id][*dst_mesh_id] = connection_data.count;
        }
    }

    // Populate mesh_host_ranks_
    // Populate mesh_host_rank_coord_ranges_
    // Populate mesh_to_chip_ids_
    auto all_meshes = mgd.all_meshes();

    // Populate with empty containers
    this->mesh_host_ranks_.clear();
    for ([[maybe_unused]] const auto& mesh : all_meshes) {
        this->mesh_host_ranks_.emplace_back(MeshShape{1, 1}, MeshHostRankId{0});
    }

    // Set up the mesh_edge_ports_to_chip_id_ with empty containers for all meshes
    mesh_edge_ports_to_chip_id_.resize(mgd.all_meshes().size());

    for (const auto& mesh : all_meshes) {
        const auto& mesh_instance = mgd.get_instance(mesh);
        TT_FATAL(
            std::holds_alternative<const proto::MeshDescriptor*>(mesh_instance.desc),
            "MeshGraph: Instance {} is not a mesh",
            mesh_instance.name);
        const auto& mesh_desc = std::get<const proto::MeshDescriptor*>(mesh_instance.desc);

        MeshId mesh_id(mesh_instance.local_id);
        MeshShape mesh_shape(mesh_desc->device_topology().dims().at(0), mesh_desc->device_topology().dims().at(1));

        // Build intra-mesh connectivity based on FabricConfig override (if provided) or MGD's fabric type
        FabricType mgd_fabric_type = MeshGraphDescriptor::infer_fabric_type_from_dim_types(mesh_desc);
        FabricType effective_fabric_type;

        if (fabric_config.has_value()) {
            FabricType requested_fabric_type = get_fabric_type(*fabric_config);
            // Validate that FabricConfig doesn't try to create connections that don't exist
            if (requires_more_connectivity(requested_fabric_type, mgd_fabric_type, mesh_shape)) {
                TT_THROW(
                    "FabricConfig requests topology {} which requires more connectivity than MGD provides {}. "
                    "FabricConfig can only restrict topology (e.g., torus→mesh), not create new connections.",
                    enchantum::to_string(requested_fabric_type),
                    enchantum::to_string(mgd_fabric_type));
            }
            effective_fabric_type = requested_fabric_type;
        } else {
            effective_fabric_type = mgd_fabric_type;
        }

        // Build connectivity using effective_fabric_type
        MeshCoordinateRange mesh_coord_range(mesh_shape);
        uint32_t mesh_size = mesh_shape[0] * mesh_shape[1];
        this->intra_mesh_connectivity_[*mesh_id].resize(mesh_size);
        for (const auto& src_mesh_coord : mesh_coord_range) {
            ChipId src_chip_id = (src_mesh_coord[0] * mesh_shape[1]) + src_mesh_coord[1];
            this->intra_mesh_connectivity_[*mesh_id][src_chip_id] =
                this->get_valid_connections(src_mesh_coord, mesh_coord_range, effective_fabric_type);
        }

        MeshShape host_shape(mesh_desc->host_topology().dims().at(0), mesh_desc->host_topology().dims().at(1));

        std::vector<MeshHostRankId> mesh_host_ranks_values;
        uint32_t next_rank = 0;
        for (const auto& host_coord : MeshCoordinateRange(host_shape)) {
            mesh_host_ranks_values.push_back(MeshHostRankId{next_rank++});

            std::uint32_t board_ns_size = mesh_shape[0] / host_shape[0];
            std::uint32_t board_ew_size = mesh_shape[1] / host_shape[1];

            TT_FATAL(
                mesh_shape[0] % host_shape[0] == 0 && mesh_shape[1] % host_shape[1] == 0,
                "MeshGraph: Mesh shape {}x{} must be divisible by host shape {}x{}",
                mesh_shape[0],
                mesh_shape[1],
                host_shape[0],
                host_shape[1]);

            // Populate mesh_host_rank_coord_ranges_
            this->mesh_host_rank_coord_ranges_.emplace(
                std::make_pair(*mesh_id, mesh_host_ranks_values.back()),
                MeshCoordinateRange(
                    MeshCoordinate(host_coord[0] * board_ns_size, host_coord[1] * board_ew_size),
                    MeshCoordinate(
                        ((host_coord[0] + 1) * board_ns_size) - 1, ((host_coord[1] + 1) * board_ew_size) - 1)));
        }

        // Populate mesh_host_ranks_
        this->mesh_host_ranks_[*mesh_id] = tt_metal::distributed::MeshContainer<MeshHostRankId>(host_shape, mesh_host_ranks_values);

        // Populate mesh_to_chip_ids
        std::vector<ChipId> chip_ids(mesh_shape[0] * mesh_shape[1]);
        std::iota(chip_ids.begin(), chip_ids.end(), 0);
        this->mesh_to_chip_ids_.emplace(
            mesh_instance.local_id, tt_metal::distributed::MeshContainer<ChipId>(mesh_shape, chip_ids));

        // Get the edge ports of each mesh
        // North, start from NW corner
        std::uint32_t chan_id = 0;
        for (std::uint32_t chip_id = 0; chip_id < mesh_shape[1]; chip_id++) {
            for (std::uint32_t i = 0; i < chip_spec_.num_eth_ports_per_direction; i++) {
                mesh_edge_ports_to_chip_id_[*mesh_id][{RoutingDirection::N, chan_id++}] = chip_id;
            }
        }
        // South, start from SW corner
        chan_id = 0;
        for (std::uint32_t chip_id = ((mesh_shape[0] * mesh_shape[1]) - mesh_shape[1]);
             chip_id < (mesh_shape[0] * mesh_shape[1]);
             chip_id++) {
            for (std::uint32_t i = 0; i < chip_spec_.num_eth_ports_per_direction; i++) {
                mesh_edge_ports_to_chip_id_[*mesh_id][{RoutingDirection::S, chan_id++}] = chip_id;
            }
        }
        // East, start from NE corner
        chan_id = 0;
        for (std::uint32_t chip_id = (mesh_shape[1] - 1); chip_id < (mesh_shape[0] * mesh_shape[1]);
             chip_id += mesh_shape[1]) {
            for (std::uint32_t i = 0; i < chip_spec_.num_eth_ports_per_direction; i++) {
                mesh_edge_ports_to_chip_id_[*mesh_id][{RoutingDirection::E, chan_id++}] = chip_id;
            }
        }
        // West, start from NW corner
        chan_id = 0;
        for (std::uint32_t chip_id = 0; chip_id < (mesh_shape[0] * mesh_shape[1]); chip_id += mesh_shape[1]) {
            for (std::uint32_t i = 0; i < chip_spec_.num_eth_ports_per_direction; i++) {
                mesh_edge_ports_to_chip_id_[*mesh_id][{RoutingDirection::W, chan_id++}] = chip_id;
            }
        }
    }
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

const RequestedIntermeshConnections& MeshGraph::get_requested_intermesh_connections() const {
    return requested_intermesh_connections_;
}

const RequestedIntermeshPorts& MeshGraph::get_requested_intermesh_ports() const { return requested_intermesh_ports_; }

const std::vector<std::unordered_map<port_id_t, ChipId, hash_pair>>& MeshGraph::get_mesh_edge_ports_to_chip_id() const {
    return mesh_edge_ports_to_chip_id_;
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

MeshContainer<ChipId> MeshGraph::get_chip_ids(MeshId mesh_id, std::optional<MeshHostRankId> host_rank) const {
    auto it = mesh_to_chip_ids_.find(mesh_id);
    TT_FATAL(it != mesh_to_chip_ids_.end(), "MeshGraph: mesh_id {} not found", mesh_id);

    if (!host_rank.has_value()) {
        // Return the entire mesh
        return it->second;
    }

    // Return submesh for the specific host rank
    MeshCoordinateRange coord_range = get_coord_range(mesh_id, host_rank);
    MeshShape submesh_shape = coord_range.shape();

    std::vector<ChipId> submesh_chip_ids;
    submesh_chip_ids.reserve(submesh_shape.mesh_size());

    for (const auto& coord : coord_range) {
        submesh_chip_ids.push_back(it->second.at(coord));
    }

    return MeshContainer<ChipId>(submesh_shape, submesh_chip_ids);
}

MeshCoordinate MeshGraph::chip_to_coordinate(MeshId mesh_id, ChipId chip_id) const {
    const auto& mesh_shape = mesh_to_chip_ids_.at(mesh_id).shape();
    int ns = chip_id / mesh_shape[1];
    int ew = chip_id % mesh_shape[1];
    return MeshCoordinate(ns, ew);
}

ChipId MeshGraph::coordinate_to_chip(MeshId mesh_id, MeshCoordinate coordinate) const {
    const auto& mesh_shape = mesh_to_chip_ids_.at(mesh_id).shape();
    return (coordinate[0] * mesh_shape[1]) + coordinate[1];
}

std::optional<MeshHostRankId> MeshGraph::get_host_rank_for_chip(MeshId mesh_id, ChipId chip_id) const {
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
    const tt::tt_metal::ClusterType cluster_type,
    const std::string& root_dir,
    const tt::tt_fabric::FabricType fabric_type) {
    auto& fabric_to_cluster_map = cluster_type_to_mesh_graph_descriptor.get();
    auto fabric_it = fabric_to_cluster_map.find(fabric_type);
    if (fabric_it != fabric_to_cluster_map.end()) {
        const auto& cluster_to_descriptor_map = fabric_it->second;
        auto cluster_it = cluster_to_descriptor_map.find(cluster_type);
        if (cluster_it != cluster_to_descriptor_map.end()) {
            return std::filesystem::path(root_dir) / MESH_GRAPH_DESCRIPTOR_DIR / cluster_it->second;
        }
    }

    // Fallback: if a torus fabric type was requested but not found, try MESH fabric type.
    if (fabric_type != FabricType::MESH) {
        auto mesh_fabric_it = fabric_to_cluster_map.find(FabricType::MESH);
        const auto& cluster_to_descriptor_map = mesh_fabric_it->second;
        auto cluster_it = cluster_to_descriptor_map.find(cluster_type);
        if (cluster_it != cluster_to_descriptor_map.end()) {
            log_warning(
                tt::LogFabric,
                "Mesh Graph Descriptor for fabric type {} and cluster type {} not found. Picking mesh graph descriptor "
                "for MESH fabric type.",
                enchantum::to_string(fabric_type),
                enchantum::to_string(cluster_type));
            return std::filesystem::path(root_dir) / MESH_GRAPH_DESCRIPTOR_DIR / cluster_it->second;
        }
    }

    TT_THROW("Cannot find mesh graph descriptor for fabric type {} and cluster type {}", fabric_type, cluster_type);
}

}  // namespace tt::tt_fabric
