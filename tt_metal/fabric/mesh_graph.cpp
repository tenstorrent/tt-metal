// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include "experimental/fabric/routing_table_generator.hpp"
#include "fabric_host_utils.hpp"
#include <tt-metalium/experimental/fabric/topology_mapper_utils.hpp>

#include <enchantum/enchantum.hpp>
#include <yaml-cpp/yaml.h>
#include <algorithm>
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
#include <tt-metalium/experimental/fabric/mesh_graph_descriptor.hpp>
#include "physical_system_descriptor.hpp"
#include "protobuf/mesh_graph_descriptor.pb.h"
#include "impl/context/metal_context.hpp"
#include <numeric>
#include <set>
#include <cmath>

// Implementation of hash function for port_id_t
std::size_t std::hash<tt::tt_fabric::port_id_t>::operator()(const tt::tt_fabric::port_id_t& p) const {
    return tt::stl::hash::hash_objects_with_default_seed(p.first, p.second);
}

namespace tt::tt_fabric {

constexpr const char* MESH_GRAPH_DESCRIPTOR_DIR = "tt_metal/fabric/mesh_graph_descriptors";

/**
 * @brief Determines the maximum number of local Ethernet connections per direction between ASICs in the system.
 *
 * For each ASIC in the provided PhysicalSystemDescriptor, this function examines all neighboring ASICs and counts
 * the number of Ethernet connections to each neighbor that are marked as local (i.e., connection.is_local is true).
 * It returns the maximum number of such local connections found in any direction for any ASIC.
 *
 * @param psd The PhysicalSystemDescriptor representing the system's ASICs and their interconnections.
 * @return The maximum number of local Ethernet connections per direction between any two ASICs.
 */

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
             {tt::tt_metal::ClusterType::BLACKHOLE_GALAXY, "single_bh_galaxy_torus_x_graph_descriptor.textproto"}}},
        {tt::tt_fabric::FabricType::TORUS_Y,
         ClusterToDescriptorMap{
             {tt::tt_metal::ClusterType::GALAXY, "single_galaxy_torus_y_graph_descriptor.textproto"},
             {tt::tt_metal::ClusterType::BLACKHOLE_GALAXY, "single_bh_galaxy_torus_y_graph_descriptor.textproto"}}},
        {tt::tt_fabric::FabricType::TORUS_XY,
         ClusterToDescriptorMap{
             {tt::tt_metal::ClusterType::GALAXY, "single_galaxy_torus_xy_graph_descriptor.textproto"},
             {tt::tt_metal::ClusterType::BLACKHOLE_GALAXY, "single_bh_galaxy_torus_xy_graph_descriptor.textproto"}}}});

MeshGraph::MeshGraph(const std::string& mesh_graph_desc_file_path, std::optional<FabricConfig> fabric_config) {
    log_debug(tt::LogFabric, "mesh_graph_desc_file_path: {}", mesh_graph_desc_file_path);
    if (mesh_graph_desc_file_path.ends_with(".textproto")) {
        auto filepath = std::filesystem::path(mesh_graph_desc_file_path);
        mesh_graph_descriptor_.emplace(filepath, true);
        this->initialize_from_mgd(mesh_graph_descriptor_.value(), fabric_config);
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

    if (has_flag(fabric_type, FabricType::TORUS_X) and mesh_shape[1] > 1) {
        E = MeshCoordinate(src_mesh_coord[0], (src_mesh_coord[1] + 1) % mesh_shape[1]);
        W = MeshCoordinate(src_mesh_coord[0], (src_mesh_coord[1] - 1 + mesh_shape[1]) % mesh_shape[1]);
    }
    if (has_flag(fabric_type, FabricType::TORUS_Y) and mesh_shape[0] > 1) {
        N = MeshCoordinate((src_mesh_coord[0] - 1 + mesh_shape[0]) % mesh_shape[0], src_mesh_coord[1]);
        S = MeshCoordinate((src_mesh_coord[0] + 1) % mesh_shape[0], src_mesh_coord[1]);
    }
    for (const auto& [coord, direction] :
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
        .num_z_ports = mgd.get_num_eth_ports_per_direction()};

    // Count total meshes including switches (switches are treated as meshes internally)
    uint32_t total_mesh_count = mgd.all_meshes().size() + mgd.all_switches().size();

    // Make intramesh connectivity
    // NOTE: Building connectivity based on FabricConfig override (if provided) or MGD's fabric type
    this->intra_mesh_connectivity_.resize(total_mesh_count);

    this->inter_mesh_connectivity_.resize(total_mesh_count);

    // This is to make sure emtpy elements are filled
    for (const auto& mesh : mgd.all_meshes()) {
        const auto& mesh_instance = mgd.get_instance(mesh);
        this->inter_mesh_connectivity_[mesh_instance.local_id].resize(mesh_instance.sub_instances.size());
    }

    // Fill inter-mesh connectivity for switches
    for (const auto& switch_inst : mgd.all_switches()) {
        const auto& switch_instance = mgd.get_instance(switch_inst);
        MeshId switch_mesh_id(switch_instance.local_id);
        this->inter_mesh_connectivity_[*switch_mesh_id].resize(switch_instance.sub_instances.size());
    }

    for (const auto& connection : mgd.connections_by_type("FABRIC")) {
        const auto& connection_data = mgd.get_connection(connection);

        const auto& src_instance = mgd.get_instance(connection_data.nodes[0]);
        const auto& dst_instance = mgd.get_instance(connection_data.nodes[1]);

        bool is_device_level = (src_instance.kind == NodeKind::Device) && (dst_instance.kind == NodeKind::Device);

        if (is_device_level) {
            const auto& src_mesh_instance = mgd.get_instance(src_instance.hierarchy.back());
            const auto& dst_mesh_instance = mgd.get_instance(dst_instance.hierarchy.back());

            MeshId src_mesh_id(src_mesh_instance.local_id);
            MeshId dst_mesh_id(dst_mesh_instance.local_id);

            const ChipId src_chip_id = src_instance.local_id;
            const ChipId dst_chip_id = dst_instance.local_id;

            requested_intermesh_ports_[*src_mesh_id][*dst_mesh_id].push_back(
                {src_chip_id, dst_chip_id, connection_data.count});

            // Track mesh pairs that should use Z direction
            if (connection_data.assign_z_direction) {
                mesh_pairs_assign_z_direction_[*src_mesh_id].insert(*dst_mesh_id);
                mesh_pairs_assign_z_direction_[*dst_mesh_id].insert(*src_mesh_id);
            }

            // Track switch-to-mesh connections
            if (src_mesh_instance.kind == NodeKind::Switch && dst_mesh_instance.kind == NodeKind::Mesh) {
                switch_to_connected_meshes_[src_mesh_id].push_back(dst_mesh_id);
            }
            if (dst_mesh_instance.kind == NodeKind::Switch && src_mesh_instance.kind == NodeKind::Mesh) {
                switch_to_connected_meshes_[dst_mesh_id].push_back(src_mesh_id);
            }
        } else {
            MeshId src_mesh_id(src_instance.local_id);
            MeshId dst_mesh_id(dst_instance.local_id);

            requested_intermesh_connections_[*src_mesh_id][*dst_mesh_id] = connection_data.count;

            // Track mesh pairs that should use Z direction
            if (connection_data.assign_z_direction) {
                mesh_pairs_assign_z_direction_[*src_mesh_id].insert(*dst_mesh_id);
                mesh_pairs_assign_z_direction_[*dst_mesh_id].insert(*src_mesh_id);
            }

            // Track switch-to-mesh connections
            if (src_instance.kind == NodeKind::Switch && dst_instance.kind == NodeKind::Mesh) {
                switch_to_connected_meshes_[src_mesh_id].push_back(dst_mesh_id);
            }
            if (dst_instance.kind == NodeKind::Switch && src_instance.kind == NodeKind::Mesh) {
                switch_to_connected_meshes_[dst_mesh_id].push_back(src_mesh_id);
            }
        }
    }

    // Populate mesh_host_ranks_
    // Populate mesh_host_rank_coord_ranges_
    // Populate mesh_to_chip_ids_
    auto all_meshes = mgd.all_meshes();
    auto all_switches = mgd.all_switches();

    // Populate with empty containers
    this->mesh_host_ranks_.clear();
    for ([[maybe_unused]] const auto& mesh : all_meshes) {
        this->mesh_host_ranks_.emplace_back(MeshShape{1, 1}, MeshHostRankId{0});
    }
    for ([[maybe_unused]] const auto& swtch : all_switches) {
        this->mesh_host_ranks_.emplace_back(MeshShape{1, 1}, MeshHostRankId{0});
    }

    // Set up the mesh_edge_ports_to_chip_id_ with empty containers for all meshes
    mesh_edge_ports_to_chip_id_.resize(all_meshes.size() + all_switches.size());

    for (const auto& mesh : all_meshes) {
        const auto& mesh_instance = mgd.get_instance(mesh);
        TT_FATAL(
            std::holds_alternative<const proto::MeshDescriptor*>(mesh_instance.desc),
            "MeshGraph: Instance {} is not a mesh",
            mesh_instance.name);
        const auto& mesh_desc = std::get<const proto::MeshDescriptor*>(mesh_instance.desc);

        MeshId mesh_id(mesh_instance.local_id);

        // Set intra-mesh relaxed policy based on channels policy from MGD
        bool is_relaxed = (mesh_desc->channels().policy() == proto::Policy::RELAXED);
        this->intra_mesh_relaxed_policy_[mesh_id] = is_relaxed;

        MeshShape mesh_shape(mesh_desc->device_topology().dims().at(0), mesh_desc->device_topology().dims().at(1));

        // Validate mesh shape dimensions are valid (must be positive and even for WORMHOLE_B0)
        TT_FATAL(
            mesh_shape[0] > 0 && mesh_shape[1] > 0,
            "MeshGraph: Mesh shape dimensions must be positive, got {}x{}",
            mesh_shape[0],
            mesh_shape[1]);

        // For WORMHOLE_B0 architecture, if both dimensions are odd, they must both be 1
        // (i.e., 1x1 is valid, but 3x5 is not)
        if (mesh_desc->arch() == proto::Architecture::WORMHOLE_B0) {
            bool both_odd = (mesh_shape[0] % 2 != 0) && (mesh_shape[1] % 2 != 0);
            if (both_odd) {
                TT_FATAL(
                    mesh_shape[0] == 1 && mesh_shape[1] == 1,
                    "MeshGraph: For WORMHOLE_B0 architecture, if both mesh dimensions are odd, they must both be 1, "
                    "got {}x{}",
                    mesh_shape[0],
                    mesh_shape[1]);
            }
        }

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

        // Validate that mesh shape is divisible by host shape before processing
        TT_FATAL(
            mesh_shape[0] % host_shape[0] == 0 && mesh_shape[1] % host_shape[1] == 0,
            "MeshGraph: Mesh shape {}x{} must be divisible by host shape {}x{}",
            mesh_shape[0],
            mesh_shape[1],
            host_shape[0],
            host_shape[1]);

        std::vector<MeshHostRankId> mesh_host_ranks_values;
        uint32_t next_rank = 0;
        for (const auto& host_coord : MeshCoordinateRange(host_shape)) {
            mesh_host_ranks_values.push_back(MeshHostRankId{next_rank++});

            std::uint32_t board_ns_size = mesh_shape[0] / host_shape[0];
            std::uint32_t board_ew_size = mesh_shape[1] / host_shape[1];

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
        // Z, for all chips (only if using blackhole)
        if (chip_spec_.num_z_ports > 0) {
            chan_id = 0;
            for (std::uint32_t chip_id = 0; chip_id < (mesh_shape[0] * mesh_shape[1]); chip_id++) {
                for (std::uint32_t i = 0; i < chip_spec_.num_z_ports; i++) {
                    mesh_edge_ports_to_chip_id_[*mesh_id][{RoutingDirection::Z, chan_id++}] = chip_id;
                }
            }
        }
    }

    // Populate switches as meshes (switches are treated as meshes internally)
    for (const auto& switch_inst : mgd.all_switches()) {
        const auto& switch_instance = mgd.get_instance(switch_inst);
        TT_FATAL(
            std::holds_alternative<const proto::SwitchDescriptor*>(switch_instance.desc),
            "MeshGraph: Instance {} is not a switch",
            switch_instance.name);
        const auto& switch_desc = std::get<const proto::SwitchDescriptor*>(switch_instance.desc);

        MeshId switch_mesh_id(switch_instance.local_id);

        // Set intra-mesh relaxed policy based on channels policy from MGD
        bool is_relaxed = (switch_desc->channels().policy() == proto::Policy::RELAXED);
        this->intra_mesh_relaxed_policy_[switch_mesh_id] = is_relaxed;

        MeshShape switch_shape(
            switch_desc->device_topology().dims().at(0), switch_desc->device_topology().dims().at(1));

        // Build intra-mesh connectivity based on FabricConfig override (if provided) or MGD's fabric type
        FabricType mgd_fabric_type;
        const auto& dim_types = switch_desc->device_topology().dim_types();
        if (dim_types.size() < 2) {
            mgd_fabric_type = FabricType::MESH;
        } else {
            bool y_is_ring = (dim_types[0] == proto::TorusTopology::RING);
            bool x_is_ring = (dim_types[1] == proto::TorusTopology::RING);
            if (y_is_ring && x_is_ring) {
                mgd_fabric_type = FabricType::TORUS_XY;
            } else if (y_is_ring) {
                mgd_fabric_type = FabricType::TORUS_Y;
            } else if (x_is_ring) {
                mgd_fabric_type = FabricType::TORUS_X;
            } else {
                mgd_fabric_type = FabricType::MESH;
            }
        }
        FabricType effective_fabric_type;

        if (fabric_config.has_value()) {
            FabricType requested_fabric_type = get_fabric_type(*fabric_config);
            // Validate that FabricConfig doesn't try to create connections that don't exist
            if (requires_more_connectivity(requested_fabric_type, mgd_fabric_type, switch_shape)) {
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
        MeshCoordinateRange switch_coord_range(switch_shape);
        uint32_t switch_size = switch_shape[0] * switch_shape[1];
        this->intra_mesh_connectivity_[*switch_mesh_id].resize(switch_size);
        for (const auto& src_switch_coord : switch_coord_range) {
            ChipId src_chip_id = (src_switch_coord[0] * switch_shape[1]) + src_switch_coord[1];
            this->intra_mesh_connectivity_[*switch_mesh_id][src_chip_id] =
                this->get_valid_connections(src_switch_coord, switch_coord_range, effective_fabric_type);
        }

        // Switches are always single host, so host_shape is [1, 1]
        MeshShape host_shape(1, 1);

        // Populate mesh_host_ranks_ for switch (single host)
        std::vector<MeshHostRankId> switch_host_ranks_values;
        switch_host_ranks_values.push_back(MeshHostRankId{0});
        this->mesh_host_ranks_[*switch_mesh_id] =
            tt_metal::distributed::MeshContainer<MeshHostRankId>(host_shape, switch_host_ranks_values);

        // Populate mesh_host_rank_coord_ranges_ for switch
        this->mesh_host_rank_coord_ranges_.emplace(
            std::make_pair(*switch_mesh_id, MeshHostRankId{0}),
            MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(switch_shape[0] - 1, switch_shape[1] - 1)));

        // Populate switch_to_chip_ids
        std::vector<ChipId> chip_ids(switch_shape[0] * switch_shape[1]);
        std::iota(chip_ids.begin(), chip_ids.end(), 0);
        this->switch_to_chip_ids_.emplace(
            switch_mesh_id, tt_metal::distributed::MeshContainer<ChipId>(switch_shape, chip_ids));
        this->mesh_to_chip_ids_.emplace(
            *switch_mesh_id, tt_metal::distributed::MeshContainer<ChipId>(switch_shape, chip_ids));

        // Track this switch in switch_ids_
        this->switch_ids_.push_back(switch_mesh_id);

        // Get the edge ports of each switch (same as mesh)
        mesh_edge_ports_to_chip_id_.resize(
            std::max(mesh_edge_ports_to_chip_id_.size(), static_cast<size_t>(*switch_mesh_id + 1)));
        std::uint32_t chan_id = 0;
        // North
        for (std::uint32_t chip_id = 0; chip_id < switch_shape[1]; chip_id++) {
            for (std::uint32_t i = 0; i < chip_spec_.num_eth_ports_per_direction; i++) {
                mesh_edge_ports_to_chip_id_[*switch_mesh_id][{RoutingDirection::N, chan_id++}] = chip_id;
            }
        }
        // South
        chan_id = 0;
        for (std::uint32_t chip_id = ((switch_shape[0] * switch_shape[1]) - switch_shape[1]);
             chip_id < (switch_shape[0] * switch_shape[1]);
             chip_id++) {
            for (std::uint32_t i = 0; i < chip_spec_.num_eth_ports_per_direction; i++) {
                mesh_edge_ports_to_chip_id_[*switch_mesh_id][{RoutingDirection::S, chan_id++}] = chip_id;
            }
        }
        // East
        chan_id = 0;
        for (std::uint32_t chip_id = (switch_shape[1] - 1); chip_id < (switch_shape[0] * switch_shape[1]);
             chip_id += switch_shape[1]) {
            for (std::uint32_t i = 0; i < chip_spec_.num_eth_ports_per_direction; i++) {
                mesh_edge_ports_to_chip_id_[*switch_mesh_id][{RoutingDirection::E, chan_id++}] = chip_id;
            }
        }
        // West
        chan_id = 0;
        for (std::uint32_t chip_id = 0; chip_id < (switch_shape[0] * switch_shape[1]); chip_id += switch_shape[1]) {
            for (std::uint32_t i = 0; i < chip_spec_.num_eth_ports_per_direction; i++) {
                mesh_edge_ports_to_chip_id_[*switch_mesh_id][{RoutingDirection::W, chan_id++}] = chip_id;
            }
        }
        // Z, for all chips (only if using blackhole)
        if (chip_spec_.num_z_ports > 0) {
            chan_id = 0;
            for (std::uint32_t chip_id = 0; chip_id < (switch_shape[0] * switch_shape[1]); chip_id++) {
                for (std::uint32_t i = 0; i < chip_spec_.num_z_ports; i++) {
                    mesh_edge_ports_to_chip_id_[*switch_mesh_id][{RoutingDirection::Z, chan_id++}] = chip_id;
                }
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

bool MeshGraph::should_assign_z_direction(MeshId src_mesh_id, MeshId dst_mesh_id) const {
    auto it = mesh_pairs_assign_z_direction_.find(*src_mesh_id);
    if (it != mesh_pairs_assign_z_direction_.end()) {
        return it->second.contains(*dst_mesh_id);
    }
    return false;
}

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
    TT_FATAL(this->mesh_to_chip_ids_.contains(mesh_id), "MeshGraph: mesh_id {} not found", mesh_id);
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

std::vector<SwitchId> MeshGraph::get_switch_ids() const {
    // Convert internal MeshId representation to SwitchId for API
    std::vector<SwitchId> result;
    result.reserve(switch_ids_.size());
    for (const auto& mesh_id : switch_ids_) {
        result.push_back(SwitchId(*mesh_id));
    }
    return result;
}

std::unordered_set<MeshId> MeshGraph::get_meshes_connected_to_switch(SwitchId switch_id) const {
    // Convert SwitchId to MeshId for internal lookup
    MeshId mesh_id(*switch_id);
    auto it = switch_to_connected_meshes_.find(mesh_id);
    if (it != switch_to_connected_meshes_.end()) {
        // Convert vector to unordered_set to automatically deduplicate (bidirectional connections may add the same mesh
        // twice)
        return std::unordered_set<MeshId>(it->second.begin(), it->second.end());
    }
    return {};
}

bool MeshGraph::is_mesh_connected_to_switch(MeshId mesh_id, SwitchId switch_id) const {
    // Convert SwitchId to MeshId for internal lookup
    MeshId switch_mesh_id(*switch_id);
    auto it = switch_to_connected_meshes_.find(switch_mesh_id);
    if (it != switch_to_connected_meshes_.end()) {
        const auto& meshes = it->second;
        return std::find(meshes.begin(), meshes.end(), mesh_id) != meshes.end();
    }
    return false;
}

std::optional<SwitchId> MeshGraph::get_switch_for_mesh(MeshId mesh_id) const {
    // Check if mesh_id corresponds to a switch (internal representation uses MeshId)
    for (const auto& switch_mesh_id : switch_ids_) {
        if (*switch_mesh_id == *mesh_id) {
            return SwitchId(*switch_mesh_id);
        }
    }
    // Check if any switch connects to this mesh
    for (const auto& [switch_mesh_id, meshes] : switch_to_connected_meshes_) {
        if (std::find(meshes.begin(), meshes.end(), mesh_id) != meshes.end()) {
            return SwitchId(*switch_mesh_id);
        }
    }
    return std::nullopt;
}

std::vector<MeshId> MeshGraph::get_all_mesh_ids() const {
    std::vector<MeshId> mesh_ids;
    mesh_ids.reserve(this->mesh_to_chip_ids_.size());
    for (const auto& [mesh_id, _] : this->mesh_to_chip_ids_) {
        mesh_ids.push_back(mesh_id);
    }
    return mesh_ids;
}

std::vector<MeshId> MeshGraph::get_mesh_ids() const {
    std::vector<MeshId> mesh_ids;
    mesh_ids.reserve(this->mesh_to_chip_ids_.size() - switch_ids_.size());
    for (const auto& [mesh_id, _] : this->mesh_to_chip_ids_) {
        if (!this->is_switch_mesh(mesh_id)) {
            mesh_ids.push_back(mesh_id);
        }
    }
    return mesh_ids;
}

bool MeshGraph::is_switch_mesh(MeshId mesh_id) const {
    return std::find(switch_ids_.begin(), switch_ids_.end(), mesh_id) != switch_ids_.end();
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

bool MeshGraph::is_intra_mesh_policy_relaxed(MeshId mesh_id) const {
    auto it = intra_mesh_relaxed_policy_.find(mesh_id);
    TT_FATAL(it != intra_mesh_relaxed_policy_.end(), "No mode for mesh_id {}", *mesh_id);
    return it->second;
}

/**
 * Generate all possible mesh shapes that can be formed from a given number of chips.
 *
 * The function considers all pairs of positive integers (x, y) such that x * y = N,
 * where N ranges from total_number_of_chips down to 1. Each pair represents a possible
 * mesh shape (rows, columns). To avoid duplicates, shapes are normalized so that the
 * larger dimension is always second (i.e., MeshShape(smaller_dim, larger_dim)).
 *
 * Filtering rules:
 *   - Only shapes where both dimensions are even or 1 are considered (except for 1D shapes).
 *     Odd dimensions (other than 1) are skipped to ensure compatibility with hardware constraints.
 *   - 1D shapes (where one dimension is 1) are collected separately and appended at the end
 *     of the result, so that 2D shapes are prioritized.
 *   - Duplicate shapes are avoided.
 *
 * @param total_number_of_chips The total number of chips to partition into mesh shapes.
 * @return A vector of possible MeshShape objects, with 2D shapes first (sorted by decreasing chip count),
 *         followed by 1D shapes. Each shape appears only once.
 */

MeshGraph MeshGraph::generate_mesh_graph_of_shape(
    MeshShape mesh_shape, tt::tt_fabric::FabricType fabric_type, std::uint32_t num_connections_per_direction) {
    MeshGraph mesh_graph;

    // Get chip spec from MetalContext
    const auto& metal_context = tt::tt_metal::MetalContext::instance();
    const auto& cluster = metal_context.get_cluster();
    tt::ARCH arch = cluster.get_cluster_desc()->get_arch();

    // Get reliability mode from fabric config (stored in MetalContext)
    tt::tt_fabric::FabricReliabilityMode reliability_mode = metal_context.get_fabric_reliability_mode();

    // Use the provided num_connections_per_direction
    std::uint32_t num_eth_ports_per_direction = num_connections_per_direction;

    // Set chip spec
    mesh_graph.chip_spec_ = ChipSpec{
        .arch = arch,
        .num_eth_ports_per_direction = num_eth_ports_per_direction,
        .num_z_ports = (arch == tt::ARCH::BLACKHOLE) ? num_eth_ports_per_direction : 0,
    };

    // Validate mesh shape dimensions
    TT_FATAL(
        mesh_shape[0] > 0 && mesh_shape[1] > 0,
        "MeshGraph: Mesh shape dimensions must be positive, got {}x{}",
        mesh_shape[0],
        mesh_shape[1]);

    // For WORMHOLE_B0 architecture, if both dimensions are odd, they must both be 1
    if (arch == tt::ARCH::WORMHOLE_B0) {
        bool both_odd = (mesh_shape[0] % 2 != 0) && (mesh_shape[1] % 2 != 0);
        if (both_odd) {
            TT_FATAL(
                mesh_shape[0] == 1 && mesh_shape[1] == 1,
                "MeshGraph: For WORMHOLE_B0 architecture, if both mesh dimensions are odd, they must both be 1, "
                "got {}x{}",
                mesh_shape[0],
                mesh_shape[1]);
        }
    }

    // Initialize for a single mesh (mesh_id = 0)
    MeshId mesh_id(0);
    uint32_t total_mesh_count = 1;

    // Resize intra-mesh and inter-mesh connectivity (inter-mesh will remain empty)
    mesh_graph.intra_mesh_connectivity_.resize(total_mesh_count);
    mesh_graph.inter_mesh_connectivity_.resize(total_mesh_count);
    mesh_graph.inter_mesh_connectivity_[0].resize(mesh_shape[0] * mesh_shape[1]);

    // Set intra-mesh relaxed policy based on reliability mode
    // RELAXED_SYSTEM_HEALTH_SETUP_MODE -> relaxed = true
    // STRICT_SYSTEM_HEALTH_SETUP_MODE -> relaxed = false
    bool is_relaxed = (reliability_mode == FabricReliabilityMode::RELAXED_SYSTEM_HEALTH_SETUP_MODE);
    mesh_graph.intra_mesh_relaxed_policy_[mesh_id] = is_relaxed;

    // Build intra-mesh connectivity using fabric_type
    MeshCoordinateRange mesh_coord_range(mesh_shape);
    uint32_t mesh_size = mesh_shape[0] * mesh_shape[1];
    mesh_graph.intra_mesh_connectivity_[*mesh_id].resize(mesh_size);
    for (const auto& src_mesh_coord : mesh_coord_range) {
        ChipId src_chip_id = (src_mesh_coord[0] * mesh_shape[1]) + src_mesh_coord[1];
        mesh_graph.intra_mesh_connectivity_[*mesh_id][src_chip_id] =
            mesh_graph.get_valid_connections(src_mesh_coord, mesh_coord_range, fabric_type);
    }

    // TODO: Enable this for multi-host meshes

    // For auto-generated meshes, assume single host (host_shape = [1, 1])
    MeshShape host_shape(1, 1);

    // Populate mesh_host_ranks_ (single host)
    std::vector<MeshHostRankId> mesh_host_ranks_values;
    mesh_host_ranks_values.push_back(MeshHostRankId{0});

    // Populate mesh_host_rank_coord_ranges_ (entire mesh belongs to host rank 0)
    mesh_graph.mesh_host_rank_coord_ranges_.emplace(
        std::make_pair(*mesh_id, MeshHostRankId{0}),
        MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(mesh_shape[0] - 1, mesh_shape[1] - 1)));

    // Populate mesh_host_ranks_
    mesh_graph.mesh_host_ranks_.clear();
    mesh_graph.mesh_host_ranks_.emplace_back(host_shape, mesh_host_ranks_values);

    // Populate mesh_to_chip_ids
    std::vector<ChipId> chip_ids(mesh_shape[0] * mesh_shape[1]);
    std::iota(chip_ids.begin(), chip_ids.end(), 0);
    mesh_graph.mesh_to_chip_ids_.emplace(*mesh_id, tt_metal::distributed::MeshContainer<ChipId>(mesh_shape, chip_ids));

    // Set up mesh_edge_ports_to_chip_id_ with empty container
    mesh_graph.mesh_edge_ports_to_chip_id_.resize(total_mesh_count);

    // Get the edge ports of the mesh
    // North, start from NW corner
    std::uint32_t chan_id = 0;
    for (std::uint32_t chip_id = 0; chip_id < mesh_shape[1]; chip_id++) {
        for (std::uint32_t i = 0; i < mesh_graph.chip_spec_.num_eth_ports_per_direction; i++) {
            mesh_graph.mesh_edge_ports_to_chip_id_[*mesh_id][{RoutingDirection::N, chan_id++}] = chip_id;
        }
    }
    // South, start from SW corner
    chan_id = 0;
    for (std::uint32_t chip_id = ((mesh_shape[0] * mesh_shape[1]) - mesh_shape[1]);
         chip_id < (mesh_shape[0] * mesh_shape[1]);
         chip_id++) {
        for (std::uint32_t i = 0; i < mesh_graph.chip_spec_.num_eth_ports_per_direction; i++) {
            mesh_graph.mesh_edge_ports_to_chip_id_[*mesh_id][{RoutingDirection::S, chan_id++}] = chip_id;
        }
    }
    // East, start from NE corner
    chan_id = 0;
    for (std::uint32_t chip_id = (mesh_shape[1] - 1); chip_id < (mesh_shape[0] * mesh_shape[1]);
         chip_id += mesh_shape[1]) {
        for (std::uint32_t i = 0; i < mesh_graph.chip_spec_.num_eth_ports_per_direction; i++) {
            mesh_graph.mesh_edge_ports_to_chip_id_[*mesh_id][{RoutingDirection::E, chan_id++}] = chip_id;
        }
    }
    // West, start from NW corner
    chan_id = 0;
    for (std::uint32_t chip_id = 0; chip_id < (mesh_shape[0] * mesh_shape[1]); chip_id += mesh_shape[1]) {
        for (std::uint32_t i = 0; i < mesh_graph.chip_spec_.num_eth_ports_per_direction; i++) {
            mesh_graph.mesh_edge_ports_to_chip_id_[*mesh_id][{RoutingDirection::W, chan_id++}] = chip_id;
        }
    }

    return mesh_graph;
}

std::filesystem::path MeshGraph::get_mesh_graph_descriptor_path_for_cluster_type(
    const tt::tt_metal::ClusterType cluster_type,
    const std::string& root_dir,
    const tt::tt_fabric::FabricType fabric_type) {
    const auto& fabric_to_cluster_map = cluster_type_to_mesh_graph_descriptor.get();
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
