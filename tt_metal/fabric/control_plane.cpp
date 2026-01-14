// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <enchantum/enchantum.hpp>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iomanip>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <ostream>
#include <queue>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <yaml-cpp/yaml.h>
#include <tt_stl/assert.hpp>

#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include "core_coord.hpp"
#include "compressed_direction_table.hpp"
#include "compressed_routing_path.hpp"
#include "tools/scaleout/factory_system_descriptor/utils.hpp"
#include "hostdevcommon/fabric_common.h"
#include "fabric_host_utils.hpp"
#include <tt-metalium/experimental/fabric/fabric_telemetry.hpp>
#include "tt_metal/llrt/hal/generated/fabric_telemetry.hpp"
#include "distributed_context.hpp"
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include "hal_types.hpp"
#include "host_api.hpp"
#include "impl/context/metal_context.hpp"
#include "tt_metal/common/env_lib.hpp"
#include <tt-logger/tt-logger.hpp>
#include "mesh_coord.hpp"
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include "llrt/metal_soc_descriptor.hpp"
#include <tt-metalium/experimental/fabric/routing_table_generator.hpp>
#include <umd/device/types/core_coordinates.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>
#include <umd/device/types/xy_pair.hpp>
#include <umd/device/cluster.hpp>
#include "tt_metal/fabric/fabric_context.hpp"
#include "tt_metal/fabric/fabric_builder_context.hpp"
#include "tt_metal/fabric/fabric_tensix_builder_impl.hpp"
#include "tt_metal/fabric/serialization/router_port_directions.hpp"
#include "tt_stl/small_vector.hpp"
#include "tt_metal/fabric/physical_system_descriptor.hpp"
#include "tt_metal/fabric/serialization/port_descriptor_serialization.hpp"
#include "tt_metal/fabric/serialization/intermesh_connections_serialization.hpp"
#include <tt-metalium/experimental/fabric/topology_mapper.hpp>
#include "tt_metal/fabric/builder/fabric_static_sized_channels_allocator.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_interface.hpp"

namespace tt::tt_fabric {

namespace {

// Get the physical chip ids for a mesh
std::unordered_map<ChipId, std::vector<CoreCoord>> get_ethernet_cores_grouped_by_connected_chips(ChipId chip_id) {
    return tt::tt_metal::MetalContext::instance().get_cluster().get_ethernet_cores_grouped_by_connected_chips(chip_id);
}

// Generate fixed ASIC position pinnings for Galaxy topology to ensure QSFP links align with fabric mesh corner nodes.
// This is a performance optimization to ensure that MGD mapping does not bisect a device.
//
// * o o o < Top left corner pinned with *
// o o o o
// o o o o
// o o o o
// o o o o
// o o o o
// o o o o
// o o o * < Bottom right corner pinned with *
std::vector<std::pair<AsicPosition, FabricNodeId>> get_galaxy_fixed_asic_position_pinnings(size_t board_size) {
    std::vector<std::pair<AsicPosition, FabricNodeId>> fixed_asic_position_pinnings;
    // Top left corner: index 0
    fixed_asic_position_pinnings.push_back({AsicPosition{1, 1}, FabricNodeId(MeshId{0}, 0)});
    // Bottom right corner: last device index
    fixed_asic_position_pinnings.push_back({AsicPosition{4, 1}, FabricNodeId(MeshId{0}, board_size - 1)});
    return fixed_asic_position_pinnings;
}

template <typename CONNECTIVITY_MAP_T>
void build_golden_link_counts(
    CONNECTIVITY_MAP_T const& golden_connectivity_map,
    std::unordered_map<MeshId, std::unordered_map<ChipId, std::unordered_map<RoutingDirection, size_t>>>&
        golden_link_counts_out) {
    static_assert(
        std::is_same_v<CONNECTIVITY_MAP_T, IntraMeshConnectivity> ||
            std::is_same_v<CONNECTIVITY_MAP_T, InterMeshConnectivity>,
        "Invalid connectivity map type");
    for (std::uint32_t mesh_id = 0; mesh_id < golden_connectivity_map.size(); mesh_id++) {
        for (std::uint32_t chip_id = 0; chip_id < golden_connectivity_map[mesh_id].size(); chip_id++) {
            for (const auto& [remote_connected_id, router_edge] : golden_connectivity_map[mesh_id][chip_id]) {
                TT_FATAL(
                    golden_link_counts_out[MeshId{mesh_id}][chip_id][router_edge.port_direction] == 0,
                    "Golden link counts already set for chip {} in mesh {}",
                    chip_id,
                    mesh_id);
                golden_link_counts_out[MeshId{mesh_id}][chip_id][router_edge.port_direction] =
                    router_edge.connected_chip_ids.size();
            }
        }
    }
}

bool check_connection_requested(
    MeshId my_mesh_id,
    MeshId neighbor_mesh_id,
    const RequestedIntermeshConnections& requested_intermesh_connections,
    const RequestedIntermeshPorts& requested_intermesh_ports) {
    if (!requested_intermesh_ports.empty()) {
        return requested_intermesh_ports.contains(*my_mesh_id) &&
               requested_intermesh_ports.at(*my_mesh_id).contains(*neighbor_mesh_id);
    }
    return requested_intermesh_connections.contains(*my_mesh_id) &&
           requested_intermesh_connections.at(*my_mesh_id).contains(*neighbor_mesh_id);
}

[[maybe_unused]] std::string create_port_tag(port_id_t port_id) {
    return std::string(enchantum::to_string(port_id.first)) + std::to_string(port_id.second);
}

}  // namespace

const std::unordered_map<tt::ARCH, std::vector<std::uint16_t>> ubb_bus_ids = {
    {tt::ARCH::WORMHOLE_B0, {0xC0, 0x80, 0x00, 0x40}},
    {tt::ARCH::BLACKHOLE, {0x00, 0x40, 0xC0, 0x80}},
};

uint16_t get_bus_id(tt::umd::Cluster& cluster, ChipId chip_id) {
    // Prefer cached value from cluster descriptor (available for silicon and our simulator/mock descriptors)
    auto* cluster_desc = cluster.get_cluster_description();
    uint16_t bus_id = cluster_desc->get_bus_id(chip_id);
    return bus_id;
}

UbbId get_ubb_id(tt::umd::Cluster& cluster, ChipId chip_id) {
    auto* cluster_desc = cluster.get_cluster_description();
    const auto& tray_bus_ids = ubb_bus_ids.at(cluster_desc->get_arch());
    const auto bus_id = get_bus_id(cluster, chip_id);
    auto tray_bus_id_it = std::find(tray_bus_ids.begin(), tray_bus_ids.end(), bus_id & 0xF0);
    if (tray_bus_id_it != tray_bus_ids.end()) {
        auto ubb_asic_id = bus_id & 0x0F;
        return UbbId{
            static_cast<uint32_t>(tray_bus_id_it - tray_bus_ids.begin() + 1), static_cast<uint32_t>(ubb_asic_id)};
    }
    return UbbId{0, 0};  // Invalid UBB ID if not found
}

void ControlPlane::initialize_dynamic_routing_plane_counts(
    const IntraMeshConnectivity& intra_mesh_connectivity,
    tt_fabric::FabricConfig fabric_config,
    tt_fabric::FabricReliabilityMode reliability_mode) {
    if (fabric_config == tt_fabric::FabricConfig::CUSTOM || fabric_config == tt_fabric::FabricConfig::DISABLED) {
        return;
    }

    this->router_port_directions_to_num_routing_planes_map_.clear();

    auto topology = FabricContext::get_topology_from_config(fabric_config);
    auto apply_min =
        [&](const std::unordered_map<tt::tt_fabric::RoutingDirection, std::vector<tt::tt_fabric::chan_id_t>>&
                port_direction_eth_chans,
            tt::tt_fabric::RoutingDirection direction,
            const std::unordered_map<tt::tt_fabric::RoutingDirection, size_t>& /*golden_link_counts*/,
            size_t& val) {
            if (auto it = port_direction_eth_chans.find(direction); it != port_direction_eth_chans.end()) {
                val = std::min(val, it->second.size());
            }
        };

    // For each mesh in the system
    auto user_meshes = this->get_user_physical_mesh_ids();
    if (reliability_mode == tt::tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE) {
        for (const auto& [fabric_node_id, directions_and_eth_chans] :
             this->router_port_directions_to_physical_eth_chan_map_) {
            for (const auto& [direction, eth_chans] : directions_and_eth_chans) {
                this->router_port_directions_to_num_routing_planes_map_[fabric_node_id][direction] = eth_chans.size();
            }
        }
    }

    std::unordered_map<MeshId, std::unordered_map<ChipId, std::unordered_map<RoutingDirection, size_t>>>
        golden_link_counts;
    TT_FATAL(this->mesh_graph_ != nullptr, "Routing table generator not initialized");
    build_golden_link_counts(this->mesh_graph_->get_intra_mesh_connectivity(), golden_link_counts);
    build_golden_link_counts(this->mesh_graph_->get_inter_mesh_connectivity(), golden_link_counts);

    auto apply_count = [&](FabricNodeId fabric_node_id, RoutingDirection direction, size_t count) {
        if (this->router_port_directions_to_physical_eth_chan_map_.contains(fabric_node_id) &&
            this->router_port_directions_to_physical_eth_chan_map_.at(fabric_node_id).contains(direction) &&
            !this->router_port_directions_to_physical_eth_chan_map_.at(fabric_node_id).at(direction).empty()) {
            this->router_port_directions_to_num_routing_planes_map_[fabric_node_id][direction] = count;
        }
    };

    const auto& distributed_context = tt::tt_metal::MetalContext::instance().full_world_distributed_context();
    // For each mesh in the system
    for (auto mesh_id : user_meshes) {
        const auto& mesh_shape = this->get_physical_mesh_shape(MeshId{mesh_id});
        TT_FATAL(mesh_shape.dims() == 2, "ControlPlane: Only 2D meshes are supported");
        TT_FATAL(mesh_shape[0] > 0, "ControlPlane: Mesh width must be greater than 0");
        TT_FATAL(mesh_shape[1] > 0, "ControlPlane: Mesh height must be greater than 0");

        std::vector<size_t> row_min_planes(mesh_shape[0], std::numeric_limits<size_t>::max());
        std::vector<size_t> col_min_planes(mesh_shape[1], std::numeric_limits<size_t>::max());

        // First pass: Calculate minimums for each row/column
        size_t num_chips_in_mesh = intra_mesh_connectivity[mesh_id.get()].size();
        bool is_single_chip = num_chips_in_mesh == 1 && user_meshes.size() == 1;
        bool may_have_intra_mesh_connectivity = !is_single_chip;

        if (may_have_intra_mesh_connectivity) {
            const auto& local_mesh_coord_range = this->get_coord_range(mesh_id, MeshScope::LOCAL);
            for (const auto& mesh_coord : local_mesh_coord_range) {
                auto fabric_chip_id = this->mesh_graph_->coordinate_to_chip(mesh_id, mesh_coord);
                const auto fabric_node_id = FabricNodeId(mesh_id, fabric_chip_id);
                auto mesh_coord_x = mesh_coord[0];
                auto mesh_coord_y = mesh_coord[1];

                const auto& port_directions = this->router_port_directions_to_physical_eth_chan_map_.at(fabric_node_id);

                const auto& golden_counts = golden_link_counts.at(MeshId{mesh_id}).at(fabric_chip_id);
                apply_min(port_directions, RoutingDirection::E, golden_counts, row_min_planes.at(mesh_coord_x));
                apply_min(port_directions, RoutingDirection::W, golden_counts, row_min_planes.at(mesh_coord_x));
                apply_min(port_directions, RoutingDirection::N, golden_counts, col_min_planes.at(mesh_coord_y));
                apply_min(port_directions, RoutingDirection::S, golden_counts, col_min_planes.at(mesh_coord_y));
            }

            // Collect row and column mins from all hosts in a BigMesh
            auto rows_min = *std::min_element(row_min_planes.begin(), row_min_planes.end());
            auto cols_min = *std::min_element(col_min_planes.begin(), col_min_planes.end());
            std::vector<size_t> rows_min_buf(*distributed_context.size());
            std::vector<size_t> cols_min_buf(*distributed_context.size());
            distributed_context.all_gather(
                tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&rows_min), sizeof(size_t)),
                tt::stl::as_writable_bytes(tt::stl::Span<size_t>{rows_min_buf.data(), rows_min_buf.size()}));
            distributed_context.all_gather(
                tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&cols_min), sizeof(size_t)),
                tt::stl::as_writable_bytes(tt::stl::Span<size_t>{cols_min_buf.data(), cols_min_buf.size()}));
            distributed_context.barrier();
            const auto global_rows_min = std::min_element(rows_min_buf.begin(), rows_min_buf.end());
            const auto global_cols_min = std::min_element(cols_min_buf.begin(), cols_min_buf.end());
            // TODO: specialize by topology for better perf
            if (topology == Topology::Mesh || topology == Topology::Torus) {
                auto global_mesh_min = std::min(*global_rows_min, *global_cols_min);
                std::fill(row_min_planes.begin(), row_min_planes.end(), global_mesh_min);
                std::fill(col_min_planes.begin(), col_min_planes.end(), global_mesh_min);
            } else {
                std::fill(row_min_planes.begin(), row_min_planes.end(), *global_rows_min);
                std::fill(col_min_planes.begin(), col_min_planes.end(), *global_cols_min);
            }

            // Second pass: Apply minimums to each device
            for (const auto& mesh_coord : local_mesh_coord_range) {
                auto fabric_chip_id = this->mesh_graph_->coordinate_to_chip(mesh_id, mesh_coord);
                const auto fabric_node_id = FabricNodeId(mesh_id, fabric_chip_id);
                auto mesh_coord_x = mesh_coord[0];
                auto mesh_coord_y = mesh_coord[1];

                apply_count(fabric_node_id, RoutingDirection::E, row_min_planes.at(mesh_coord_x));
                apply_count(fabric_node_id, RoutingDirection::W, row_min_planes.at(mesh_coord_x));
                apply_count(fabric_node_id, RoutingDirection::N, col_min_planes.at(mesh_coord_y));
                apply_count(fabric_node_id, RoutingDirection::S, col_min_planes.at(mesh_coord_y));
            }
        }
    }
}

LocalMeshBinding ControlPlane::initialize_local_mesh_binding() {
    // When unset, assume host rank 0.
    const char* host_rank_str = std::getenv("TT_MESH_HOST_RANK");
    const MeshHostRankId host_rank = (host_rank_str == nullptr)
                                         ? MeshHostRankId{0}
                                         : MeshHostRankId{static_cast<unsigned int>(std::stoi(host_rank_str))};

    // If TT_MESH_ID is unset, assume this host is the only host in the system and owns all Meshes in
    // the MeshGraphDescriptor. Single Host Multi-Mesh is only used for testing purposes.
    const char* mesh_id_str = std::getenv("TT_MESH_ID");
    if (mesh_id_str == nullptr) {
        const auto& ctx = tt::tt_metal::MetalContext::instance().full_world_distributed_context();
        TT_FATAL(
            *ctx.size() == 1 && *ctx.rank() == 0,
            "Not specifying both TT_MESH_ID and TT_MESH_HOST_RANK is only supported for single host systems.");
        std::vector<MeshId> local_mesh_ids;
        for (const auto& mesh_id : this->mesh_graph_->get_all_mesh_ids()) {
            // TODO: #24528 - Move this to use TopologyMapper once Topology mapper works for multi-mesh systems
            const auto& host_ranks = this->mesh_graph_->get_host_ranks(mesh_id);
            TT_FATAL(
                host_ranks.size() == 1 && *host_ranks.values().front() == 0,
                "Mesh {} has {} host ranks, expected 1",
                *mesh_id,
                host_ranks.size());
            local_mesh_ids.push_back(mesh_id);
        }
        TT_FATAL(!local_mesh_ids.empty(), "No local meshes found.");
        return LocalMeshBinding{.mesh_ids = std::move(local_mesh_ids), .host_rank = MeshHostRankId{0}};
    }

    // Otherwise, use the value from the environment variable.
    auto local_mesh_binding = LocalMeshBinding{
        .mesh_ids = {MeshId{static_cast<unsigned int>(std::stoi(mesh_id_str))}}, .host_rank = host_rank};

    log_debug(
        tt::LogDistributed,
        "Local mesh binding: mesh_id: {}, host_rank: {}",
        local_mesh_binding.mesh_ids[0],
        local_mesh_binding.host_rank);

    // Validate the local mesh binding exists in the mesh graph descriptor
    const auto mesh_ids = this->mesh_graph_->get_all_mesh_ids();
    TT_FATAL(
        std::find(mesh_ids.begin(), mesh_ids.end(), local_mesh_binding.mesh_ids[0]) != mesh_ids.end(),
        "Invalid TT_MESH_ID: Local mesh binding mesh_id {} not found in mesh graph descriptor",
        *local_mesh_binding.mesh_ids[0]);

    // Validate host rank (only if mesh_id is valid)
    const auto& host_ranks = this->mesh_graph_->get_host_ranks(local_mesh_binding.mesh_ids[0]).values();
    if (host_rank_str == nullptr) {
        local_mesh_binding.host_rank = MeshHostRankId{0};
        TT_FATAL(
            host_ranks.size() == 1 && *host_ranks.front() == 0,
            "TT_MESH_HOST_RANK must be set when multiple host ranks are present in the mesh graph descriptor for mesh "
            "ID {}",
            *local_mesh_binding.mesh_ids[0]);
    } else {
        TT_FATAL(
            std::find(host_ranks.begin(), host_ranks.end(), local_mesh_binding.host_rank) != host_ranks.end(),
            "Invalid TT_MESH_HOST_RANK: Local mesh binding host_rank {} not found in mesh graph descriptor",
            *local_mesh_binding.host_rank);
    }

    return local_mesh_binding;
}

void ControlPlane::initialize_distributed_contexts() {
    const auto& global_context = tt::tt_metal::distributed::multihost::DistributedContext::get_current_world();
    if (*global_context->size() == 1) {
        host_local_context_ = global_context;
        std::transform(
            local_mesh_binding_.mesh_ids.begin(),
            local_mesh_binding_.mesh_ids.end(),
            std::inserter(distributed_contexts_, distributed_contexts_.end()),
            [&](const MeshId& mesh_id) { return std::make_pair(mesh_id, global_context); });
        return;
    }

    std::array this_host = {*global_context->rank()};
    host_local_context_ = global_context->create_sub_context(this_host);

    // Use mesh_graph to get all (mesh_id, host_rank) pairs (this follows topology_mapper's mesh_rank_bindings),
    // then use topology_mapper's helper function to get the MPI rank for each (mesh_id, host_rank) pair.
    for (const auto& mesh_id : this->mesh_graph_->get_all_mesh_ids()) {
        const auto& host_ranks = this->mesh_graph_->get_host_ranks(mesh_id);
        for (const auto& [_, mesh_host_rank] : host_ranks) {
            int mpi_rank = topology_mapper_->get_mpi_rank_for_mesh_host_rank(mesh_id, mesh_host_rank);
            mpi_ranks_[mesh_id][mesh_host_rank] = tt::tt_metal::distributed::multihost::Rank{mpi_rank};
            global_logical_bindings_[tt::tt_metal::distributed::multihost::Rank{mpi_rank}] = {mesh_id, mesh_host_rank};
        }
    }

    // Create a sub-context for each mesh-host-rank pair.
    for (const auto local_mesh_id : local_mesh_binding_.mesh_ids) {
        auto mesh_host_ranks = mpi_ranks_.find(local_mesh_id);
        TT_FATAL(mesh_host_ranks != mpi_ranks_.end(), "Mesh {} not found in mpi_ranks.", local_mesh_id);
        if (mesh_host_ranks->second.size() == 1) {
            distributed_contexts_.emplace(local_mesh_id, host_local_context_);
        } else {
            std::vector<int> mpi_neighbors;
            // Sort mesh_host_ranks->second for deterministic iteration across hosts
            std::vector<std::pair<MeshHostRankId, tt::tt_metal::distributed::multihost::Rank>> sorted_host_ranks(
                mesh_host_ranks->second.begin(), mesh_host_ranks->second.end());
            std::sort(sorted_host_ranks.begin(), sorted_host_ranks.end(), [](const auto& a, const auto& b) {
                return a.first.get() < b.first.get();
            });
            std::transform(
                sorted_host_ranks.begin(),
                sorted_host_ranks.end(),
                std::back_inserter(mpi_neighbors),
                [](const auto& p) { return p.second.get(); });
            std::sort(mpi_neighbors.begin(), mpi_neighbors.end());
            distributed_contexts_.emplace(local_mesh_id, global_context->create_sub_context(mpi_neighbors));
        }
    }
}

FabricNodeId ControlPlane::get_fabric_node_id_from_asic_id(uint64_t asic_id) const {
    // Check cache first for faster lookup
    auto cache_it = asic_id_to_fabric_node_cache_.find(asic_id);
    if (cache_it != asic_id_to_fabric_node_cache_.end()) {
        return cache_it->second;
    }

    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto& chip_unique_ids = cluster.get_unique_chip_ids();

    for (const auto& [physical_chip_id, unique_id] : chip_unique_ids) {
        if (unique_id == asic_id) {
            FabricNodeId fabric_node_id = this->get_fabric_node_id_from_physical_chip_id(physical_chip_id);
            // Cache the result for future lookups
            asic_id_to_fabric_node_cache_.emplace(asic_id, fabric_node_id);
            return fabric_node_id;
        }
    }

    TT_FATAL(false, "FabricNodeId not found for ASIC ID {}", asic_id);
    return FabricNodeId(MeshId{0}, 0);
}

void ControlPlane::init_control_plane(
    const std::string& mesh_graph_desc_file,
    std::optional<std::reference_wrapper<const std::map<FabricNodeId, ChipId>>>
        logical_mesh_chip_id_to_physical_chip_id_mapping) {
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto& driver = cluster.get_driver();
    const auto& distributed_context = tt_metal::distributed::multihost::DistributedContext::get_current_world();
    const auto& rtoptions = tt::tt_metal::MetalContext::instance().rtoptions();
    auto fabric_config = tt::tt_metal::MetalContext::instance().get_fabric_config();

    // Create mesh_graph first
    this->mesh_graph_ = std::make_unique<MeshGraph>(mesh_graph_desc_file, fabric_config);

    this->physical_system_descriptor_ = std::make_unique<tt::tt_metal::PhysicalSystemDescriptor>(
        driver, distributed_context, &tt::tt_metal::MetalContext::instance().hal(), rtoptions);
    this->local_mesh_binding_ = this->initialize_local_mesh_binding();

    if (logical_mesh_chip_id_to_physical_chip_id_mapping.has_value()) {
        // Initialize topology mapper with provided mapping, skipping discovery
        this->topology_mapper_ = std::make_unique<tt::tt_fabric::TopologyMapper>(
            *this->mesh_graph_,
            *this->physical_system_descriptor_,
            this->local_mesh_binding_,
            logical_mesh_chip_id_to_physical_chip_id_mapping->get());
        this->load_physical_chip_mapping(logical_mesh_chip_id_to_physical_chip_id_mapping->get());
    } else {
        std::vector<std::pair<AsicPosition, FabricNodeId>> fixed_asic_position_pinnings;

        // Pin the start of the mesh to match the Galaxy Topology, ensuring that external QSFP links align with the
        // corner node IDs of the fabric mesh. This is a performance optimization to ensure that MGD mapping does not
        // bisect a device.
        const bool is_1d = this->mesh_graph_->get_mesh_shape(MeshId{0})[0] == 1 ||
                           this->mesh_graph_->get_mesh_shape(MeshId{0})[1] == 1;
        const size_t board_size = cluster.get_unique_chip_ids().size();
        const size_t distributed_size = *distributed_context->size();

        // Limiting this for single-host galaxy systems only because the dateline could be placed differently,
        // multi-host machines should be limited via rank bindings so should be ok
        if (cluster.is_ubb_galaxy() && !is_1d && board_size == 32 &&
            distributed_size == 1) {  // Using full board size for UBB Galaxy
            fixed_asic_position_pinnings = get_galaxy_fixed_asic_position_pinnings(board_size);
        }
        this->topology_mapper_ = std::make_unique<tt::tt_fabric::TopologyMapper>(
            *this->mesh_graph_,
            *this->physical_system_descriptor_,
            this->local_mesh_binding_,
            fixed_asic_position_pinnings);
        this->load_physical_chip_mapping(
            topology_mapper_->get_local_logical_mesh_chip_id_to_physical_chip_id_mapping());
    }

    // Automatically export physical chip mesh coordinate mapping to generated/fabric directory after topology mapper is
    // created This ensures ttnn-visualizer topology remains functional
    const auto& global_context = tt::tt_metal::distributed::multihost::DistributedContext::get_current_world();
    int world_size = *global_context->size();
    int rank = *global_context->rank();
    std::filesystem::path output_file = std::filesystem::path(rtoptions.get_root_dir()) / "generated" / "fabric" /
                                        ("physical_chip_mesh_coordinate_mapping_" + std::to_string(rank + 1) + "_of_" +
                                         std::to_string(world_size) + ".yaml");
    try {
        tt::tt_fabric::serialize_mesh_coordinates_to_file(*this->topology_mapper_, output_file);
    } catch (const std::exception& e) {
        log_warning(tt::LogFabric, "Failed to export physical chip mesh coordinate mapping: {}", e.what());
    }

    // Initialize routing table generator after topology_mapper is created
    this->routing_table_generator_ = std::make_unique<RoutingTableGenerator>(*this->topology_mapper_);

    // Initialize distributed contexts after topology_mapper is created so we can use its helper function
    this->initialize_distributed_contexts();
    this->generate_intermesh_connectivity();

    // Printing, only enabled with log_debug
    this->mesh_graph_->print_connectivity();
}

void ControlPlane::init_control_plane_auto_discovery() {
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto& driver = cluster.get_driver();
    const auto& distributed_context = tt_metal::distributed::multihost::DistributedContext::get_current_world();
    const auto& rtoptions = tt::tt_metal::MetalContext::instance().rtoptions();
    auto fabric_config = tt::tt_metal::MetalContext::instance().get_fabric_config();

    // NOTE: This algorithm is only supported for single host systems for now
    TT_FATAL(
        *distributed_context->size() == 1,
        "Auto discovery is only supported for single host systems, since you are running on a {} process,"
        " please specify a rank binding file via the tt-run argument --rank-binding argument",
        *distributed_context->size());

    // Initialize physical system descriptor
    this->physical_system_descriptor_ = std::make_unique<tt::tt_metal::PhysicalSystemDescriptor>(
        driver, distributed_context, &tt::tt_metal::MetalContext::instance().hal(), rtoptions);

    // Generate Mesh graph based on physical system descriptor
    // Reliability mode is obtained from MetalContext inside the function
    this->mesh_graph_ = std::make_unique<tt::tt_fabric::MeshGraph>(
        tt::tt_fabric::TopologyMapper::generate_mesh_graph_from_physical_system_descriptor(
            *this->physical_system_descriptor_, fabric_config));

    this->local_mesh_binding_ = this->initialize_local_mesh_binding();

    std::vector<std::pair<AsicPosition, FabricNodeId>> fixed_asic_position_pinnings;

    // Pin the start of the mesh to match the Galaxy Topology, ensuring that external QSFP links align with the
    // corner node IDs of the fabric mesh. This is a performance optimization to ensure that MGD mapping does not
    // bisect a device.
    const bool is_1d =
        this->mesh_graph_->get_mesh_shape(MeshId{0})[0] == 1 || this->mesh_graph_->get_mesh_shape(MeshId{0})[1] == 1;
    const size_t board_size = cluster.get_unique_chip_ids().size();
    const size_t distributed_size = *distributed_context->size();

    // Limiting this for single-host galaxy systems only because the dateline could be placed differently,
    // multi-host machines should be limited via rank bindings so should be ok
    if (cluster.is_ubb_galaxy() && !is_1d && board_size == 32 &&
        distributed_size == 1) {  // Using full board size for UBB Galaxy
        fixed_asic_position_pinnings = get_galaxy_fixed_asic_position_pinnings(board_size);
    }
    this->topology_mapper_ = std::make_unique<tt::tt_fabric::TopologyMapper>(
        *this->mesh_graph_,
        *this->physical_system_descriptor_,
        this->local_mesh_binding_,
        fixed_asic_position_pinnings);
    this->load_physical_chip_mapping(topology_mapper_->get_local_logical_mesh_chip_id_to_physical_chip_id_mapping());

    // Automatically export physical chip mesh coordinate mapping to generated/fabric directory after topology mapper is
    // created This ensures ttnn-visualizer topology remains functional
    const auto& global_context = tt::tt_metal::distributed::multihost::DistributedContext::get_current_world();
    int world_size = *global_context->size();
    int rank = *global_context->rank();
    std::filesystem::path output_file = std::filesystem::path(rtoptions.get_root_dir()) / "generated" / "fabric" /
                                        ("physical_chip_mesh_coordinate_mapping_" + std::to_string(rank + 1) + "_of_" +
                                         std::to_string(world_size) + ".yaml");
    try {
        tt::tt_fabric::serialize_mesh_coordinates_to_file(*this->topology_mapper_, output_file);
    } catch (const std::exception& e) {
        log_warning(tt::LogFabric, "Failed to export physical chip mesh coordinate mapping: {}", e.what());
    }

    // Initialize routing table generator after topology_mapper is created
    this->routing_table_generator_ = std::make_unique<RoutingTableGenerator>(*this->topology_mapper_);

    // Initialize distributed contexts after topology_mapper is created so we can use its helper function
    this->initialize_distributed_contexts();
    this->generate_intermesh_connectivity();

    // Printing, only enabled with log_debug
    this->mesh_graph_->print_connectivity();
}

ControlPlane::ControlPlane() { init_control_plane_auto_discovery(); }

ControlPlane::ControlPlane(const std::string& mesh_graph_desc_file) {
    init_control_plane(mesh_graph_desc_file, std::nullopt);
}

ControlPlane::ControlPlane(
    const std::string& mesh_graph_desc_file,
    const std::map<FabricNodeId, ChipId>& logical_mesh_chip_id_to_physical_chip_id_mapping) {
    init_control_plane(mesh_graph_desc_file, logical_mesh_chip_id_to_physical_chip_id_mapping);
}

void ControlPlane::load_physical_chip_mapping(
    const std::map<FabricNodeId, ChipId>& logical_mesh_chip_id_to_physical_chip_id_mapping) {
    this->logical_mesh_chip_id_to_physical_chip_id_mapping_ = logical_mesh_chip_id_to_physical_chip_id_mapping;
    this->validate_mesh_connections();
}

void ControlPlane::validate_mesh_connections(MeshId mesh_id) const {
    MeshShape mesh_shape = mesh_graph_->get_mesh_shape(mesh_id);
    auto get_physical_chip_id = [&](const MeshCoordinate& mesh_coord) {
        auto fabric_chip_id = this->mesh_graph_->coordinate_to_chip(mesh_id, mesh_coord);
        return logical_mesh_chip_id_to_physical_chip_id_mapping_.at(FabricNodeId(mesh_id, fabric_chip_id));
    };
    auto validate_chip_connections = [&](const MeshCoordinate& mesh_coord, const MeshCoordinate& other_mesh_coord) {
        ChipId physical_chip_id = get_physical_chip_id(mesh_coord);
        ChipId physical_chip_id_other = get_physical_chip_id(other_mesh_coord);
        auto eth_links = get_ethernet_cores_grouped_by_connected_chips(physical_chip_id);
        auto eth_links_to_other = eth_links.find(physical_chip_id_other);
        TT_FATAL(
            eth_links_to_other != eth_links.end(),
            "Chip {} not connected to chip {}",
            physical_chip_id,
            physical_chip_id_other);
    };
    const auto& mesh_coord_range = this->get_coord_range(mesh_id, MeshScope::LOCAL);
    for (const auto& mesh_coord : mesh_coord_range) {
        auto mode = mesh_coord_range.get_boundary_mode();

        auto col_neighbor = mesh_coord.get_neighbor(mesh_shape, 1, 1, mode);
        auto row_neighbor = mesh_coord.get_neighbor(mesh_shape, 1, 0, mode);

        if (col_neighbor.has_value() && mesh_coord_range.contains(*col_neighbor)) {
            validate_chip_connections(mesh_coord, *col_neighbor);
        }
        if (row_neighbor.has_value() && mesh_coord_range.contains(*row_neighbor)) {
            validate_chip_connections(mesh_coord, *row_neighbor);
        }
    }
}

void ControlPlane::validate_mesh_connections() const {
    for (const auto& mesh_id : this->mesh_graph_->get_all_mesh_ids()) {
        if (this->is_local_mesh(mesh_id)) {
            this->validate_mesh_connections(mesh_id);
        }
    }
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

    // TODO: for now disable collapsing routing planes until we add the corresponding logic for
    //     connecting the routers on these planes
    // If no match found, return a channel from candidate_target_chans
    // Enabled for TG Dispatch on Fabric
    // TODO: https://github.com/tenstorrent/tt-metal/issues/24413
    if (tt::tt_metal::MetalContext::instance().get_cluster().get_cluster_type() == tt::tt_metal::ClusterType::TG) {
        while (src_routing_plane_id >= candidate_target_chans.size()) {
            src_routing_plane_id = src_routing_plane_id % candidate_target_chans.size();
        }
        return candidate_target_chans[src_routing_plane_id];
    }

    return eth_chan_magic_values::INVALID_DIRECTION;
};

void ControlPlane::convert_fabric_routing_table_to_chip_routing_table() {
    // Routing tables contain direction from chip to chip
    // Convert it to be unique per ethernet channel

    auto host_rank_id = this->get_local_host_rank_id_binding();
    const auto& router_intra_mesh_routing_table = this->routing_table_generator_->get_intra_mesh_table();
    // Get the number of ports per chip from a local mesh
    std::uint32_t num_ports_per_chip = 0;
    for (std::uint32_t mesh_id_val = 0; mesh_id_val < router_intra_mesh_routing_table.size(); mesh_id_val++) {
        MeshId mesh_id{mesh_id_val};
        if (this->is_local_mesh(mesh_id)) {
            // Get the number of ports per chip from any chip in the local mesh
            auto local_mesh_chip_id_container = this->topology_mapper_->get_chip_ids(mesh_id, host_rank_id);

            for (const auto& [_, src_fabric_chip_id] : local_mesh_chip_id_container) {
                const auto src_fabric_node_id = FabricNodeId(mesh_id, src_fabric_chip_id);
                auto physical_chip_id = get_physical_chip_id_from_fabric_node_id(src_fabric_node_id);
                num_ports_per_chip = tt::tt_metal::MetalContext::instance()
                                         .get_cluster()
                                         .get_soc_desc(physical_chip_id)
                                         .get_cores(CoreType::ETH)
                                         .size();
                break;
            }
        }
    }
    for (std::uint32_t mesh_id_val = 0; mesh_id_val < router_intra_mesh_routing_table.size(); mesh_id_val++) {
        MeshId mesh_id{mesh_id_val};
        const auto& global_mesh_chip_id_container = this->mesh_graph_->get_chip_ids(mesh_id);
        for (const auto& [_, src_fabric_chip_id] : global_mesh_chip_id_container) {
            const auto src_fabric_node_id = FabricNodeId(mesh_id, src_fabric_chip_id);
            this->intra_mesh_routing_tables_[src_fabric_node_id].resize(
                num_ports_per_chip);  // contains more entries than needed, this size is for all eth channels on chip
            for (int i = 0; i < num_ports_per_chip; i++) {
                // Size the routing table to the number of chips in the mesh
                this->intra_mesh_routing_tables_[src_fabric_node_id][i].resize(
                    router_intra_mesh_routing_table[mesh_id_val][src_fabric_chip_id].size());
            }
            // Dst is looped over all chips in the mesh, regardless of whether they are local or not
            for (ChipId dst_fabric_chip_id = 0;
                 dst_fabric_chip_id < router_intra_mesh_routing_table[mesh_id_val][src_fabric_chip_id].size();
                 dst_fabric_chip_id++) {
                // Target direction is the direction to the destination chip for all ethernet channesl
                const auto& target_direction =
                    router_intra_mesh_routing_table[mesh_id_val][src_fabric_chip_id][dst_fabric_chip_id];
                // We view ethernet channels on one side of the chip as parallel planes. So N[0] talks to S[0], E[0],
                // W[0] and so on For all live ethernet channels on this chip, set the routing table entry to the
                // destination chip as the ethernet channel on the same plane
                for (const auto& [direction, eth_chans_on_side] :
                     this->router_port_directions_to_physical_eth_chan_map_.at(src_fabric_node_id)) {
                    for (const auto& src_chan_id : eth_chans_on_side) {
                        if (src_fabric_chip_id == dst_fabric_chip_id) {
                            TT_ASSERT(
                                (target_direction == RoutingDirection::C),
                                "Expecting same direction for intra mesh routing");
                            // This entry represents chip to itself, should not be used by FW
                            this->intra_mesh_routing_tables_.at(src_fabric_node_id)[src_chan_id][dst_fabric_chip_id] =
                                src_chan_id;
                        } else if (target_direction == direction) {
                            // This entry represents an outgoing eth channel
                            this->intra_mesh_routing_tables_.at(src_fabric_node_id)[src_chan_id][dst_fabric_chip_id] =
                                src_chan_id;
                        } else {
                            const auto& eth_chans_in_target_direction =
                                this->router_port_directions_to_physical_eth_chan_map_.at(
                                    src_fabric_node_id)[target_direction];
                            const auto src_routing_plane_id =
                                this->get_routing_plane_id(src_chan_id, eth_chans_on_side);
                            this->intra_mesh_routing_tables_.at(src_fabric_node_id)[src_chan_id][dst_fabric_chip_id] =
                                this->get_downstream_eth_chan_id(src_routing_plane_id, eth_chans_in_target_direction);
                        }
                    }
                }
            }
        }
    }
    const auto& router_inter_mesh_routing_table = this->routing_table_generator_->get_inter_mesh_table();
    for (std::uint32_t src_mesh_id_val = 0; src_mesh_id_val < router_inter_mesh_routing_table.size();
         src_mesh_id_val++) {
        MeshId src_mesh_id{src_mesh_id_val};
        const auto& global_mesh_chip_id_container = this->mesh_graph_->get_chip_ids(src_mesh_id);
        for (const auto& [_, src_fabric_chip_id] : global_mesh_chip_id_container) {
            const auto src_fabric_node_id = FabricNodeId(src_mesh_id, src_fabric_chip_id);
            this->inter_mesh_routing_tables_[src_fabric_node_id].resize(
                num_ports_per_chip);  // contains more entries than needed
            for (int i = 0; i < num_ports_per_chip; i++) {
                // Size the routing table to the number of meshes
                this->inter_mesh_routing_tables_[src_fabric_node_id][i].resize(
                    router_inter_mesh_routing_table[src_mesh_id_val][src_fabric_chip_id].size());
            }
            for (ChipId dst_mesh_id_val = 0;
                 dst_mesh_id_val < router_inter_mesh_routing_table[src_mesh_id_val][src_fabric_chip_id].size();
                 dst_mesh_id_val++) {
                // Target direction is the direction to the destination mesh for all ethernet channesl
                const auto& target_direction =
                    router_inter_mesh_routing_table[src_mesh_id_val][src_fabric_chip_id][dst_mesh_id_val];

                // We view ethernet channels on one side of the chip as parallel planes. So N[0] talks to S[0], E[0],
                // W[0] and so on For all live ethernet channels on this chip, set the routing table entry to the
                // destination mesh as the ethernet channel on the same plane
                for (const auto& [direction, eth_chans_on_side] :
                     this->router_port_directions_to_physical_eth_chan_map_.at(src_fabric_node_id)) {
                    for (const auto& src_chan_id : eth_chans_on_side) {
                        if (src_mesh_id_val == dst_mesh_id_val) {
                            TT_ASSERT(
                                (target_direction == RoutingDirection::C),
                                "ControlPlane: Expecting same direction for inter mesh routing");
                            // This entry represents mesh to itself, should not be used by FW
                            this->inter_mesh_routing_tables_.at(src_fabric_node_id)[src_chan_id][dst_mesh_id_val] =
                                src_chan_id;
                        } else if (target_direction == RoutingDirection::NONE) {
                            // This entry represents a mesh to mesh connection that is not reachable
                            // Set to an invalid channel id
                            this->inter_mesh_routing_tables_.at(src_fabric_node_id)[src_chan_id][dst_mesh_id_val] =
                                eth_chan_magic_values::INVALID_DIRECTION;
                        } else if (target_direction == direction) {
                            // This entry represents an outgoing eth channel
                            this->inter_mesh_routing_tables_.at(src_fabric_node_id)[src_chan_id][dst_mesh_id_val] =
                                src_chan_id;
                        } else {
                            const auto& eth_chans_in_target_direction =
                                this->router_port_directions_to_physical_eth_chan_map_.at(
                                    src_fabric_node_id)[target_direction];
                            const auto src_routing_plane_id =
                                this->get_routing_plane_id(src_chan_id, eth_chans_on_side);
                            this->inter_mesh_routing_tables_.at(src_fabric_node_id)[src_chan_id][dst_mesh_id_val] =
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

// order ethernet channels using translated coordinates
void ControlPlane::order_ethernet_channels() {
    for (auto& [fabric_node_id, eth_chans_by_dir] : this->router_port_directions_to_physical_eth_chan_map_) {
        auto phys_chip_id = this->get_physical_chip_id_from_fabric_node_id(fabric_node_id);
        const auto src_asic_id = tt::tt_metal::AsicID{
            tt::tt_metal::MetalContext::instance().get_cluster().get_unique_chip_ids().at(phys_chip_id)};
        const auto& asic_neighbors = physical_system_descriptor_->get_asic_neighbors(src_asic_id);
        for (auto& [direction, eth_chans] : eth_chans_by_dir) {
            std::optional<tt::tt_metal::AsicID> neighbor_asic_id;
            std::vector<tt::tt_metal::EthConnection> eth_connections;
            for (const auto& asic_neighbor : asic_neighbors) {
                eth_connections = physical_system_descriptor_->get_eth_connections(src_asic_id, asic_neighbor);
                for (const auto& eth_connection : eth_connections) {
                    if (std::find(eth_chans.begin(), eth_chans.end(), eth_connection.src_chan) != eth_chans.end() &&
                        eth_connections.size() == eth_chans.size()) {
                        neighbor_asic_id = asic_neighbor;
                        break;
                    }
                }
                if (neighbor_asic_id.has_value()) {
                    break;
                }
            }
            const auto& soc_desc = tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(phys_chip_id);
            if (neighbor_asic_id.has_value() && src_asic_id > neighbor_asic_id.value()) {
                std::sort(eth_chans.begin(), eth_chans.end(), [&soc_desc](const auto& a, const auto& b) {
                    auto translated_coords_a = soc_desc.get_eth_core_for_channel(a, CoordSystem::TRANSLATED);
                    auto translated_coords_b = soc_desc.get_eth_core_for_channel(b, CoordSystem::TRANSLATED);
                    return translated_coords_a.x < translated_coords_b.x;
                });
            } else if (neighbor_asic_id.has_value()) {
                // Find the physical chip ID for the neighbor AsicID
                ChipId neighbor_phys_chip_id = 0;
                const auto& chip_unique_ids =
                    tt::tt_metal::MetalContext::instance().get_cluster().get_unique_chip_ids();
                for (const auto& [physical_chip_id, unique_id] : chip_unique_ids) {
                    if (tt::tt_metal::AsicID{unique_id} == neighbor_asic_id.value()) {
                        neighbor_phys_chip_id = physical_chip_id;
                        break;
                    }
                }
                // Get the soc_desc for the neighbor chip
                const auto& neighbor_soc_desc =
                    tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(neighbor_phys_chip_id);
                std::sort(
                    eth_connections.begin(), eth_connections.end(), [&neighbor_soc_desc](const auto& a, const auto& b) {
                        auto translated_coords_a =
                            neighbor_soc_desc.get_eth_core_for_channel(a.dst_chan, CoordSystem::TRANSLATED);
                        auto translated_coords_b =
                            neighbor_soc_desc.get_eth_core_for_channel(b.dst_chan, CoordSystem::TRANSLATED);
                        return translated_coords_a.x < translated_coords_b.x;
                    });
                for (uint32_t i = 0; i < eth_connections.size(); i++) {
                    eth_chans[i] = eth_connections[i].src_chan;
                }
            }
        }
    }
}

void ControlPlane::trim_ethernet_channels_not_mapped_to_live_routing_planes() {
    auto user_mesh_ids = this->get_user_physical_mesh_ids();
    std::unordered_set<MeshId> user_mesh_ids_set(user_mesh_ids.begin(), user_mesh_ids.end());
    if (tt::tt_metal::MetalContext::instance().get_fabric_config() != tt_fabric::FabricConfig::CUSTOM) {
        for (auto& [fabric_node_id, directional_eth_chans] : this->router_port_directions_to_physical_eth_chan_map_) {
            if (!user_mesh_ids_set.contains(fabric_node_id.mesh_id)) {
                continue;
            }
            for (auto direction :
                 {RoutingDirection::N, RoutingDirection::S, RoutingDirection::E, RoutingDirection::W}) {
                if (directional_eth_chans.contains(direction)) {
                    size_t num_available_routing_planes = this->get_num_live_routing_planes(fabric_node_id, direction);
                    TT_FATAL(
                        directional_eth_chans.at(direction).size() >= num_available_routing_planes,
                        "Expected {} eth channels on M{}D{} in direction {}, but got {}",
                        num_available_routing_planes,
                        fabric_node_id.mesh_id,
                        fabric_node_id.chip_id,
                        direction,
                        directional_eth_chans.at(direction).size());
                    bool trim = directional_eth_chans.at(direction).size() > num_available_routing_planes;
                    auto physical_chip_id = this->logical_mesh_chip_id_to_physical_chip_id_mapping_.at(fabric_node_id);
                    if (trim) {
                        log_warning(
                            tt::LogFabric,
                            "phys {} M{}D{} in direction {} has {} eth channels, but only {} routing planes are "
                            "available",
                            physical_chip_id,
                            fabric_node_id.mesh_id,
                            fabric_node_id.chip_id,
                            direction,
                            directional_eth_chans.at(direction).size(),
                            num_available_routing_planes);
                    }
                    directional_eth_chans.at(direction).resize(num_available_routing_planes);
                }
            }
        }
    }
}

size_t ControlPlane::get_num_live_routing_planes(
    FabricNodeId fabric_node_id, RoutingDirection routing_direction) const {
    TT_FATAL(
        this->router_port_directions_to_num_routing_planes_map_.contains(fabric_node_id),
        "Fabric node id (mesh={}, chip={}) not found in router port directions to num routing planes map",
        fabric_node_id.mesh_id,
        fabric_node_id.chip_id);
    TT_FATAL(
        this->router_port_directions_to_num_routing_planes_map_.at(fabric_node_id).contains(routing_direction),
        "Routing direction {} not found in router port directions to num routing planes map for fabric node id "
        "(mesh={}, chip={})",
        routing_direction,
        fabric_node_id.mesh_id,
        fabric_node_id.chip_id);
    return this->router_port_directions_to_num_routing_planes_map_.at(fabric_node_id).at(routing_direction);
}

// Only builds the routing table representation, does not actually populate the routing tables in memory of the
// fabric routers on device
void ControlPlane::configure_routing_tables_for_fabric_ethernet_channels(
    tt::tt_fabric::FabricConfig fabric_config, tt_fabric::FabricReliabilityMode reliability_mode) {
    this->intra_mesh_routing_tables_.clear();
    this->inter_mesh_routing_tables_.clear();
    this->router_port_directions_to_physical_eth_chan_map_.clear();

    const auto& intra_mesh_connectivity = this->mesh_graph_->get_intra_mesh_connectivity();
    // Initialize the bookkeeping for mapping from mesh/chip/direction to physical ethernet channels
    for (const auto& [fabric_node_id, _] : this->logical_mesh_chip_id_to_physical_chip_id_mapping_) {
        if (!this->router_port_directions_to_physical_eth_chan_map_.contains(fabric_node_id)) {
            this->router_port_directions_to_physical_eth_chan_map_[fabric_node_id] = {};
        }
    }

    auto host_rank_id = this->get_local_host_rank_id_binding();
    const auto& my_host = physical_system_descriptor_->my_host_name();
    const auto& neighbor_hosts = physical_system_descriptor_->get_host_neighbors(my_host);

    for (std::uint32_t mesh_id_val = 0; mesh_id_val < intra_mesh_connectivity.size(); mesh_id_val++) {
        // run for all meshes. intra_mesh_connectivity.size() == number of meshes in the system
        // TODO: we can probably remove this check, in general should update these loops to iterate over local meshes
        MeshId mesh_id{mesh_id_val};
        if (!this->is_local_mesh(mesh_id)) {
            continue;
        }
        const auto& local_mesh_coord_range = this->get_coord_range(mesh_id, MeshScope::LOCAL);

        MeshContainer<ChipId> local_mesh_chip_id_container =
            this->topology_mapper_->get_chip_ids(mesh_id, host_rank_id);

        for (const auto& [_, fabric_chip_id] : local_mesh_chip_id_container) {
            const auto fabric_node_id = FabricNodeId(mesh_id, fabric_chip_id);
            auto physical_chip_id = this->get_physical_chip_id_from_fabric_node_id(fabric_node_id);
            auto asic_id = this->topology_mapper_->get_asic_id_from_fabric_node_id(fabric_node_id);

            for (const auto& [logical_connected_chip_id, edge] : intra_mesh_connectivity[*mesh_id][fabric_chip_id]) {
                auto connected_mesh_coord = this->mesh_graph_->chip_to_coordinate(mesh_id, logical_connected_chip_id);
                if (local_mesh_coord_range.contains(connected_mesh_coord)) {
                    // This is a local chip, so we can use the logical chip id directly
                    TT_ASSERT(
                        this->logical_mesh_chip_id_to_physical_chip_id_mapping_.contains(
                            FabricNodeId(mesh_id, logical_connected_chip_id)),
                        "Mesh {} Chip {} not found in logical mesh chip id to physical chip id mapping",
                        mesh_id,
                        logical_connected_chip_id);
                    const auto& physical_connected_chip_id = this->logical_mesh_chip_id_to_physical_chip_id_mapping_.at(
                        FabricNodeId(mesh_id, logical_connected_chip_id));

                    const auto& connected_chips_and_eth_cores =
                        tt::tt_metal::MetalContext::instance()
                            .get_cluster()
                            .get_ethernet_cores_grouped_by_connected_chips(physical_chip_id);

                    // If connected_chips_and_eth_cores contains physical_connected_chip_id then atleast one connection
                    // exists to physical_connected_chip_id
                    bool connections_exist = connected_chips_and_eth_cores.contains(physical_connected_chip_id);
                    TT_FATAL(
                        connections_exist ||
                            reliability_mode != tt::tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE,
                        "Expected connections to exist for M{}D{} to D{}",
                        mesh_id,
                        fabric_chip_id,
                        logical_connected_chip_id);
                    if (!connections_exist) {
                        continue;
                    }

                    const auto& connected_eth_cores = connected_chips_and_eth_cores.at(physical_connected_chip_id);
                    if (reliability_mode == tt::tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE) {
                        TT_FATAL(
                            connected_eth_cores.size() >= edge.connected_chip_ids.size(),
                            "Expected {} eth links from physical chip {} to physical chip {}",
                            edge.connected_chip_ids.size(),
                            physical_chip_id,
                            physical_connected_chip_id);
                    }

                    for (const auto& eth_core : connected_eth_cores) {
                        // There could be an optimization here to create entry for both chips here, assuming links are
                        // bidirectional
                        this->assign_direction_to_fabric_eth_core(fabric_node_id, eth_core, edge.port_direction);
                    }
                } else {
                    auto host_rank_for_chip =
                        this->topology_mapper_->get_host_rank_for_chip(mesh_id, logical_connected_chip_id);

                    TT_ASSERT(
                        host_rank_for_chip.has_value(),
                        "Mesh {} Chip {} does not have a host rank associated with it",
                        *mesh_id,
                        logical_connected_chip_id);
                    auto connected_host_rank_id = host_rank_for_chip.value();

                    // Iterate over all neighboring hosts
                    // Check if the neighbor belongs to the same mesh and owns the connected chip
                    // If so, iterate over all cross host connections between the neighbors
                    // Assign this edge to all links on the local chip part of this intramesh connection
                    for (const auto& neighbor_host : neighbor_hosts) {
                        auto neighbor_host_rank = physical_system_descriptor_->get_rank_for_hostname(neighbor_host);
                        auto neighbor_mesh_id =
                            this->global_logical_bindings_
                                .at(tt::tt_metal::distributed::multihost::Rank{static_cast<int>(neighbor_host_rank)})
                                .first;
                        auto neighbor_mesh_host_rank =
                            this->global_logical_bindings_
                                .at(tt::tt_metal::distributed::multihost::Rank{static_cast<int>(neighbor_host_rank)})
                                .second;
                        if (neighbor_mesh_id == mesh_id && neighbor_mesh_host_rank == connected_host_rank_id) {
                            const auto& neighbor_exit_nodes =
                                physical_system_descriptor_->get_connecting_exit_nodes(my_host, neighbor_host);
                            for (const auto& exit_node : neighbor_exit_nodes) {
                                if (*exit_node.src_exit_node == *asic_id) {
                                    this->assign_direction_to_fabric_eth_chan(
                                        fabric_node_id, exit_node.eth_conn.src_chan, edge.port_direction);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    for (const auto& [exit_node_fabric_node_id, exit_node_directions] : this->exit_node_directions_) {
        for (const auto& [src_eth_chan, port_direction] : exit_node_directions) {
            this->assign_direction_to_fabric_eth_chan(exit_node_fabric_node_id, src_eth_chan, port_direction);
        }
    }

    this->initialize_dynamic_routing_plane_counts(intra_mesh_connectivity, fabric_config, reliability_mode);

    // Order the ethernet channels so that when we use them for deciding connections, indexing into ports per direction
    // is consistent for each each neighbouring chip.
    this->order_ethernet_channels();

    // Trim the ethernet channels that don't map to live fabric routing planes.
    // NOTE: This MUST be called after ordering ethernet channels
    this->trim_ethernet_channels_not_mapped_to_live_routing_planes();

    this->collect_and_merge_router_port_directions_from_all_hosts();

    this->convert_fabric_routing_table_to_chip_routing_table();
    // After this, router_port_directions_to_physical_eth_chan_map_, intra_mesh_routing_tables_,
    // inter_mesh_routing_tables_ should be populated for all hosts in BigMesh
}

FabricNodeId ControlPlane::get_fabric_node_id_from_physical_chip_id(ChipId physical_chip_id) const {
    for (const auto& [fabric_node_id, mapped_physical_chip_id] :
         this->logical_mesh_chip_id_to_physical_chip_id_mapping_) {
        if (mapped_physical_chip_id == physical_chip_id) {
            return fabric_node_id;
        }
    }
    TT_FATAL(
        false,
        "Physical chip id {} not found in control plane chip mapping. You are calling for a chip outside of the fabric "
        "cluster. Check that your mesh graph descriptor specifies the correct topology",
        physical_chip_id);
    return FabricNodeId(MeshId{0}, 0);
}

ChipId ControlPlane::get_physical_chip_id_from_fabric_node_id(const FabricNodeId& fabric_node_id) const {
    TT_ASSERT(logical_mesh_chip_id_to_physical_chip_id_mapping_.contains(fabric_node_id));
    return logical_mesh_chip_id_to_physical_chip_id_mapping_.at(fabric_node_id);
}

std::pair<FabricNodeId, chan_id_t> ControlPlane::get_connected_mesh_chip_chan_ids(
    FabricNodeId fabric_node_id, chan_id_t chan_id) const {
    // TODO: simplify this and use Global Physical Desc in ControlPlane soon
    const auto& intra_mesh_connectivity = this->mesh_graph_->get_intra_mesh_connectivity();
    const auto& inter_mesh_connectivity = this->mesh_graph_->get_inter_mesh_connectivity();
    RoutingDirection port_direction = RoutingDirection::NONE;
    routing_plane_id_t routing_plane_id = 0;
    for (const auto& [direction, eth_chans] :
         this->router_port_directions_to_physical_eth_chan_map_.at(fabric_node_id)) {
        for (const auto& eth_chan : eth_chans) {
            if (eth_chan == chan_id) {
                port_direction = direction;
                routing_plane_id = this->get_routing_plane_id(eth_chan, eth_chans);
                break;
            }
        }
    }

    // Try to find the connected mesh chip chan ids for the given port direction in intra mesh connectivity
    const auto& intra_mesh_node = intra_mesh_connectivity[*fabric_node_id.mesh_id][fabric_node_id.chip_id];
    for (const auto& [dst_fabric_chip_id, edge] : intra_mesh_node) {
        if (edge.port_direction == port_direction) {
            // Get reverse port direction
            TT_ASSERT(
                intra_mesh_connectivity[*fabric_node_id.mesh_id][dst_fabric_chip_id].contains(fabric_node_id.chip_id),
                "Intra mesh connectivity from {} to {} not found",
                dst_fabric_chip_id,
                fabric_node_id.chip_id);
            RoutingDirection reverse_port_direction =
                intra_mesh_connectivity[*fabric_node_id.mesh_id][dst_fabric_chip_id]
                    .at(fabric_node_id.chip_id)
                    .port_direction;
            // Find the eth chan on connected dst_fabric_chip_id based on routing_plane_id
            const auto& dst_fabric_node = FabricNodeId(fabric_node_id.mesh_id, dst_fabric_chip_id);
            const auto& dst_fabric_chip_eth_chans =
                this->router_port_directions_to_physical_eth_chan_map_.at(dst_fabric_node);
            for (const auto& [direction, eth_chans] : dst_fabric_chip_eth_chans) {
                if (direction == reverse_port_direction) {
                    return std::make_pair(dst_fabric_node, eth_chans[routing_plane_id]);
                }
            }
        }
    }

    // Try to find the connected mesh chip chan ids for the given port direction in inter mesh connectivity
    const auto& inter_mesh_node = inter_mesh_connectivity[*fabric_node_id.mesh_id][fabric_node_id.chip_id];
    for (const auto& [dst_fabric_mesh_id, edge] : inter_mesh_node) {
        if (edge.port_direction == port_direction) {
            // Get reverse port direction
            const auto& dst_connected_fabric_chip_id = edge.connected_chip_ids[0];
            TT_ASSERT(
                inter_mesh_connectivity[*dst_fabric_mesh_id][dst_connected_fabric_chip_id].contains(
                    fabric_node_id.mesh_id),
                "Inter mesh connectivity from {} to {} not found",
                dst_fabric_mesh_id,
                fabric_node_id.mesh_id);
            RoutingDirection reverse_port_direction =
                inter_mesh_connectivity[*dst_fabric_mesh_id][dst_connected_fabric_chip_id]
                    .at(fabric_node_id.mesh_id)
                    .port_direction;
            // Find the eth chan on connected dst_fabric_mesh_id based on routing_plane_id
            const auto& dst_fabric_node = FabricNodeId(dst_fabric_mesh_id, dst_connected_fabric_chip_id);
            const auto& dst_fabric_chip_eth_chans =
                this->router_port_directions_to_physical_eth_chan_map_.at(dst_fabric_node);
            for (const auto& [direction, eth_chans] : dst_fabric_chip_eth_chans) {
                if (direction == reverse_port_direction) {
                    if (routing_plane_id >= eth_chans.size()) {
                        // Only TG non-standard intermesh connections hits this
                        return std::make_pair(dst_fabric_node, eth_chans[0]);
                    }
                    return std::make_pair(dst_fabric_node, eth_chans[routing_plane_id]);
                }
            }
        }
    }
    TT_FATAL(false, "Could not find connected mesh chip chan ids for {} on chan {}", fabric_node_id, chan_id);
    return std::make_pair(FabricNodeId(MeshId{0}, 0), 0);
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
        case RoutingDirection::Z: return static_cast<eth_chan_directions>(eth_chan_magic_values::INVALID_DIRECTION);
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

std::vector<chan_id_t> ControlPlane::get_intermesh_facing_eth_chans(FabricNodeId fabric_node_id) const {
    std::vector<chan_id_t> channels;
    auto it = this->exit_node_directions_.find(fabric_node_id);
    if (it == this->exit_node_directions_.end()) {
        return channels;
    }
    channels.reserve(it->second.size());
    for (const auto& [chan_id, _] : it->second) {
        channels.push_back(chan_id);
    }
    return channels;
}

std::vector<chan_id_t> ControlPlane::get_intramesh_facing_eth_chans(FabricNodeId fabric_node_id) const {
    std::vector<chan_id_t> channels;
    if (!this->router_port_directions_to_physical_eth_chan_map_.contains(fabric_node_id)) {
        return channels;
    }

    std::unordered_set<chan_id_t> intermesh_channels;
    if (auto it = this->exit_node_directions_.find(fabric_node_id); it != this->exit_node_directions_.end()) {
        for (const auto& [chan_id, _] : it->second) {
            intermesh_channels.insert(chan_id);
        }
    }

    const auto& dir_map = this->router_port_directions_to_physical_eth_chan_map_.at(fabric_node_id);
    for (const auto& [_, eth_chans] : dir_map) {
        for (const auto& chan_id : eth_chans) {
            if (!intermesh_channels.contains(chan_id)) {
                channels.push_back(chan_id);
            }
        }
    }

    std::sort(channels.begin(), channels.end());
    channels.erase(std::unique(channels.begin(), channels.end()), channels.end());
    return channels;
}

std::vector<std::pair<FabricNodeId, chan_id_t>> ControlPlane::get_fabric_route(
    FabricNodeId src_fabric_node_id, FabricNodeId dst_fabric_node_id, chan_id_t src_chan_id) const {
    // Query the mesh coord range owned by the current host
    auto host_local_coord_range = this->get_coord_range(this->get_local_mesh_id_bindings()[0], MeshScope::LOCAL);
    auto src_mesh_coord = this->mesh_graph_->chip_to_coordinate(src_fabric_node_id.mesh_id, src_fabric_node_id.chip_id);
    auto dst_mesh_coord = this->mesh_graph_->chip_to_coordinate(dst_fabric_node_id.mesh_id, dst_fabric_node_id.chip_id);

    std::vector<std::pair<FabricNodeId, chan_id_t>> route;
    int i = 0;
    while (src_fabric_node_id != dst_fabric_node_id) {
        i++;
        auto src_mesh_id = src_fabric_node_id.mesh_id;
        auto src_chip_id = src_fabric_node_id.chip_id;
        auto dst_mesh_id = dst_fabric_node_id.mesh_id;
        auto dst_chip_id = dst_fabric_node_id.chip_id;
        if (i >= tt::tt_fabric::MAX_MESH_SIZE * tt::tt_fabric::MAX_NUM_MESHES) {
            log_warning(
                tt::LogFabric, "Could not find a route between {} and {}", src_fabric_node_id, dst_fabric_node_id);
            return {};
        }
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
            log_warning(
                tt::LogFabric, "Could not find a route between {} and {}", src_fabric_node_id, dst_fabric_node_id);
            return {};
        }
        if (src_chan_id != next_chan_id) {
            // Chan to chan within chip
            route.push_back({src_fabric_node_id, next_chan_id});
        }
        std::tie(src_fabric_node_id, src_chan_id) =
            this->get_connected_mesh_chip_chan_ids(src_fabric_node_id, next_chan_id);
        route.push_back({src_fabric_node_id, src_chan_id});
    }
    return route;
}

std::optional<RoutingDirection> ControlPlane::get_forwarding_direction(
    FabricNodeId src_fabric_node_id, FabricNodeId dst_fabric_node_id) const {
    auto src_mesh_id = src_fabric_node_id.mesh_id;
    auto src_chip_id = src_fabric_node_id.chip_id;
    auto dst_mesh_id = dst_fabric_node_id.mesh_id;
    auto dst_chip_id = dst_fabric_node_id.chip_id;
    // TODO: remove returning of std::nullopt, and just return NONE value
    // Tests and usage should check for NONE value
    if (src_mesh_id != dst_mesh_id) {
        const auto& inter_mesh_routing_table = this->routing_table_generator_->get_inter_mesh_table();
        if (inter_mesh_routing_table[*src_mesh_id][src_chip_id][*dst_mesh_id] != RoutingDirection::NONE) {
            return inter_mesh_routing_table[*src_mesh_id][src_chip_id][*dst_mesh_id];
        }
    } else if (src_chip_id != dst_chip_id) {
        const auto& intra_mesh_routing_table = this->routing_table_generator_->get_intra_mesh_table();
        if (intra_mesh_routing_table[*src_mesh_id][src_chip_id][dst_chip_id] != RoutingDirection::NONE) {
            return intra_mesh_routing_table[*src_mesh_id][src_chip_id][dst_chip_id];
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

stl::Span<const ChipId> ControlPlane::get_intra_chip_neighbors(
    FabricNodeId src_fabric_node_id, RoutingDirection routing_direction) const {
    for (const auto& [_, routing_edge] :
         this->mesh_graph_->get_intra_mesh_connectivity()[*src_fabric_node_id.mesh_id][src_fabric_node_id.chip_id]) {
        if (routing_edge.port_direction == routing_direction) {
            return routing_edge.connected_chip_ids;
        }
    }
    return {};
}

std::unordered_map<MeshId, std::vector<ChipId>> ControlPlane::get_chip_neighbors(
    FabricNodeId src_fabric_node_id, RoutingDirection routing_direction) const {
    std::unordered_map<MeshId, std::vector<ChipId>> neighbors;
    auto intra_neighbors = this->get_intra_chip_neighbors(src_fabric_node_id, routing_direction);
    auto src_mesh_id = src_fabric_node_id.mesh_id;
    auto src_chip_id = src_fabric_node_id.chip_id;
    if (!intra_neighbors.empty()) {
        neighbors[src_mesh_id].insert(neighbors[src_mesh_id].end(), intra_neighbors.begin(), intra_neighbors.end());
    }
    for (const auto& [mesh_id, routing_edge] :
         this->mesh_graph_->get_inter_mesh_connectivity()[*src_mesh_id][src_chip_id]) {
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

void write_to_worker_or_fabric_tensix_cores(
    const void* worker_data,
    const void* dispatcher_data,
    const void* tensix_extension_data,
    size_t size,
    tt::tt_metal::HalL1MemAddrType addr_type,
    ChipId physical_chip_id) {
    TT_FATAL(
        size ==
            tt_metal::MetalContext::instance().hal().get_dev_size(tt_metal::HalProgrammableCoreType::TENSIX, addr_type),
        "ControlPlane: Tensix core data size mismatch expected {} but got {}",
        size,
        tt_metal::MetalContext::instance().hal().get_dev_size(tt_metal::HalProgrammableCoreType::TENSIX, addr_type));

    const auto& soc_desc = tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(physical_chip_id);
    const std::vector<tt::umd::CoreCoord>& all_tensix_cores =
        soc_desc.get_cores(CoreType::TENSIX, CoordSystem::TRANSLATED);

    // Check if tensix config is enabled
    bool tensix_config_enabled = tt::tt_metal::MetalContext::instance().get_fabric_tensix_config() !=
                                 tt::tt_fabric::FabricTensixConfig::DISABLED;

    // Get pre-computed translated fabric mux cores from tensix config
    std::unordered_set<CoreCoord> fabric_mux_cores_translated;
    std::unordered_set<CoreCoord> dispatch_mux_cores_translated;
    if (tensix_config_enabled) {
        const auto& fabric_context = tt::tt_metal::MetalContext::instance().get_control_plane().get_fabric_context();
        const auto& tensix_config = fabric_context.get_builder_context().get_tensix_config();
        fabric_mux_cores_translated = tensix_config.get_translated_fabric_mux_cores();
        dispatch_mux_cores_translated = tensix_config.get_translated_dispatch_mux_cores();
    }

    enum class CoreType { Worker, FabricTensixExtension, DispatcherMux };

    auto get_core_type = [&](const CoreCoord& core_coord) -> CoreType {
        if (fabric_mux_cores_translated.contains(core_coord)) {
            return CoreType::FabricTensixExtension;
        }
        if (dispatch_mux_cores_translated.contains(core_coord)) {
            return CoreType::DispatcherMux;
        }
        return CoreType::Worker;
    };

    auto select_data = [&](CoreType core_type) -> const void* {
        if (tensix_config_enabled) {
            switch (core_type) {
                case CoreType::FabricTensixExtension: return worker_data;
                case CoreType::DispatcherMux: return dispatcher_data;
                case CoreType::Worker: return tensix_extension_data;
                default: TT_THROW("unknown core type: {}", core_type);
            }
        } else {
            return worker_data;
        }
    };

    for (const auto& tensix_core : all_tensix_cores) {
        CoreCoord core_coord(tensix_core.x, tensix_core.y);
        CoreType core_type = get_core_type(core_coord);
        const void* data_to_write = select_data(core_type);

        tt::tt_metal::MetalContext::instance().get_cluster().write_core(
            data_to_write,
            size,
            tt_cxy_pair(physical_chip_id, core_coord),
            tt_metal::MetalContext::instance().hal().get_dev_addr(
                tt_metal::HalProgrammableCoreType::TENSIX, addr_type));
    }
}

static void write_to_all_cores(
    const void* data,
    size_t size,
    tt::tt_metal::HalL1MemAddrType addr_type,
    ChipId physical_chip_id,
    tt::tt_metal::HalProgrammableCoreType core_type) {
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();

    const char* type_label = "Unknown";
    switch (core_type) {
        case tt::tt_metal::HalProgrammableCoreType::TENSIX: type_label = "Tensix"; break;
        case tt::tt_metal::HalProgrammableCoreType::IDLE_ETH: type_label = "Idle ETH"; break;
        case tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH: type_label = "Active ETH"; break;
        default: break;
    }

    TT_FATAL(
        size == tt::tt_metal::MetalContext::instance().hal().get_dev_size(core_type, addr_type),
        "ControlPlane: {} core data size mismatch expected {} but got {}",
        type_label,
        tt::tt_metal::MetalContext::instance().hal().get_dev_size(core_type, addr_type),
        size);

    switch (core_type) {
        case tt::tt_metal::HalProgrammableCoreType::TENSIX: {
            const auto& soc_desc = cluster.get_soc_desc(physical_chip_id);
            const std::vector<tt::umd::CoreCoord>& tensix_cores =
                soc_desc.get_cores(CoreType::TENSIX, CoordSystem::TRANSLATED);
            for (const auto& tensix_core : tensix_cores) {
                tt::tt_metal::MetalContext::instance().get_cluster().write_core(
                    data,
                    size,
                    tt_cxy_pair(physical_chip_id, CoreCoord(tensix_core.x, tensix_core.y)),
                    tt::tt_metal::MetalContext::instance().hal().get_dev_addr(core_type, addr_type));
            }
            break;
        }
        case tt::tt_metal::HalProgrammableCoreType::IDLE_ETH:
        case tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH: {
            std::unordered_set<CoreCoord> logical_eth_cores =
                (core_type == tt::tt_metal::HalProgrammableCoreType::IDLE_ETH)
                    ? tt::tt_metal::MetalContext::instance().get_control_plane().get_inactive_ethernet_cores(
                          physical_chip_id)
                    : tt::tt_metal::MetalContext::instance().get_control_plane().get_active_ethernet_cores(
                          physical_chip_id);
            for (const CoreCoord& logical_eth_core : logical_eth_cores) {
                CoreCoord virtual_eth_core = cluster.get_virtual_coordinate_from_logical_coordinates(
                    physical_chip_id, logical_eth_core, CoreType::ETH);
                tt::tt_metal::MetalContext::instance().get_cluster().write_core(
                    data,
                    size,
                    tt_cxy_pair(physical_chip_id, CoreCoord(virtual_eth_core.x, virtual_eth_core.y)),
                    tt::tt_metal::MetalContext::instance().hal().get_dev_addr(core_type, addr_type));
            }
            break;
        }
        default: TT_THROW("Unsupported core type {}", enchantum::to_string(core_type));
    }
}

// Helper functions to compute and embed routing path tables
void ControlPlane::compute_and_embed_1d_routing_path_table(MeshId mesh_id, routing_l1_info_t& routing_info) const {
    auto host_rank_id = this->get_local_host_rank_id_binding();
    const auto& local_mesh_chip_id_container = this->topology_mapper_->get_chip_ids(mesh_id, host_rank_id);
    uint16_t num_chips = MAX_CHIPS_LOWLAT_1D < local_mesh_chip_id_container.size()
                             ? MAX_CHIPS_LOWLAT_1D
                             : static_cast<uint16_t>(local_mesh_chip_id_container.size());

    intra_mesh_routing_path_t<1, false> routing_path_1d;
    routing_path_1d.calculate_chip_to_all_routing_fields(FabricNodeId(mesh_id, 0), num_chips);

    std::memcpy(&routing_info.routing_path_table_1d, &routing_path_1d, sizeof(intra_mesh_routing_path_t<1, false>));
}

void ControlPlane::compute_and_embed_2d_routing_path_table(
    MeshId mesh_id, ChipId chip_id, routing_l1_info_t& routing_info) const {
    auto host_rank_id = this->get_local_host_rank_id_binding();
    auto local_mesh_chip_id_container = this->topology_mapper_->get_chip_ids(mesh_id, host_rank_id);

    bool chip_is_local_to_host = false;
    for (const auto& [_, local_chip_id] : local_mesh_chip_id_container) {
        if (local_chip_id == chip_id) {
            chip_is_local_to_host = true;
            break;
        }
    }
    TT_ASSERT(
        chip_is_local_to_host,
        "2D routing path: chip {} is not owned by local host_rank {} for mesh {}",
        chip_id,
        *host_rank_id,
        *mesh_id);

    // Calculate routing using global mesh geometry (device tables are indexed by global chip ids)
    MeshShape mesh_shape = this->get_physical_mesh_shape(mesh_id, MeshScope::GLOBAL);
    uint16_t num_chips = mesh_shape[0] * mesh_shape[1];
    TT_ASSERT(num_chips <= 256, "Number of chips exceeds 256 for mesh {}", *mesh_id);
    TT_ASSERT(
        mesh_shape[0] <= 32 && mesh_shape[1] <= 32,
        "One or both of mesh axis exceed 32 for mesh {}: {}x{}",
        *mesh_id,
        mesh_shape[0],
        mesh_shape[1]);

    intra_mesh_routing_path_t<2, true> routing_path_2d;
    routing_path_2d.calculate_chip_to_all_routing_fields(FabricNodeId(mesh_id, chip_id), num_chips);

    std::memcpy(&routing_info.routing_path_table_2d, &routing_path_2d, sizeof(intra_mesh_routing_path_t<2, true>));

    // Build per-dst-mesh exit node table (1 byte per mesh) for this src chip
    std::uint8_t exit_table[MAX_NUM_MESHES];
    std::fill_n(exit_table, MAX_NUM_MESHES, eth_chan_magic_values::INVALID_ROUTING_TABLE_ENTRY);
    const auto& inter_mesh_table = this->routing_table_generator_->get_inter_mesh_table();
    for (const auto& dst_mesh_id : this->mesh_graph_->get_all_mesh_ids()) {
        auto direction = inter_mesh_table[*mesh_id][chip_id][*dst_mesh_id];
        if (direction == RoutingDirection::NONE) {
            continue;
        }
        auto exit_node = this->routing_table_generator_->get_exit_node_from_mesh_to_mesh(mesh_id, chip_id, dst_mesh_id);
        exit_table[*dst_mesh_id] = static_cast<std::uint8_t>(exit_node.chip_id);
    }
    std::memcpy(&routing_info.exit_node_table, &exit_table, sizeof(std::uint8_t[MAX_NUM_MESHES]));
}

// Write routing table to Tensix cores' L1 on a specific chip
void ControlPlane::write_routing_info_to_devices(MeshId mesh_id, ChipId chip_id) const {
    FabricNodeId src_fabric_node_id{mesh_id, static_cast<uint32_t>(chip_id)};
    auto physical_chip_id = this->logical_mesh_chip_id_to_physical_chip_id_mapping_.at(src_fabric_node_id);

    routing_l1_info_t routing_info = {};
    routing_info.my_mesh_id = *mesh_id;
    routing_info.my_device_id = chip_id;

    // Build intra-mesh routing entries (chip-to-chip routing)
    const auto& router_intra_mesh_routing_table = this->routing_table_generator_->get_intra_mesh_table();
    TT_FATAL(
        router_intra_mesh_routing_table[*mesh_id][chip_id].size() <= tt::tt_fabric::MAX_MESH_SIZE,
        "ControlPlane: Intra mesh routing table size exceeds maximum allowed size");

    // Initialize all entries to INVALID_ROUTING_TABLE_ENTRY first
    for (std::uint32_t i = 0; i < tt::tt_fabric::MAX_MESH_SIZE; i++) {
        routing_info.intra_mesh_direction_table.set_original_direction(
            i, static_cast<std::uint8_t>(eth_chan_magic_values::INVALID_ROUTING_TABLE_ENTRY));
    }

    for (ChipId dst_chip_id = 0; dst_chip_id < router_intra_mesh_routing_table[*mesh_id][chip_id].size();
         dst_chip_id++) {
        if (chip_id == dst_chip_id) {
            routing_info.intra_mesh_direction_table.set_original_direction(
                dst_chip_id, static_cast<std::uint8_t>(eth_chan_magic_values::INVALID_DIRECTION));
            continue;
        }
        auto forwarding_direction = router_intra_mesh_routing_table[*mesh_id][chip_id][dst_chip_id];
        std::uint8_t direction_value =
            forwarding_direction != RoutingDirection::NONE
                ? static_cast<std::uint8_t>(this->routing_direction_to_eth_direction(forwarding_direction))
                : static_cast<std::uint8_t>(eth_chan_magic_values::INVALID_DIRECTION);
        routing_info.intra_mesh_direction_table.set_original_direction(dst_chip_id, direction_value);
    }

    // Build inter-mesh routing entries (mesh-to-mesh routing)
    const auto& router_inter_mesh_routing_table = this->routing_table_generator_->get_inter_mesh_table();
    TT_FATAL(
        router_inter_mesh_routing_table[*mesh_id][chip_id].size() <= tt::tt_fabric::MAX_NUM_MESHES,
        "ControlPlane: Inter mesh routing table size exceeds maximum allowed size");

    // Initialize all entries to INVALID_ROUTING_TABLE_ENTRY first
    for (std::uint32_t i = 0; i < tt::tt_fabric::MAX_NUM_MESHES; i++) {
        routing_info.inter_mesh_direction_table.set_original_direction(
            i, static_cast<std::uint8_t>(eth_chan_magic_values::INVALID_ROUTING_TABLE_ENTRY));
    }

    for (std::uint32_t dst_mesh_id = 0; dst_mesh_id < router_inter_mesh_routing_table[*mesh_id][chip_id].size();
         dst_mesh_id++) {
        if (*mesh_id == dst_mesh_id) {
            routing_info.inter_mesh_direction_table.set_original_direction(
                dst_mesh_id, static_cast<std::uint8_t>(eth_chan_magic_values::INVALID_DIRECTION));
            continue;
        }
        auto forwarding_direction = router_inter_mesh_routing_table[*mesh_id][chip_id][dst_mesh_id];
        std::uint8_t direction_value =
            forwarding_direction != RoutingDirection::NONE
                ? static_cast<std::uint8_t>(this->routing_direction_to_eth_direction(forwarding_direction))
                : static_cast<std::uint8_t>(eth_chan_magic_values::INVALID_DIRECTION);
        routing_info.inter_mesh_direction_table.set_original_direction(dst_mesh_id, direction_value);
    }

    if (this->get_fabric_context().is_2D_routing_enabled()) {
        // Compute and embed 2D routing path table and exit node table (per src chip id)
        compute_and_embed_2d_routing_path_table(mesh_id, chip_id, routing_info);
    } else {
        // Compute and embed 1D routing path table (independent of src chip id)
        compute_and_embed_1d_routing_path_table(mesh_id, routing_info);
    }

    // Finally, write the full routing info to all Tensix cores and mirror to IDLE_ETH routing table
    write_to_all_cores(
        &routing_info,
        sizeof(routing_info),
        tt::tt_metal::HalL1MemAddrType::ROUTING_TABLE,
        physical_chip_id,
        tt::tt_metal::HalProgrammableCoreType::TENSIX);
    write_to_all_cores(
        &routing_info,
        sizeof(routing_info),
        tt::tt_metal::HalL1MemAddrType::ROUTING_TABLE,
        physical_chip_id,
        tt::tt_metal::HalProgrammableCoreType::IDLE_ETH);
    write_to_all_cores(
        &routing_info,
        sizeof(routing_info),
        tt::tt_metal::HalL1MemAddrType::ROUTING_TABLE,
        physical_chip_id,
        tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH);
    tt::tt_metal::MetalContext::instance().get_cluster().l1_barrier(physical_chip_id);
}

// Write connection info to Tensix cores' L1 on a specific chip
void ControlPlane::write_fabric_connections_to_tensix_cores(MeshId mesh_id, ChipId chip_id) const {
    if (this->fabric_context_ == nullptr) {
        log_warning(
            tt::LogFabric,
            "ControlPlane: Fabric context is not set, cannot write fabric connections to Tensix cores for M%dD%d",
            *mesh_id,
            chip_id);
        return;
    }
    FabricNodeId src_fabric_node_id{mesh_id, static_cast<uint32_t>(chip_id)};
    auto physical_chip_id = this->logical_mesh_chip_id_to_physical_chip_id_mapping_.at(src_fabric_node_id);

    tt::tt_fabric::tensix_fabric_connections_l1_info_t fabric_worker_connections = {};
    tt::tt_fabric::tensix_fabric_connections_l1_info_t fabric_dispatcher_connections = {};
    tt::tt_fabric::tensix_fabric_connections_l1_info_t fabric_tensix_connections = {};

    // Get all physically connected ethernet channels directly from the cluster
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto& connected_chips_and_eth_cores = cluster.get_ethernet_cores_grouped_by_connected_chips(physical_chip_id);

    size_t num_eth_endpoint = 0;
    for (const auto& [direction, eth_chans] :
         this->router_port_directions_to_physical_eth_chan_map_.at(src_fabric_node_id)) {
        for (auto eth_channel_id : eth_chans) {
            eth_chan_directions router_direction = this->routing_direction_to_eth_direction(direction);
            if (num_eth_endpoint >= tt::tt_fabric::tensix_fabric_connections_l1_info_t::MAX_FABRIC_ENDPOINTS) {
                log_warning(
                    tt::LogFabric,
                    "ControlPlane: Maximum number of fabric endpoints exceeded for M%dD%d, skipping further "
                    "connections",
                    *mesh_id,
                    chip_id);
                break;
            }

            // Populate connection info for regular fabric connections (for tensix mux cores)
            auto& worker_connection_info = fabric_worker_connections.read_only[eth_channel_id];
            worker_connection_info.edm_direction = router_direction;

            // Populate connection info for dispatcher fabric connections
            auto& dispatcher_connection_info = fabric_dispatcher_connections.read_only[eth_channel_id];
            dispatcher_connection_info.edm_direction = router_direction;

            // Populate connection info for tensix mux connections (for normal worker cores)
            auto& tensix_connection_info = fabric_tensix_connections.read_only[eth_channel_id];
            tensix_connection_info.edm_direction = router_direction;

            // Use helper function to populate both connection types
            this->populate_fabric_connection_info(
                worker_connection_info,
                dispatcher_connection_info,
                tensix_connection_info,
                physical_chip_id,
                eth_channel_id);

            // Mark this connection as valid for fabric communication
            fabric_worker_connections.valid_connections_mask |= (1u << eth_channel_id);
            fabric_dispatcher_connections.valid_connections_mask |= (1u << eth_channel_id);
            fabric_tensix_connections.valid_connections_mask |= (1u << eth_channel_id);
            num_eth_endpoint++;
        }
    }

    const auto& fabric_tensix_config = tt::tt_metal::MetalContext::instance().get_fabric_tensix_config();
    if (fabric_tensix_config == tt::tt_fabric::FabricTensixConfig::UDM) {
        // UDM mode: use per-worker connections
        this->write_udm_fabric_connections_to_tensix_cores(
            physical_chip_id, fabric_worker_connections, fabric_dispatcher_connections);
    } else {
        // Non UDM mode: same connection info for all workers
        write_to_worker_or_fabric_tensix_cores(
            &fabric_worker_connections,      // worker_data - goes to mux cores
            &fabric_dispatcher_connections,  // dispatcher_data - goes to dispatcher cores
            &fabric_tensix_connections,      // tensix_extension_data - goes to worker cores
            sizeof(tt::tt_fabric::tensix_fabric_connections_l1_info_t),
            tt::tt_metal::HalL1MemAddrType::TENSIX_FABRIC_CONNECTIONS,
            physical_chip_id);
    }
}

std::vector<chan_id_t> ControlPlane::get_active_fabric_eth_routing_planes_in_direction(
    FabricNodeId fabric_node_id, RoutingDirection routing_direction) const {
    auto eth_chans = get_active_fabric_eth_channels_in_direction(fabric_node_id, routing_direction);
    size_t num_routing_planes = 0;
    if (this->router_port_directions_to_num_routing_planes_map_.contains(fabric_node_id) &&
        this->router_port_directions_to_num_routing_planes_map_.at(fabric_node_id).contains(routing_direction)) {
        num_routing_planes =
            this->router_port_directions_to_num_routing_planes_map_.at(fabric_node_id).at(routing_direction);
        TT_FATAL(
            eth_chans.size() >= num_routing_planes,
            "Not enough active fabric eth channels for node {} in direction {}. Requested {} routing planes but only "
            "have {} eth channels",
            fabric_node_id,
            routing_direction,
            num_routing_planes,
            eth_chans.size());
        eth_chans.resize(num_routing_planes);
    }
    return eth_chans;
}

size_t ControlPlane::get_num_available_routing_planes_in_direction(
    FabricNodeId fabric_node_id, RoutingDirection routing_direction) const {
    if (this->router_port_directions_to_num_routing_planes_map_.contains(fabric_node_id) &&
        this->router_port_directions_to_num_routing_planes_map_.at(fabric_node_id).contains(routing_direction)) {
        return this->router_port_directions_to_num_routing_planes_map_.at(fabric_node_id).at(routing_direction);
    }
    return 0;
}

void ControlPlane::write_fabric_telemetry_to_all_chips(const FabricNodeId& fabric_node_id) const {
    auto physical_chip_id = this->logical_mesh_chip_id_to_physical_chip_id_mapping_.at(fabric_node_id);
    auto active_ethernet_cores = this->get_active_ethernet_cores(physical_chip_id);

    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    const auto& factory = hal.get_fabric_telemetry_factory(tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH);

    auto telemetry = factory.create<::tt::tt_fabric::fabric_telemetry::FabricTelemetryStaticOnly>();
    auto telemetry_view = telemetry.view();
    auto static_view = telemetry_view.static_info();
    static_view.mesh_id() = static_cast<std::uint16_t>(*fabric_node_id.mesh_id);
    static_view.device_id() = static_cast<std::uint8_t>(fabric_node_id.chip_id);
    static_view.direction() = 0;  // TODO: populate from routing direction when available.
    static_view.fabric_config() =
        static_cast<std::uint32_t>(tt::tt_metal::MetalContext::instance().get_fabric_config());
    static_view.supported_stats() = ::tt::tt_fabric::fabric_telemetry::DynamicStatistics::NONE;

    for (const auto& active_ethernet_core : active_ethernet_cores) {
        auto chan_id = tt::tt_metal::MetalContext::instance()
                           .get_cluster()
                           .get_soc_desc(physical_chip_id)
                           .logical_eth_core_to_chan_map.at(active_ethernet_core);

        // auto routing_direction = get_eth_chan_direction(fabric_node_id, chan_id);
        // static_view.direction() = static_cast<std::uint8_t>(routing_direction);

        CoreCoord virtual_eth_core =
            tt::tt_metal::MetalContext::instance().get_cluster().get_virtual_eth_core_from_channel(
                physical_chip_id, chan_id);
        tt::tt_metal::MetalContext::instance().get_cluster().write_core(
            telemetry.data(),
            telemetry.size(),
            tt_cxy_pair(physical_chip_id, virtual_eth_core),
            hal.get_dev_addr(
                tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::FABRIC_TELEMETRY));
    }
    tt::tt_metal::MetalContext::instance().get_cluster().l1_barrier(physical_chip_id);
}

void ControlPlane::write_routing_tables_to_all_chips() const {
    // Configure the routing tables on the chips
    TT_ASSERT(
        this->intra_mesh_routing_tables_.size() == this->inter_mesh_routing_tables_.size(),
        "Intra mesh routing tables size mismatch with inter mesh routing tables");
    auto user_meshes = this->get_user_physical_mesh_ids();
    for (auto mesh_id : user_meshes) {
        const auto& local_mesh_coord_range = this->get_coord_range(mesh_id, MeshScope::LOCAL);
        for (const auto& mesh_coord : local_mesh_coord_range) {
            auto fabric_chip_id = this->mesh_graph_->coordinate_to_chip(mesh_id, mesh_coord);
            auto fabric_node_id = FabricNodeId(mesh_id, fabric_chip_id);
            TT_ASSERT(
                this->inter_mesh_routing_tables_.contains(fabric_node_id),
                "Intra mesh routing tables keys mismatch with inter mesh routing tables");
            this->write_routing_info_to_devices(fabric_node_id.mesh_id, fabric_node_id.chip_id);
            this->write_fabric_connections_to_tensix_cores(fabric_node_id.mesh_id, fabric_node_id.chip_id);
            this->write_fabric_telemetry_to_all_chips(fabric_node_id);
        }
    }
}

// TODO: remove this after TG is deprecated
std::vector<MeshId> ControlPlane::get_user_physical_mesh_ids() const {
    std::vector<MeshId> physical_mesh_ids;
    const auto user_chips = tt::tt_metal::MetalContext::instance().get_cluster().user_exposed_chip_ids();
    for (const auto& [fabric_node_id, physical_chip_id] : this->logical_mesh_chip_id_to_physical_chip_id_mapping_) {
        if (user_chips.contains(physical_chip_id) and
            std::find(physical_mesh_ids.begin(), physical_mesh_ids.end(), fabric_node_id.mesh_id) ==
                physical_mesh_ids.end()) {
            physical_mesh_ids.push_back(fabric_node_id.mesh_id);
        }
    }
    return physical_mesh_ids;
}

MeshShape ControlPlane::get_physical_mesh_shape(MeshId mesh_id, MeshScope scope) const {
    std::optional<MeshHostRankId> local_host_rank_id =
        MeshScope::LOCAL == scope ? std::make_optional(this->get_local_host_rank_id_binding()) : std::nullopt;

    return this->topology_mapper_->get_mesh_shape(mesh_id, local_host_rank_id);
}

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
            ss << "   " << enchantum::to_string(direction) << ":";
            for (const auto& eth_chan : eth_chans) {
                ss << " " << (std::uint16_t)eth_chan;
            }
            ss << std::endl;
        }
    }
    log_debug(tt::LogFabric, "{}", ss.str());
}

void ControlPlane::initialize_fabric_context(tt_fabric::FabricConfig fabric_config) {
    TT_FATAL(this->fabric_context_ == nullptr, "Trying to re-initialize fabric context");
    this->fabric_context_ = std::make_unique<FabricContext>(fabric_config);
}

FabricContext& ControlPlane::get_fabric_context() const {
    TT_FATAL(this->fabric_context_ != nullptr, "Trying to get un-initialized fabric context");
    return *this->fabric_context_;
}

std::map<std::string, std::string> ControlPlane::get_fabric_kernel_defines() const {
    if (this->fabric_context_ == nullptr) {
        return {};
    }

    return this->fabric_context_->get_fabric_kernel_defines();
}

void ControlPlane::clear_fabric_context() {
    this->fabric_context_.reset(nullptr);
    asic_id_to_fabric_node_cache_.clear();
}

void ControlPlane::initialize_fabric_tensix_datamover_config() {
    TT_FATAL(this->fabric_context_ != nullptr, "Fabric context must be initialized first");
    this->fabric_context_->get_builder_context().initialize_tensix_config();
}

bool ControlPlane::is_cross_host_eth_link(ChipId chip_id, chan_id_t chan_id) const {
    auto asic_id = tt::tt_metal::MetalContext::instance().get_cluster().get_unique_chip_ids().at(chip_id);
    return this->physical_system_descriptor_->is_cross_host_eth_link(tt::tt_metal::AsicID{asic_id}, chan_id);
}

std::unordered_set<CoreCoord> ControlPlane::get_active_ethernet_cores(ChipId chip_id, bool skip_reserved_cores) const {
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();

    std::unordered_set<CoreCoord> active_ethernet_cores;
    const auto& cluster_desc = cluster.get_cluster_desc();
    const auto& soc_desc = cluster.get_soc_desc(chip_id);

    // Check if there are any ethernet cores available on this chip
    if (soc_desc.logical_eth_core_to_chan_map.empty()) {
        return active_ethernet_cores;  // Return empty set if no ethernet cores
    }

    if (cluster.arch() == ARCH::BLACKHOLE) {
        // Can't just use `get_ethernet_cores_grouped_by_connected_chips` because there are some active ethernet cores
        // without links. Only risc1 on these cores is available for Metal and should not be classified as idle
        // to ensure that Metal does not try to program both riscs.
        std::set<uint32_t> logical_active_eth_channels = cluster_desc->get_active_eth_channels(chip_id);
        for (auto logical_active_eth_channel : logical_active_eth_channels) {
            tt::umd::CoreCoord logical_active_eth =
                soc_desc.get_eth_core_for_channel(logical_active_eth_channel, CoordSystem::LOGICAL);
            active_ethernet_cores.insert(CoreCoord(logical_active_eth.x, logical_active_eth.y));
        }
    } else {
        std::set<uint32_t> logical_active_eth_channels = cluster_desc->get_active_eth_channels(chip_id);
        const auto& freq_retrain_eth_cores = cluster.get_eth_cores_with_frequent_retraining(chip_id);
        const auto& eth_routing_info = cluster.get_eth_routing_info(chip_id);
        for (const auto& eth_channel : logical_active_eth_channels) {
            tt::umd::CoreCoord eth_core = soc_desc.get_eth_core_for_channel(eth_channel, CoordSystem::LOGICAL);
            const auto& routing_info = eth_routing_info.at(eth_core);
            if (routing_info == EthRouterMode::FABRIC_ROUTER && skip_reserved_cores) {
                continue;
            }
            if (freq_retrain_eth_cores.contains(eth_core)) {
                continue;
            }

            active_ethernet_cores.insert(eth_core);
        }
        // WH has a special case where mmio chips with remote connections must always have certain channels active
        if (cluster.arch() == tt::ARCH::WORMHOLE_B0 && cluster_desc->is_chip_mmio_capable(chip_id) &&
            !cluster.get_tunnels_from_mmio_device(chip_id).empty()) {
            // UMD routing FW uses these cores for base routing
            // channel 15 is used by syseng tools
            std::unordered_set<int> channels_to_skip = {};
            if (cluster.is_galaxy_cluster()) {
                // TODO: This may need to change, if we need additional eth cores for dispatch on Galaxy
                channels_to_skip = {0, 1, 2, 3, 15};
            } else {
                channels_to_skip = {15};
            }
            for (const auto& eth_channel : channels_to_skip) {
                if (!logical_active_eth_channels.contains(eth_channel)) {
                    tt::umd::CoreCoord eth_core = soc_desc.get_eth_core_for_channel(eth_channel, CoordSystem::LOGICAL);
                    active_ethernet_cores.insert(eth_core);
                }
            }
        }
    }
    return active_ethernet_cores;
}

std::unordered_set<CoreCoord> ControlPlane::get_inactive_ethernet_cores(ChipId chip_id) const {
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    std::unordered_set<CoreCoord> active_ethernet_cores = this->get_active_ethernet_cores(chip_id);
    std::unordered_set<CoreCoord> inactive_ethernet_cores;

    for (const auto& [eth_core, chan] : cluster.get_soc_desc(chip_id).logical_eth_core_to_chan_map) {
        if (!active_ethernet_cores.contains(eth_core)) {
            inactive_ethernet_cores.insert(eth_core);
        }
    }
    return inactive_ethernet_cores;
}

void ControlPlane::assign_direction_to_fabric_eth_chan(
    const FabricNodeId& fabric_node_id, chan_id_t chan_id, RoutingDirection direction) {
    auto physical_chip_id = this->logical_mesh_chip_id_to_physical_chip_id_mapping_.at(fabric_node_id);
    // TODO: get_fabric_ethernet_channels accounts for down links, but we should manage down links in control plane
    auto fabric_router_channels_on_chip =
        tt::tt_metal::MetalContext::instance().get_cluster().get_fabric_ethernet_channels(physical_chip_id);

    // TODO: add logic here to disable unsed routers, e.g. Mesh on Torus system
    if (fabric_router_channels_on_chip.contains(chan_id)) {
        this->router_port_directions_to_physical_eth_chan_map_.at(fabric_node_id)[direction].push_back(chan_id);
    } else {
        log_debug(
            tt::LogFabric,
            "Control Plane: Disabling router on M{}D{} eth channel {}",
            fabric_node_id.mesh_id,
            fabric_node_id.chip_id,
            chan_id);
    }
}

void ControlPlane::assign_direction_to_fabric_eth_core(
    const FabricNodeId& fabric_node_id, const CoreCoord& eth_core, RoutingDirection direction) {
    auto physical_chip_id = this->logical_mesh_chip_id_to_physical_chip_id_mapping_.at(fabric_node_id);
    auto chan_id = tt::tt_metal::MetalContext::instance()
                       .get_cluster()
                       .get_soc_desc(physical_chip_id)
                       .logical_eth_core_to_chan_map.at(eth_core);
    this->assign_direction_to_fabric_eth_chan(fabric_node_id, chan_id, direction);
}

const MeshGraph& ControlPlane::get_mesh_graph() const { return *mesh_graph_; }

std::vector<MeshId> ControlPlane::get_local_mesh_id_bindings() const {
    const auto& mesh_id_bindings = this->local_mesh_binding_.mesh_ids;
    const auto& user_mesh_ids = this->get_user_physical_mesh_ids();
    std::vector<MeshId> local_mesh_ids;
    for (const auto& mesh_id : mesh_id_bindings) {
        if (std::find(user_mesh_ids.begin(), user_mesh_ids.end(), mesh_id) != user_mesh_ids.end()) {
            local_mesh_ids.push_back(mesh_id);
        }
    }
    TT_FATAL(!local_mesh_ids.empty(), "No local mesh ids found");
    return local_mesh_ids;
}

MeshHostRankId ControlPlane::get_local_host_rank_id_binding() const { return this->local_mesh_binding_.host_rank; }

MeshCoordinate ControlPlane::get_local_mesh_offset() const {
    auto coord_range = this->get_coord_range(this->get_local_mesh_id_bindings()[0], MeshScope::LOCAL);
    return coord_range.start_coord();
}

MeshCoordinateRange ControlPlane::get_coord_range(MeshId mesh_id, MeshScope scope) const {
    std::optional<MeshHostRankId> local_host_rank_id =
        MeshScope::LOCAL == scope ? std::make_optional(this->get_local_host_rank_id_binding()) : std::nullopt;

    return this->topology_mapper_->get_coord_range(mesh_id, local_host_rank_id);
}

bool ControlPlane::is_local_mesh(MeshId mesh_id) const {
    const auto& local_mesh_ids = local_mesh_binding_.mesh_ids;
    return std::find(local_mesh_ids.begin(), local_mesh_ids.end(), mesh_id) != local_mesh_ids.end();
}

const std::shared_ptr<tt::tt_metal::distributed::multihost::DistributedContext>& ControlPlane::get_distributed_context(
    MeshId mesh_id) const {
    auto distributed_context = distributed_contexts_.find(mesh_id);
    TT_FATAL(distributed_context != distributed_contexts_.end(), "Unknown mesh id: {}", mesh_id);
    return distributed_context->second;
}

const std::shared_ptr<tt::tt_metal::distributed::multihost::DistributedContext>& ControlPlane::get_host_local_context()
    const {
    return host_local_context_;
}

const std::unordered_map<tt_metal::distributed::multihost::Rank, std::pair<MeshId, MeshHostRankId>>&
ControlPlane::get_global_logical_bindings() const {
    return global_logical_bindings_;
}

// Helper function to fill connection info with common fields for fabric router configs
void fill_connection_info_fields(
    tt::tt_fabric::fabric_connection_info_t& connection_info,
    const CoreCoord& virtual_core,
    const FabricEriscDatamoverConfig& config,
    uint32_t sender_channel,
    uint16_t worker_free_slots_stream_id) {
    auto* channel_allocator = config.channel_allocator.get();
    auto* const static_channel_allocator =
        dynamic_cast<tt::tt_fabric::FabricStaticSizedChannelsAllocator*>(channel_allocator);
    TT_FATAL(static_channel_allocator != nullptr, "Channel allocator must be a FabricStaticSizedChannelsAllocator.");
    connection_info.edm_noc_x = static_cast<uint8_t>(virtual_core.x);
    connection_info.edm_noc_y = static_cast<uint8_t>(virtual_core.y);
    connection_info.edm_buffer_base_addr = static_channel_allocator->get_sender_channel_base_address(sender_channel);
    connection_info.num_buffers_per_channel =
        static_channel_allocator->get_sender_channel_number_of_slots(sender_channel);
    connection_info.edm_connection_handshake_addr = config.sender_channels_connection_semaphore_address[sender_channel];
    connection_info.edm_worker_location_info_addr =
        config.sender_channels_worker_conn_info_base_address[sender_channel];
    connection_info.buffer_size_bytes = config.channel_buffer_size_bytes;
    connection_info.buffer_index_semaphore_id = config.sender_channels_buffer_index_semaphore_address[sender_channel];
    connection_info.worker_free_slots_stream_id = worker_free_slots_stream_id;
}

// Helper function to fill tensix connection info with tensix-specific configuration
void fill_tensix_connection_info_fields(
    tt::tt_fabric::fabric_connection_info_t& connection_info,
    const CoreCoord& mux_core_virtual,
    const tt::tt_fabric::FabricTensixDatamoverConfig& tensix_config,
    uint32_t sender_channel,
    tt::tt_fabric::FabricTensixCoreType core_id) {
    connection_info.edm_noc_x = static_cast<uint8_t>(mux_core_virtual.x);
    connection_info.edm_noc_y = static_cast<uint8_t>(mux_core_virtual.y);
    connection_info.edm_buffer_base_addr = tensix_config.get_channels_base_address(core_id, sender_channel);
    connection_info.num_buffers_per_channel = tensix_config.get_num_buffers_per_channel();
    connection_info.buffer_size_bytes = tensix_config.get_buffer_size_bytes_full_size_channel();
    connection_info.edm_connection_handshake_addr =
        tensix_config.get_connection_semaphore_address(sender_channel, core_id);
    connection_info.edm_worker_location_info_addr =
        tensix_config.get_worker_conn_info_base_address(sender_channel, core_id);
    connection_info.buffer_index_semaphore_id =
        tensix_config.get_buffer_index_semaphore_address(sender_channel, core_id);
    connection_info.worker_free_slots_stream_id = tensix_config.get_channel_credits_stream_id(sender_channel, core_id);
}

void ControlPlane::populate_fabric_connection_info(
    tt::tt_fabric::fabric_connection_info_t& worker_connection_info,
    tt::tt_fabric::fabric_connection_info_t& dispatcher_connection_info,
    tt::tt_fabric::fabric_connection_info_t& tensix_connection_info,
    ChipId physical_chip_id,
    chan_id_t eth_channel_id) const {
    constexpr uint16_t WORKER_FREE_SLOTS_STREAM_ID =
        tt::tt_fabric::connection_interface::sender_channel_0_free_slots_stream_id;
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto& fabric_context = this->get_fabric_context();
    const auto& builder_context = fabric_context.get_builder_context();
    // Sender channel 0 is always for local worker in the new design
    const auto sender_channel = 0;

    const auto& fabric_tensix_config = tt::tt_metal::MetalContext::instance().get_fabric_tensix_config();
    // Always populate fabric router config for normal workers
    const auto& edm_config = builder_context.get_fabric_router_config(
        fabric_tensix_config, static_cast<eth_chan_directions>(sender_channel));
    CoreCoord fabric_router_virtual_core = cluster.get_virtual_eth_core_from_channel(physical_chip_id, eth_channel_id);

    fill_connection_info_fields(
        worker_connection_info, fabric_router_virtual_core, edm_config, sender_channel, WORKER_FREE_SLOTS_STREAM_ID);

    // Check if fabric tensix config is enabled, if so populate different configs for dispatcher and tensix
    if (fabric_tensix_config != tt::tt_fabric::FabricTensixConfig::DISABLED) {
        // dispatcher uses different fabric router, which still has the default buffer size.
        const auto& default_edm_config = builder_context.get_fabric_router_config();
        fill_connection_info_fields(
            dispatcher_connection_info,
            fabric_router_virtual_core,
            default_edm_config,
            sender_channel,
            WORKER_FREE_SLOTS_STREAM_ID);

        const auto& tensix_config = builder_context.get_tensix_config();
        CoreCoord mux_core_logical = tensix_config.get_core_for_channel(physical_chip_id, eth_channel_id);
        CoreCoord mux_core_virtual = cluster.get_virtual_coordinate_from_logical_coordinates(
            physical_chip_id, mux_core_logical, CoreType::WORKER);
        // Get the RISC ID that handles this ethernet channel
        auto core_id = tensix_config.get_core_id_for_channel(physical_chip_id, eth_channel_id);
        // In UDM mode, get the first channel for worker connection for now.
        // TODO: have a vector of worker channels based on the current core and connected eth_channel_id
        uint32_t tensix_sender_channel = sender_channel;

        fill_tensix_connection_info_fields(
            tensix_connection_info, mux_core_virtual, tensix_config, tensix_sender_channel, core_id);
    } else {
        dispatcher_connection_info = worker_connection_info;
    }
}

// UDM-specific: write per-worker connection info to each worker core's L1
void ControlPlane::write_udm_fabric_connections_to_tensix_cores(
    ChipId physical_chip_id,
    const tt::tt_fabric::tensix_fabric_connections_l1_info_t& fabric_mux_connections,
    const tt::tt_fabric::tensix_fabric_connections_l1_info_t& fabric_dispatcher_connections) const {
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto& fabric_context = this->get_fabric_context();
    const auto& tensix_config = fabric_context.get_builder_context().get_tensix_config();

    // Get mux and dispatcher cores
    std::unordered_set<CoreCoord> fabric_mux_cores_translated = tensix_config.get_translated_fabric_mux_cores();
    std::unordered_set<CoreCoord> dispatch_mux_cores_translated = tensix_config.get_translated_dispatch_mux_cores();

    const auto& soc_desc = cluster.get_soc_desc(physical_chip_id);
    const std::vector<tt::umd::CoreCoord>& all_tensix_cores =
        soc_desc.get_cores(CoreType::TENSIX, CoordSystem::TRANSLATED);

    // Build per-worker connection info and write to each worker core
    for (const auto& tensix_core : all_tensix_cores) {
        CoreCoord core_coord(tensix_core.x, tensix_core.y);

        // Determine core type
        const void* data_to_write = nullptr;
        if (fabric_mux_cores_translated.contains(core_coord)) {
            // Mux core: write fabric_mux_connections (passed in from caller)
            data_to_write = &fabric_mux_connections;
        } else if (dispatch_mux_cores_translated.contains(core_coord)) {
            // Dispatcher core: write fabric_dispatcher_connections (passed in from caller)
            data_to_write = &fabric_dispatcher_connections;
        } else {
            // Worker core: build per-worker connection info
            tt::tt_fabric::tensix_fabric_connections_l1_info_t worker_connections = {};

            // Get worker assignment info (tensix core + channel index) with a single lookup
            auto tensix_info = tensix_config.get_worker_tensix_info(physical_chip_id, core_coord);

            // Populate worker-specific tensix mux connection for ALL eth channel indices
            for (auto& connection_info : worker_connections.read_only) {
                fill_tensix_connection_info_fields(
                    connection_info,
                    tensix_info.tensix_core,
                    tensix_config,
                    tensix_info.channel_index,
                    FabricTensixCoreType::MUX);
            }

            data_to_write = &worker_connections;
        }

        // Write to L1
        cluster.write_core(
            data_to_write,
            sizeof(tt::tt_fabric::tensix_fabric_connections_l1_info_t),
            tt_cxy_pair(physical_chip_id, core_coord),
            tt_metal::MetalContext::instance().hal().get_dev_addr(
                tt_metal::HalProgrammableCoreType::TENSIX, tt::tt_metal::HalL1MemAddrType::TENSIX_FABRIC_CONNECTIONS));

        // Initialize fabric connection sync region (lock=0, initialized=0, connection_storage zeroed)
        tt::tt_fabric::fabric_connection_sync_t sync_init = {};
        cluster.write_core(
            &sync_init,
            sizeof(sync_init),
            tt_cxy_pair(physical_chip_id, core_coord),
            tt_metal::MetalContext::instance().hal().get_dev_addr(
                tt_metal::HalProgrammableCoreType::TENSIX, tt::tt_metal::HalL1MemAddrType::FABRIC_CONNECTION_LOCK));
    }
}

void ControlPlane::collect_and_merge_router_port_directions_from_all_hosts() {
    const auto& distributed_context = tt::tt_metal::MetalContext::instance().full_world_distributed_context();
    if (*distributed_context.size() == 1) {
        // No need to collect from other hosts when running a single process
        return;
    }

    // Create RouterPortDirectionsData from local data
    RouterPortDirectionsData local_data;
    local_data.local_mesh_id = local_mesh_binding_.mesh_ids[0];
    local_data.local_host_rank_id = this->get_local_host_rank_id_binding();
    local_data.router_port_directions_map = router_port_directions_to_physical_eth_chan_map_;

    auto serialized_data = tt::tt_fabric::serialize_router_port_directions_to_bytes(local_data);
    std::vector<uint8_t> serialized_remote_data;
    auto my_rank = *(distributed_context.rank());

    for (std::size_t bcast_root = 0; bcast_root < *(distributed_context.size()); ++bcast_root) {
        if (my_rank == bcast_root) {
            // Issue the broadcast from the current process to all other processes in the world
            int local_data_size_bytes = serialized_data.size();  // Send data size first
            distributed_context.broadcast(
                tt::stl::Span<std::byte>(
                    reinterpret_cast<std::byte*>(&local_data_size_bytes), sizeof(local_data_size_bytes)),
                distributed_context.rank());

            distributed_context.broadcast(
                tt::stl::as_writable_bytes(tt::stl::Span<uint8_t>(serialized_data.data(), serialized_data.size())),
                distributed_context.rank());
        } else {
            // Acknowledge the broadcast issued by the root
            int remote_data_size_bytes = 0;  // Receive the size of the serialized data
            distributed_context.broadcast(
                tt::stl::Span<std::byte>(
                    reinterpret_cast<std::byte*>(&remote_data_size_bytes), sizeof(remote_data_size_bytes)),
                tt::tt_metal::distributed::multihost::Rank{static_cast<int>(bcast_root)});
            serialized_remote_data.clear();
            serialized_remote_data.resize(remote_data_size_bytes);
            distributed_context.broadcast(
                tt::stl::as_writable_bytes(
                    tt::stl::Span<uint8_t>(serialized_remote_data.data(), serialized_remote_data.size())),
                tt::tt_metal::distributed::multihost::Rank{static_cast<int>(bcast_root)});

            RouterPortDirectionsData deserialized_remote_data =
                tt::tt_fabric::deserialize_router_port_directions_from_bytes(serialized_remote_data);

            // Merge remote data into local router_port_directions_to_physical_eth_chan_map_
            for (const auto& [fabric_node_id, direction_map] : deserialized_remote_data.router_port_directions_map) {
                // Only merge if this fabric node is not already in our local map
                if (!router_port_directions_to_physical_eth_chan_map_.contains(fabric_node_id)) {
                    router_port_directions_to_physical_eth_chan_map_[fabric_node_id] = direction_map;
                } else {
                    // If fabric node exists, merge direction maps
                    for (const auto& [direction, channels] : direction_map) {
                        auto& local_direction_map = router_port_directions_to_physical_eth_chan_map_[fabric_node_id];
                        if (!local_direction_map.contains(direction)) {
                            local_direction_map[direction] = channels;
                        } else {
                            // Merge channels, avoiding duplicates
                            auto& local_channels = local_direction_map[direction];
                            for (const auto& channel : channels) {
                                if (std::find(local_channels.begin(), local_channels.end(), channel) ==
                                    local_channels.end()) {
                                    local_channels.push_back(channel);
                                }
                            }
                        }
                    }
                }
            }
        }
        // Barrier here for safety - Ensure that all ranks have completed the bcast op before proceeding to the next
        // root
        distributed_context.barrier();
    }
}

// Intermesh Connectivity Generation Functions

void ControlPlane::generate_intermesh_connectivity() {
    AnnotatedIntermeshConnections intermesh_connections;

    auto generate_mapping_locally_ = (this->mesh_graph_->get_all_mesh_ids().size() == 1) &&
                                     (this->mesh_graph_->get_host_ranks(local_mesh_binding_.mesh_ids[0]).size() == 1);

    auto get_num_requested_intermesh_connections = [&]() -> size_t {
        const auto& mesh_graph = *this->mesh_graph_;
        const auto& requested_intermesh_connections = mesh_graph.get_requested_intermesh_connections();
        const auto& requested_intermesh_ports = mesh_graph.get_requested_intermesh_ports();
        TT_FATAL(
            requested_intermesh_connections.empty() || requested_intermesh_ports.empty(),
            "Mesh Graph Descriptor must specify either RelaxedGraph or Graph connections, not both.");
        return !requested_intermesh_connections.empty() ? requested_intermesh_connections.size()
                                                        : requested_intermesh_ports.size();
    };

    if (!generate_mapping_locally_ &&
        *(tt_metal::MetalContext::instance().full_world_distributed_context().size()) > 1) {
        // Intermesh Connectivity generation for the multi-host case
        auto exit_node_port_descriptors = this->generate_port_descriptors_for_exit_nodes();
        intermesh_connections = this->convert_port_desciptors_to_intermesh_connections(exit_node_port_descriptors);
    } else {
        // Intermesh Connectivity generation for the single-host case
        intermesh_connections = this->generate_intermesh_connections_on_local_host();
    }
    // Divide by 2 here, since the intermesh_connections data structure stores connections
    // bidirectionally.
    auto num_assigned_intermesh_connections = intermesh_connections.size() / 2;

    TT_FATAL(
        num_assigned_intermesh_connections >= get_num_requested_intermesh_connections(),
        "Unable to bind the intermesh connections requested in the Mesh Graph Descriptor to physical links."
        " Found {} intermesh connections, but {} were requested",
        num_assigned_intermesh_connections,
        get_num_requested_intermesh_connections());

    this->routing_table_generator_->load_intermesh_connections(intermesh_connections);
}

std::vector<PortDescriptor> ControlPlane::assign_logical_ports_to_exit_nodes(
    const std::string& my_host,
    const std::string& neighbor_host,
    bool strict_binding,
    const std::unordered_set<FabricNodeId>& requested_exit_nodes,
    std::unordered_set<port_id_t>& assigned_port_ids) {
    const auto my_mesh_id = local_mesh_binding_.mesh_ids[0];
    auto neighbor_host_rank = physical_system_descriptor_->get_rank_for_hostname(neighbor_host);
    const auto& neighbor_binding = this->global_logical_bindings_.at(
        tt::tt_metal::distributed::multihost::Rank{static_cast<int>(neighbor_host_rank)});
    const auto neighbor_mesh_id = neighbor_binding.first;

    const auto& exit_nodes = physical_system_descriptor_->get_connecting_exit_nodes(my_host, neighbor_host);
    const auto& mesh_edge_ports_to_chip_id = this->mesh_graph_->get_mesh_edge_ports_to_chip_id();

    std::vector<PortDescriptor> ports_to_neighbor;

    std::unordered_map<uint64_t, RoutingDirection> curr_exit_node_direction;
    for (const auto& exit_node : exit_nodes) {
        FabricNodeId exit_node_fabric_node_id = this->get_fabric_node_id_from_asic_id(*exit_node.src_exit_node);

        TT_FATAL(exit_node_fabric_node_id.mesh_id == my_mesh_id, "Exit node is not on my mesh");
        if (strict_binding) {
            if (!requested_exit_nodes.contains(exit_node_fabric_node_id)) {
                continue;
            }
        }
        auto assoc_connection_hash = std::hash<tt::tt_metal::ExitNodeConnection>{}(exit_node);
        auto exit_node_hash = (*exit_node.src_exit_node) + (*exit_node.dst_exit_node);
        auto src_eth_chan = exit_node.eth_conn.src_chan;
        auto exit_node_chip = exit_node_fabric_node_id.chip_id;

        bool should_assign_z = this->mesh_graph_->should_assign_z_direction(my_mesh_id, neighbor_mesh_id);

        for (const auto& [port_id, chip_id] : mesh_edge_ports_to_chip_id[*my_mesh_id]) {
            if (exit_node_chip == chip_id) {
                auto port_direction = port_id.first;
                auto logical_chan_id = port_id.second;

                // Blackhole Z-channels must be assigned the Z Routing Direction.
                // All other channels must avoid using the Z-direction (they are used for routing along the X/Y
                // directions). This is to ensure that logical and physical channel assignments are consistent.
                bool is_z_direction = (port_direction == RoutingDirection::Z);
                if (should_assign_z != is_z_direction) {
                    continue;
                }

                port_id_t port_id = {port_direction, logical_chan_id};
                // Assign this port id to the exit node if it is not already assigned
                bool valid_direction = !curr_exit_node_direction.contains(exit_node_hash) ||
                                       curr_exit_node_direction.at(exit_node_hash) == port_direction;
                if (!assigned_port_ids.contains(port_id) && valid_direction) {
                    assigned_port_ids.insert(port_id);
                    ports_to_neighbor.push_back(PortDescriptor{port_id, assoc_connection_hash});
                    // Override direction to Z if this is a Z channel on BLACKHOLE or should assign Z direction
                    RoutingDirection final_direction = (should_assign_z) ? RoutingDirection::Z : port_direction;
                    exit_node_directions_[exit_node_fabric_node_id][src_eth_chan] = final_direction;
                    logical_port_to_eth_chan_[exit_node_fabric_node_id][port_id] = src_eth_chan;
                    curr_exit_node_direction[exit_node_hash] = final_direction;
                    break;
                }
            }
        }
    }
    return ports_to_neighbor;
}

PortDescriptorTable ControlPlane::generate_port_descriptors_for_exit_nodes() {
    const auto& mesh_graph = *this->mesh_graph_;
    const auto& requested_intermesh_connections = mesh_graph.get_requested_intermesh_connections();
    const auto& requested_intermesh_ports = mesh_graph.get_requested_intermesh_ports();
    const auto& my_host = physical_system_descriptor_->my_host_name();
    const auto my_mesh_id = local_mesh_binding_.mesh_ids[0];

    TT_FATAL(
        requested_intermesh_connections.empty() || requested_intermesh_ports.empty(),
        "Mesh Graph Descriptor must specify either RelaxedGraph or Graph connections, not both.");

    bool strict_binding = !requested_intermesh_ports.empty();

    // Track the Logical Ethernet Ports connecting to all neighbors of my_mesh
    PortDescriptorTable port_descriptors;
    // Track Direction and Logical Ports already assigned for intermesh links
    std::unordered_set<port_id_t> assigned_port_ids;
    port_descriptors[my_mesh_id] = {};

    for (const auto& neighbor_host : physical_system_descriptor_->get_host_neighbors(my_host)) {
        auto neighbor_host_rank = physical_system_descriptor_->get_rank_for_hostname(neighbor_host);
        // Skip if neighbor host is not in our global logical bindings
        if (!this->global_logical_bindings_.contains(
                tt::tt_metal::distributed::multihost::Rank{static_cast<int>(neighbor_host_rank)})) {
            continue;
        }
        auto neighbor_mesh_id =
            this->global_logical_bindings_
                .at(tt::tt_metal::distributed::multihost::Rank{static_cast<int>(neighbor_host_rank)})
                .first;
        bool connection_requested = check_connection_requested(
            my_mesh_id, neighbor_mesh_id, requested_intermesh_connections, requested_intermesh_ports);
        if (!connection_requested) {
            continue;
        }
        const auto& exit_nodes = physical_system_descriptor_->get_connecting_exit_nodes(my_host, neighbor_host);
        std::vector<uint64_t> src_exit_node_chips;
        src_exit_node_chips.reserve(exit_nodes.size());
        std::transform(
            exit_nodes.begin(), exit_nodes.end(), std::back_inserter(src_exit_node_chips), [](const auto& exit_node) {
                return *exit_node.src_exit_node;
            });
        std::unordered_set<FabricNodeId> requested_exit_nodes = this->get_requested_exit_nodes(
            my_mesh_id, neighbor_mesh_id, requested_intermesh_ports, src_exit_node_chips);
        port_descriptors[my_mesh_id][neighbor_mesh_id] = this->assign_logical_ports_to_exit_nodes(
            my_host, neighbor_host, strict_binding, requested_exit_nodes, assigned_port_ids);
    }
    return port_descriptors;
}

void ControlPlane::validate_requested_intermesh_connections(
    const RequestedIntermeshConnections& requested_intermesh_connections, const PortDescriptorTable& port_descriptors) {
    bool strict_binding = requested_intermesh_connections.empty();
    if (strict_binding) {
        return;
    }
    for (const auto& [src_mesh, dst_mesh_map] : requested_intermesh_connections) {
        auto src_mesh_id = MeshId(src_mesh);
        for (const auto& [dst_mesh, num_channels] : dst_mesh_map) {
            auto dst_mesh_id = MeshId(dst_mesh);
            TT_FATAL(
                num_channels <= port_descriptors.at(src_mesh_id).at(dst_mesh_id).size(),
                "Requested {} channels between {} and {}, but only have {} physical links",
                num_channels,
                src_mesh,
                dst_mesh,
                port_descriptors.at(src_mesh_id).at(dst_mesh_id).size());
        }
    }
}

std::unordered_set<FabricNodeId> ControlPlane::get_requested_exit_nodes(
    MeshId my_mesh_id,
    MeshId neighbor_mesh_id,
    const RequestedIntermeshPorts& requested_intermesh_ports,
    const std::vector<uint64_t>& src_exit_node_chips) const {
    std::unordered_set<FabricNodeId> requested_exit_nodes;
    const auto& local_coord_range = this->get_coord_range(my_mesh_id, MeshScope::LOCAL);
    if (!requested_intermesh_ports.empty()) {
        for (const auto& port : requested_intermesh_ports.at(*my_mesh_id).at(*neighbor_mesh_id)) {
            auto src_device = std::get<0>(port);
            auto requested_coordinate = this->mesh_graph_->chip_to_coordinate(my_mesh_id, src_device);
            if (!local_coord_range.contains(requested_coordinate)) {
                continue;
            }
            uint32_t num_physical_channels_found = 0;
            uint32_t num_channels_requested = std::get<2>(port);
            for (const auto& src_exit_node_chip : src_exit_node_chips) {
                if (this->get_fabric_node_id_from_asic_id(src_exit_node_chip) == FabricNodeId(my_mesh_id, src_device)) {
                    requested_exit_nodes.insert(FabricNodeId(my_mesh_id, src_device));
                    num_physical_channels_found++;
                }
            }
            TT_FATAL(
                num_physical_channels_found >= num_channels_requested,
                "Requested {} channels between {} and {} on src FabricNodeId {}, but only have {} physical channels",
                num_channels_requested,
                *my_mesh_id,
                *neighbor_mesh_id,
                FabricNodeId(my_mesh_id, src_device),
                num_physical_channels_found);
        }
    }
    return requested_exit_nodes;
}

void ControlPlane::forward_descriptors_to_controller(
    PortDescriptorTable& port_descriptors, uint32_t my_rank, const std::string& my_host) {
    using namespace tt::tt_metal::distributed::multihost;
    constexpr uint32_t CONTROLLER_RANK = 0;
    const auto& distributed_context = tt::tt_metal::MetalContext::instance().full_world_distributed_context();
    const auto& physical_system_descriptor = this->physical_system_descriptor_;
    std::size_t serialized_table_size = 0;
    std::vector<uint8_t> serialized_table;
    if (my_rank != CONTROLLER_RANK) {
        serialized_table = serialize_to_bytes(port_descriptors);
        serialized_table_size = serialized_table.size();
        distributed_context.send(
            tt::stl::Span<std::byte>(
                reinterpret_cast<std::byte*>(&serialized_table_size), sizeof(serialized_table_size)),
            Rank{CONTROLLER_RANK},
            Tag{0});
        distributed_context.send(
            tt::stl::as_writable_bytes(tt::stl::Span<uint8_t>(serialized_table.data(), serialized_table.size())),
            Rank{CONTROLLER_RANK},
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
                Rank{static_cast<int>(peer_rank)},
                Tag{0});
            serialized_table.resize(serialized_table_size);
            distributed_context.recv(
                tt::stl::as_writable_bytes(tt::stl::Span<uint8_t>(serialized_table.data(), serialized_table.size())),
                Rank{static_cast<int>(peer_rank)},
                Tag{0});
            auto peer_port_descriptors = deserialize_port_descriptors_from_bytes(serialized_table);
            TT_FATAL(peer_port_descriptors.size() == 1, "Expecting peer port id table to have exactly one mesh");

            // Extract the single mesh entry from peer port descriptors
            const auto& neighbor_mesh_id = peer_port_descriptors.begin()->first;
            auto& neighbor_connections = peer_port_descriptors.begin()->second;

            // Check if we already have entries for this neighbor mesh
            auto& neighbor_mesh_descriptors = port_descriptors[neighbor_mesh_id];
            if (neighbor_mesh_descriptors.empty()) {
                // First time seeing this neighbor mesh - move all connections
                neighbor_mesh_descriptors = std::move(neighbor_connections);
            } else {
                // Merge connections from this neighbor mesh
                for (auto&& [dest_mesh_id, dest_port_descriptors] : neighbor_connections) {
                    auto& dest_descriptors = neighbor_mesh_descriptors[dest_mesh_id];
                    if (dest_descriptors.empty()) {
                        // First time seeing this destination on the neighbor mesh - move the descriptors
                        dest_descriptors = std::move(dest_port_descriptors);
                    } else {
                        // Append to existing descriptors for this destination
                        dest_descriptors.insert(
                            dest_descriptors.end(),
                            std::make_move_iterator(dest_port_descriptors.begin()),
                            std::make_move_iterator(dest_port_descriptors.end()));
                    }
                }
            }
        }
    }
    distributed_context.barrier();
}

void ControlPlane::forward_intermesh_connections_from_controller(AnnotatedIntermeshConnections& intermesh_connections) {
    using namespace tt::tt_metal::distributed::multihost;
    const auto& distributed_context = tt::tt_metal::MetalContext::instance().full_world_distributed_context();
    constexpr uint32_t CONTROLLER_RANK = 0;
    const auto& my_host = physical_system_descriptor_->my_host_name();
    auto my_rank = physical_system_descriptor_->get_rank_for_hostname(my_host);
    std::size_t serialized_table_size = 0;
    std::vector<uint8_t> serialized_connections;
    if (my_rank == CONTROLLER_RANK) {
        for (const auto& hostname : physical_system_descriptor_->get_all_hostnames()) {
            if (hostname == my_host) {
                continue;
            }
            auto peer_rank = physical_system_descriptor_->get_rank_for_hostname(hostname);
            serialized_connections = serialize_intermesh_connections_to_bytes(intermesh_connections);
            serialized_table_size = serialized_connections.size();
            distributed_context.send(
                tt::stl::Span<std::byte>(
                    reinterpret_cast<std::byte*>(&serialized_table_size), sizeof(serialized_table_size)),
                Rank{static_cast<int>(peer_rank)},
                Tag{0});
            distributed_context.send(
                tt::stl::as_writable_bytes(
                    tt::stl::Span<uint8_t>(serialized_connections.data(), serialized_connections.size())),
                Rank{static_cast<int>(peer_rank)},
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
        intermesh_connections = deserialize_intermesh_connections_from_bytes(serialized_connections);
    }
    distributed_context.barrier();
}

AnnotatedIntermeshConnections ControlPlane::pair_logical_intermesh_ports(const PortDescriptorTable& port_descriptors) {
    AnnotatedIntermeshConnections intermesh_connections;

    const auto& mesh_graph = *this->mesh_graph_;
    const auto& requested_intermesh_connections = mesh_graph.get_requested_intermesh_connections();
    const auto& requested_intermesh_ports = mesh_graph.get_requested_intermesh_ports();
    const auto& mesh_edge_ports_to_chip_id = mesh_graph.get_mesh_edge_ports_to_chip_id();

    bool strict_binding = !requested_intermesh_ports.empty();
    std::set<std::pair<uint32_t, uint32_t>> processed_neighbors;

    validate_requested_intermesh_connections(requested_intermesh_connections, port_descriptors);

    for (const auto& [src_mesh, port_identifiers] : port_descriptors) {
        for (const auto& [dest_mesh, src_ports] : port_identifiers) {
            if (processed_neighbors.contains({*dest_mesh, *src_mesh})) {
                // Connections for these neighbors have already been setup - skip
                continue;
            }
            std::size_t num_ports_assigned = 0;
            std::size_t num_ports_requested = 0;
            std::unordered_map<FabricNodeId, uint32_t> num_ports_requested_at_exit_node;
            std::unordered_map<FabricNodeId, uint32_t> num_ports_assigned_at_exit_node;
            if (strict_binding) {
                for (const auto& port : requested_intermesh_ports.at(*src_mesh).at(*dest_mesh)) {
                    num_ports_requested_at_exit_node[FabricNodeId(src_mesh, std::get<0>(port))] += std::get<2>(port);
                    num_ports_assigned_at_exit_node[FabricNodeId(src_mesh, std::get<0>(port))] = 0;
                }
            } else {
                num_ports_requested = requested_intermesh_connections.at(*src_mesh).at(*dest_mesh);
            }

            const auto& dest_ports = port_descriptors.at(dest_mesh).at(src_mesh);
            // Iterate over src ports. For each src port, determine which dst port it connects to
            for (const auto& src_port : src_ports) {
                const auto& src_port_id = src_port.port_id;
                auto src_chip = mesh_edge_ports_to_chip_id.at(*src_mesh).at(src_port_id);
                if (strict_binding) {
                    if (num_ports_assigned_at_exit_node.at(FabricNodeId(src_mesh, src_chip)) >=
                        num_ports_requested_at_exit_node.at(FabricNodeId(src_mesh, src_chip))) {
                        continue;
                    }
                } else {
                    if (num_ports_assigned == num_ports_requested) {
                        break;
                    }
                }
                const auto& connection_hash = src_port.connection_hash;
                for (const auto& dest_port : dest_ports) {
                    if (dest_port.connection_hash == connection_hash) {
                        auto src_port_id = src_port.port_id;
                        auto dest_port_id = dest_port.port_id;
                        log_debug(
                            tt::LogDistributed,
                            "Connecting Meshes {} {} over Logical Ports {} {}",
                            *src_mesh,
                            *dest_mesh,
                            create_port_tag(src_port_id),
                            create_port_tag(dest_port_id));

                        intermesh_connections.push_back({{*src_mesh, src_port_id}, {*dest_mesh, dest_port_id}});
                        intermesh_connections.push_back({{*dest_mesh, dest_port_id}, {*src_mesh, src_port_id}});
                        num_ports_assigned++;
                        num_ports_assigned_at_exit_node[FabricNodeId(src_mesh, src_chip)]++;
                        break;
                    }
                }
            }
            processed_neighbors.insert({*src_mesh, *dest_mesh});
        }
    }
    return intermesh_connections;
}

AnnotatedIntermeshConnections ControlPlane::convert_port_desciptors_to_intermesh_connections(
    PortDescriptorTable& port_descriptors) {
    const auto& my_host = physical_system_descriptor_->my_host_name();
    auto my_rank = physical_system_descriptor_->get_rank_for_hostname(my_host);

    this->forward_descriptors_to_controller(port_descriptors, my_rank, my_host);

    AnnotatedIntermeshConnections intermesh_connections;
    if (my_rank == 0) {
        intermesh_connections = this->pair_logical_intermesh_ports(port_descriptors);
    }
    this->forward_intermesh_connections_from_controller(intermesh_connections);

    const auto my_mesh_id = local_mesh_binding_.mesh_ids[0];
    // Track all logical ports with active intermesh connections
    std::set<port_id_t> active_logical_ports;
    for (const auto& connection : intermesh_connections) {
        if (std::get<0>(connection).first == *my_mesh_id) {
            active_logical_ports.insert(std::get<0>(connection).second);
        }
    }
    // Remove directions from all logical ports not being actively used
    for (const auto& [exit_node, port] : logical_port_to_eth_chan_) {
        for (const auto& [port_id, physical_chan] : port) {
            if (!active_logical_ports.contains(port_id)) {
                exit_node_directions_.at(exit_node).erase(physical_chan);
            }
        }
    }
    return intermesh_connections;
}

AnnotatedIntermeshConnections ControlPlane::generate_intermesh_connections_on_local_host() {
    const auto& mesh_graph = *this->mesh_graph_;
    const auto& physical_system_descriptor = this->physical_system_descriptor_;

    std::unordered_map<uint32_t, std::set<port_id_t>> assigned_ports_per_mesh;
    std::set<std::pair<uint32_t, uint32_t>> processed_neighbors;
    AnnotatedIntermeshConnections intermesh_connections;
    std::unordered_map<uint64_t, uint32_t> num_connections;

    const auto& requested_intermesh_connections = mesh_graph.get_requested_intermesh_connections();
    const auto& requested_intermesh_ports = mesh_graph.get_requested_intermesh_ports();

    TT_FATAL(
        requested_intermesh_connections.empty() || requested_intermesh_ports.empty(),
        "Mesh Graph Descriptor must specify either RelaxedGraph or Graph connections, not both.");

    bool strict_binding = !requested_intermesh_ports.empty();

    auto should_process_direction_for_chip = [&](const FabricNodeId& edge_node,
                                                 ChipId candidate_chip_id,
                                                 std::optional<RoutingDirection> current_dir,
                                                 RoutingDirection candidate_dir) -> bool {
        return edge_node.chip_id == candidate_chip_id &&
               ((!current_dir.has_value()) || current_dir.value() == candidate_dir);
    };

    auto compute_mesh_connectivity_hash = [&](MeshId src_mesh_id, MeshId dst_mesh_id) -> uint64_t {
        return (1 << *src_mesh_id) | (1 << *dst_mesh_id);
    };

    for (const auto& local_mesh_id : local_mesh_binding_.mesh_ids) {
        const auto& mesh_edges = mesh_graph.get_mesh_edge_ports_to_chip_id().at(*local_mesh_id);

        std::unordered_set<FabricNodeId> exit_nodes;
        for (const auto& [port_id, edge_chip] : mesh_edges) {
            auto node = FabricNodeId(local_mesh_id, edge_chip);
            exit_nodes.insert(node);
        }
        // Pair the exit nodes from the current mesh with the exit nodes from the neighboring meshes
        for (const auto& node : exit_nodes) {
            auto physical_chip_id = logical_mesh_chip_id_to_physical_chip_id_mapping_.at(node);
            auto asic_id =
                tt::tt_metal::MetalContext::instance().get_cluster().get_unique_chip_ids().at(physical_chip_id);
            const auto& asic_neighbors = physical_system_descriptor->get_asic_neighbors(tt::tt_metal::AsicID{asic_id});

            for (const auto& asic_neighbor : asic_neighbors) {
                // if the asic neighbor is not on the same host skip
                if (physical_system_descriptor->get_host_name_for_asic(tt::tt_metal::AsicID{asic_id}) !=
                    physical_system_descriptor->get_host_name_for_asic(asic_neighbor)) {
                    continue;
                }

                auto neighbor_node = this->get_fabric_node_id_from_asic_id(*asic_neighbor);
                if (neighbor_node.mesh_id == local_mesh_id ||
                    processed_neighbors.contains({*neighbor_node.mesh_id, *local_mesh_id})) {
                    continue;
                }
                if (!check_connection_requested(
                        local_mesh_id,
                        neighbor_node.mesh_id,
                        requested_intermesh_connections,
                        requested_intermesh_ports)) {
                    continue;
                }
                if (!strict_binding and
                    num_connections[compute_mesh_connectivity_hash(local_mesh_id, neighbor_node.mesh_id)] >=
                        requested_intermesh_connections.at(*local_mesh_id).at(*neighbor_node.mesh_id)) {
                    continue;
                }

                auto connected_eth_chans =
                    physical_system_descriptor_->get_eth_connections(tt::tt_metal::AsicID{asic_id}, asic_neighbor);
                uint32_t num_connections_assigned = 0;
                std::unordered_map<FabricNodeId, uint32_t> num_ports_requested_at_exit_node;
                std::unordered_map<FabricNodeId, uint32_t> num_ports_assigned_at_exit_node;

                if (strict_binding) {
                    for (const auto& port : requested_intermesh_ports.at(*local_mesh_id).at(*neighbor_node.mesh_id)) {
                        num_ports_requested_at_exit_node[node] += std::get<2>(port);
                        num_ports_assigned_at_exit_node[node] = 0;
                    }
                }
                std::optional<RoutingDirection> local_dir = std::nullopt;
                std::optional<RoutingDirection> neighbor_dir = std::nullopt;
                for (const auto& [local_port_id, local_chip_id] : mesh_edges) {
                    if (strict_binding &&
                        num_ports_assigned_at_exit_node[node] >= num_ports_requested_at_exit_node[node]) {
                        continue;
                    }
                    if (num_connections_assigned >= connected_eth_chans.size()) {
                        break;
                    }
                    // Skip if this port doesn't match our node and direction constraints
                    if (!should_process_direction_for_chip(node, local_chip_id, local_dir, local_port_id.first)) {
                        continue;
                    }

                    // Try to assign local port
                    if (assigned_ports_per_mesh[*local_mesh_id].contains(local_port_id)) {
                        continue;
                    }

                    // Local port is available - assign it
                    assigned_ports_per_mesh[*local_mesh_id].insert(local_port_id);

                    // Find matching neighbor port
                    bool found_neighbor = false;
                    for (const auto& [neighbor_port_id, neighbor_chip_id] :
                         mesh_graph.get_mesh_edge_ports_to_chip_id().at(*neighbor_node.mesh_id)) {
                        if (!should_process_direction_for_chip(
                                neighbor_node, neighbor_chip_id, neighbor_dir, neighbor_port_id.first)) {
                            continue;
                        }

                        if (assigned_ports_per_mesh[*neighbor_node.mesh_id].contains(neighbor_port_id)) {
                            continue;
                        }

                        // Found available neighbor port - create connection
                        assigned_ports_per_mesh[*neighbor_node.mesh_id].insert(neighbor_port_id);
                        processed_neighbors.insert({*local_mesh_id, *neighbor_node.mesh_id});

                        // Add bidirectional connections
                        intermesh_connections.push_back(
                            {{*local_mesh_id, local_port_id}, {*neighbor_node.mesh_id, neighbor_port_id}});
                        intermesh_connections.push_back(
                            {{*neighbor_node.mesh_id, neighbor_port_id}, {*local_mesh_id, local_port_id}});

                        // Update exit node directions
                        auto& current_eth_conn = connected_eth_chans[num_connections_assigned];
                        exit_node_directions_[node][current_eth_conn.src_chan] = local_port_id.first;
                        exit_node_directions_[neighbor_node][current_eth_conn.dst_chan] = neighbor_port_id.first;

                        // Update counters
                        num_connections[compute_mesh_connectivity_hash(local_mesh_id, neighbor_node.mesh_id)]++;
                        num_connections_assigned++;
                        num_ports_assigned_at_exit_node[node]++;
                        local_dir = local_port_id.first;
                        neighbor_dir = neighbor_port_id.first;
                        found_neighbor = true;
                        break;
                    }

                    if (!found_neighbor) {
                        // No neighbor port found, release the local port
                        assigned_ports_per_mesh[*local_mesh_id].erase(local_port_id);
                    }
                }
            }
        }
    }
    return intermesh_connections;
}

bool ControlPlane::is_fabric_config_valid(tt::tt_fabric::FabricConfig fabric_config) const {
    if (fabric_config == tt::tt_fabric::FabricConfig::DISABLED) {
        return false;
    }

    static const std::unordered_set<tt::tt_fabric::FabricConfig> torus_fabric_configs = {
        tt::tt_fabric::FabricConfig::FABRIC_2D_TORUS_X,
        tt::tt_fabric::FabricConfig::FABRIC_2D_TORUS_Y,
        tt::tt_fabric::FabricConfig::FABRIC_2D_TORUS_XY,
    };

    if (torus_fabric_configs.contains(fabric_config)) {
        validate_torus_setup(fabric_config);
        return true;  // Validation passed if no exception was thrown
    }

    // Non-torus configurations are valid by default since we always have at least mesh topology,
    // and mesh configurations don't require special validation like torus does
    return true;
}

void ControlPlane::validate_torus_setup(tt::tt_fabric::FabricConfig fabric_config) const {
    TT_ASSERT(physical_system_descriptor_ != nullptr, "Physical system descriptor not initialized");

    auto all_hostnames = physical_system_descriptor_->get_all_hostnames();
    auto cabling_descriptor_path = get_galaxy_cabling_descriptor_path(fabric_config);
    // Check if the cabling descriptor file exists
    TT_ASSERT(
        std::filesystem::exists(cabling_descriptor_path),
        "Cabling descriptor file not found: {}",
        cabling_descriptor_path);

    // Generate GSD YAML from the current physical system descriptor
    YAML::Node gsd_yaml = physical_system_descriptor_->generate_yaml_node();

    // Use the new validation function that handles CablingGenerator internally
    tt::scaleout_tools::validate_cabling_descriptor_against_gsd(
        cabling_descriptor_path,
        all_hostnames,
        gsd_yaml,
        false,  // strict_validation
        true    // assert_on_connection_mismatch
    );

    log_debug(tt::LogFabric, "Torus validation passed for configuration: {}", enchantum::to_string(fabric_config));
}

std::string ControlPlane::get_galaxy_cabling_descriptor_path(tt::tt_fabric::FabricConfig fabric_config) const {
    auto cluster_type = tt::tt_metal::MetalContext::instance().get_cluster().get_cluster_type();
    TT_FATAL(
        cluster_type == tt::tt_metal::ClusterType::GALAXY,
        "get_galaxy_cabling_descriptor_path is only supported on Galaxy systems, but cluster type is {}",
        enchantum::to_string(cluster_type));

    static constexpr std::string_view X_TORUS_PATH = "tt_metal/fabric/cabling_descriptors/wh_galaxy_x_torus.textproto";
    static constexpr std::string_view Y_TORUS_PATH = "tt_metal/fabric/cabling_descriptors/wh_galaxy_y_torus.textproto";
    static constexpr std::string_view XY_TORUS_PATH =
        "tt_metal/fabric/cabling_descriptors/wh_galaxy_xy_torus.textproto";

    // Get fabric type from config and map to cabling descriptor paths
    FabricType fabric_type = get_fabric_type(fabric_config);

    static constexpr std::array<std::pair<FabricType, std::string_view>, 3> cabling_map = {
        {{FabricType::TORUS_X, X_TORUS_PATH},
         {FabricType::TORUS_Y, Y_TORUS_PATH},
         {FabricType::TORUS_XY, XY_TORUS_PATH}}};

    const auto* it = std::find_if(
        cabling_map.begin(), cabling_map.end(), [fabric_type](const auto& pair) { return pair.first == fabric_type; });
    TT_FATAL(it != cabling_map.end(), "Unknown torus configuration: {}", enchantum::to_string(fabric_config));

    const auto& root_dir = tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir();
    return root_dir + std::string(it->second);
}

bool ControlPlane::is_local_host_on_switch_mesh() const {
    const auto& local_mesh_ids = this->get_local_mesh_id_bindings();
    const auto& mesh_graph = this->get_mesh_graph();

    std::optional<MeshId> local_switch_mesh_id = std::nullopt;
    std::vector<MeshId> local_compute_mesh_ids;
    for (const auto& mesh_id : local_mesh_ids) {
        if (mesh_graph.is_switch_mesh(mesh_id)) {
            if (local_switch_mesh_id.has_value()) {
                TT_THROW("Local host is on multiple switch meshes: {} and {}", *local_switch_mesh_id, *mesh_id);
            }
            local_switch_mesh_id = mesh_id;
        } else {
            // This is a compute mesh
            local_compute_mesh_ids.push_back(mesh_id);
        }
    }

    // Guard against host being bound to both switch and compute meshes
    TT_FATAL(
        !(local_switch_mesh_id.has_value() && !local_compute_mesh_ids.empty()),
        "Local host cannot be bound to both a switch mesh and a compute mesh.");

    return local_switch_mesh_id.has_value();
}

std::vector<ChipId> ControlPlane::get_switch_mesh_device_ids() const {
    const auto& local_mesh_ids = this->get_local_mesh_id_bindings();
    const auto& mesh_graph = this->get_mesh_graph();

    std::vector<ChipId> switch_device_ids;
    for (const auto& mesh_id : local_mesh_ids) {
        if (mesh_graph.is_switch_mesh(mesh_id)) {
            const auto& chip_ids = mesh_graph.get_chip_ids(mesh_id);
            for (const auto& chip_id : chip_ids.values()) {
                auto fabric_node_id = FabricNodeId(mesh_id, chip_id);
                auto physical_chip_id = this->get_physical_chip_id_from_fabric_node_id(fabric_node_id);
                switch_device_ids.push_back(physical_chip_id);
            }
        }
    }
    return switch_device_ids;
}

tt::tt_metal::AsicID ControlPlane::get_asic_id_from_fabric_node_id(const FabricNodeId& fabric_node_id) const {
    return topology_mapper_->get_asic_id_from_fabric_node_id(fabric_node_id);
}

ControlPlane::~ControlPlane() = default;

}  // namespace tt::tt_fabric
