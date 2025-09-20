// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <enchantum/enchantum.hpp>
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
#include "compressed_routing_table.hpp"
#include "compressed_routing_path.hpp"
#include "hostdevcommon/fabric_common.h"
#include "distributed_context.hpp"
#include "fabric_types.hpp"
#include "hal_types.hpp"
#include "host_api.hpp"
#include "impl/context/metal_context.hpp"
#include "tt_metal/common/env_lib.hpp"
#include <tt-logger/tt-logger.hpp>
#include "mesh_coord.hpp"
#include "mesh_graph.hpp"
#include "metal_soc_descriptor.h"
#include "routing_table_generator.hpp"
#include <umd/device/types/core_coordinates.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>
#include <umd/device/types/xy_pair.hpp>
#include "tt_metal/fabric/fabric_context.hpp"
#include "tt_metal/fabric/serialization/intermesh_link_table.hpp"
#include "tt_metal/fabric/serialization/router_port_directions.hpp"
#include "tt_stl/small_vector.hpp"

namespace tt::tt_fabric {

namespace {
// TODO: Support custom operator< for eth_coord_t to allow usage in std::set
struct EthCoordComparator {
    bool operator()(const eth_coord_t& eth_coord_a, const eth_coord_t& eth_coord_b) const {
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
    }
};

// Get the physical chip ids for a mesh
std::unordered_map<chip_id_t, std::vector<CoreCoord>> get_ethernet_cores_grouped_by_connected_chips(chip_id_t chip_id) {
    return tt::tt_metal::MetalContext::instance().get_cluster().get_ethernet_cores_grouped_by_connected_chips(chip_id);
}

template <typename CONNECTIVITY_MAP_T>
void build_golden_link_counts(
    CONNECTIVITY_MAP_T const& golden_connectivity_map,
    std::unordered_map<MeshId, std::unordered_map<chip_id_t, std::unordered_map<RoutingDirection, size_t>>>&
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

std::vector<chip_id_t> get_adjacent_chips_from_ethernet_connections(
    chip_id_t chip_id, std::uint32_t num_ports_per_side) {
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    auto eth_links = cluster.get_ethernet_cores_grouped_by_connected_chips(chip_id);
    bool is_ubb = cluster.get_board_type(chip_id) == BoardType::UBB;

    auto mmio_chip_ids = cluster.mmio_chip_ids();

    std::vector<chip_id_t> adjacent_chips;

    for (const auto& [connected_chip_id, eth_ports] : eth_links) {
        // Do not include any corner to corner links on UBB
        if (is_ubb && cluster.is_external_cable(chip_id, eth_ports[0])) {
            continue;
        }
        if (eth_ports.size() > 0) {
            // Special case for TG not to include MMIO devices in adjacency map because they are control chips
            if (cluster.get_cluster_type() == tt::tt_metal::ClusterType::TG &&
                mmio_chip_ids.contains(connected_chip_id)) {
                continue;
            }

            if (eth_ports.size() < num_ports_per_side) {
                log_warning(
                    tt::LogFabric,
                    "Ethernet between chip {} and chip {} have {} expected ethernet ports, but only {} present",
                    chip_id,
                    connected_chip_id,
                    num_ports_per_side,
                    eth_ports.size());
            }
            adjacent_chips.push_back(connected_chip_id);
        }
    }

    return adjacent_chips;
}

std::uint64_t encode_mesh_id_and_rank(MeshId mesh_id, MeshHostRankId host_rank) {
    return (static_cast<uint64_t>(mesh_id.get()) << 32) | static_cast<uint64_t>(host_rank.get());
}

std::pair<MeshId, MeshHostRankId> decode_mesh_id_and_rank(std::uint64_t encoded_value) {
    return {
        MeshId{static_cast<std::uint32_t>(encoded_value >> 32)},
        MeshHostRankId{static_cast<std::uint32_t>(encoded_value & 0xFFFFFFFF)}};
}

}  // namespace

const std::unordered_map<tt::ARCH, std::vector<std::uint16_t>> ubb_bus_ids = {
    {tt::ARCH::WORMHOLE_B0, {0xC0, 0x80, 0x00, 0x40}},
    {tt::ARCH::BLACKHOLE, {0x00, 0x40, 0xC0, 0x80}},
};

UbbId get_ubb_id(chip_id_t chip_id) {
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto& tray_bus_ids = ubb_bus_ids.at(cluster.arch());
    const auto bus_id = cluster.get_bus_id(chip_id);
    auto tray_bus_id_it = std::find(tray_bus_ids.begin(), tray_bus_ids.end(), bus_id & 0xF0);
    if (tray_bus_id_it != tray_bus_ids.end()) {
        auto ubb_asic_id = bus_id & 0x0F;
        return UbbId{tray_bus_id_it - tray_bus_ids.begin() + 1, ubb_asic_id};
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

    // For TG need to skip the direction on the remote devices directly connected to the MMIO devices as we have only
    // one outgoing eth chan to the mmio device
    // TODO: https://github.com/tenstorrent/tt-metal/issues/24413
    auto skip_direction = [&](const FabricNodeId& node_id, const RoutingDirection direction) -> bool {
        const auto& neighbors = this->get_chip_neighbors(node_id, direction);
        if (neighbors.empty()) {
            return false;
        }

        // The remote devices connected directly to the mmio will have both intra-mesh and inter-mesh neighbors
        if (neighbors.size() > 1 || neighbors.begin()->first != node_id.mesh_id) {
            return true;
        }

        return false;
    };

    auto apply_min =
        [&](FabricNodeId fabric_node_id,
            const std::unordered_map<tt::tt_fabric::RoutingDirection, std::vector<tt::tt_fabric::chan_id_t>>&
                port_direction_eth_chans,
            tt::tt_fabric::RoutingDirection direction,
            const std::unordered_map<tt::tt_fabric::RoutingDirection, size_t>& golden_link_counts,
            size_t& val) {
            if (skip_direction(fabric_node_id, direction)) {
                return;
            }
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

    std::unordered_map<MeshId, std::unordered_map<chip_id_t, std::unordered_map<RoutingDirection, size_t>>>
        golden_link_counts;
    TT_FATAL(
        this->routing_table_generator_ != nullptr && this->routing_table_generator_->mesh_graph != nullptr,
        "Routing table generator not initialized");
    build_golden_link_counts(
        this->routing_table_generator_->mesh_graph->get_intra_mesh_connectivity(), golden_link_counts);
    build_golden_link_counts(
        this->routing_table_generator_->mesh_graph->get_inter_mesh_connectivity(), golden_link_counts);

    auto apply_count = [&](FabricNodeId fabric_node_id, RoutingDirection direction, size_t count) {
        if (skip_direction(fabric_node_id, direction)) {
            return;
        }
        if (this->router_port_directions_to_physical_eth_chan_map_.contains(fabric_node_id) &&
            this->router_port_directions_to_physical_eth_chan_map_.at(fabric_node_id).contains(direction) &&
            this->router_port_directions_to_physical_eth_chan_map_.at(fabric_node_id).at(direction).size() > 0) {
            this->router_port_directions_to_num_routing_planes_map_[fabric_node_id][direction] = count;
        }
    };

    const auto& distributed_context = tt::tt_metal::MetalContext::instance().global_distributed_context();
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
                auto fabric_chip_id =
                    this->routing_table_generator_->mesh_graph->coordinate_to_chip(mesh_id, mesh_coord);
                const auto fabric_node_id = FabricNodeId(mesh_id, fabric_chip_id);
                auto mesh_coord_x = mesh_coord[0];
                auto mesh_coord_y = mesh_coord[1];

                const auto& port_directions = this->router_port_directions_to_physical_eth_chan_map_.at(fabric_node_id);

                const auto& golden_counts = golden_link_counts.at(MeshId{mesh_id}).at(fabric_chip_id);
                apply_min(
                    fabric_node_id,
                    port_directions,
                    RoutingDirection::E,
                    golden_counts,
                    row_min_planes.at(mesh_coord_x));
                apply_min(
                    fabric_node_id,
                    port_directions,
                    RoutingDirection::W,
                    golden_counts,
                    row_min_planes.at(mesh_coord_x));
                apply_min(
                    fabric_node_id,
                    port_directions,
                    RoutingDirection::N,
                    golden_counts,
                    col_min_planes.at(mesh_coord_y));
                apply_min(
                    fabric_node_id,
                    port_directions,
                    RoutingDirection::S,
                    golden_counts,
                    col_min_planes.at(mesh_coord_y));
            }

            // TODO: specialize by topology for better perf
            if (topology == Topology::Mesh || topology == Topology::Torus) {
                const auto rows_min = std::min_element(row_min_planes.begin(), row_min_planes.end());
                const auto cols_min = std::min_element(col_min_planes.begin(), col_min_planes.end());
                auto mesh_min = std::min(*rows_min, *cols_min);

                std::vector<size_t> recv_buf(*distributed_context.size());
                distributed_context.all_gather(
                    tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&mesh_min), sizeof(size_t)),
                    tt::stl::as_writable_bytes(tt::stl::Span<size_t>{recv_buf.data(), recv_buf.size()}));

                distributed_context.barrier();

                auto global_mesh_min = std::min(recv_buf.begin(), recv_buf.end());
                std::fill(row_min_planes.begin(), row_min_planes.end(), *global_mesh_min);
                std::fill(col_min_planes.begin(), col_min_planes.end(), *global_mesh_min);
            }

            // Second pass: Apply minimums to each device
            for (const auto& mesh_coord : local_mesh_coord_range) {
                auto fabric_chip_id =
                    this->routing_table_generator_->mesh_graph->coordinate_to_chip(mesh_id, mesh_coord);
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
    const MeshHostRankId host_rank =
        (host_rank_str == nullptr) ? MeshHostRankId{0} : MeshHostRankId{std::stoi(host_rank_str)};

    // If TT_MESH_ID is unset, assume this host is the only host in the system and owns all Meshes in
    // the MeshGraphDescriptor. Single Host Multi-Mesh is only used for testing purposes.
    const char* mesh_id_str = std::getenv("TT_MESH_ID");
    if (mesh_id_str == nullptr) {
        auto& ctx = tt::tt_metal::MetalContext::instance().global_distributed_context();
        TT_FATAL(
            *ctx.size() == 1 && *ctx.rank() == 0,
            "Not specifying both TT_MESH_ID and TT_MESH_HOST_RANK is only supported for single host systems.");
        std::vector<MeshId> local_mesh_ids;
        for (const auto& mesh_id : this->routing_table_generator_->mesh_graph->get_mesh_ids()) {
            const auto& host_ranks = this->routing_table_generator_->mesh_graph->get_host_ranks(mesh_id);
            TT_FATAL(
                host_ranks.size() == 1 && *host_ranks.values().front() == 0,
                "Mesh {} has {} host ranks, expected 1",
                *mesh_id,
                host_ranks.size());
            local_mesh_ids.push_back(mesh_id);
        }
        TT_FATAL(local_mesh_ids.size() > 0, "No local meshes found.");
        return LocalMeshBinding{.mesh_ids = std::move(local_mesh_ids), .host_rank = MeshHostRankId{0}};
    }

    // Otherwise, use the value from the environment variable.
    auto local_mesh_binding = LocalMeshBinding{.mesh_ids = {MeshId{std::stoi(mesh_id_str)}}, .host_rank = host_rank};

    log_debug(
        tt::LogDistributed,
        "Local mesh binding: mesh_id: {}, host_rank: {}",
        local_mesh_binding.mesh_ids[0],
        local_mesh_binding.host_rank);

    // Validate the local mesh binding exists in the mesh graph descriptor
    const auto mesh_ids = this->routing_table_generator_->mesh_graph->get_mesh_ids();
    TT_FATAL(
        std::find(mesh_ids.begin(), mesh_ids.end(), local_mesh_binding.mesh_ids[0]) != mesh_ids.end(),
        "Invalid TT_MESH_ID: Local mesh binding mesh_id {} not found in mesh graph descriptor",
        *local_mesh_binding.mesh_ids[0]);

    // Validate host rank (only if mesh_id is valid)
    const auto& host_ranks =
        this->routing_table_generator_->mesh_graph->get_host_ranks(local_mesh_binding.mesh_ids[0]).values();
    if (host_rank_str == nullptr) {
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

    // Find out which MPI ranks manage the same meshes as this host.
    uint64_t this_host_encoded_ids =
        encode_mesh_id_and_rank(local_mesh_binding_.mesh_ids[0], local_mesh_binding_.host_rank);
    std::vector<std::uint64_t> encoded_mesh_ids(*global_context->size());
    global_context->all_gather(
        ttsl::Span<std::byte>(reinterpret_cast<std::byte*>(&this_host_encoded_ids), sizeof(std::uint64_t)),
        ttsl::as_writable_bytes(ttsl::make_span(encoded_mesh_ids)));

    int mpi_rank = 0;
    for (std::uint64_t encoded_value : encoded_mesh_ids) {
        const auto [mesh_id, mesh_host_rank] = decode_mesh_id_and_rank(encoded_value);
        mpi_ranks_[mesh_id][mesh_host_rank] = tt::tt_metal::distributed::multihost::Rank{mpi_rank++};
    }

    // Create a sub-context for each mesh-host-rank pair.
    for (const auto local_mesh_id : local_mesh_binding_.mesh_ids) {
        auto mesh_host_ranks = mpi_ranks_.find(local_mesh_id);
        TT_FATAL(mesh_host_ranks != mpi_ranks_.end(), "Mesh {} not found in mpi_ranks.", local_mesh_id);
        if (mesh_host_ranks->second.size() == 1) {
            distributed_contexts_.emplace(local_mesh_id, host_local_context_);
        } else {
            std::vector<int> mpi_neighbors;
            std::transform(
                mesh_host_ranks->second.begin(),
                mesh_host_ranks->second.end(),
                std::back_inserter(mpi_neighbors),
                [](const auto& p) { return p.second.get(); });
            std::sort(mpi_neighbors.begin(), mpi_neighbors.end());
            distributed_contexts_.emplace(local_mesh_id, global_context->create_sub_context(mpi_neighbors));
        }
    }

    global_context->barrier();
}

void ControlPlane::init_control_plane(
    const std::string& mesh_graph_desc_file,
    std::optional<std::reference_wrapper<const std::map<FabricNodeId, chip_id_t>>>
        logical_mesh_chip_id_to_physical_chip_id_mapping) {

    this->routing_table_generator_ = std::make_unique<RoutingTableGenerator>(mesh_graph_desc_file);
    this->local_mesh_binding_ = this->initialize_local_mesh_binding();

    this->initialize_distributed_contexts();

    // Printing, only enabled with log_debug
    this->routing_table_generator_->mesh_graph->print_connectivity();

    if (logical_mesh_chip_id_to_physical_chip_id_mapping.has_value()) {
        this->load_physical_chip_mapping(logical_mesh_chip_id_to_physical_chip_id_mapping->get());
    } else {
        this->load_physical_chip_mapping(get_logical_chip_to_physical_chip_mapping(mesh_graph_desc_file));
    }
    this->initialize_intermesh_eth_links();
    this->generate_local_intermesh_link_table();
}

ControlPlane::ControlPlane(const std::string& mesh_graph_desc_file) {
    init_control_plane(mesh_graph_desc_file, std::nullopt);
}

ControlPlane::ControlPlane(
    const std::string& mesh_graph_desc_file,
    const std::map<FabricNodeId, chip_id_t>& logical_mesh_chip_id_to_physical_chip_id_mapping) {
    init_control_plane(mesh_graph_desc_file, logical_mesh_chip_id_to_physical_chip_id_mapping);
}

void ControlPlane::load_physical_chip_mapping(
    const std::map<FabricNodeId, chip_id_t>& logical_mesh_chip_id_to_physical_chip_id_mapping) {
    this->logical_mesh_chip_id_to_physical_chip_id_mapping_ = logical_mesh_chip_id_to_physical_chip_id_mapping;
    this->validate_mesh_connections();
}

void ControlPlane::validate_mesh_connections(MeshId mesh_id) const {
    MeshShape mesh_shape = routing_table_generator_->mesh_graph->get_mesh_shape(mesh_id);
    auto get_physical_chip_id = [&](const MeshCoordinate& mesh_coord) {
        auto fabric_chip_id = this->routing_table_generator_->mesh_graph->coordinate_to_chip(mesh_id, mesh_coord);
        return logical_mesh_chip_id_to_physical_chip_id_mapping_.at(FabricNodeId(mesh_id, fabric_chip_id));
    };
    auto validate_chip_connections = [&](const MeshCoordinate& mesh_coord, const MeshCoordinate& other_mesh_coord) {
        chip_id_t physical_chip_id = get_physical_chip_id(mesh_coord);
        chip_id_t physical_chip_id_other = get_physical_chip_id(other_mesh_coord);
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
        chip_id_t physical_chip_id = get_physical_chip_id(mesh_coord);
        MeshCoordinate mesh_coord_next{mesh_coord[0], mesh_coord[1] + 1};
        MeshCoordinate mesh_coord_next_row{mesh_coord[0] + 1, mesh_coord[1]};
        const auto& eth_links = get_ethernet_cores_grouped_by_connected_chips(physical_chip_id);
        if (mesh_coord_range.contains(mesh_coord_next)) {
            validate_chip_connections(mesh_coord, mesh_coord_next);
        }
        if (mesh_coord_range.contains(mesh_coord_next_row)) {
            validate_chip_connections(mesh_coord, mesh_coord_next_row);
        }
    }
}

void ControlPlane::validate_mesh_connections() const {
    for (const auto& mesh_id : this->routing_table_generator_->mesh_graph->get_mesh_ids()) {
        if (this->is_local_mesh(mesh_id)) {
            this->validate_mesh_connections(mesh_id);
        }
    }
}

// TODO: refactor mesh_ns_size/mesh_ew_size to use MeshCoordinateRange
// TODO: update logical_mesh_chip_id_to_physical_chip_id_mapping_ to be updated here probably
std::vector<chip_id_t> ControlPlane::get_mesh_physical_chip_ids(
    const tt::tt_metal::distributed::MeshContainer<chip_id_t>& mesh_container,
    std::optional<chip_id_t> nw_corner_chip_id) const {
    // Convert the coordinate range to a set of chip IDs using MeshContainer iterator
    const auto& user_chip_ids = tt::tt_metal::MetalContext::instance().get_cluster().user_exposed_chip_ids();
    TT_FATAL(
        user_chip_ids.size() >= mesh_container.size(),
        "Number of chips visible ({}) is less than the number of chips specified in mesh graph descriptor ({}), check system status with tt-smi that all chips are visible.",
        user_chip_ids.size(),
        mesh_container.size());

    // Special case for 1x1 mesh
    if (mesh_container.shape() == tt::tt_metal::distributed::MeshShape(1, 1)) {
        std::vector<chip_id_t> physical_chip_ids(1);
        physical_chip_ids[0] = *user_chip_ids.begin();
        return physical_chip_ids;
    }

    // Build mesh adjacency map using BFS
    std::uint32_t num_ports_per_side =
        routing_table_generator_->mesh_graph->get_chip_spec().num_eth_ports_per_direction;
    auto topology_info = build_mesh_adjacency_map(
        user_chip_ids,
        mesh_container.shape(),
        [num_ports_per_side](chip_id_t chip_id) {
            return get_adjacent_chips_from_ethernet_connections(chip_id, num_ports_per_side);
        },
        nw_corner_chip_id);

    // Handle 1D meshes (1xN or Nx1)
    bool is_1d_mesh = (topology_info.ns_size == 1) || (topology_info.ew_size == 1);
    if (is_1d_mesh) {
        return convert_1d_mesh_adjacency_to_row_major_vector(topology_info);
    }

    // Handle 2D meshes
    return convert_2d_mesh_adjacency_to_row_major_vector(topology_info, nw_corner_chip_id);
}

std::map<FabricNodeId, chip_id_t> ControlPlane::get_logical_chip_to_physical_chip_mapping(
    const std::string& mesh_graph_desc_file) {
    std::map<FabricNodeId, chip_id_t> logical_mesh_chip_id_to_physical_chip_id_mapping;

    std::string mesh_graph_desc_filename = std::filesystem::path(mesh_graph_desc_file).filename().string();

    // NOTE: This is a special case for the TG mesh graph descriptor.
    // It has to use Ethernet coordinates because ethernet coordinates must be mapped manually to physical chip IDs
    // because the TG intermesh ethernet links could be inverted when mapped to physical chip IDs.
    if (mesh_graph_desc_filename.starts_with("tg_mesh_graph_descriptor.")) {
        // Add the N150 MMIO devices
        auto eth_coords_per_chip =
            tt::tt_metal::MetalContext::instance().get_cluster().get_all_chip_ethernet_coordinates();
        std::unordered_map<int, chip_id_t> eth_coord_y_for_gateway_chips = {};
        for (const auto& [chip_id, eth_coord] : eth_coords_per_chip) {
            if (tt::tt_metal::MetalContext::instance().get_cluster().get_board_type(chip_id) == BoardType::N150) {
                eth_coord_y_for_gateway_chips[eth_coord.y] = chip_id;
            }
        }
        logical_mesh_chip_id_to_physical_chip_id_mapping.insert(
            {FabricNodeId(MeshId{0}, 0), eth_coord_y_for_gateway_chips[3]});
        logical_mesh_chip_id_to_physical_chip_id_mapping.insert(
            {FabricNodeId(MeshId{1}, 0), eth_coord_y_for_gateway_chips[2]});
        logical_mesh_chip_id_to_physical_chip_id_mapping.insert(
            {FabricNodeId(MeshId{2}, 0), eth_coord_y_for_gateway_chips[1]});
        logical_mesh_chip_id_to_physical_chip_id_mapping.insert(
            {FabricNodeId(MeshId{3}, 0), eth_coord_y_for_gateway_chips[0]});

        auto nw_chip_physical_id =
            tt::tt_metal::MetalContext::instance().get_cluster().get_physical_chip_id_from_eth_coord({0, 3, 7, 0, 1});
        // Main board
        const auto& mesh_container = this->routing_table_generator_->mesh_graph->get_chip_ids(MeshId{4});
        const auto& physical_chip_ids = this->get_mesh_physical_chip_ids(mesh_container, nw_chip_physical_id);
        for (std::uint32_t i = 0; i < physical_chip_ids.size(); i++) {
            logical_mesh_chip_id_to_physical_chip_id_mapping.insert({FabricNodeId(MeshId{4}, i), physical_chip_ids[i]});
        }
        // This case can be depreciated once we have multi-host testing and validate it working
    } else {
        // Iterate over every mesh defined in the mesh-graph descriptor and embed it on top of
        // the physical cluster using the generic helper.
        for (const auto& mesh_id : this->routing_table_generator_->mesh_graph->get_mesh_ids()) {
            if (!this->is_local_mesh(mesh_id)) {
                continue;
            }
            auto host_rank_id = this->get_local_host_rank_id_binding();
            const auto& mesh_container = this->routing_table_generator_->mesh_graph->get_chip_ids(mesh_id, host_rank_id);

            std::optional<chip_id_t> nw_chip_physical_id = std::nullopt;
            const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
            // TODO: remove once we use global physical graph to map logical big mesh to physical chips
            // NOTE: This nw chip may not be set the same for UBB devices when using the Mock Cluster Descriptor
            if (cluster.get_board_type(0) == BoardType::UBB) {
                for (const auto& chip_id : cluster.all_chip_ids()) {
                    auto candidate_ubb_id = tt::tt_fabric::get_ubb_id(chip_id);
                    if (candidate_ubb_id.tray_id == 1 && candidate_ubb_id.asic_id == 1) {
                        nw_chip_physical_id = chip_id;
                    }
                }
            }

            const auto& physical_chip_ids = this->get_mesh_physical_chip_ids(mesh_container, nw_chip_physical_id);
            std::uint32_t i = 0;
            for (const auto& [_, fabric_chip_id] : mesh_container) {
                logical_mesh_chip_id_to_physical_chip_id_mapping.emplace(
                    FabricNodeId(mesh_id, fabric_chip_id), physical_chip_ids[i]);
                i++;
            }
        }
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
            const auto& local_mesh_chip_id_container =
                this->routing_table_generator_->mesh_graph->get_chip_ids(mesh_id, host_rank_id);
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
        const auto& global_mesh_chip_id_container = this->routing_table_generator_->mesh_graph->get_chip_ids(mesh_id);
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
            for (chip_id_t dst_fabric_chip_id = 0;
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
        const auto& global_mesh_chip_id_container =
            this->routing_table_generator_->mesh_graph->get_chip_ids(src_mesh_id);
        for (const auto& [_, src_fabric_chip_id] : global_mesh_chip_id_container) {
            const auto src_fabric_node_id = FabricNodeId(src_mesh_id, src_fabric_chip_id);
            this->inter_mesh_routing_tables_[src_fabric_node_id].resize(
                num_ports_per_chip);  // contains more entries than needed
            for (int i = 0; i < num_ports_per_chip; i++) {
                // Size the routing table to the number of meshes
                this->inter_mesh_routing_tables_[src_fabric_node_id][i].resize(
                    router_inter_mesh_routing_table[src_mesh_id_val][src_fabric_chip_id].size());
            }
            for (chip_id_t dst_mesh_id_val = 0;
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

void ControlPlane::trim_ethernet_channels_not_mapped_to_live_routing_planes() {
    auto user_mesh_ids = this->get_user_physical_mesh_ids();
    std::unordered_set<MeshId> user_mesh_ids_set(user_mesh_ids.begin(), user_mesh_ids.end());
    if (tt::tt_metal::MetalContext::instance().get_fabric_config() != tt_fabric::FabricConfig::CUSTOM) {
        for (auto& [fabric_node_id, directional_eth_chans] : this->router_port_directions_to_physical_eth_chan_map_) {
            if (user_mesh_ids_set.count(fabric_node_id.mesh_id) == 0) {
                continue;
            }
            for (auto direction :
                 {RoutingDirection::N, RoutingDirection::S, RoutingDirection::E, RoutingDirection::W}) {
                if (directional_eth_chans.find(direction) != directional_eth_chans.end()) {
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
        this->router_port_directions_to_num_routing_planes_map_.find(fabric_node_id) !=
            this->router_port_directions_to_num_routing_planes_map_.end(),
        "Fabric node id (mesh={}, chip={}) not found in router port directions to num routing planes map",
        fabric_node_id.mesh_id,
        fabric_node_id.chip_id);
    TT_FATAL(
        this->router_port_directions_to_num_routing_planes_map_.at(fabric_node_id).find(routing_direction) !=
            this->router_port_directions_to_num_routing_planes_map_.at(fabric_node_id).end(),
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

    // When running multi-host workloads, have all hosts in the system exchange their local intermesh link tables
    // with all other hosts in the system. This information is used to assign directions to intermesh links.
    this->exchange_intermesh_link_tables();

    const auto& intra_mesh_connectivity = this->routing_table_generator_->mesh_graph->get_intra_mesh_connectivity();
    const auto& inter_mesh_connectivity = this->routing_table_generator_->mesh_graph->get_inter_mesh_connectivity();
    // Initialize the bookkeeping for mapping from mesh/chip/direction to physical ethernet channels
    for (const auto& [fabric_node_id, _] : this->logical_mesh_chip_id_to_physical_chip_id_mapping_) {
        if (!this->router_port_directions_to_physical_eth_chan_map_.contains(fabric_node_id)) {
            this->router_port_directions_to_physical_eth_chan_map_[fabric_node_id] = {};
        }
    }

    auto host_rank_id = this->get_local_host_rank_id_binding();
    for (std::uint32_t mesh_id_val = 0; mesh_id_val < intra_mesh_connectivity.size(); mesh_id_val++) {
        // TODO: we can probably remove this check, in general should update these loops to iterate over local meshes
        MeshId mesh_id{mesh_id_val};
        if (!this->is_local_mesh(mesh_id)) {
            continue;
        }
        const auto& local_mesh_coord_range = this->get_coord_range(mesh_id, MeshScope::LOCAL);
        const auto& local_mesh_chip_id_container =
            this->routing_table_generator_->mesh_graph->get_chip_ids(mesh_id, host_rank_id);
        for (const auto& [_, fabric_chip_id] : local_mesh_chip_id_container) {
            const auto fabric_node_id = FabricNodeId(mesh_id, fabric_chip_id);
            auto physical_chip_id = this->get_physical_chip_id_from_fabric_node_id(fabric_node_id);

            for (const auto& [logical_connected_chip_id, edge] : intra_mesh_connectivity[*mesh_id][fabric_chip_id]) {
                auto connected_mesh_coord =
                    this->routing_table_generator_->mesh_graph->chip_to_coordinate(mesh_id, logical_connected_chip_id);
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
                        tt::tt_metal::MetalContext::instance().get_cluster().get_ethernet_cores_grouped_by_connected_chips(
                            physical_chip_id);

                    bool connections_exist = connected_chips_and_eth_cores.find(physical_connected_chip_id) !=
                                             connected_chips_and_eth_cores.end();
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
                    TT_ASSERT(
                        this->routing_table_generator_->mesh_graph
                            ->get_host_rank_for_chip(mesh_id, logical_connected_chip_id)
                            .has_value(),
                        "Mesh {} Chip {} does not have a host rank associated with it",
                        mesh_id,
                        fabric_chip_id);
                    auto connected_host_rank_id = this->routing_table_generator_->mesh_graph
                                                      ->get_host_rank_for_chip(mesh_id, logical_connected_chip_id)
                                                      .value();
                    auto unique_chip_id =
                        tt::tt_metal::MetalContext::instance().get_cluster().get_unique_chip_ids().at(physical_chip_id);
                    // Look up connected chip's intermesh link table and grab local desc channel
                    // TODO: need to add validate to make sure there is bidrectional traffic
                    for (const auto& [local_desc, peer_desc] :
                         peer_intermesh_link_tables_[mesh_id][connected_host_rank_id]) {
                        if (peer_desc.board_id == unique_chip_id) {
                            tt::umd::CoreCoord eth_core =
                                tt::tt_metal::MetalContext::instance()
                                    .get_cluster()
                                    .get_soc_desc(physical_chip_id)
                                    .get_eth_core_for_channel(local_desc.chan_id, CoordSystem::LOGICAL);
                            this->assign_direction_to_fabric_eth_core(fabric_node_id, eth_core, edge.port_direction);
                        }
                    }
                }
            }
        }
    }

    const auto& distributed_context = tt::tt_metal::MetalContext::instance().global_distributed_context();
    for (std::uint32_t mesh_id_val = 0; mesh_id_val < inter_mesh_connectivity.size(); mesh_id_val++) {
        MeshId mesh_id{mesh_id_val};
        if (this->is_local_mesh(mesh_id)) {
            const auto& local_mesh_chip_id_container =
                this->routing_table_generator_->mesh_graph->get_chip_ids(mesh_id, host_rank_id);
            for (const auto& [_, fabric_chip_id] : local_mesh_chip_id_container) {
                const auto fabric_node_id = FabricNodeId(mesh_id, fabric_chip_id);
                if (*(distributed_context.size()) > 1) {
                    this->assign_intermesh_link_directions_to_remote_host(fabric_node_id);
                } else {
                    this->assign_intermesh_link_directions_to_local_host(fabric_node_id);
                }
            }
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

void ControlPlane::write_routing_tables_to_eth_cores(MeshId mesh_id, chip_id_t chip_id) const {
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
            tt::tt_fabric::fabric_router_l1_config_t fabric_router_config{};
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
    TT_FATAL(false, "Physical chip id {} not found in control plane chip mapping. You are calling for a chip outside of the fabric cluster. Check that your mesh graph descriptor specifies the correct topology", physical_chip_id);
    return FabricNodeId(MeshId{0}, 0);
}

chip_id_t ControlPlane::get_physical_chip_id_from_fabric_node_id(const FabricNodeId& fabric_node_id) const {
    TT_ASSERT(logical_mesh_chip_id_to_physical_chip_id_mapping_.contains(fabric_node_id));
    return logical_mesh_chip_id_to_physical_chip_id_mapping_.at(fabric_node_id);
}

std::pair<FabricNodeId, chan_id_t> ControlPlane::get_connected_mesh_chip_chan_ids(
    FabricNodeId fabric_node_id, chan_id_t chan_id) const {
    // TODO: simplify this and use Global Physical Desc in ControlPlane soon
    const auto& intra_mesh_connectivity = this->routing_table_generator_->mesh_graph->get_intra_mesh_connectivity();
    const auto& inter_mesh_connectivity = this->routing_table_generator_->mesh_graph->get_inter_mesh_connectivity();
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

std::vector<std::pair<FabricNodeId, chan_id_t>> ControlPlane::get_fabric_route(
    FabricNodeId src_fabric_node_id, FabricNodeId dst_fabric_node_id, chan_id_t src_chan_id) const {
    // Query the mesh coord range owned by the current host
    auto host_local_coord_range = this->get_coord_range(this->get_local_mesh_id_bindings()[0], MeshScope::LOCAL);
    auto src_mesh_coord = this->routing_table_generator_->mesh_graph->chip_to_coordinate(
        src_fabric_node_id.mesh_id, src_fabric_node_id.chip_id);
    auto dst_mesh_coord = this->routing_table_generator_->mesh_graph->chip_to_coordinate(
        dst_fabric_node_id.mesh_id, dst_fabric_node_id.chip_id);

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

void write_to_worker_or_fabric_tensix_cores(
    const void* worker_data,
    const void* dispatcher_data,
    const void* tensix_extension_data,
    size_t size,
    tt::tt_metal::HalL1MemAddrType addr_type,
    chip_id_t physical_chip_id) {
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
        const auto& tensix_config = fabric_context.get_tensix_config();
        fabric_mux_cores_translated = tensix_config.get_translated_fabric_mux_cores();
        dispatch_mux_cores_translated = tensix_config.get_translated_dispatch_mux_cores();
    }

    enum class CoreType { Worker, FabricTensixExtension, DispatcherMux };

    auto get_core_type = [&](const CoreCoord& core_coord) -> CoreType {
        if (fabric_mux_cores_translated.find(core_coord) != fabric_mux_cores_translated.end()) {
            return CoreType::FabricTensixExtension;
        }
        if (dispatch_mux_cores_translated.find(core_coord) != dispatch_mux_cores_translated.end()) {
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

void write_to_all_tensix_cores(
    const void* data, size_t size, tt::tt_metal::HalL1MemAddrType addr_type, chip_id_t physical_chip_id) {
    TT_FATAL(
        size ==
            tt_metal::MetalContext::instance().hal().get_dev_size(tt_metal::HalProgrammableCoreType::TENSIX, addr_type),
        "ControlPlane: Tensix core data size mismatch expected {} but got {}",
        size,
        tt_metal::MetalContext::instance().hal().get_dev_size(tt_metal::HalProgrammableCoreType::TENSIX, addr_type));
    const auto& soc_desc = tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(physical_chip_id);
    const std::vector<tt::umd::CoreCoord>& tensix_cores = soc_desc.get_cores(CoreType::TENSIX, CoordSystem::TRANSLATED);
    // Write to all Tensix cores
    for (const auto& tensix_core : tensix_cores) {
        tt::tt_metal::MetalContext::instance().get_cluster().write_core(
            data,
            size,
            tt_cxy_pair(physical_chip_id, CoreCoord{tensix_core.x, tensix_core.y}),
            tt_metal::MetalContext::instance().hal().get_dev_addr(
                tt_metal::HalProgrammableCoreType::TENSIX, addr_type));
    }
}

// Write routing table to Tensix cores' L1 on a specific chip
void ControlPlane::write_routing_tables_to_tensix_cores(MeshId mesh_id, chip_id_t chip_id) const {
    FabricNodeId src_fabric_node_id{mesh_id, chip_id};
    auto physical_chip_id = this->logical_mesh_chip_id_to_physical_chip_id_mapping_.at(src_fabric_node_id);

    tensix_routing_l1_info_t tensix_routing_info = {};
    tensix_routing_info.my_mesh_id = *mesh_id;
    tensix_routing_info.my_device_id = chip_id;

    // Build intra-mesh routing entries (chip-to-chip routing)
    const auto& router_intra_mesh_routing_table = this->routing_table_generator_->get_intra_mesh_table();
    TT_FATAL(
        router_intra_mesh_routing_table[*mesh_id][chip_id].size() <= tt::tt_fabric::MAX_MESH_SIZE,
        "ControlPlane: Intra mesh routing table size exceeds maximum allowed size");

    // Initialize all entries to INVALID_ROUTING_TABLE_ENTRY first
    for (std::uint32_t i = 0; i < tt::tt_fabric::MAX_MESH_SIZE; i++) {
        tensix_routing_info.intra_mesh_routing_table.set_original_direction(
            i, static_cast<std::uint8_t>(eth_chan_magic_values::INVALID_ROUTING_TABLE_ENTRY));
    }

    for (chip_id_t dst_chip_id = 0; dst_chip_id < router_intra_mesh_routing_table[*mesh_id][chip_id].size();
         dst_chip_id++) {
        if (chip_id == dst_chip_id) {
            tensix_routing_info.intra_mesh_routing_table.set_original_direction(
                dst_chip_id, static_cast<std::uint8_t>(eth_chan_magic_values::INVALID_DIRECTION));
            continue;
        }
        auto forwarding_direction = router_intra_mesh_routing_table[*mesh_id][chip_id][dst_chip_id];
        std::uint8_t direction_value =
            forwarding_direction != RoutingDirection::NONE
                ? static_cast<std::uint8_t>(this->routing_direction_to_eth_direction(forwarding_direction))
                : static_cast<std::uint8_t>(eth_chan_magic_values::INVALID_DIRECTION);
        tensix_routing_info.intra_mesh_routing_table.set_original_direction(dst_chip_id, direction_value);
    }

    // Build inter-mesh routing entries (mesh-to-mesh routing)
    const auto& router_inter_mesh_routing_table = this->routing_table_generator_->get_inter_mesh_table();
    TT_FATAL(
        router_inter_mesh_routing_table[*mesh_id][chip_id].size() <= tt::tt_fabric::MAX_NUM_MESHES,
        "ControlPlane: Inter mesh routing table size exceeds maximum allowed size");

    // Initialize all entries to INVALID_ROUTING_TABLE_ENTRY first
    for (std::uint32_t i = 0; i < tt::tt_fabric::MAX_NUM_MESHES; i++) {
        tensix_routing_info.inter_mesh_routing_table.set_original_direction(
            i, static_cast<std::uint8_t>(eth_chan_magic_values::INVALID_ROUTING_TABLE_ENTRY));
    }

    for (std::uint32_t dst_mesh_id = 0; dst_mesh_id < router_inter_mesh_routing_table[*mesh_id][chip_id].size();
         dst_mesh_id++) {
        if (*mesh_id == dst_mesh_id) {
            tensix_routing_info.inter_mesh_routing_table.set_original_direction(
                dst_mesh_id, static_cast<std::uint8_t>(eth_chan_magic_values::INVALID_DIRECTION));
            continue;
        }
        auto forwarding_direction = router_inter_mesh_routing_table[*mesh_id][chip_id][dst_mesh_id];
        std::uint8_t direction_value =
            forwarding_direction != RoutingDirection::NONE
                ? static_cast<std::uint8_t>(this->routing_direction_to_eth_direction(forwarding_direction))
                : static_cast<std::uint8_t>(eth_chan_magic_values::INVALID_DIRECTION);
        tensix_routing_info.inter_mesh_routing_table.set_original_direction(dst_mesh_id, direction_value);
    }

    write_to_all_tensix_cores(
        &tensix_routing_info,
        sizeof(tensix_routing_l1_info_t),
        tt::tt_metal::HalL1MemAddrType::TENSIX_ROUTING_TABLE,
        physical_chip_id);
}

// Write connection info to Tensix cores' L1 on a specific chip
void ControlPlane::write_fabric_connections_to_tensix_cores(MeshId mesh_id, chip_id_t chip_id) const {
    if (this->fabric_context_ == nullptr) {
        log_warning(
            tt::LogFabric,
            "ControlPlane: Fabric context is not set, cannot write fabric connections to Tensix cores for M%dD%d",
            *mesh_id,
            chip_id);
        return;
    }
    FabricNodeId src_fabric_node_id{mesh_id, chip_id};
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
                eth_channel_id,
                router_direction);

            // Mark this connection as valid for fabric communication
            fabric_worker_connections.valid_connections_mask |= (1u << eth_channel_id);
            fabric_dispatcher_connections.valid_connections_mask |= (1u << eth_channel_id);
            fabric_tensix_connections.valid_connections_mask |= (1u << eth_channel_id);
            num_eth_endpoint++;
        }
    }

    // Write fabric connections (fabric router config) to mux cores and tensix connections (tensix config) to worker
    // cores
    write_to_worker_or_fabric_tensix_cores(
        &fabric_worker_connections,      // worker_data - goes to mux cores
        &fabric_dispatcher_connections,  // dispatcher_data - goes to dispatcher cores
        &fabric_tensix_connections,      // tensix_extension_data - goes to worker cores
        sizeof(tt::tt_fabric::tensix_fabric_connections_l1_info_t),
        tt::tt_metal::HalL1MemAddrType::TENSIX_FABRIC_CONNECTIONS,
        physical_chip_id);
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

template <>
void ControlPlane::write_all_to_all_routing_fields<1, false>(MeshId mesh_id) const {
    auto host_rank_id = this->get_local_host_rank_id_binding();
    const auto& local_mesh_chip_id_container =
        this->routing_table_generator_->mesh_graph->get_chip_ids(mesh_id, host_rank_id);
    uint16_t num_chips = MAX_CHIPS_LOWLAT_1D < local_mesh_chip_id_container.size()
                             ? MAX_CHIPS_LOWLAT_1D
                             : static_cast<uint16_t>(local_mesh_chip_id_container.size());

    routing_path_t<1, false> routing_path;
    routing_path.calculate_chip_to_all_routing_fields(0, num_chips);

    // For each source chip in the current mesh
    for (const auto& [_, src_chip_id] : local_mesh_chip_id_container) {
        FabricNodeId src_fabric_node_id(mesh_id, src_chip_id);
        auto physical_chip_id = this->logical_mesh_chip_id_to_physical_chip_id_mapping_.at(src_fabric_node_id);

        write_to_all_tensix_cores(
            &routing_path,
            sizeof(routing_path),
            tt::tt_metal::HalL1MemAddrType::TENSIX_ROUTING_PATH_1D,
            physical_chip_id);
    }
}

template <>
void ControlPlane::write_all_to_all_routing_fields<2, true>(MeshId mesh_id) const {
    auto host_rank_id = this->get_local_host_rank_id_binding();
    const auto& local_mesh_chip_id_container =
        this->routing_table_generator_->mesh_graph->get_chip_ids(mesh_id, host_rank_id);

    // Get mesh shape for 2D routing calculation
    MeshShape mesh_shape = this->get_physical_mesh_shape(mesh_id);
    uint16_t num_chips = mesh_shape[0] * mesh_shape[1];
    uint16_t ew_dim = mesh_shape[1];  // east-west dimension
    TT_ASSERT(num_chips <= 256, "Number of chips exceeds 256 for mesh {}", *mesh_id);
    TT_ASSERT(
        mesh_shape[0] <= 16 && mesh_shape[1] <= 16,
        "One or both of mesh axis exceed 16 for mesh {}: {}x{}",
        *mesh_id,
        mesh_shape[0],
        mesh_shape[1]);

    for (const auto& [_, src_chip_id] : local_mesh_chip_id_container) {
        routing_path_t<2, true> routing_path;
        FabricNodeId src_fabric_node_id(mesh_id, src_chip_id);

        routing_path.calculate_chip_to_all_routing_fields(src_chip_id, num_chips, ew_dim);
        auto physical_chip_id = this->logical_mesh_chip_id_to_physical_chip_id_mapping_.at(src_fabric_node_id);

        write_to_all_tensix_cores(
            &routing_path,
            sizeof(routing_path),
            tt::tt_metal::HalL1MemAddrType::TENSIX_ROUTING_PATH_2D,
            physical_chip_id);
    }
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
            auto fabric_chip_id = this->routing_table_generator_->mesh_graph->coordinate_to_chip(mesh_id, mesh_coord);
            auto fabric_node_id = FabricNodeId(mesh_id, fabric_chip_id);
            TT_ASSERT(
                this->inter_mesh_routing_tables_.contains(fabric_node_id),
                "Intra mesh routing tables keys mismatch with inter mesh routing tables");
            this->write_routing_tables_to_tensix_cores(fabric_node_id.mesh_id, fabric_node_id.chip_id);
            this->write_fabric_connections_to_tensix_cores(fabric_node_id.mesh_id, fabric_node_id.chip_id);
            this->write_routing_tables_to_eth_cores(fabric_node_id.mesh_id, fabric_node_id.chip_id);
        }
    }

    for (const auto& mesh_id : this->get_local_mesh_id_bindings()) {
        this->write_all_to_all_routing_fields<1, false>(mesh_id);
        this->write_all_to_all_routing_fields<2, true>(mesh_id);
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

MeshShape ControlPlane::get_physical_mesh_shape(MeshId mesh_id, MeshScope scope) const {
    std::optional<MeshHostRankId> local_host_rank_id =
        MeshScope::LOCAL == scope ? std::make_optional(this->get_local_host_rank_id_binding()) : std::nullopt;
    return this->routing_table_generator_->mesh_graph->get_mesh_shape(mesh_id, local_host_rank_id);
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

void ControlPlane::initialize_fabric_context(tt_fabric::FabricConfig fabric_config) {
    TT_FATAL(this->fabric_context_ == nullptr, "Trying to re-initialize fabric context");
    this->fabric_context_ = std::make_unique<FabricContext>(fabric_config);
}

FabricContext& ControlPlane::get_fabric_context() const {
    TT_FATAL(this->fabric_context_ != nullptr, "Trying to get un-initialized fabric context");
    return *this->fabric_context_;
}

void ControlPlane::clear_fabric_context() { this->fabric_context_.reset(nullptr); }

void ControlPlane::initialize_fabric_tensix_datamover_config() {
    TT_FATAL(this->fabric_context_ != nullptr, "Fabric context must be initialized first");
    this->fabric_context_->initialize_tensix_config();
}

void ControlPlane::initialize_intermesh_eth_links() {
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();

    // Iterate over all chips in the cluster and populate the intermesh_eth_links
    for (const auto& chip_id : cluster.all_chip_ids()) {
        auto& intermesh_eth_links = intermesh_eth_links_[chip_id];
        // Remote connections visible to UMD
        auto remote_connections = cluster.get_ethernet_connections_to_remote_devices().find(chip_id);
        if (remote_connections != cluster.get_ethernet_connections_to_remote_devices().end()) {
            const auto& soc_desc = cluster.get_soc_desc(chip_id);
            for (auto [link, _] : remote_connections->second) {
                // Find the CoreCoord for this channel
                for (const auto& [core_coord, channel] : soc_desc.logical_eth_core_to_chan_map) {
                    if (channel == link) {
                        intermesh_eth_links.push_back({core_coord, link});
                        break;
                    }
                }
            }
        }
    }
}

bool ControlPlane::system_has_intermesh_links() const { return !this->get_all_intermesh_eth_links().empty(); }

bool ControlPlane::has_intermesh_links(chip_id_t chip_id) const {
    return !this->get_intermesh_eth_links(chip_id).empty();
}

bool ControlPlane::is_intermesh_eth_link(chip_id_t chip_id, CoreCoord eth_core) const {
    for (const auto& [link_eth_core, channel] : this->get_intermesh_eth_links(chip_id)) {
        if (link_eth_core == eth_core) {
            return true;
        }
    }
    return false;
}

const std::vector<std::pair<CoreCoord, chan_id_t>>& ControlPlane::get_intermesh_eth_links(chip_id_t chip_id) const {
    return intermesh_eth_links_.at(chip_id);
}

const std::unordered_map<chip_id_t, std::vector<std::pair<CoreCoord, chan_id_t>>>&
ControlPlane::get_all_intermesh_eth_links() const {
    return intermesh_eth_links_;
}

std::unordered_set<CoreCoord> ControlPlane::get_active_ethernet_cores(
    chip_id_t chip_id, bool skip_reserved_cores) const {
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
            if (freq_retrain_eth_cores.find(eth_core) != freq_retrain_eth_cores.end()) {
                continue;
            }

            active_ethernet_cores.insert(eth_core);
        }
        // WH has a special case where mmio chips with remote connections must always have certain channels active
        if (cluster.arch() == tt::ARCH::WORMHOLE_B0 && cluster_desc->is_chip_mmio_capable(chip_id) &&
            cluster.get_tunnels_from_mmio_device(chip_id).size() > 0) {
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
                if (logical_active_eth_channels.find(eth_channel) == logical_active_eth_channels.end()) {
                    tt::umd::CoreCoord eth_core = soc_desc.get_eth_core_for_channel(eth_channel, CoordSystem::LOGICAL);
                    active_ethernet_cores.insert(eth_core);
                }
            }
        }
    }
    return active_ethernet_cores;
}

std::unordered_set<CoreCoord> ControlPlane::get_inactive_ethernet_cores(chip_id_t chip_id) const {
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    std::unordered_set<CoreCoord> active_ethernet_cores = this->get_active_ethernet_cores(chip_id);
    std::unordered_set<CoreCoord> inactive_ethernet_cores;

    for (const auto& [eth_core, chan] : cluster.get_soc_desc(chip_id).logical_eth_core_to_chan_map) {
        if (active_ethernet_cores.find(eth_core) == active_ethernet_cores.end()) {
            inactive_ethernet_cores.insert(eth_core);
        }
    }
    return inactive_ethernet_cores;
}

void ControlPlane::generate_local_intermesh_link_table() {
    // Populate the local to remote mapping for all intermesh links
    // This cannot be done by UMD, since it has no knowledge of links marked
    // for intermesh routing (these links are hidden from UMD).
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    intermesh_link_table_.local_mesh_id = local_mesh_binding_.mesh_ids[0];
    intermesh_link_table_.local_host_rank_id = this->get_local_host_rank_id_binding();
    for (const auto& chip_id : cluster.user_exposed_chip_ids()) {
        auto local_board_id = cluster.get_unique_chip_ids().find(chip_id);
        if (local_board_id == cluster.get_unique_chip_ids().end()) {
            chip_id_to_asic_id_[chip_id] = chip_id;
            continue;
        }
        chip_id_to_asic_id_[chip_id] = local_board_id->second;
        for (const auto& [eth_core, chan_id] : this->get_intermesh_eth_links(chip_id)) {
            auto [remote_board_id, remote_chan_id] =
                cluster.get_ethernet_connections_to_remote_devices().at(chip_id).at(chan_id);
            auto local_eth_chan_desc = EthChanDescriptor{
                .board_id = local_board_id->second,
                .chan_id = chan_id,
            };
            auto remote_eth_chan_desc = EthChanDescriptor{
                .board_id = remote_board_id,
                .chan_id = remote_chan_id,
            };
            intermesh_link_table_.intermesh_links[local_eth_chan_desc] = remote_eth_chan_desc;
        }
    }
}

void ControlPlane::exchange_intermesh_link_tables() {
    const auto& distributed_context = tt::tt_metal::MetalContext::instance().global_distributed_context();
    if (*distributed_context.size() == 1) {
        // No need to exchange intermesh link tables when running a single process
        return;
    }

    auto serialized_table = tt::tt_fabric::serialize_to_bytes(intermesh_link_table_);
    std::vector<uint8_t> serialized_remote_table;
    auto my_rank = *(distributed_context.rank());

    for (std::size_t bcast_root = 0; bcast_root < *(distributed_context.size()); ++bcast_root) {
        if (my_rank == bcast_root) {
            // Issue the broadcast from the current process to all other processes in the world
            int local_table_size_bytes = serialized_table.size();  // Send txn size first
            distributed_context.broadcast(
                tt::stl::Span<std::byte>(
                    reinterpret_cast<std::byte*>(&local_table_size_bytes), sizeof(local_table_size_bytes)),
                distributed_context.rank());

            distributed_context.broadcast(
                tt::stl::as_writable_bytes(tt::stl::Span<uint8_t>(serialized_table.data(), serialized_table.size())),
                distributed_context.rank());
        } else {
            // Acknowledge the broadcast issued by the root
            int remote_table_size_bytes = 0;  // Receive the size of the serialized descriptor
            distributed_context.broadcast(
                tt::stl::Span<std::byte>(
                    reinterpret_cast<std::byte*>(&remote_table_size_bytes), sizeof(remote_table_size_bytes)),
                tt::tt_metal::distributed::multihost::Rank{bcast_root});
            serialized_remote_table.clear();
            serialized_remote_table.resize(remote_table_size_bytes);
            distributed_context.broadcast(
                tt::stl::as_writable_bytes(
                    tt::stl::Span<uint8_t>(serialized_remote_table.data(), serialized_remote_table.size())),
                tt::tt_metal::distributed::multihost::Rank{bcast_root});
            tt_fabric::IntermeshLinkTable deserialized_remote_table =
                tt::tt_fabric::deserialize_from_bytes(serialized_remote_table);
            peer_intermesh_link_tables_[deserialized_remote_table.local_mesh_id]
                                       [deserialized_remote_table.local_host_rank_id] =
                                           std::move(deserialized_remote_table.intermesh_links);
        }
        // Barrier here for safety - Ensure that all ranks have completed the bcast op before proceeding to the next
        // root
        distributed_context.barrier();
    }
}

void ControlPlane::assign_direction_to_fabric_eth_core(
    const FabricNodeId& fabric_node_id, const CoreCoord& eth_core, RoutingDirection direction) {
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
            tt::LogFabric,
            "Control Plane: Disabling router on M{}D{} eth channel {}",
            fabric_node_id.mesh_id,
            fabric_node_id.chip_id,
            chan_id);
    }
}

void ControlPlane::assign_intermesh_link_directions_to_local_host(const FabricNodeId& fabric_node_id) {
    const auto& inter_mesh_connectivity = this->routing_table_generator_->mesh_graph->get_inter_mesh_connectivity();
    auto physical_chip_id = this->logical_mesh_chip_id_to_physical_chip_id_mapping_.at(fabric_node_id);
    const auto& connected_chips_and_eth_cores =
        tt::tt_metal::MetalContext::instance().get_cluster().get_ethernet_cores_grouped_by_connected_chips(
            physical_chip_id);

    for (const auto& [connected_mesh_id, edge] :
         inter_mesh_connectivity[*fabric_node_id.mesh_id][fabric_node_id.chip_id]) {
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
                this->assign_direction_to_fabric_eth_core(fabric_node_id, eth_core, edge.port_direction);
            }
        }
    }
}

void ControlPlane::assign_intermesh_link_directions_to_remote_host(const FabricNodeId& fabric_node_id) {
    const auto& inter_mesh_connectivity = this->routing_table_generator_->mesh_graph->get_inter_mesh_connectivity();
    auto physical_chip_id = this->logical_mesh_chip_id_to_physical_chip_id_mapping_.at(fabric_node_id);
    auto board_id = chip_id_to_asic_id_.at(physical_chip_id);
    auto intermesh_links = this->get_intermesh_eth_links(physical_chip_id);

    // Used to track the number of directions that could be assigned to intermesh links on this node
    uint32_t num_directions_assigned = 0;

    for (const auto& [eth_core, eth_chan] : intermesh_links) {
        auto intermesh_routing_direction = RoutingDirection::NONE;
        auto curr_eth_chan_desc = EthChanDescriptor{.board_id = board_id, .chan_id = eth_chan};
        const auto& remote_eth_chan_desc = intermesh_link_table_.intermesh_links.at(curr_eth_chan_desc);
        for (const auto& [connected_mesh_id, edge] :
             inter_mesh_connectivity[*fabric_node_id.mesh_id][fabric_node_id.chip_id]) {
            bool connection_found = false;
            // TODO: untested, but should work. We would need two big meshes connected to test this
            auto connected_host_rank_id = this->routing_table_generator_->mesh_graph
                                              ->get_host_rank_for_chip(connected_mesh_id, fabric_node_id.chip_id)
                                              .value();
            for (const auto& [candidate_desc, candidate_peer_desc] :
                 peer_intermesh_link_tables_[connected_mesh_id][connected_host_rank_id]) {
                if (candidate_desc == remote_eth_chan_desc && candidate_peer_desc == curr_eth_chan_desc) {
                    // Found the matching intermesh link
                    num_directions_assigned++;
                    intermesh_routing_direction = edge.port_direction;
                    connection_found = true;
                    break;
                }
            }
            if (connection_found) {
                break;  // No need to check other edges, we found the matching intermesh link
            }
        }
        if (intermesh_routing_direction != RoutingDirection::NONE) {
            auto& direction_to_channel_map = router_port_directions_to_physical_eth_chan_map_.at(fabric_node_id);
            direction_to_channel_map[intermesh_routing_direction].push_back(eth_chan);
        }
    }
    // Compute the number of intermesh links requsted by the user and ensure that they could be mapped to physical links
    // on the fabric node
    uint32_t num_links_requested_on_node = 0;
    for (const auto& [connected_mesh_id, edge] :
         inter_mesh_connectivity[*fabric_node_id.mesh_id][fabric_node_id.chip_id]) {
        num_links_requested_on_node += edge.connected_chip_ids.size();
    }
    TT_FATAL(
        num_directions_assigned == num_links_requested_on_node,
        "Could not bind all edges in the Mesh Graph to an intermesh link.");
}

const IntermeshLinkTable& ControlPlane::get_local_intermesh_link_table() const { return intermesh_link_table_; }

const MeshGraph& ControlPlane::get_mesh_graph() const { return *routing_table_generator_->mesh_graph; }

uint64_t ControlPlane::get_asic_id(chip_id_t chip_id) const { return chip_id_to_asic_id_.at(chip_id); }

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
    return this->routing_table_generator_->mesh_graph->get_coord_range(mesh_id, local_host_rank_id);
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

// Helper function to fill connection info with common fields for fabric router configs
void fill_connection_info_fields(
    tt::tt_fabric::fabric_connection_info_t& connection_info,
    const CoreCoord& virtual_core,
    const FabricEriscDatamoverConfig& config,
    uint32_t sender_channel,
    uint16_t worker_free_slots_stream_id) {
    connection_info.edm_noc_x = static_cast<uint8_t>(virtual_core.x);
    connection_info.edm_noc_y = static_cast<uint8_t>(virtual_core.y);
    connection_info.edm_buffer_base_addr = config.sender_channels_base_address[sender_channel];
    connection_info.num_buffers_per_channel = config.sender_channels_num_buffers[sender_channel];
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
    chip_id_t physical_chip_id,
    chan_id_t eth_channel_id,
    uint32_t sender_channel,
    uint32_t risc_id) {
    connection_info.edm_noc_x = static_cast<uint8_t>(mux_core_virtual.x);
    connection_info.edm_noc_y = static_cast<uint8_t>(mux_core_virtual.y);
    connection_info.edm_buffer_base_addr = tensix_config.get_channels_base_address(risc_id, sender_channel);
    connection_info.num_buffers_per_channel = tensix_config.get_num_buffers_per_channel();
    connection_info.buffer_size_bytes = tensix_config.get_buffer_size_bytes_full_size_channel();
    connection_info.edm_connection_handshake_addr =
        tensix_config.get_connection_semaphore_address(physical_chip_id, eth_channel_id, sender_channel);
    connection_info.edm_worker_location_info_addr =
        tensix_config.get_worker_conn_info_base_address(physical_chip_id, eth_channel_id, sender_channel);
    connection_info.buffer_index_semaphore_id =
        tensix_config.get_buffer_index_semaphore_address(physical_chip_id, eth_channel_id, sender_channel);
    connection_info.worker_free_slots_stream_id =
        tensix_config.get_channel_credits_stream_id(physical_chip_id, eth_channel_id, sender_channel);
}

void ControlPlane::populate_fabric_connection_info(
    tt::tt_fabric::fabric_connection_info_t& worker_connection_info,
    tt::tt_fabric::fabric_connection_info_t& dispatcher_connection_info,
    tt::tt_fabric::fabric_connection_info_t& tensix_connection_info,
    chip_id_t physical_chip_id,
    chan_id_t eth_channel_id,
    eth_chan_directions router_direction) const {
    constexpr uint16_t WORKER_FREE_SLOTS_STREAM_ID = 17;
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto& fabric_context = this->get_fabric_context();
    const bool is_2d_fabric = fabric_context.is_2D_routing_enabled();
    const auto sender_channel = is_2d_fabric ? router_direction : 0;

    const auto& fabric_tensix_config = tt::tt_metal::MetalContext::instance().get_fabric_tensix_config();
    // Always populate fabric router config for normal workers
    const auto& edm_config = fabric_context.get_fabric_router_config(
        tt::tt_fabric::FabricEriscDatamoverType::Default,
        tt::tt_fabric::FabricEriscDatamoverAxis::Short,
        fabric_tensix_config,
        static_cast<eth_chan_directions>(sender_channel));
    CoreCoord fabric_router_virtual_core = cluster.get_virtual_eth_core_from_channel(physical_chip_id, eth_channel_id);

    fill_connection_info_fields(
        worker_connection_info, fabric_router_virtual_core, edm_config, sender_channel, WORKER_FREE_SLOTS_STREAM_ID);

    // Check if fabric tensix config is enabled, if so populate different configs for dispatcher and tensix
    if (fabric_tensix_config != tt::tt_fabric::FabricTensixConfig::DISABLED) {
        // dispatcher uses different fabric router, which still has the default buffer size.
        const auto& default_edm_config = fabric_context.get_fabric_router_config();
        fill_connection_info_fields(
            dispatcher_connection_info,
            fabric_router_virtual_core,
            default_edm_config,
            sender_channel,
            WORKER_FREE_SLOTS_STREAM_ID);

        const auto& tensix_config = fabric_context.get_tensix_config();
        CoreCoord mux_core_logical = tensix_config.get_core_for_channel(physical_chip_id, eth_channel_id);
        CoreCoord mux_core_virtual = cluster.get_virtual_coordinate_from_logical_coordinates(
            physical_chip_id, mux_core_logical, CoreType::WORKER);
        // Get the RISC ID that handles this ethernet channel
        auto risc_id = tensix_config.get_risc_id_for_channel(physical_chip_id, eth_channel_id);

        fill_tensix_connection_info_fields(
            tensix_connection_info,
            mux_core_virtual,
            tensix_config,
            physical_chip_id,
            eth_channel_id,
            sender_channel,
            risc_id);
    } else {
        dispatcher_connection_info = worker_connection_info;
    }
}

void ControlPlane::collect_and_merge_router_port_directions_from_all_hosts() {
    const auto& distributed_context = tt::tt_metal::MetalContext::instance().global_distributed_context();
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
                tt::tt_metal::distributed::multihost::Rank{bcast_root});
            serialized_remote_data.clear();
            serialized_remote_data.resize(remote_data_size_bytes);
            distributed_context.broadcast(
                tt::stl::as_writable_bytes(
                    tt::stl::Span<uint8_t>(serialized_remote_data.data(), serialized_remote_data.size())),
                tt::tt_metal::distributed::multihost::Rank{bcast_root});

            RouterPortDirectionsData deserialized_remote_data =
                tt::tt_fabric::deserialize_router_port_directions_from_bytes(serialized_remote_data);

            // Merge remote data into local router_port_directions_to_physical_eth_chan_map_
            for (const auto& [fabric_node_id, direction_map] : deserialized_remote_data.router_port_directions_map) {
                // Only merge if this fabric node is not already in our local map
                if (router_port_directions_to_physical_eth_chan_map_.find(fabric_node_id) ==
                    router_port_directions_to_physical_eth_chan_map_.end()) {
                    router_port_directions_to_physical_eth_chan_map_[fabric_node_id] = direction_map;
                } else {
                    // If fabric node exists, merge direction maps
                    for (const auto& [direction, channels] : direction_map) {
                        auto& local_direction_map = router_port_directions_to_physical_eth_chan_map_[fabric_node_id];
                        if (local_direction_map.find(direction) == local_direction_map.end()) {
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

ControlPlane::~ControlPlane() = default;

}  // namespace tt::tt_fabric
