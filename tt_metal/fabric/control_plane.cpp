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
#include "fabric_types.hpp"
#include "hal_types.hpp"
#include "host_api.hpp"
#include "intermesh_constants.hpp"
#include "impl/context/metal_context.hpp"
#include "tt_metal/common/env_lib.hpp"
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
#include "tt_metal/fabric/serialization/intermesh_link_table.hpp"

namespace tt::tt_fabric {

namespace {

// Helper to extract intermesh ports from config value
std::vector<chan_id_t> extract_intermesh_eth_links(uint32_t config_value, chip_id_t chip_id) {
    std::vector<chan_id_t> intermesh_eth_links;
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto& soc_desc = cluster.get_soc_desc(chip_id);
    uint32_t intermesh_eth_links_bits = (config_value >> intermesh_constants::INTERMESH_ETH_LINK_BITS_SHIFT) &
                                        intermesh_constants::INTERMESH_ETH_LINK_BITS_MASK;
    for (chan_id_t link = 0; link < static_cast<chan_id_t>(soc_desc.get_num_eth_channels()); ++link) {
        if (intermesh_eth_links_bits & (1 << link)) {
            intermesh_eth_links.push_back(link);
        }
    }
    return intermesh_eth_links;
}

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
    std::vector<chip_id_t> adjacent_chips;

    for (const auto& [connected_chip_id, eth_ports] : eth_links) {
        // Do not include any corner to corner links on UBB
        if (is_ubb && cluster.is_external_cable(chip_id, eth_ports[0])) {
            continue;
        }
        if (eth_ports.size() >= num_ports_per_side) {
            adjacent_chips.push_back(connected_chip_id);
        }
    }

    return adjacent_chips;
}

}  // namespace

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

    auto get_chip_coord = [this](FabricNodeId fabric_node_id) -> MeshCoordinate {
        auto mesh_id = fabric_node_id.mesh_id;
        auto chip_id = fabric_node_id.chip_id;

        // Get mesh dimensions from the mesh graph descriptor
        auto mesh_ew_size = this->routing_table_generator_->mesh_graph->get_mesh_shape(mesh_id)[1];

        // Convert linear chip_id to 2D mesh coordinates
        auto coord_y = chip_id / mesh_ew_size;
        auto coord_x = chip_id % mesh_ew_size;

        return MeshCoordinate(coord_x, coord_y);
    };

    const auto user_meshes = this->get_user_physical_mesh_ids();
    if (reliability_mode == tt::tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE) {
        for (auto mesh_id : user_meshes) {
            size_t num_chips_in_mesh = intra_mesh_connectivity[mesh_id.get()].size();
            for (std::uint32_t chip_id = 0; chip_id < num_chips_in_mesh; chip_id++) {
                const auto fabric_node_id = FabricNodeId(MeshId{mesh_id}, chip_id);
                for (const auto& [direction, eth_chans] :
                     this->router_port_directions_to_physical_eth_chan_map_.at(fabric_node_id)) {
                    this->router_port_directions_to_num_routing_planes_map_[fabric_node_id][direction] =
                        eth_chans.size();
                }
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
            for (std::uint32_t chip_id = 0; chip_id < num_chips_in_mesh; chip_id++) {
                const auto fabric_node_id = FabricNodeId(MeshId{mesh_id}, chip_id);
                const auto chip_coord = get_chip_coord(fabric_node_id);
                auto chip_coord_x = chip_coord[0];
                auto chip_coord_y = chip_coord[1];

                const auto& port_directions = this->router_port_directions_to_physical_eth_chan_map_.at(fabric_node_id);

                const auto& golden_counts = golden_link_counts.at(MeshId{mesh_id}).at(chip_id);
                apply_min(
                    fabric_node_id,
                    port_directions,
                    RoutingDirection::E,
                    golden_counts,
                    row_min_planes.at(chip_coord_y));
                apply_min(
                    fabric_node_id,
                    port_directions,
                    RoutingDirection::W,
                    golden_counts,
                    row_min_planes.at(chip_coord_y));
                apply_min(
                    fabric_node_id,
                    port_directions,
                    RoutingDirection::N,
                    golden_counts,
                    col_min_planes.at(chip_coord_x));
                apply_min(
                    fabric_node_id,
                    port_directions,
                    RoutingDirection::S,
                    golden_counts,
                    col_min_planes.at(chip_coord_x));
            }

            // TODO: specialize by topology for better perf
            if (topology == Topology::Mesh || topology == Topology::Torus) {
                const auto rows_min = std::min_element(row_min_planes.begin(), row_min_planes.end());
                const auto cols_min = std::min_element(col_min_planes.begin(), col_min_planes.end());
                const auto mesh_min = std::min(*rows_min, *cols_min);
                std::fill(row_min_planes.begin(), row_min_planes.end(), mesh_min);
                std::fill(col_min_planes.begin(), col_min_planes.end(), mesh_min);
            }

            // Second pass: Apply minimums to each device
            for (std::uint32_t chip_id = 0; chip_id < num_chips_in_mesh; chip_id++) {
                const auto fabric_node_id = FabricNodeId(MeshId{mesh_id}, chip_id);
                const auto chip_coord = get_chip_coord(fabric_node_id);
                auto chip_coord_x = chip_coord[0];
                auto chip_coord_y = chip_coord[1];

                apply_count(fabric_node_id, RoutingDirection::E, row_min_planes.at(chip_coord_y));
                apply_count(fabric_node_id, RoutingDirection::W, row_min_planes.at(chip_coord_y));
                apply_count(fabric_node_id, RoutingDirection::N, col_min_planes.at(chip_coord_x));
                apply_count(fabric_node_id, RoutingDirection::S, col_min_planes.at(chip_coord_x));
            }
        }
    }
}

LocalMeshBinding ControlPlane::initialize_local_mesh_binding() {
    const char* mesh_id_str = std::getenv("TT_MESH_ID");
    const char* host_rank_str = std::getenv("TT_HOST_RANK");
    if (mesh_id_str == nullptr ^ host_rank_str == nullptr) {
        TT_THROW("Both TT_MESH_ID and TT_HOST_RANK environment variables must be set together or both unset");
    }

    // If both TT_MESH_ID and TT_HOST_RANK are unset, we don't initialzie the local mesh binding.
    // A nullopt here indicates that the host this ControlPlane is runnning on owns all Meshes in
    // the MeshGraphDescriptor. Single Host Multi-Mesh is only used for testing purposes.
    if (mesh_id_str == nullptr && host_rank_str == nullptr) {
        auto& ctx = tt::tt_metal::MetalContext::instance().get_distributed_context();
        auto mpi_rank = *ctx.rank();
        std::vector<MeshId> local_mesh_ids;
        for (const auto& mesh_id : this->routing_table_generator_->mesh_graph->get_mesh_ids()) {
            const auto& host_ranks = this->routing_table_generator_->mesh_graph->get_host_ranks(mesh_id);
            for (const auto& [coord, rank] : host_ranks) {
                if (mpi_rank == *rank) {
                    local_mesh_ids.push_back(mesh_id);
                }
            }
        }
        TT_FATAL(local_mesh_ids.size() > 0, "No local meshes found for host rank {}", mpi_rank);
        return LocalMeshBinding{.mesh_ids = std::move(local_mesh_ids), .host_rank = HostRankId{mpi_rank}};
    }

    // If both TT_MESH_ID and TT_HOST_RANK are set, we'll use the values from the environment variables.
    auto local_mesh_binding = LocalMeshBinding{
        .mesh_ids = {MeshId{std::stoi(mesh_id_str)}}, .host_rank = HostRankId{std::stoi(host_rank_str)}};

    log_debug(
        tt::LogDistributed,
        "Local mesh binding: mesh_id: {}, host_rank: {}",
        local_mesh_binding.mesh_ids[0],
        local_mesh_binding.host_rank);

    // Validate the local mesh binding exists in the mesh graph descriptor
    auto mesh_ids = this->routing_table_generator_->mesh_graph->get_mesh_ids();
    if (std::find(mesh_ids.begin(), mesh_ids.end(), local_mesh_binding.mesh_ids[0]) == mesh_ids.end()) {
        TT_THROW(
            "Invalid TT_MESH_ID: Local mesh binding mesh_id {} not found in mesh graph descriptor",
            local_mesh_binding.mesh_ids[0]);
    }

    // Validate host rank (only if mesh_id is valid)
    const auto& host_ranks = this->routing_table_generator_->mesh_graph->get_host_ranks(local_mesh_binding.mesh_ids[0]);
    bool is_valid_host_rank = std::find_if(host_ranks.begin(), host_ranks.end(), [&](const auto& coord_rank_pair) {
                                  return coord_rank_pair.value() == local_mesh_binding.host_rank;
                              }) != host_ranks.end();

    TT_FATAL(
        is_valid_host_rank,
        "Invalid TT_HOST_RANK: Local mesh binding host_rank {} not found in mesh graph descriptor",
        local_mesh_binding.host_rank);

    return local_mesh_binding;
}

ControlPlane::ControlPlane(const std::string& mesh_graph_desc_file) {
    this->routing_table_generator_ = std::make_unique<RoutingTableGenerator>(mesh_graph_desc_file);
    this->local_mesh_binding_ = this->initialize_local_mesh_binding();
    // Printing, only enabled with log_debug
    this->routing_table_generator_->mesh_graph->print_connectivity();
    // Printing, only enabled with log_debug
    this->routing_table_generator_->print_routing_tables();

    // Initialize the control plane routers based on mesh graph
    const auto& logical_mesh_chip_id_to_physical_chip_id_mapping =
        this->get_logical_chip_to_physical_chip_mapping(mesh_graph_desc_file);
    this->load_physical_chip_mapping(logical_mesh_chip_id_to_physical_chip_id_mapping);
    // Query and generate intermesh ethernet links per physical chip
    this->initialize_intermesh_eth_links();
    this->generate_local_intermesh_link_table();
}

ControlPlane::ControlPlane(
    const std::string& mesh_graph_desc_file,
    const std::map<FabricNodeId, chip_id_t>& logical_mesh_chip_id_to_physical_chip_id_mapping) {
    this->routing_table_generator_ = std::make_unique<RoutingTableGenerator>(mesh_graph_desc_file);
    this->local_mesh_binding_ = this->initialize_local_mesh_binding();
    // Printing, only enabled with log_debug
    this->routing_table_generator_->mesh_graph->print_connectivity();
    // Printing, only enabled with log_debug
    this->routing_table_generator_->print_routing_tables();

    // Initialize the control plane routers based on mesh graph
    this->load_physical_chip_mapping(logical_mesh_chip_id_to_physical_chip_id_mapping);
    // Query and generate intermesh ethernet links per physical chip
    this->initialize_intermesh_eth_links();
    this->generate_local_intermesh_link_table();
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
        if (this->is_local_mesh(mesh_id)) {
            this->validate_mesh_connections(mesh_id);
        }
    }
}

std::vector<chip_id_t> ControlPlane::get_mesh_physical_chip_ids(
    const tt::tt_metal::distributed::MeshContainer<chip_id_t>& mesh_container,
    std::optional<chip_id_t> nw_corner_chip_id) const {
    // Convert the coordinate range to a set of chip IDs using MeshContainer iterator
    std::set<chip_id_t> user_chip_ids;
    for (const auto& [coord, chip_id] : mesh_container) {
        user_chip_ids.insert(chip_id);
    }

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
        auto mesh_shape = routing_table_generator_->mesh_graph->get_mesh_shape(MeshId{4});
        // Main board
        const auto& mesh_container = this->routing_table_generator_->mesh_graph->get_chip_ids(MeshId{4});
        const auto& physical_chip_ids = this->get_mesh_physical_chip_ids(mesh_container, nw_chip_physical_id);
        for (std::uint32_t i = 0; i < physical_chip_ids.size(); i++) {
            logical_mesh_chip_id_to_physical_chip_id_mapping.insert({FabricNodeId(MeshId{4}, i), physical_chip_ids[i]});
        }
        // This case can be depreciated once we have multi-host testing and validate it working
    } else if (mesh_graph_desc_filename == "t3k_dual_host_mesh_graph_descriptor.yaml") {
        // TODO(#24230): This path will soon be deprecated once we generalize logical mesh_chip_id to physical chip_id
        // mapping
        auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
        auto chip_eth_coords = cluster.get_user_chip_ethernet_coordinates();
        std::vector<eth_coord_t> eth_coords;
        eth_coords.reserve(chip_eth_coords.size());
        for (const auto& [_, eth_coord] : chip_eth_coords) {
            eth_coords.push_back(eth_coord);
        }
        std::sort(eth_coords.begin(), eth_coords.end(), EthCoordComparator());

        auto mesh_ids = this->get_local_mesh_id_bindings();
        auto mesh_id = mesh_ids.at(0);  // Use the first mesh ID
        auto host_rank_id = this->get_local_host_rank_id_binding();
        auto fabric_chip_ids = this->routing_table_generator_->mesh_graph->get_chip_ids(mesh_id, host_rank_id).values();

        TT_FATAL(
            fabric_chip_ids.size() == eth_coords.size(),
            "Number of fabric chip ids {} does not match number of eth coords {}",
            fabric_chip_ids.size(),
            eth_coords.size());
        for (std::uint32_t idx = 0; idx < fabric_chip_ids.size(); idx++) {
            auto fabric_chip_id = fabric_chip_ids.at(idx);
            auto eth_coord = eth_coords.at(idx);
            logical_mesh_chip_id_to_physical_chip_id_mapping.insert(
                {tt_fabric::FabricNodeId(mesh_id, fabric_chip_id),
                 cluster.get_physical_chip_id_from_eth_coord(eth_coord)});
        }
    } else {
        // Iterate over every mesh defined in the mesh-graph descriptor and embed it on top of
        // the physical cluster using the generic helper.
        for (const auto& mesh_id : this->routing_table_generator_->mesh_graph->get_mesh_ids()) {
            const auto& mesh_container = this->routing_table_generator_->mesh_graph->get_chip_ids(mesh_id);
            const auto& physical_chip_ids = this->get_mesh_physical_chip_ids(mesh_container);

            for (std::uint32_t i = 0; i < physical_chip_ids.size(); ++i) {
                logical_mesh_chip_id_to_physical_chip_id_mapping.emplace(
                    FabricNodeId(mesh_id, i), physical_chip_ids[i]);
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
    if (tt::tt_metal::MetalContext::instance().get_cluster().get_cluster_type() == tt::ClusterType::TG) {
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

    const auto& router_intra_mesh_routing_table = this->routing_table_generator_->get_intra_mesh_table();
    for (std::uint32_t mesh_id = 0; mesh_id < router_intra_mesh_routing_table.size(); mesh_id++) {
        if (!this->is_local_mesh(MeshId{mesh_id})) {
            continue;
        }
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
        if (!this->is_local_mesh(MeshId{src_mesh_id})) {
            continue;
        }
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
                    TT_FATAL(
                        num_available_routing_planes <= 4,
                        "Expected at most 4 routing planes for M{}D{} in direction {}",
                        fabric_node_id.mesh_id,
                        fabric_node_id.chip_id,
                        direction);
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

    for (std::uint32_t mesh_id = 0; mesh_id < intra_mesh_connectivity.size(); mesh_id++) {
        if (!this->is_local_mesh(MeshId{mesh_id})) {
            continue;
        }
        for (std::uint32_t chip_id = 0; chip_id < intra_mesh_connectivity[mesh_id].size(); chip_id++) {
            const auto fabric_node_id = FabricNodeId(MeshId{mesh_id}, chip_id);
            auto physical_chip_id = this->logical_mesh_chip_id_to_physical_chip_id_mapping_.at(fabric_node_id);
            const auto& connected_chips_and_eth_cores =
                tt::tt_metal::MetalContext::instance().get_cluster().get_ethernet_cores_grouped_by_connected_chips(
                    physical_chip_id);
            for (const auto& [logical_connected_chip_id, edge] : intra_mesh_connectivity[mesh_id][chip_id]) {
                const auto& physical_connected_chip_id = this->logical_mesh_chip_id_to_physical_chip_id_mapping_.at(
                    FabricNodeId(MeshId{mesh_id}, logical_connected_chip_id));

                bool connections_exist = connected_chips_and_eth_cores.find(physical_connected_chip_id) !=
                                         connected_chips_and_eth_cores.end();
                TT_FATAL(
                    connections_exist ||
                        reliability_mode != tt::tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE,
                    "Expected connections to exist for M{}D{} to D{}",
                    mesh_id,
                    chip_id,
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
            }
        }
    }

    const auto& distributed_context = tt::tt_metal::MetalContext::instance().get_distributed_context();
    for (std::uint32_t mesh_id = 0; mesh_id < inter_mesh_connectivity.size(); mesh_id++) {
        for (std::uint32_t chip_id = 0; chip_id < inter_mesh_connectivity[mesh_id].size(); chip_id++) {
            if (this->is_local_mesh(MeshId{mesh_id})) {
                const auto fabric_node_id = FabricNodeId(MeshId{mesh_id}, chip_id);
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

    this->convert_fabric_routing_table_to_chip_routing_table();
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

std::vector<std::pair<FabricNodeId, chan_id_t>> ControlPlane::get_fabric_route(
    FabricNodeId src_fabric_node_id, FabricNodeId dst_fabric_node_id, chan_id_t src_chan_id) const {
    std::vector<std::pair<FabricNodeId, chan_id_t>> route;
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

static void write_to_all_tensix_cores(
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
    tensix_routing_info.mesh_id = *mesh_id;
    tensix_routing_info.device_id = chip_id;

    // Build intra-mesh routing entries (chip-to-chip routing)
    std::fill_n(
        tensix_routing_info.intra_mesh_routing_table,
        tt::tt_fabric::MAX_MESH_SIZE,
        (eth_chan_directions)eth_chan_magic_values::INVALID_ROUTING_TABLE_ENTRY);
    const auto& router_intra_mesh_routing_table = this->routing_table_generator_->get_intra_mesh_table();
    TT_FATAL(
        router_intra_mesh_routing_table[*mesh_id][chip_id].size() <= tt::tt_fabric::MAX_MESH_SIZE,
        "ControlPlane: Intra mesh routing table size exceeds maximum allowed size");
    for (chip_id_t dst_chip_id = 0; dst_chip_id < router_intra_mesh_routing_table[*mesh_id][chip_id].size();
         dst_chip_id++) {
        if (chip_id == dst_chip_id) {
            tensix_routing_info.intra_mesh_routing_table[dst_chip_id] =
                (eth_chan_directions)eth_chan_magic_values::INVALID_DIRECTION;
            continue;
        }
        auto forwarding_direction = router_intra_mesh_routing_table[*mesh_id][chip_id][dst_chip_id];
        tensix_routing_info.intra_mesh_routing_table[dst_chip_id] =
            forwarding_direction != RoutingDirection::NONE
                ? this->routing_direction_to_eth_direction(forwarding_direction)
                : (eth_chan_directions)eth_chan_magic_values::INVALID_DIRECTION;
    }

    // Build inter-mesh routing entries (mesh-to-mesh routing)
    std::fill_n(
        tensix_routing_info.inter_mesh_routing_table,
        tt::tt_fabric::MAX_NUM_MESHES,
        (eth_chan_directions)eth_chan_magic_values::INVALID_ROUTING_TABLE_ENTRY);
    const auto& router_inter_mesh_routing_table = this->routing_table_generator_->get_inter_mesh_table();
    TT_FATAL(
        router_inter_mesh_routing_table[*mesh_id][chip_id].size() <= tt::tt_fabric::MAX_NUM_MESHES,
        "ControlPlane: Inter mesh routing table size exceeds maximum allowed size");
    for (std::uint32_t dst_mesh_id = 0; dst_mesh_id < router_inter_mesh_routing_table[*mesh_id][chip_id].size();
         dst_mesh_id++) {
        if (*mesh_id == dst_mesh_id) {
            tensix_routing_info.inter_mesh_routing_table[dst_mesh_id] =
                (eth_chan_directions)eth_chan_magic_values::INVALID_DIRECTION;
            continue;
        }
        auto forwarding_direction = router_inter_mesh_routing_table[*mesh_id][chip_id][dst_mesh_id];
        tensix_routing_info.inter_mesh_routing_table[dst_mesh_id] =
            forwarding_direction != RoutingDirection::NONE
                ? this->routing_direction_to_eth_direction(forwarding_direction)
                : (eth_chan_directions)eth_chan_magic_values::INVALID_DIRECTION;
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

    const auto& fabric_context = this->get_fabric_context();
    const auto& edm_config = fabric_context.get_fabric_router_config();
    const auto topology = fabric_context.get_fabric_topology();
    const bool is_2d_fabric = topology == Topology::Mesh;

    tt::tt_fabric::tensix_fabric_connections_l1_info_t fabric_connections = {};

    // Get all physically connected ethernet channels directly from the cluster
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto& soc_desc = cluster.get_soc_desc(physical_chip_id);
    const auto& connected_chips_and_eth_cores = cluster.get_ethernet_cores_grouped_by_connected_chips(physical_chip_id);

    size_t num_eth_endpoint = 0;
    for (const auto& [connected_chip_id, eth_cores] : connected_chips_and_eth_cores) {
        // iterate all physically connected ethernet cores
        for (const auto& eth_core : eth_cores) {
            auto eth_channel_id = soc_desc.logical_eth_core_to_chan_map.at(eth_core);
            bool is_fabric_connected = false;
            eth_chan_directions router_direction = eth_chan_directions::COUNT;
            for (const auto& [direction, eth_chans] :
                 this->router_port_directions_to_physical_eth_chan_map_.at(src_fabric_node_id)) {
                // Check if the physically connected channel is part of fabric channel
                if (std::find(eth_chans.begin(), eth_chans.end(), eth_channel_id) != eth_chans.end()) {
                    is_fabric_connected = true;
                    router_direction = this->routing_direction_to_eth_direction(direction);
                    break;
                }
            }
            if (!is_fabric_connected) {
                continue;
            }
            if (num_eth_endpoint >= tt::tt_fabric::tensix_fabric_connections_l1_info_t::MAX_FABRIC_ENDPOINTS) {
                log_warning(
                    tt::LogFabric,
                    "ControlPlane: Maximum number of fabric endpoints exceeded for M%dD%d, skipping further "
                    "connections",
                    *mesh_id,
                    chip_id);
                break;
            }

            CoreCoord fabric_router_virtual_core =
                cluster.get_virtual_eth_core_from_channel(physical_chip_id, eth_channel_id);

            // Populate connection info for fabric-routed channels
            const auto sender_channel = is_2d_fabric ? router_direction : 0;
            auto& connection_info = fabric_connections.connections[eth_channel_id];
            connection_info.edm_direction = router_direction;
            connection_info.edm_noc_xy =
                tt::tt_fabric::WorkerXY(fabric_router_virtual_core.x, fabric_router_virtual_core.y).to_uint32();
            connection_info.edm_buffer_base_addr = edm_config.sender_channels_base_address[sender_channel];
            connection_info.num_buffers_per_channel = edm_config.sender_channels_num_buffers[sender_channel];
            connection_info.edm_l1_sem_addr =
                edm_config.sender_channels_local_flow_control_semaphore_address[sender_channel];
            connection_info.edm_connection_handshake_addr =
                edm_config.sender_channels_connection_semaphore_address[sender_channel];
            connection_info.edm_worker_location_info_addr =
                edm_config.sender_channels_worker_conn_info_base_address[sender_channel];
            connection_info.buffer_size_bytes = edm_config.channel_buffer_size_bytes;
            connection_info.buffer_index_semaphore_id =
                edm_config.sender_channels_buffer_index_semaphore_address[sender_channel];

            // Mark this connection as valid for fabric communication
            fabric_connections.valid_connections_mask |= (1u << eth_channel_id);
            num_eth_endpoint++;
        }
    }

    write_to_all_tensix_cores(
        &fabric_connections,
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

void ControlPlane::write_routing_tables_to_all_chips() const {
    // Configure the routing tables on the chips
    TT_ASSERT(
        this->intra_mesh_routing_tables_.size() == this->inter_mesh_routing_tables_.size(),
        "Intra mesh routing tables size mismatch with inter mesh routing tables");
    for (const auto& [fabric_node_id, _] : this->intra_mesh_routing_tables_) {
        TT_ASSERT(
            this->inter_mesh_routing_tables_.contains(fabric_node_id),
            "Intra mesh routing tables keys mismatch with inter mesh routing tables");
        this->write_routing_tables_to_tensix_cores(fabric_node_id.mesh_id, fabric_node_id.chip_id);
        this->write_fabric_connections_to_tensix_cores(fabric_node_id.mesh_id, fabric_node_id.chip_id);
        this->write_routing_tables_to_eth_cores(fabric_node_id.mesh_id, fabric_node_id.chip_id);
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
    std::optional<HostRankId> local_host_rank_id =
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

void ControlPlane::initialize_fabric_context(tt_fabric::FabricConfig fabric_config) {
    TT_FATAL(this->fabric_context_ == nullptr, "Trying to re-initialize fabric context");
    this->fabric_context_ = std::make_unique<FabricContext>(fabric_config);
}

FabricContext& ControlPlane::get_fabric_context() const {
    TT_FATAL(this->fabric_context_ != nullptr, "Trying to get un-initialized fabric context");
    return *this->fabric_context_.get();
}

void ControlPlane::clear_fabric_context() { this->fabric_context_.reset(nullptr); }

void ControlPlane::initialize_intermesh_eth_links() {
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    // If intermesh links are not enabled, set all intermesh_eth_links_ to empty
    if (not this->is_intermesh_enabled()) {
        for (const auto& chip_id : cluster.all_chip_ids()) {
            intermesh_eth_links_[chip_id] = {};
        }
        return;
    }

    // Iterate over all chips in the cluster and populate the intermesh_eth_links
    for (const auto& chip_id : cluster.all_chip_ids()) {
        const auto& soc_desc = cluster.get_soc_desc(chip_id);
        if (soc_desc.logical_eth_core_to_chan_map.empty()) {
            intermesh_eth_links_[chip_id] = {};
            continue;
        }
        // Remote connections not visible to UMD
        // Read multi-mesh configuration from the first available eth core
        auto first_eth_core = soc_desc.logical_eth_core_to_chan_map.begin()->first;
        tt_cxy_pair virtual_eth_core(
            chip_id, cluster.get_virtual_coordinate_from_logical_coordinates(chip_id, first_eth_core, CoreType::ETH));

        std::vector<uint32_t> config_data(1, 0);
        auto multi_mesh_config_addr = tt_metal::MetalContext::instance().hal().get_dev_addr(
            tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt_metal::HalL1MemAddrType::INTERMESH_ETH_LINK_CONFIG);
        cluster.read_core(config_data, sizeof(uint32_t), virtual_eth_core, multi_mesh_config_addr);
        std::vector<std::pair<CoreCoord, chan_id_t>> intermesh_eth_links;
        for (auto link : extract_intermesh_eth_links(config_data[0], chip_id)) {
            // Find the CoreCoord for this channel
            for (const auto& [core_coord, channel] : soc_desc.logical_eth_core_to_chan_map) {
                if (channel == link) {
                    intermesh_eth_links.push_back({core_coord, link});
                    break;
                }
            }
        }

        // Remote connections visible to UMD
        auto remote_connections = cluster.get_ethernet_connections_to_remote_devices().find(chip_id);
        if (remote_connections != cluster.get_ethernet_connections_to_remote_devices().end()) {
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

        intermesh_eth_links_[chip_id] = intermesh_eth_links;
    }
}

bool ControlPlane::is_intermesh_enabled() const {
    // Check if the architecture and system support intermesh routing
    if (not tt_metal::MetalContext::instance().hal().intermesh_eth_links_enabled()) {
        return false;
    }

    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    auto first_chip_id = *(cluster.all_pci_chip_ids().begin());

    // Check if there are any ethernet cores available on the first chip
    const auto& soc_desc = cluster.get_soc_desc(first_chip_id);
    if (soc_desc.logical_eth_core_to_chan_map.empty()) {
        return false;
    }

    // UMD Visible Intermesh Links
    if (!cluster.get_ethernet_connections_to_remote_devices().empty()) {
        return true;
    }

    // UMD Hidden Intermesh Links
    std::vector<uint32_t> config_data(1, 0);
    auto first_eth_core = soc_desc.logical_eth_core_to_chan_map.begin()->first;
    tt_cxy_pair virtual_eth_core(
        first_chip_id,
        cluster.get_virtual_coordinate_from_logical_coordinates(first_chip_id, first_eth_core, CoreType::ETH));
    auto multi_mesh_config_addr = tt_metal::MetalContext::instance().hal().get_dev_addr(
        tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt_metal::HalL1MemAddrType::INTERMESH_ETH_LINK_CONFIG);
    cluster.read_core(config_data, sizeof(uint32_t), virtual_eth_core, multi_mesh_config_addr);
    bool intermesh_enabled =
        (config_data[0] & intermesh_constants::MULTI_MESH_MODE_MASK) == intermesh_constants::MULTI_MESH_ENABLED_VALUE;
    return intermesh_enabled;
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

// TODO: Support Intramesh links through this API as well
bool ControlPlane::is_intermesh_eth_link_trained(chip_id_t chip_id, CoreCoord eth_core) const {
    TT_FATAL(
        this->is_intermesh_eth_link(chip_id, eth_core), "Can only call {} on intermesh ethernet links.", __FUNCTION__);
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    // Read the link status from designated L1 address
    tt_cxy_pair virtual_eth_core(
        chip_id, cluster.get_virtual_coordinate_from_logical_coordinates(chip_id, eth_core, CoreType::ETH));
    std::vector<uint32_t> status_data(1, 0);
    auto multi_mesh_link_status_addr = tt_metal::MetalContext::instance().hal().get_dev_addr(
        tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt_metal::HalL1MemAddrType::INTERMESH_ETH_LINK_STATUS);
    cluster.read_core(status_data, sizeof(uint32_t), virtual_eth_core, multi_mesh_link_status_addr);

    // Check if the link is trained
    return (status_data[0] & intermesh_constants::LINK_CONNECTED_MASK) == intermesh_constants::LINK_CONNECTED_MASK;
}

const std::vector<std::pair<CoreCoord, chan_id_t>>& ControlPlane::get_intermesh_eth_links(chip_id_t chip_id) const {
    return this->intermesh_eth_links_.at(chip_id);
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

        if (cluster.get_board_type(chip_id) != BoardType::UBB) {
            // For Non-UBB Wormhole systems, intermesh links must also be marked as active ethernet cores
            // These cores are not seen by UMD or the cluster descriptor as active. Control Plane is
            // responsible for querying this information.
            // Note: On UBB systems, intermesh links are already identified as active by UMD, so control
            // plane does not need to do this.
            auto intermesh_links = this->get_intermesh_eth_links(chip_id);
            for (const auto& [eth_coord, eth_chan] : intermesh_links) {
                active_ethernet_cores.insert(eth_coord);
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
    const auto& distributed_context = tt::tt_metal::MetalContext::instance().get_distributed_context();
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    intermesh_link_table_.local_mesh_id = local_mesh_binding_.mesh_ids[0];
    const uint32_t remote_config_base_addr = tt_metal::MetalContext::instance().hal().get_dev_addr(
        tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt_metal::HalL1MemAddrType::ETH_LINK_REMOTE_INFO);
    for (const auto& chip_id : cluster.user_exposed_chip_ids()) {
        if (this->has_intermesh_links(chip_id)) {
            for (const auto& [eth_core, chan_id] : this->get_intermesh_eth_links(chip_id)) {
                if (not this->is_intermesh_eth_link_trained(chip_id, eth_core)) {
                    // Link is untrained/unusuable
                    continue;
                }
                tt_cxy_pair virtual_eth_core(
                    chip_id, cluster.get_virtual_coordinate_from_logical_coordinates(chip_id, eth_core, CoreType::ETH));
                uint64_t local_board_id = 0;
                uint64_t remote_board_id = 0;
                uint32_t remote_chan_id = 0;
                cluster.read_core(
                    &local_board_id,
                    sizeof(uint64_t),
                    virtual_eth_core,
                    remote_config_base_addr + intermesh_constants::LOCAL_BOARD_ID_OFFSET);
                cluster.read_core(
                    &remote_board_id,
                    sizeof(uint64_t),
                    virtual_eth_core,
                    remote_config_base_addr + intermesh_constants::REMOTE_BOARD_ID_OFFSET);
                cluster.read_core(
                    &remote_chan_id,
                    sizeof(uint32_t),
                    virtual_eth_core,
                    remote_config_base_addr + intermesh_constants::REMOTE_ETH_CHAN_ID_OFFSET);
                auto local_eth_chan_desc = EthChanDescriptor{
                    .board_id = local_board_id,
                    .chan_id = chan_id,
                };
                auto remote_eth_chan_desc = EthChanDescriptor{
                    .board_id = remote_board_id,
                    .chan_id = remote_chan_id,
                };
                intermesh_link_table_.intermesh_links[local_eth_chan_desc] = remote_eth_chan_desc;
                chip_id_to_asic_id_[chip_id] = local_board_id;
            }
        } else if (cluster.arch() != ARCH::BLACKHOLE) {
            // For chips without intermesh links, we still need to populate the asic IDs
            // for consistency.
            // Skip this on Blackhole for now.
            if (this->get_active_ethernet_cores(chip_id).size() == 0) {
                // No Active Ethernet Cores found. Not querying the board id off ethernet cores.
                chip_id_to_asic_id_[chip_id] = chip_id;
            } else {
                auto first_eth_core = *(this->get_active_ethernet_cores(chip_id).begin());
                tt_cxy_pair virtual_eth_core(
                    chip_id,
                    cluster.get_virtual_coordinate_from_logical_coordinates(chip_id, first_eth_core, CoreType::ETH));
                uint64_t local_board_id = 0;
                cluster.read_core(
                    &local_board_id,
                    sizeof(uint64_t),
                    virtual_eth_core,
                    remote_config_base_addr + intermesh_constants::LOCAL_BOARD_ID_OFFSET);
                chip_id_to_asic_id_[chip_id] = local_board_id;
            }
        }
    }
}

void ControlPlane::exchange_intermesh_link_tables() {
    const auto& distributed_context = tt::tt_metal::MetalContext::instance().get_distributed_context();
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
            peer_intermesh_link_tables_[deserialized_remote_table.local_mesh_id] =
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
            for (const auto& [candidate_desc, candidate_peer_desc] : peer_intermesh_link_tables_[connected_mesh_id]) {
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
        router_port_directions_to_physical_eth_chan_map_.at(fabric_node_id)[intermesh_routing_direction].push_back(
            eth_chan);
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

HostRankId ControlPlane::get_local_host_rank_id_binding() const { return this->local_mesh_binding_.host_rank; }

MeshCoordinate ControlPlane::get_local_mesh_offset() const {
    auto coord_range = this->get_coord_range(this->get_local_mesh_id_bindings()[0], MeshScope::LOCAL);
    return coord_range.start_coord();
}

MeshCoordinateRange ControlPlane::get_coord_range(MeshId mesh_id, MeshScope scope) const {
    std::optional<HostRankId> local_host_rank_id =
        MeshScope::LOCAL == scope ? std::make_optional(this->get_local_host_rank_id_binding()) : std::nullopt;
    return this->routing_table_generator_->mesh_graph->get_coord_range(mesh_id, local_host_rank_id);
}

bool ControlPlane::is_local_mesh(MeshId mesh_id) const {
    const auto& local_mesh_ids = local_mesh_binding_.mesh_ids;
    return std::find(local_mesh_ids.begin(), local_mesh_ids.end(), mesh_id) != local_mesh_ids.end();
}

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
