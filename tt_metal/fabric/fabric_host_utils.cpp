// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include "fabric_host_utils.hpp"

#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/experimental/fabric/topology_mapper.hpp>
#include <tt_stl/assert.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>  // ChipId
#include <tt-metalium/metal_soc_descriptor.h>
#include "impl/context/metal_context.hpp"
#include "erisc_datamover_builder.hpp"
#include <set>
#include <vector>
#include <algorithm>
#include "fabric_context.hpp"
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <filesystem>
#include <fstream>
#include <yaml-cpp/yaml.h>
#include <tt-logger/tt-logger.hpp>

namespace tt::tt_fabric {

bool is_tt_fabric_config(tt::tt_fabric::FabricConfig fabric_config) {
    return is_1d_fabric_config(fabric_config) || is_2d_fabric_config(fabric_config);
}

FabricType get_fabric_type(tt::tt_fabric::FabricConfig fabric_config) {
    switch (fabric_config) {
        // Issue: 32146, Special case for T3k WH devices to use Mesh fabric type instead of Torus_XY
        // WH T3K currently do not support Torus_XY fabric type, because they do not have wrapping connections.
        // If you want to use 1D Ring on t3k please use 1x8 MGD.
        case tt::tt_fabric::FabricConfig::FABRIC_1D_NEIGHBOR_EXCHANGE:
        case tt::tt_fabric::FabricConfig::FABRIC_1D_RING: {
            if (tt::tt_metal::MetalContext::instance().get_cluster().is_ubb_galaxy()) {
                return FabricType::TORUS_XY;
            }
            return FabricType::MESH;
        }
        case tt::tt_fabric::FabricConfig::FABRIC_2D_TORUS_X: return FabricType::TORUS_X;
        case tt::tt_fabric::FabricConfig::FABRIC_2D_TORUS_Y: return FabricType::TORUS_Y;
        case tt::tt_fabric::FabricConfig::FABRIC_2D_TORUS_XY: return FabricType::TORUS_XY;
        default: return FabricType::MESH;
    }
}

bool requires_more_connectivity(FabricType requested_type, FabricType available_type, const MeshShape& mesh_shape) {
    // Requesting MESH is always valid (can restrict any topology to MESH)
    if (requested_type == FabricType::MESH) {
        return false;
    }

    // Check if available topology can satisfy the requested topology
    if (available_type == FabricType::MESH) {
        // Special case: 2-element dimensions make torus wrap-around equivalent to mesh neighbor connections
        // E.g., in a 2-row mesh, north/south wrap-around just connects to the adjacent row
        bool has_two_rows = (mesh_shape[0] == 2);
        bool has_two_cols = (mesh_shape[1] == 2);

        if (has_flag(requested_type, FabricType::TORUS_Y) && !has_two_rows) {
            return true;
        }
        if (has_flag(requested_type, FabricType::TORUS_X) && !has_two_cols) {
            return true;
        }
        return false;
    }

    // For non-MESH available types, check if requested features are present
    if (requested_type == FabricType::TORUS_XY) {
        return available_type != FabricType::TORUS_XY;
    }
    if (requested_type == FabricType::TORUS_X) {
        return !has_flag(available_type, FabricType::TORUS_X);
    }
    if (requested_type == FabricType::TORUS_Y) {
        return !has_flag(available_type, FabricType::TORUS_Y);
    }

    return false;
}

uint32_t compute_max_1d_hops(const std::vector<MeshShape>& mesh_shapes) {
    if (mesh_shapes.empty()) {
        return 0;
    }

    uint32_t max_dimension = 0;
    for (const auto& shape : mesh_shapes) {
        // For 1D routing, find the maximum dimension (either rows or cols)
        // Hops = max_dimension - 1 (e.g., 8 chips in a line = 7 hops)
        uint32_t rows = shape[0];
        uint32_t cols = shape[1];
        uint32_t mesh_max_dim = std::max(rows, cols);
        max_dimension = std::max(max_dimension, mesh_max_dim);
    }

    return (max_dimension > 0) ? (max_dimension - 1) : 0;
}

uint32_t compute_max_2d_hops(const std::vector<MeshShape>& mesh_shapes) {
    if (mesh_shapes.empty()) {
        return 0;
    }

    uint32_t max_hops = 0;
    for (const auto& shape : mesh_shapes) {
        // For 2D routing, compute Manhattan distance from corner to corner
        // Hops = (rows - 1) + (cols - 1)
        uint32_t rows = shape[0];
        uint32_t cols = shape[1];
        uint32_t mesh_hops = (rows - 1) + (cols - 1);
        max_hops = std::max(max_hops, mesh_hops);
    }

    return max_hops;
}

std::vector<uint32_t> get_forwarding_link_indices_in_direction(
    const FabricNodeId& src_fabric_node_id, const FabricNodeId& dst_fabric_node_id, RoutingDirection direction) {
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();

    const std::vector<chan_id_t>& fabric_channels =
        control_plane.get_active_fabric_eth_channels_in_direction(src_fabric_node_id, direction);

    // the subset of routers that support forwarding b/w those chips
    std::vector<chan_id_t> forwarding_channels;
    forwarding_channels =
        control_plane.get_forwarding_eth_chans_to_chip(src_fabric_node_id, dst_fabric_node_id, direction);

    std::vector<uint32_t> link_indices;
    for (uint32_t i = 0; i < fabric_channels.size(); i++) {
        if (std::find(forwarding_channels.begin(), forwarding_channels.end(), fabric_channels[i]) !=
            forwarding_channels.end()) {
            link_indices.push_back(i);
        }
    }

    return link_indices;
}

void serialize_mesh_coordinates_to_file(
    const TopologyMapper& topology_mapper, const std::filesystem::path& output_file_path) {
    // Ensure output directory exists
    std::filesystem::create_directories(output_file_path.parent_path());

    // Get the mapping from TopologyMapper
    const auto& mapping = topology_mapper.get_local_logical_mesh_chip_id_to_physical_chip_id_mapping();
    const auto& mesh_graph = topology_mapper.get_mesh_graph();

    // Write to file using emitter with Flow style for inline sequences
    std::ofstream out_file(output_file_path);
    if (!out_file.is_open()) {
        TT_THROW("Failed to open output file: {}", output_file_path.string());
    }

    YAML::Emitter emitter;
    emitter << YAML::BeginMap;
    emitter << YAML::Key << "chips";
    emitter << YAML::Value << YAML::BeginMap;

    // Emit each chip with flow style for the coordinate array
    for (const auto& [fabric_node_id, physical_chip_id] : mapping) {
        MeshCoordinate mesh_coord = mesh_graph.chip_to_coordinate(fabric_node_id.mesh_id, fabric_node_id.chip_id);
        emitter << YAML::Key << physical_chip_id;
        emitter << YAML::Value;
        emitter << YAML::Flow << YAML::BeginSeq;
        for (size_t dim = 0; dim < mesh_coord.dims(); ++dim) {
            emitter << mesh_coord[dim];
        }
        emitter << YAML::EndSeq;
    }

    emitter << YAML::EndMap;
    emitter << YAML::EndMap;
    out_file << emitter.c_str();
    out_file.close();

    log_debug(tt::LogFabric, "Serialized physical chip mesh coordinate mapping to file: {}", output_file_path.string());
}

}  // namespace tt::tt_fabric
