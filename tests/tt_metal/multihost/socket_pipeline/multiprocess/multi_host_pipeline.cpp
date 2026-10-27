
// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"

#include "tt_metal/multihost/socket_pipeline/multiprocess/utils/mesh_socket_send_recv.hpp"
#include "tt_metal/multihost/socket_pipeline/multiprocess/utils/mesh_socket_forward.hpp"
#include "tt_metal/multihost/socket_pipeline/multiprocess/utils/mesh_socket_rate.hpp"

#include "tt_metal/multihost/fabric_tests/multihost_fabric_fixtures.hpp"
#include <tt-metalium/experimental/sockets/mesh_socket.hpp>
#include <tt-metalium/experimental/fabric/pipeline_builder.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph_descriptor.hpp>
#include <tt-metalium/distributed_context.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <chrono>
#include <filesystem>
#include <numeric>
#include <optional>
#include <tt-metalium/experimental/fabric/physical_system_descriptor.hpp>
#include "tt_metal/fabric/physical_system_discovery.hpp"
#include "tt_metal/llrt/tt_cluster.hpp"

namespace tt::tt_metal {

// Pipeline config structs.

// User builds a pipeline in physical space (Host Ranks, Tray IDs, ASIC Locations)
struct PhysicalPipelineStageConfig {
    uint32_t entry_node_tray_id;
    uint32_t exit_node_tray_id;
    uint32_t entry_node_asic_location;
    uint32_t exit_node_asic_location;
};

// Logical Coords for start, intermed and end nodes in the pipeline are derived from the physical config.
struct LogicalPipelineStageConfig {
    std::size_t stage_index;
    uint32_t rank;
    distributed::MeshCoordinate entry_node_coord;
    distributed::MeshCoordinate exit_node_coord;
};

struct LogicalPipelineConfig {
    std::vector<LogicalPipelineStageConfig> stages;
    distributed::MeshCoordinate start_coord;
};

// Determine how the Multi Mesh Coordinate system is instantiated on the physical cluster.
std::unordered_map<tt::tt_metal::AsicID, distributed::MeshCoordinate> get_asic_id_to_mesh_coord_map(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device) {
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    std::unordered_map<tt::tt_metal::AsicID, distributed::MeshCoordinate> asic_id_to_mesh_coord_map;

    for (const auto& coord : distributed::MeshCoordinateRange(mesh_device->shape())) {
        tt_fabric::FabricNodeId fabric_node_id = mesh_device->get_fabric_node_id(coord);
        tt_metal::AsicID asic_id = control_plane.get_asic_id_from_fabric_node_id(fabric_node_id);
        asic_id_to_mesh_coord_map.emplace(asic_id, coord);
    }
    // Exchange this map across all hosts using distributed context
    const auto& distributed_context = tt_metal::distributed::multihost::DistributedContext::get_current_world();
    for (auto rank = 0; rank < *(distributed_context->size()); rank++) {
        if (rank == *(distributed_context->rank())) {
            // Loop over all entries of the map and send them to the other hosts
            std::size_t num_entries = asic_id_to_mesh_coord_map.size();
            distributed_context->broadcast(
                ttsl::Span<std::byte>(reinterpret_cast<std::byte*>(&num_entries), sizeof(num_entries)),
                distributed::multihost::Rank{rank});
            for (auto& [asic_id, mesh_coord] : asic_id_to_mesh_coord_map) {
                distributed_context->broadcast(
                    ttsl::Span<std::byte>(
                        reinterpret_cast<std::byte*>(const_cast<tt_metal::AsicID*>(&asic_id)), sizeof(asic_id)),
                    distributed::multihost::Rank{rank});
                distributed_context->broadcast(
                    ttsl::Span<std::byte>(reinterpret_cast<std::byte*>(&(mesh_coord[0])), sizeof(mesh_coord[0])),
                    distributed::multihost::Rank{rank});
                distributed_context->broadcast(
                    ttsl::Span<std::byte>(reinterpret_cast<std::byte*>(&(mesh_coord[1])), sizeof(mesh_coord[1])),
                    distributed::multihost::Rank{rank});
            }
        } else {
            // Receive the map from the other host
            std::size_t num_entries = 0;
            distributed_context->broadcast(
                ttsl::Span<std::byte>(reinterpret_cast<std::byte*>(&num_entries), sizeof(num_entries)),
                distributed::multihost::Rank{rank});
            for (auto i = 0; i < num_entries; i++) {
                tt_metal::AsicID asic_id;
                distributed::MeshCoordinate mesh_coord = distributed::MeshCoordinate(0, 0);
                distributed_context->broadcast(
                    ttsl::Span<std::byte>(reinterpret_cast<std::byte*>(&asic_id), sizeof(asic_id)),
                    distributed::multihost::Rank{rank});
                distributed_context->broadcast(
                    ttsl::Span<std::byte>(reinterpret_cast<std::byte*>(&(mesh_coord[0])), sizeof(mesh_coord[0])),
                    distributed::multihost::Rank{rank});
                distributed_context->broadcast(
                    ttsl::Span<std::byte>(reinterpret_cast<std::byte*>(&(mesh_coord[1])), sizeof(mesh_coord[1])),
                    distributed::multihost::Rank{rank});
                asic_id_to_mesh_coord_map.emplace(asic_id, mesh_coord);
            }
        }
    }
    return asic_id_to_mesh_coord_map;
}

// Pipeline type enum to toggle between different pipeline configurations.
enum class PipelineType {
    SINGLE_GALAXY,  // Single-galaxy pipeline (4 stages, 9 hops across 4 trays)
    DUAL_GALAXY,
    QUAD_GALAXY,
    SINGLE_POD,
    SUPERPOD_2_POD,
    SUPERPOD_4
};

// Maps number of processes (distributed context size) to pipeline config.
// 4 -> Single Galaxy, 16 -> Single Pod (Superpod 4), 32 -> Superpod 2 Pod, 64 -> Superpod 4 Pod.
inline std::optional<PipelineType> pipeline_type_from_num_ranks(size_t num_ranks) {
    switch (num_ranks) {
        case 4u: return PipelineType::SINGLE_GALAXY;
        case 16u: return PipelineType::SINGLE_POD;
        case 32u: return PipelineType::SUPERPOD_2_POD;
        case 64u: return PipelineType::SUPERPOD_4;
        default: return std::nullopt;
    }
}

tt::tt_fabric::FabricConfig fabric_config_for_active_pipeline_mgd() {
    using tt::tt_fabric::FabricConfig;
    auto& rtoptions = MetalContext::instance().rtoptions();
    if (!rtoptions.is_custom_fabric_mesh_graph_desc_path_specified()) {
        return FabricConfig::FABRIC_2D;
    }

    const tt::tt_fabric::MeshGraphDescriptor mgd(
        std::filesystem::path(rtoptions.get_custom_fabric_mesh_graph_desc_path()));
    bool any_mesh = false;
    bool ring_ns = true;
    bool ring_ew = true;
    for (const auto& mesh_name : mgd.get_all_mesh_names()) {
        const auto topology = mgd.get_effective_declared_topology(mesh_name);
        if (!topology.has_value()) {
            continue;
        }
        any_mesh = true;
        ring_ns = ring_ns && (!topology->ring_dims.empty() && topology->ring_dims[0]);
        ring_ew = ring_ew && (topology->ring_dims.size() > 1 && topology->ring_dims[1]);
    }
    if (any_mesh && ring_ns && ring_ew) {
        return FabricConfig::FABRIC_2D_TORUS_XY;
    }
    if (any_mesh && ring_ns) {
        return FabricConfig::FABRIC_2D_TORUS_Y;
    }
    if (any_mesh && ring_ew) {
        return FabricConfig::FABRIC_2D_TORUS_X;
    }
    return FabricConfig::FABRIC_2D;
}

// Universal pipeline fixture: selects config (Single Galaxy, Single Pod, Superpod 2 Pod, Superpod 4 Pod)
// based on distributed context size. Set TT_FABRIC_MESH_GRAPH_DESC_PATH to the matching mesh graph.
class BlitzDecodePipelineFixture : public MeshDeviceFixtureBase {
public:
    BlitzDecodePipelineFixture() :
        MeshDeviceFixtureBase(Config{.num_cqs = 1, .fabric_config = fabric_config_for_active_pipeline_mgd()}) {}

    void SetUp() override {
        if (not system_supported()) {
            GTEST_SKIP() << "Skipping: pipeline requires 4, 16, 32, or 64 ranks and matching mesh graph.";
        }
        MeshDeviceFixtureBase::SetUp();
    }

    void TearDown() override {
        if (system_supported()) {
            MeshDeviceFixtureBase::TearDown();
        }
    }

    bool system_supported() {
        const auto& cluster = MetalContext::instance().get_cluster();
        const auto& mesh_graph = MetalContext::instance().get_control_plane().get_mesh_graph();
        const auto num_ranks = *MetalContext::instance().global_distributed_context().size();
        if (num_ranks != mesh_graph.get_mesh_ids().size() || not cluster.is_ubb_galaxy()) {
            return false;
        }
        return pipeline_type_from_num_ranks(num_ranks).has_value();
    }
};

// Get physical pipeline stage configs for the specified pipeline type.
// When enable_loopback is true, returns the full path including the loopback/wrap-around stage where defined.
// When enable_loopback is false, returns a linear path (loopback stage removed or replaced with linear endpoint).
// SINGLE_GALAXY: 5 stages with loopback; 4 stages linear. SUPERPOD_2_POD has no loopback config (linear only).
std::vector<PhysicalPipelineStageConfig> get_physical_pipeline_config(PipelineType type, bool enable_loopback = true) {
    std::vector<PhysicalPipelineStageConfig> config;
    switch (type) {
        case PipelineType::SINGLE_GALAXY:
            config = {
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 2,
                 .exit_node_asic_location = 6},
                {.entry_node_tray_id = 3,
                 .exit_node_tray_id = 3,
                 .entry_node_asic_location = 6,
                 .exit_node_asic_location = 4},
                {.entry_node_tray_id = 4,
                 .exit_node_tray_id = 4,
                 .entry_node_asic_location = 4,
                 .exit_node_asic_location = 7},
                {.entry_node_tray_id = 2,
                 .exit_node_tray_id = 2,
                 .entry_node_asic_location = 7,
                 .exit_node_asic_location = 4},
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 4,
                 .exit_node_asic_location = 2},
            };
            break;
        case PipelineType::SINGLE_POD:
            config = {
                // First Tray
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 1,
                 .exit_node_asic_location = 2},
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 3,
                 .exit_node_asic_location = 4},
                {.entry_node_tray_id = 2,
                 .exit_node_tray_id = 2,
                 .entry_node_asic_location = 4,
                 .exit_node_asic_location = 3},
                {.entry_node_tray_id = 2,
                 .exit_node_tray_id = 2,
                 .entry_node_asic_location = 2,
                 .exit_node_asic_location = 1},
                // Second Tray
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 1,
                 .exit_node_asic_location = 2},
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 3,
                 .exit_node_asic_location = 4},
                {.entry_node_tray_id = 2,
                 .exit_node_tray_id = 2,
                 .entry_node_asic_location = 4,
                 .exit_node_asic_location = 3},
                {.entry_node_tray_id = 2,
                 .exit_node_tray_id = 2,
                 .entry_node_asic_location = 2,
                 .exit_node_asic_location = 1},
                // Third Tray
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 1,
                 .exit_node_asic_location = 2},
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 3,
                 .exit_node_asic_location = 4},
                {.entry_node_tray_id = 2,
                 .exit_node_tray_id = 2,
                 .entry_node_asic_location = 4,
                 .exit_node_asic_location = 3},
                {.entry_node_tray_id = 2,
                 .exit_node_tray_id = 2,
                 .entry_node_asic_location = 2,
                 .exit_node_asic_location = 1},
                // Fourth Tray
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 1,
                 .exit_node_asic_location = 2},
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 3,
                 .exit_node_asic_location = 4},
                {.entry_node_tray_id = 2,
                 .exit_node_tray_id = 2,
                 .entry_node_asic_location = 4,
                 .exit_node_asic_location = 7},
                {.entry_node_tray_id = 2,
                 .exit_node_tray_id = 2,
                 .entry_node_asic_location = 6,
                 .exit_node_asic_location = 5},
                // Wrap-around
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 5,
                 .exit_node_asic_location = 1},
            };
            break;
        // SUPERPOD_2_POD: 32 stages (linear only; no loopback config).
        case PipelineType::SUPERPOD_2_POD:
            config = {
                // First Pod
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 1,
                 .exit_node_asic_location = 2},
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 3,
                 .exit_node_asic_location = 4},
                {.entry_node_tray_id = 2,
                 .exit_node_tray_id = 2,
                 .entry_node_asic_location = 4,
                 .exit_node_asic_location = 3},
                {.entry_node_tray_id = 2,
                 .exit_node_tray_id = 2,
                 .entry_node_asic_location = 2,
                 .exit_node_asic_location = 1},
                // Second Pod
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 1,
                 .exit_node_asic_location = 2},
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 3,
                 .exit_node_asic_location = 4},
                {.entry_node_tray_id = 2,
                 .exit_node_tray_id = 2,
                 .entry_node_asic_location = 4,
                 .exit_node_asic_location = 3},
                {.entry_node_tray_id = 2,
                 .exit_node_tray_id = 2,
                 .entry_node_asic_location = 2,
                 .exit_node_asic_location = 1},
                // Third Pod
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 1,
                 .exit_node_asic_location = 2},
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 3,
                 .exit_node_asic_location = 4},
                {.entry_node_tray_id = 2,
                 .exit_node_tray_id = 2,
                 .entry_node_asic_location = 4,
                 .exit_node_asic_location = 3},
                {.entry_node_tray_id = 2,
                 .exit_node_tray_id = 2,
                 .entry_node_asic_location = 2,
                 .exit_node_asic_location = 1},
                // Fourth Pod
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 1,
                 .exit_node_asic_location = 2},
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 3,
                 .exit_node_asic_location = 4},
                {.entry_node_tray_id = 2,
                 .exit_node_tray_id = 2,
                 .entry_node_asic_location = 4,
                 .exit_node_asic_location = 7},
                {.entry_node_tray_id = 2,
                 .exit_node_tray_id = 2,
                 .entry_node_asic_location = 6,
                 .exit_node_asic_location = 5},
                // Fifth Pod
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 5,
                 .exit_node_asic_location = 6},
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 7,
                 .exit_node_asic_location = 8},
                {.entry_node_tray_id = 2,
                 .exit_node_tray_id = 2,
                 .entry_node_asic_location = 8,
                 .exit_node_asic_location = 7},
                {.entry_node_tray_id = 2,
                 .exit_node_tray_id = 2,
                 .entry_node_asic_location = 6,
                 .exit_node_asic_location = 5},
                // Sixth Pod
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 5,
                 .exit_node_asic_location = 6},
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 7,
                 .exit_node_asic_location = 8},
                {.entry_node_tray_id = 2,
                 .exit_node_tray_id = 2,
                 .entry_node_asic_location = 8,
                 .exit_node_asic_location = 7},
                {.entry_node_tray_id = 2,
                 .exit_node_tray_id = 2,
                 .entry_node_asic_location = 6,
                 .exit_node_asic_location = 5},
                // Seventh Pod
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 5,
                 .exit_node_asic_location = 6},
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 7,
                 .exit_node_asic_location = 8},
                {.entry_node_tray_id = 2,
                 .exit_node_tray_id = 2,
                 .entry_node_asic_location = 8,
                 .exit_node_asic_location = 7},
                {.entry_node_tray_id = 2,
                 .exit_node_tray_id = 2,
                 .entry_node_asic_location = 6,
                 .exit_node_asic_location = 5},
                // Eighth Pod
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 5,
                 .exit_node_asic_location = 6},
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 7,
                 .exit_node_asic_location = 8},
                {.entry_node_tray_id = 2,
                 .exit_node_tray_id = 2,
                 .entry_node_asic_location = 8,
                 .exit_node_asic_location = 7},
                {.entry_node_tray_id = 2,
                 .exit_node_tray_id = 2,
                 .entry_node_asic_location = 6,
                 .exit_node_asic_location = 5},
            };
            break;
        // SUPERPOD_4: 65 stages (16+16+16+16+1). Order: Pod1 (trays 1,2) -> Pod4 (trays 1,2) -> Pod3 (trays 1,2) ->
        // Pod2 (trays 3,4) -> wrap-around to Pod1.
        case PipelineType::SUPERPOD_4:
            config = {
                {.entry_node_tray_id = 2,
                 .exit_node_tray_id = 2,
                 .entry_node_asic_location = 2,
                 .exit_node_asic_location = 1},
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 1,
                 .exit_node_asic_location = 6},
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 7,
                 .exit_node_asic_location = 8},
                {.entry_node_tray_id = 2,
                 .exit_node_tray_id = 2,
                 .entry_node_asic_location = 8,
                 .exit_node_asic_location = 7},
                {.entry_node_tray_id = 2,
                 .exit_node_tray_id = 2,
                 .entry_node_asic_location = 6,
                 .exit_node_asic_location = 5},
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 5,
                 .exit_node_asic_location = 6},
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 7,
                 .exit_node_asic_location = 8},
                {.entry_node_tray_id = 2,
                 .exit_node_tray_id = 2,
                 .entry_node_asic_location = 8,
                 .exit_node_asic_location = 7},
                {.entry_node_tray_id = 2,
                 .exit_node_tray_id = 2,
                 .entry_node_asic_location = 6,
                 .exit_node_asic_location = 5},
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 5,
                 .exit_node_asic_location = 6},
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 7,
                 .exit_node_asic_location = 8},
                {.entry_node_tray_id = 2,
                 .exit_node_tray_id = 2,
                 .entry_node_asic_location = 8,
                 .exit_node_asic_location = 7},
                {.entry_node_tray_id = 2,
                 .exit_node_tray_id = 2,
                 .entry_node_asic_location = 6,
                 .exit_node_asic_location = 5},
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 5,
                 .exit_node_asic_location = 6},
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 7,
                 .exit_node_asic_location = 8},
                {.entry_node_tray_id = 2,
                 .exit_node_tray_id = 2,
                 .entry_node_asic_location = 8,
                 .exit_node_asic_location = 7},
                // jump to pod 4
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 7,
                 .exit_node_asic_location = 8},
                {.entry_node_tray_id = 2,
                 .exit_node_tray_id = 2,
                 .entry_node_asic_location = 8,
                 .exit_node_asic_location = 7},
                {.entry_node_tray_id = 2,
                 .exit_node_tray_id = 2,
                 .entry_node_asic_location = 6,
                 .exit_node_asic_location = 5},
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 5,
                 .exit_node_asic_location = 6},
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 7,
                 .exit_node_asic_location = 8},
                {.entry_node_tray_id = 2,
                 .exit_node_tray_id = 2,
                 .entry_node_asic_location = 8,
                 .exit_node_asic_location = 7},
                {.entry_node_tray_id = 2,
                 .exit_node_tray_id = 2,
                 .entry_node_asic_location = 6,
                 .exit_node_asic_location = 5},
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 5,
                 .exit_node_asic_location = 6},
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 7,
                 .exit_node_asic_location = 8},
                {.entry_node_tray_id = 2,
                 .exit_node_tray_id = 2,
                 .entry_node_asic_location = 8,
                 .exit_node_asic_location = 7},
                {.entry_node_tray_id = 2,
                 .exit_node_tray_id = 2,
                 .entry_node_asic_location = 6,
                 .exit_node_asic_location = 5},
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 5,
                 .exit_node_asic_location = 6},
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 7,
                 .exit_node_asic_location = 8},
                {.entry_node_tray_id = 2,
                 .exit_node_tray_id = 2,
                 .entry_node_asic_location = 8,
                 .exit_node_asic_location = 7},
                {.entry_node_tray_id = 2,
                 .exit_node_tray_id = 2,
                 .entry_node_asic_location = 6,
                 .exit_node_asic_location = 5},
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 5,
                 .exit_node_asic_location = 6},
                // jump to pod 3
                {.entry_node_tray_id = 2,
                 .exit_node_tray_id = 2,
                 .entry_node_asic_location = 6,
                 .exit_node_asic_location = 5},
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 5,
                 .exit_node_asic_location = 6},
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 7,
                 .exit_node_asic_location = 8},
                {.entry_node_tray_id = 2,
                 .exit_node_tray_id = 2,
                 .entry_node_asic_location = 8,
                 .exit_node_asic_location = 7},
                {.entry_node_tray_id = 2,
                 .exit_node_tray_id = 2,
                 .entry_node_asic_location = 6,
                 .exit_node_asic_location = 5},
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 5,
                 .exit_node_asic_location = 6},
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 7,
                 .exit_node_asic_location = 8},
                {.entry_node_tray_id = 2,
                 .exit_node_tray_id = 2,
                 .entry_node_asic_location = 8,
                 .exit_node_asic_location = 7},
                {.entry_node_tray_id = 2,
                 .exit_node_tray_id = 2,
                 .entry_node_asic_location = 6,
                 .exit_node_asic_location = 5},
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 5,
                 .exit_node_asic_location = 6},
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 7,
                 .exit_node_asic_location = 8},
                {.entry_node_tray_id = 2,
                 .exit_node_tray_id = 2,
                 .entry_node_asic_location = 8,
                 .exit_node_asic_location = 7},
                {.entry_node_tray_id = 2,
                 .exit_node_tray_id = 2,
                 .entry_node_asic_location = 6,
                 .exit_node_asic_location = 5},
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 5,
                 .exit_node_asic_location = 6},
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 7,
                 .exit_node_asic_location = 8},
                {.entry_node_tray_id = 2,
                 .exit_node_tray_id = 4,
                 .entry_node_asic_location = 8,
                 .exit_node_asic_location = 4},
                // jump to pod 2
                {.entry_node_tray_id = 3,
                 .exit_node_tray_id = 3,
                 .entry_node_asic_location = 3,
                 .exit_node_asic_location = 4},
                {.entry_node_tray_id = 4,
                 .exit_node_tray_id = 4,
                 .entry_node_asic_location = 4,
                 .exit_node_asic_location = 3},
                {.entry_node_tray_id = 4,
                 .exit_node_tray_id = 4,
                 .entry_node_asic_location = 2,
                 .exit_node_asic_location = 1},
                {.entry_node_tray_id = 3,
                 .exit_node_tray_id = 3,
                 .entry_node_asic_location = 1,
                 .exit_node_asic_location = 2},
                {.entry_node_tray_id = 3,
                 .exit_node_tray_id = 3,
                 .entry_node_asic_location = 3,
                 .exit_node_asic_location = 4},
                {.entry_node_tray_id = 4,
                 .exit_node_tray_id = 4,
                 .entry_node_asic_location = 4,
                 .exit_node_asic_location = 3},
                {.entry_node_tray_id = 4,
                 .exit_node_tray_id = 4,
                 .entry_node_asic_location = 2,
                 .exit_node_asic_location = 1},
                {.entry_node_tray_id = 3,
                 .exit_node_tray_id = 3,
                 .entry_node_asic_location = 1,
                 .exit_node_asic_location = 2},
                {.entry_node_tray_id = 3,
                 .exit_node_tray_id = 3,
                 .entry_node_asic_location = 3,
                 .exit_node_asic_location = 4},
                {.entry_node_tray_id = 4,
                 .exit_node_tray_id = 4,
                 .entry_node_asic_location = 4,
                 .exit_node_asic_location = 3},
                {.entry_node_tray_id = 4,
                 .exit_node_tray_id = 4,
                 .entry_node_asic_location = 2,
                 .exit_node_asic_location = 1},
                {.entry_node_tray_id = 3,
                 .exit_node_tray_id = 3,
                 .entry_node_asic_location = 1,
                 .exit_node_asic_location = 2},
                {.entry_node_tray_id = 3,
                 .exit_node_tray_id = 3,
                 .entry_node_asic_location = 3,
                 .exit_node_asic_location = 4},
                {.entry_node_tray_id = 4,
                 .exit_node_tray_id = 4,
                 .entry_node_asic_location = 4,
                 .exit_node_asic_location = 3},
                {.entry_node_tray_id = 4,
                 .exit_node_tray_id = 4,
                 .entry_node_asic_location = 2,
                 .exit_node_asic_location = 1},
                {.entry_node_tray_id = 3,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 1,
                 .exit_node_asic_location = 5},
                // wrap-around
                {.entry_node_tray_id = 2,
                 .exit_node_tray_id = 2,
                 .entry_node_asic_location = 6,
                 .exit_node_asic_location = 2},
            };
            break;
        default: return {};
    }
    // No loopback config for SUPERPOD_2_POD (linear-only topology).
    if (type == PipelineType::SUPERPOD_2_POD && enable_loopback) {
        return {};
    }
    if (enable_loopback) {
        return config;
    }
    // Linear pipeline: remove or replace loopback stage.
    switch (type) {
        case PipelineType::SUPERPOD_4:
            config.pop_back();
            config.push_back(
                {.entry_node_tray_id = 2,
                 .exit_node_tray_id = 2,
                 .entry_node_asic_location = 5,
                 .exit_node_asic_location = 6});
            break;
        default: break;
    }
    return config;
}

// Overloaded build_pipeline that accepts an external physical pipeline config.
// Same pattern as ClosetBox: stage_index % num_procs for hostname. For loopback (last stage same tray
// as first), resolve last stage's ASICs using rank 0 host so they exist in asic_id_to_mesh_coord.
std::vector<LogicalPipelineStageConfig> build_pipeline(
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    const std::unordered_map<tt::tt_metal::AsicID, distributed::MeshCoordinate>& asic_id_to_mesh_coord,
    const std::vector<PhysicalPipelineStageConfig>& physical_pipeline_stage_configs) {
    const auto num_procs = *(tt::tt_metal::MetalContext::instance().get_distributed_context_ptr()->size());
    const std::size_t num_stages = physical_pipeline_stage_configs.size();
    const bool last_stage_loopback =
        (num_stages > 1u && physical_pipeline_stage_configs.back().exit_node_tray_id ==
                                physical_pipeline_stage_configs[0].entry_node_tray_id);
    std::vector<LogicalPipelineStageConfig> logical_pipeline_stage_configs;
    for (std::size_t stage_index = 0; stage_index < num_stages; stage_index++) {
        uint32_t rank_for_host = (last_stage_loopback && stage_index == num_stages - 1u)
                                     ? 0u
                                     : static_cast<uint32_t>(stage_index % num_procs);
        auto stage_hostname = physical_system_descriptor.get_hostname_for_rank(rank_for_host);
        const auto& phys = physical_pipeline_stage_configs[stage_index];
        auto entry_node_asic_id = physical_system_descriptor.get_asic_id(
            stage_hostname,
            tt::tt_metal::TrayID(phys.entry_node_tray_id),
            tt::tt_metal::ASICLocation(phys.entry_node_asic_location));
        auto exit_node_asic_id = physical_system_descriptor.get_asic_id(
            stage_hostname,
            tt::tt_metal::TrayID(phys.exit_node_tray_id),
            tt::tt_metal::ASICLocation(phys.exit_node_asic_location));
        logical_pipeline_stage_configs.emplace_back(LogicalPipelineStageConfig{
            .stage_index = stage_index,
            .rank = rank_for_host,
            .entry_node_coord = asic_id_to_mesh_coord.at(entry_node_asic_id),
            .exit_node_coord = asic_id_to_mesh_coord.at(exit_node_asic_id)});
    }
    return logical_pipeline_stage_configs;
}

// Resolve the single-galaxy pipeline from the automapper result. This keeps the socket
// endpoints aligned with the cross-tray submeshes selected by the MGD instead of assuming
// that rank N owns a particular physical tray.
LogicalPipelineConfig build_automapped_single_galaxy_pipeline() {
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto& mesh_graph = control_plane.get_mesh_graph();
    const auto& global_bindings = control_plane.get_global_logical_bindings();

    auto mesh_ids = mesh_graph.get_mesh_ids();
    std::sort(mesh_ids.begin(), mesh_ids.end());

    std::vector<std::vector<tt::tt_fabric::ChipTuple>> submesh_chips;
    std::vector<uint32_t> submesh_ranks;
    for (const auto mesh_id : mesh_ids) {
        for (const auto& [host_coord, host_rank] : mesh_graph.get_host_ranks(mesh_id)) {
            const auto rank_shape = mesh_graph.get_mesh_shape(mesh_id, host_rank);
            std::vector<tt::tt_fabric::ChipTuple> chips;
            chips.reserve(rank_shape.mesh_size());
            for (uint32_t row = 0; row < rank_shape[0]; ++row) {
                for (uint32_t col = 0; col < rank_shape[1]; ++col) {
                    const distributed::MeshCoordinate local_coord(row, col);
                    const auto chip_id = mesh_graph.coordinate_to_chip(mesh_id, local_coord, host_rank);
                    chips.emplace_back(*mesh_id, chip_id, row, col);
                }
            }
            submesh_chips.push_back(std::move(chips));

            const auto binding = std::make_pair(mesh_id, host_rank);
            const auto rank_it =
                std::find_if(global_bindings.begin(), global_bindings.end(), [&](const auto& rank_binding) {
                    return rank_binding.second == binding;
                });
            TT_FATAL(
                rank_it != global_bindings.end(), "No MPI rank is bound to mesh {} host rank {}", *mesh_id, *host_rank);
            submesh_ranks.push_back(static_cast<uint32_t>(*rank_it->first));
        }
    }

    constexpr std::size_t NUM_STAGES = 4;
    TT_FATAL(
        submesh_chips.size() == NUM_STAGES,
        "Single-galaxy pipeline requires four automapped submeshes, found {}",
        submesh_chips.size());

    const std::vector<std::string> nodes = {"s0", "s1", "s2", "s3"};
    const std::vector<tt::tt_fabric::EdgeInputTuple> edges = {
        {"s0", "s1", false}, {"s1", "s2", false}, {"s2", "s3", false}, {"s3", "s0", true}};
    const auto layout = tt::tt_fabric::resolve_graph_layout(nodes, edges, submesh_chips);

    LogicalPipelineConfig pipeline{
        .stages = std::vector<LogicalPipelineStageConfig>(
            NUM_STAGES,
            LogicalPipelineStageConfig{
                .stage_index = 0,
                .rank = 0,
                .entry_node_coord = distributed::MeshCoordinate(0, 0),
                .exit_node_coord = distributed::MeshCoordinate(0, 0)}),
        .start_coord = distributed::MeshCoordinate(layout.h2d_entry_row, layout.h2d_entry_col)};

    for (std::size_t stage_index = 0; stage_index < NUM_STAGES; ++stage_index) {
        const auto submesh_index = layout.node_to_submesh.at(nodes[stage_index]);
        pipeline.stages[stage_index].stage_index = stage_index;
        pipeline.stages[stage_index].rank = submesh_ranks.at(submesh_index);

        const auto& edge = layout.resolved_edges.at(stage_index);
        pipeline.stages[stage_index].exit_node_coord = distributed::MeshCoordinate(edge.exit_row, edge.exit_col);
        const auto downstream_stage = (stage_index + 1) % NUM_STAGES;
        pipeline.stages[downstream_stage].entry_node_coord =
            distributed::MeshCoordinate(edge.entry_row, edge.entry_col);
    }

    // Terminate the loopback directly on the sender at the stage-0 entry.
    pipeline.start_coord = pipeline.stages[0].entry_node_coord;

    return pipeline;
}

// Helper to get the device coords connecting the given pipeline stage and neighbor stage.
std::pair<distributed::MeshCoordinate, distributed::MeshCoordinate> get_connecting_coords(
    const std::vector<LogicalPipelineStageConfig>& pipeline_stages,
    uint32_t curr_stage_index,
    uint32_t neighbor_stage_index) {
    const auto& my_stage = pipeline_stages[curr_stage_index];
    const auto& neighbor_stage = pipeline_stages[neighbor_stage_index];

    if (curr_stage_index > neighbor_stage_index) {
        // Neighbor feeds into my stage
        return std::make_pair(my_stage.entry_node_coord, neighbor_stage.exit_node_coord);
    }
    // My stage feeds into neighbor
    return std::make_pair(my_stage.exit_node_coord, neighbor_stage.entry_node_coord);
}

// float convert_to_us(uint64_t cycles) {

// }

PhysicalSystemDescriptor create_physical_system_descriptor() {
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto& distributed_context = tt::tt_metal::MetalContext::instance().get_distributed_context_ptr();
    const auto& rtoptions = tt::tt_metal::MetalContext::instance().rtoptions();
    return tt::tt_metal::run_physical_system_discovery(
        *cluster.get_cluster_desc(), distributed_context, rtoptions.get_target_device());
}

// Multi-process loopback pipeline. On a single galaxy, the four stage placements and
// endpoints are resolved from the automapped cross-tray topology.
void run_single_galaxy_pipeline(
    std::shared_ptr<distributed::MeshDevice>& mesh_device,
    PipelineType pipeline_type,
    uint32_t num_iterations,
    bool enable_correctness_check) {
    constexpr uint32_t XFER_SIZE = 14 * 1024;  // size of data being moved across pipeline stages for the workload

    const auto& distributed_context = tt_metal::distributed::multihost::DistributedContext::get_current_world();
    const auto my_rank = *distributed_context->rank();

    const auto logical_coord = CoreCoord(0, 0);
    const uint32_t socket_fifo_size = XFER_SIZE * 16;

    std::optional<LogicalPipelineConfig> automapped_pipeline;
    std::vector<LogicalPipelineStageConfig> pipeline_stages;
    if (pipeline_type == PipelineType::SINGLE_GALAXY) {
        automapped_pipeline = build_automapped_single_galaxy_pipeline();
        pipeline_stages = automapped_pipeline->stages;
    } else {
        auto physical_system_descriptor = create_physical_system_descriptor();
        auto asic_id_to_mesh_coord = get_asic_id_to_mesh_coord_map(mesh_device);
        auto physical_config = get_physical_pipeline_config(pipeline_type);
        pipeline_stages = build_pipeline(physical_system_descriptor, asic_id_to_mesh_coord, physical_config);
    }

    const uint32_t num_stages = static_cast<uint32_t>(pipeline_stages.size());
    const uint32_t num_ranks = static_cast<uint32_t>(*(distributed_context->size()));
    const auto my_stage_it = std::find_if(
        pipeline_stages.begin(), pipeline_stages.end(), [&](const auto& stage) { return stage.rank == my_rank; });
    TT_FATAL(my_stage_it != pipeline_stages.end(), "MPI rank {} has no pipeline stage", my_rank);
    const uint32_t my_stage = static_cast<uint32_t>(std::distance(pipeline_stages.begin(), my_stage_it));
    const uint32_t downstream_stage = (my_stage + 1) % num_stages;
    const uint32_t upstream_stage = (my_stage + num_stages - 1) % num_stages;

    uint32_t downstream_rank;
    uint32_t upstream_rank;
    if (automapped_pipeline.has_value()) {
        downstream_rank = pipeline_stages[downstream_stage].rank;
        upstream_rank = pipeline_stages[upstream_stage].rank;
    } else {
        // Legacy multi-system configs model loopback as an extra local stage on rank 0.
        downstream_rank = (downstream_stage == num_stages - 1u) ? 0u : downstream_stage;
        upstream_rank = (upstream_stage == num_stages - 1u) ? (num_ranks - 1u) : upstream_stage;
    }

    const auto& global_bindings =
        tt::tt_metal::MetalContext::instance().get_control_plane().get_global_logical_bindings();
    const tt::tt_fabric::MeshId my_mesh_id = std::get<0>(global_bindings.at(distributed::multihost::Rank(my_rank)));
    const tt::tt_fabric::MeshId upstream_mesh_id =
        std::get<0>(global_bindings.at(distributed::multihost::Rank(upstream_rank)));
    const tt::tt_fabric::MeshId downstream_mesh_id =
        std::get<0>(global_bindings.at(distributed::multihost::Rank(downstream_rank)));

    const distributed::SocketMemoryConfig socket_mem_config(BufferType::L1, socket_fifo_size);

    // Metal-level buffer configuration
    const uint32_t num_elems = XFER_SIZE / sizeof(uint32_t);
    const DeviceAddr buffer_size = XFER_SIZE;
    const DeviceAddr page_size = XFER_SIZE;  // Single page buffer

    // Helper to create an intermediate socket pair for local forwarding
    auto create_intermed_socket_pair = [&](const distributed::MeshCoordinate& sender_coord,
                                           const distributed::MeshCoordinate& recv_coord) {
        auto connection = distributed::SocketConnection(
            distributed::MeshCoreCoord(sender_coord, logical_coord),
            distributed::MeshCoreCoord(recv_coord, logical_coord));
        auto config = distributed::SocketConfig({connection}, socket_mem_config);
        return distributed::MeshSocket::create_socket_pair(mesh_device, mesh_device, config);
    };

    // Helper to run warmup iteration with barrier synchronization
    auto barrier = [&]() {
        Synchronize(*mesh_device, std::nullopt);
        distributed_context->barrier();
    };

    const bool is_pipeline_start = (my_stage == 0);

    auto my_entry = pipeline_stages[my_stage].entry_node_coord;
    auto my_exit = pipeline_stages[my_stage].exit_node_coord;
    auto upstream_exit = pipeline_stages[upstream_stage].exit_node_coord;
    auto downstream_entry = pipeline_stages[downstream_stage].entry_node_coord;

    // Create Latency Measurement Buffer
    // Size: 8 bytes per iteration (uint64_t latency) + 32 bytes padding
    // First address is reused for credit/barrier synchronization
    const uint32_t latency_measurement_buffer_size = (8 * num_iterations) + 32;
    CoreRangeSet latency_core_range = CoreRangeSet(CoreRange(CoreCoord(0, 0), CoreCoord(0, 0)));
    auto shard_params = ShardSpecBuffer(latency_core_range, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});
    distributed::DeviceLocalBufferConfig latency_measurement_buffer_specs = {
        .page_size = latency_measurement_buffer_size,
        .buffer_type = BufferType::L1,
        .sharding_args = BufferShardingArgs(shard_params, TensorMemoryLayout::HEIGHT_SHARDED),
        .bottom_up = std::nullopt,
        .sub_device_id = std::nullopt,
    };
    auto latency_measurement_buffer = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = latency_measurement_buffer_size},
        latency_measurement_buffer_specs,
        mesh_device.get());
    // Write 0 to latency measurement buffer (initializes credit/barrier to 0)
    std::vector<uint32_t> latency_init_data(latency_measurement_buffer_size / sizeof(uint32_t), 0);
    distributed::EnqueueWriteMeshBuffer(
        mesh_device->mesh_command_queue(), latency_measurement_buffer, latency_init_data, true);

    const uint32_t latency_measurement_address = latency_measurement_buffer->address();

    const distributed::MeshCoordinate start_coord =
        automapped_pipeline.has_value() ? automapped_pipeline->start_coord : pipeline_stages[0].entry_node_coord;

    if (is_pipeline_start) {
        // Send path: start_coord -> my_exit -> downstream (use stage indices)
        auto [my_sender, downstream_recv] = get_connecting_coords(pipeline_stages, my_stage, downstream_stage);
        auto [intermed_send, intermed_recv] = create_intermed_socket_pair(start_coord, my_sender);

        auto fwd_connection = distributed::SocketConnection(
            distributed::MeshCoreCoord(my_sender, logical_coord),
            distributed::MeshCoreCoord(downstream_recv, logical_coord));
        auto send_socket_config = distributed::SocketConfig(
            {fwd_connection}, socket_mem_config, my_mesh_id, downstream_mesh_id, distributed_context);
        auto send_socket = distributed::MeshSocket(mesh_device, send_socket_config);

        // Recv path: terminate the automapped loopback directly on the sender.
        const auto my_recv = automapped_pipeline.has_value() ? start_coord : pipeline_stages.back().entry_node_coord;
        const auto upstream_send = automapped_pipeline.has_value() ? pipeline_stages.back().exit_node_coord
                                                                   : pipeline_stages[num_stages - 2u].exit_node_coord;
        auto bwd_connection = distributed::SocketConnection(
            distributed::MeshCoreCoord(upstream_send, logical_coord),
            distributed::MeshCoreCoord(my_recv, logical_coord));
        auto recv_socket_config = distributed::SocketConfig(
            {bwd_connection}, socket_mem_config, upstream_mesh_id, my_mesh_id, distributed_context);
        auto recv_socket = distributed::MeshSocket(mesh_device, recv_socket_config);

        // Create device buffer using metal-level API
        distributed::DeviceLocalBufferConfig buffer_config = {
            .page_size = page_size,
            .buffer_type = BufferType::DRAM,
            .sharding_args = BufferShardingArgs(std::nullopt, TensorMemoryLayout::INTERLEAVED),
            .bottom_up = std::nullopt,
            .sub_device_id = std::nullopt,
        };
        auto input_mesh_buffer = distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{.size = buffer_size}, buffer_config, mesh_device.get());

        // Initialize buffer with data (arange equivalent: 0, 1, 2, ..., num_elems-1)
        std::vector<uint32_t> host_data(num_elems);
        std::iota(host_data.begin(), host_data.end(), 0u);

        // Write data to device buffer
        distributed::EnqueueWriteMeshBuffer(mesh_device->mesh_command_queue(), input_mesh_buffer, host_data, true);

        // Extract buffer pointer for metal-level operations
        Buffer* input_buffer = input_mesh_buffer->get_reference_buffer();

        // Launch kernels:
        if (automapped_pipeline.has_value()) {
            // The sender receives the stage-3 loopback directly through recv_socket.
            tt::tt_metal::send_async(
                mesh_device.get(),
                input_buffer,
                tt::DataFormat::UInt32,
                intermed_send,
                recv_socket,
                latency_measurement_address,
                num_iterations,
                enable_correctness_check);
            tt::tt_metal::socket_forward(
                mesh_device.get(), intermed_recv, send_socket, XFER_SIZE, latency_measurement_address, num_iterations);
        } else {
            // Legacy layouts return through an extra local forwarding stage.
            auto [intermed_send_2, intermed_recv_2] = create_intermed_socket_pair(my_recv, start_coord);
            tt::tt_metal::send_async(
                mesh_device.get(),
                input_buffer,
                tt::DataFormat::UInt32,
                intermed_send,
                intermed_recv_2,
                latency_measurement_address,
                num_iterations,
                enable_correctness_check);
            tt::tt_metal::socket_forward(
                mesh_device.get(), intermed_recv, send_socket, XFER_SIZE, latency_measurement_address, num_iterations);
            tt::tt_metal::socket_forward(
                mesh_device.get(),
                recv_socket,
                intermed_send_2,
                XFER_SIZE,
                latency_measurement_address,
                num_iterations);
        }
    } else {
        // Non-start ranks: receive from upstream, forward locally, send to downstream

        // Cross-mesh recv from upstream: upstream_exit -> my_entry
        auto bwd_connection = distributed::SocketConnection(
            distributed::MeshCoreCoord(upstream_exit, logical_coord),
            distributed::MeshCoreCoord(my_entry, logical_coord));
        auto recv_socket_config = distributed::SocketConfig(
            {bwd_connection}, socket_mem_config, upstream_mesh_id, my_mesh_id, distributed_context);
        auto recv_socket = distributed::MeshSocket(mesh_device, recv_socket_config);

        // Cross-mesh send to downstream: my_exit -> downstream_entry
        auto fwd_connection = distributed::SocketConnection(
            distributed::MeshCoreCoord(my_exit, logical_coord),
            distributed::MeshCoreCoord(downstream_entry, logical_coord));
        auto send_socket_config = distributed::SocketConfig(
            {fwd_connection}, socket_mem_config, my_mesh_id, downstream_mesh_id, distributed_context);
        auto send_socket = distributed::MeshSocket(mesh_device, send_socket_config);

        // Local intermed: my_entry -> my_exit
        auto [intermed_send, intermed_recv] = create_intermed_socket_pair(my_entry, my_exit);

        // Create device buffer using metal-level API
        distributed::DeviceLocalBufferConfig buffer_config = {
            .page_size = page_size,
            .buffer_type = BufferType::DRAM,
            .sharding_args = BufferShardingArgs(std::nullopt, TensorMemoryLayout::INTERLEAVED),
            .bottom_up = std::nullopt,
            .sub_device_id = std::nullopt,
        };
        auto output_mesh_buffer = distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{.size = buffer_size}, buffer_config, mesh_device.get());

        // Launch kernels: forward from upstream to downstream through local intermed
        tt::tt_metal::socket_forward(
            mesh_device.get(), recv_socket, intermed_send, XFER_SIZE, latency_measurement_address, num_iterations);
        tt::tt_metal::socket_forward(
            mesh_device.get(), intermed_recv, send_socket, XFER_SIZE, latency_measurement_address, num_iterations);
    }

    barrier();
    if (is_pipeline_start) {
        const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
        auto start_device_id = mesh_device->get_device(start_coord)->id();
        auto start_core_coord = mesh_device->worker_core_from_logical_core(logical_coord);
        std::vector<uint64_t> latencies = std::vector<uint64_t>(num_iterations, 0);
        uint32_t base_addr = latency_measurement_address;
        cluster.read_core(
            latencies.data(),
            sizeof(uint64_t) * num_iterations,
            tt_cxy_pair(start_device_id, start_core_coord),
            base_addr);
        // Skip first iteration (often an outlier due to cold start)
        const uint32_t latency_iterations_for_avg = num_iterations > 1 ? num_iterations - 1 : 1;
        double avg_latency_cycles = 0.0;
        for (uint32_t i = 1; i < num_iterations; i++) {
            avg_latency_cycles += static_cast<double>(latencies[i]);
        }
        avg_latency_cycles /= latency_iterations_for_avg;
        double freq_mhz = static_cast<double>(cluster.get_device_aiclk(start_device_id));
        double avg_latency_us = (avg_latency_cycles / (freq_mhz * 1e6)) * 1e6;
        double avd_latency_per_stage_us = avg_latency_us / static_cast<double>(num_stages);
        log_info(tt::LogTest, "Average latency in cycles: {:.2f}", avg_latency_cycles);
        log_info(tt::LogTest, "Average latency in microseconds: {:.2f}", avg_latency_us);
        log_info(tt::LogTest, "Average latency per stage in microseconds: {:.2f}", avd_latency_per_stage_us);
    }
}

TEST_F(BlitzDecodePipelineFixture, SendRecvPipeline) {
    constexpr uint32_t NUM_ITERATIONS = 500;
    auto num_ranks = *MetalContext::instance().global_distributed_context().size();
    auto pipeline_type = pipeline_type_from_num_ranks(num_ranks);
    ASSERT_TRUE(pipeline_type.has_value()) << "Unsupported rank count";
    // SendRecv (loopback) config is not defined for SUPERPOD_2_POD (32 ranks).
    if (*pipeline_type == PipelineType::SUPERPOD_2_POD) {
        GTEST_SKIP() << "SendRecv not configured for 32-rank (SUPERPOD_2_POD).";
    }
    run_single_galaxy_pipeline(mesh_device_, *pipeline_type, NUM_ITERATIONS, /*enable_correctness_check=*/false);
}

TEST_F(BlitzDecodePipelineFixture, SendRecvPipelineWithCorrectnessCheck) {
    constexpr uint32_t NUM_ITERATIONS = 500;
    auto num_ranks = *MetalContext::instance().global_distributed_context().size();
    auto pipeline_type = pipeline_type_from_num_ranks(num_ranks);
    ASSERT_TRUE(pipeline_type.has_value()) << "Unsupported rank count";
    if (*pipeline_type == PipelineType::SUPERPOD_2_POD) {
        GTEST_SKIP() << "SendRecv not configured for 32-rank (SUPERPOD_2_POD).";
    }
    run_single_galaxy_pipeline(mesh_device_, *pipeline_type, NUM_ITERATIONS, /*enable_correctness_check=*/true);
}

// ─── Rate (throughput) pipeline test ─────────────────────────────────────────
// Linear pipeline (no loopback): data flows one-way through pipeline stages.
// Measures sustained pipeline throughput by pushing data for many iterations.

// Multi-host rate pipeline test helper. Timing is done on the host side using
// std::chrono, matching the original TTNN implementation.
void run_single_galaxy_rate_pipeline(
    std::shared_ptr<distributed::MeshDevice>& mesh_device,
    PipelineType pipeline_type,
    uint32_t num_iterations,
    bool enable_correctness_check) {
    constexpr uint32_t XFER_SIZE = 14 * 1024;

    const auto& distributed_context = tt_metal::distributed::multihost::DistributedContext::get_current_world();
    const auto my_rank = *distributed_context->rank();
    const auto num_ranks = static_cast<uint32_t>(*distributed_context->size());

    const auto logical_coord = CoreCoord(0, 0);
    const uint32_t socket_fifo_size = XFER_SIZE * 16;

    std::optional<LogicalPipelineConfig> automapped_pipeline;
    std::vector<LogicalPipelineStageConfig> pipeline_stages;
    if (pipeline_type == PipelineType::SINGLE_GALAXY) {
        automapped_pipeline = build_automapped_single_galaxy_pipeline();
        pipeline_stages = automapped_pipeline->stages;
    } else {
        auto physical_system_descriptor = create_physical_system_descriptor();
        auto asic_id_to_mesh_coord = get_asic_id_to_mesh_coord_map(mesh_device);
        auto physical_config = get_physical_pipeline_config(pipeline_type, false);
        pipeline_stages = build_pipeline(physical_system_descriptor, asic_id_to_mesh_coord, physical_config);
    }

    const auto my_stage_it = std::find_if(
        pipeline_stages.begin(), pipeline_stages.end(), [&](const auto& stage) { return stage.rank == my_rank; });
    TT_FATAL(my_stage_it != pipeline_stages.end(), "MPI rank {} has no pipeline stage", my_rank);
    const uint32_t my_stage = static_cast<uint32_t>(std::distance(pipeline_stages.begin(), my_stage_it));
    const uint32_t downstream_stage = my_stage + 1;
    const uint32_t upstream_stage = my_stage - 1;  // wraps for stage 0, but unused there
    const uint32_t downstream_rank = automapped_pipeline.has_value() && downstream_stage < pipeline_stages.size()
                                         ? pipeline_stages[downstream_stage].rank
                                         : my_rank + 1;
    const uint32_t upstream_rank =
        automapped_pipeline.has_value() && my_stage > 0 ? pipeline_stages[upstream_stage].rank : my_rank - 1;

    const auto& global_bindings =
        tt::tt_metal::MetalContext::instance().get_control_plane().get_global_logical_bindings();
    const tt::tt_fabric::MeshId my_mesh_id = std::get<0>(global_bindings.at(distributed::multihost::Rank(my_rank)));

    const distributed::SocketMemoryConfig socket_mem_config(BufferType::L1, socket_fifo_size);

    const uint32_t num_elems = XFER_SIZE / sizeof(uint32_t);
    const DeviceAddr buffer_size = XFER_SIZE;
    const DeviceAddr page_size = XFER_SIZE;

    auto create_intermed_socket_pair = [&](const distributed::MeshCoordinate& sender_coord,
                                           const distributed::MeshCoordinate& recv_coord) {
        auto connection = distributed::SocketConnection(
            distributed::MeshCoreCoord(sender_coord, logical_coord),
            distributed::MeshCoreCoord(recv_coord, logical_coord));
        auto config = distributed::SocketConfig({connection}, socket_mem_config);
        return distributed::MeshSocket::create_socket_pair(mesh_device, mesh_device, config);
    };

    auto barrier = [&]() {
        Synchronize(*mesh_device, std::nullopt);
        distributed_context->barrier();
    };

    const bool is_pipeline_start = (my_stage == 0);
    const bool is_pipeline_end = (my_stage == num_ranks - 1);

    auto my_entry = pipeline_stages[my_stage].entry_node_coord;
    auto my_exit = pipeline_stages[my_stage].exit_node_coord;

    if (is_pipeline_start) {
        // Sender: start_coord -> my_exit (local), then my_exit -> downstream_entry (cross-mesh)
        const auto start_coord = automapped_pipeline.has_value() ? automapped_pipeline->start_coord : my_entry;
        auto [my_sender, downstream_recv] = get_connecting_coords(pipeline_stages, my_stage, downstream_stage);

        auto [intermed_send, intermed_recv] = create_intermed_socket_pair(start_coord, my_sender);

        const tt::tt_fabric::MeshId downstream_mesh_id =
            std::get<0>(global_bindings.at(distributed::multihost::Rank(downstream_rank)));
        auto fwd_connection = distributed::SocketConnection(
            distributed::MeshCoreCoord(my_sender, logical_coord),
            distributed::MeshCoreCoord(downstream_recv, logical_coord));
        auto send_socket_config = distributed::SocketConfig(
            {fwd_connection}, socket_mem_config, my_mesh_id, downstream_mesh_id, distributed_context);
        auto send_socket = distributed::MeshSocket(mesh_device, send_socket_config);

        // Create device buffer
        distributed::DeviceLocalBufferConfig buffer_config = {
            .page_size = page_size,
            .buffer_type = BufferType::DRAM,
            .sharding_args = BufferShardingArgs(std::nullopt, TensorMemoryLayout::INTERLEAVED),
            .bottom_up = std::nullopt,
            .sub_device_id = std::nullopt,
        };
        auto input_mesh_buffer = distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{.size = buffer_size}, buffer_config, mesh_device.get());
        std::vector<uint32_t> host_data(num_elems);
        std::iota(host_data.begin(), host_data.end(), 0u);
        distributed::EnqueueWriteMeshBuffer(mesh_device->mesh_command_queue(), input_mesh_buffer, host_data, true);
        Buffer* input_buffer = input_mesh_buffer->get_reference_buffer();

        // Warmup: run a small number of iterations to trigger kernel compilation and caching.
        // num_iterations is a runtime arg so the same compiled kernel is reused for the timed run.
        constexpr uint32_t WARMUP_ITERATIONS = 8;
        tt::tt_metal::send_async_rate(
            mesh_device.get(), input_buffer, tt::DataFormat::UInt32, intermed_send, WARMUP_ITERATIONS);
        tt::tt_metal::socket_forward_rate(mesh_device.get(), intermed_recv, send_socket, XFER_SIZE, WARMUP_ITERATIONS);
        barrier();
        log_info(tt::LogTest, "Warmup complete ({} iterations)", WARMUP_ITERATIONS);

        // Host-side timing: record start time after warmup, just before launching timed kernels
        auto start_time = std::chrono::duration_cast<std::chrono::microseconds>(
                              std::chrono::high_resolution_clock::now().time_since_epoch())
                              .count();

        // Launch rate-mode kernels:
        // - send_async_rate on start_coord: sends data to local intermed
        // - socket_forward_rate on my_exit: forwards from local intermed to cross-mesh
        tt::tt_metal::send_async_rate(
            mesh_device.get(), input_buffer, tt::DataFormat::UInt32, intermed_send, num_iterations);
        tt::tt_metal::socket_forward_rate(mesh_device.get(), intermed_recv, send_socket, XFER_SIZE, num_iterations);
        barrier();

        auto end_time = std::chrono::duration_cast<std::chrono::microseconds>(
                            std::chrono::high_resolution_clock::now().time_since_epoch())
                            .count();

        double elapsed_us = static_cast<double>(end_time - start_time);
        double total_bytes = static_cast<double>(num_iterations) * XFER_SIZE;
        double rate_gbps = (total_bytes * 8.0) / (elapsed_us * 1e3);

        log_info(tt::LogTest, "Rate pipeline: {} iterations, {} bytes/iter", num_iterations, XFER_SIZE);
        log_info(
            tt::LogTest,
            "Sender host-side elapsed: {:.2f} us, total bytes: {:.2f} MB, {:.4f} Gbps ({:.2f} Mbps)",
            elapsed_us,
            total_bytes / (1024.0 * 1024.0),
            rate_gbps,
            rate_gbps * 1e3);
    } else if (is_pipeline_end) {
        // Receiver: upstream_exit -> my_entry (cross-mesh), then my_entry -> end_coord (local)
        auto upstream_exit = pipeline_stages[upstream_stage].exit_node_coord;
        const auto& end_coord = my_exit;

        const tt::tt_fabric::MeshId upstream_mesh_id =
            std::get<0>(global_bindings.at(distributed::multihost::Rank(upstream_rank)));
        auto bwd_connection = distributed::SocketConnection(
            distributed::MeshCoreCoord(upstream_exit, logical_coord),
            distributed::MeshCoreCoord(my_entry, logical_coord));
        auto recv_socket_config = distributed::SocketConfig(
            {bwd_connection}, socket_mem_config, upstream_mesh_id, my_mesh_id, distributed_context);
        auto recv_socket = distributed::MeshSocket(mesh_device, recv_socket_config);

        auto [intermed_send, intermed_recv] = create_intermed_socket_pair(my_entry, end_coord);

        // Warmup
        constexpr uint32_t WARMUP_ITERATIONS = 8;
        tt::tt_metal::socket_forward_rate(mesh_device.get(), recv_socket, intermed_send, XFER_SIZE, WARMUP_ITERATIONS);
        tt::tt_metal::recv_async_rate(mesh_device.get(), intermed_recv, XFER_SIZE, WARMUP_ITERATIONS, false);
        barrier();
        log_info(tt::LogTest, "Warmup complete ({} iterations)", WARMUP_ITERATIONS);

        // Host-side timing: record start time after warmup
        auto start_time = std::chrono::duration_cast<std::chrono::microseconds>(
                              std::chrono::high_resolution_clock::now().time_since_epoch())
                              .count();

        // Launch rate-mode kernels:
        // - socket_forward_rate on my_entry: forwards from cross-mesh recv to local intermed
        // - recv_async_rate on end_coord: drains data from local intermed
        tt::tt_metal::socket_forward_rate(mesh_device.get(), recv_socket, intermed_send, XFER_SIZE, num_iterations);
        tt::tt_metal::recv_async_rate(
            mesh_device.get(), intermed_recv, XFER_SIZE, num_iterations, enable_correctness_check);
        barrier();

        auto end_time = std::chrono::duration_cast<std::chrono::microseconds>(
                            std::chrono::high_resolution_clock::now().time_since_epoch())
                            .count();

        double elapsed_us = static_cast<double>(end_time - start_time);
        double total_bytes = static_cast<double>(num_iterations) * XFER_SIZE;
        double rate_gbps = (total_bytes * 8.0) / (elapsed_us * 1e3);

        log_info(tt::LogTest, "Rate pipeline: {} iterations, {} bytes/iter", num_iterations, XFER_SIZE);
        log_info(
            tt::LogTest,
            "Receiver host-side elapsed: {:.2f} us, total bytes: {:.2f} MB, {:.4f} Gbps ({:.2f} Mbps)",
            elapsed_us,
            total_bytes / (1024.0 * 1024.0),
            rate_gbps,
            rate_gbps * 1e3);
    } else {
        // Intermediate: upstream_exit -> my_entry (cross-mesh), local forward, my_exit -> downstream_entry (cross-mesh)
        auto upstream_exit = pipeline_stages[upstream_stage].exit_node_coord;
        auto downstream_entry = pipeline_stages[downstream_stage].entry_node_coord;

        const tt::tt_fabric::MeshId upstream_mesh_id =
            std::get<0>(global_bindings.at(distributed::multihost::Rank(upstream_rank)));
        const tt::tt_fabric::MeshId downstream_mesh_id =
            std::get<0>(global_bindings.at(distributed::multihost::Rank(downstream_rank)));

        auto bwd_connection = distributed::SocketConnection(
            distributed::MeshCoreCoord(upstream_exit, logical_coord),
            distributed::MeshCoreCoord(my_entry, logical_coord));
        auto recv_socket_config = distributed::SocketConfig(
            {bwd_connection}, socket_mem_config, upstream_mesh_id, my_mesh_id, distributed_context);
        auto recv_socket = distributed::MeshSocket(mesh_device, recv_socket_config);

        auto fwd_connection = distributed::SocketConnection(
            distributed::MeshCoreCoord(my_exit, logical_coord),
            distributed::MeshCoreCoord(downstream_entry, logical_coord));
        auto send_socket_config = distributed::SocketConfig(
            {fwd_connection}, socket_mem_config, my_mesh_id, downstream_mesh_id, distributed_context);
        auto send_socket = distributed::MeshSocket(mesh_device, send_socket_config);

        auto [intermed_send, intermed_recv] = create_intermed_socket_pair(my_entry, my_exit);

        // Warmup
        constexpr uint32_t WARMUP_ITERATIONS = 8;
        tt::tt_metal::socket_forward_rate(mesh_device.get(), recv_socket, intermed_send, XFER_SIZE, WARMUP_ITERATIONS);
        tt::tt_metal::socket_forward_rate(mesh_device.get(), intermed_recv, send_socket, XFER_SIZE, WARMUP_ITERATIONS);
        barrier();

        // Launch rate-mode kernels: forward from upstream to downstream through local intermed
        tt::tt_metal::socket_forward_rate(mesh_device.get(), recv_socket, intermed_send, XFER_SIZE, num_iterations);
        tt::tt_metal::socket_forward_rate(mesh_device.get(), intermed_recv, send_socket, XFER_SIZE, num_iterations);
        barrier();
    }
}

TEST_F(BlitzDecodePipelineFixture, RatePipeline) {
    constexpr uint32_t NUM_ITERATIONS = 100000;
    auto num_ranks = *MetalContext::instance().global_distributed_context().size();
    auto pipeline_type = pipeline_type_from_num_ranks(num_ranks);
    ASSERT_TRUE(pipeline_type.has_value()) << "Unsupported rank count";
    run_single_galaxy_rate_pipeline(mesh_device_, *pipeline_type, NUM_ITERATIONS, /*enable_correctness_check=*/false);
}

TEST_F(BlitzDecodePipelineFixture, RatePipelineWithCorrectnessCheck) {
    constexpr uint32_t NUM_ITERATIONS = 100;
    auto num_ranks = *MetalContext::instance().global_distributed_context().size();
    auto pipeline_type = pipeline_type_from_num_ranks(num_ranks);
    ASSERT_TRUE(pipeline_type.has_value()) << "Unsupported rank count";
    run_single_galaxy_rate_pipeline(mesh_device_, *pipeline_type, NUM_ITERATIONS, /*enable_correctness_check=*/true);
}
}  // namespace tt::tt_metal
