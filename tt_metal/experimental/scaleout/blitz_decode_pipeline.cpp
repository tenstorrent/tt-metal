// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/experimental/scaleout/blitz_decode_pipeline.hpp"

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include <tt-metalium/base_types.hpp>
#include <tt-metalium/distributed_context.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt_stl/span.hpp>

#include "tt_metal/impl/context/metal_context.hpp"
#include "tt_metal/fabric/physical_system_descriptor.hpp"
#include "tt_metal/llrt/tt_cluster.hpp"

namespace tt::tt_metal::experimental::scaleout {

namespace {

struct PhysicalPipelineStageConfig {
    uint32_t tray_id;
    uint32_t entry_node_asic_location;
    uint32_t exit_node_asic_location;
};

std::unordered_map<tt::tt_metal::AsicID, distributed::MeshCoordinate> get_asic_id_to_mesh_coord_map(
    const distributed::MeshDevice& mesh_device) {
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    std::unordered_map<tt::tt_metal::AsicID, distributed::MeshCoordinate> asic_id_to_mesh_coord_map;

    for (const auto& coord : distributed::MeshCoordinateRange(mesh_device.shape())) {
        tt_fabric::FabricNodeId fabric_node_id = mesh_device.get_fabric_node_id(coord);
        tt_metal::AsicID asic_id = control_plane.get_asic_id_from_fabric_node_id(fabric_node_id);
        asic_id_to_mesh_coord_map.emplace(asic_id, coord);
    }

    auto& distributed_context = tt_metal::distributed::multihost::DistributedContext::get_current_world();
    for (auto rank = 0; rank < *(distributed_context->size()); rank++) {
        if (rank == *(distributed_context->rank())) {
            std::size_t num_entries = asic_id_to_mesh_coord_map.size();
            distributed_context->broadcast(
                tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&num_entries), sizeof(num_entries)),
                distributed::multihost::Rank{rank});
            for (auto& [asic_id, mesh_coord] : asic_id_to_mesh_coord_map) {
                distributed_context->broadcast(
                    tt::stl::Span<std::byte>(
                        reinterpret_cast<std::byte*>(const_cast<tt_metal::AsicID*>(&asic_id)), sizeof(asic_id)),
                    distributed::multihost::Rank{rank});
                distributed_context->broadcast(
                    tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&(mesh_coord[0])), sizeof(mesh_coord[0])),
                    distributed::multihost::Rank{rank});
                distributed_context->broadcast(
                    tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&(mesh_coord[1])), sizeof(mesh_coord[1])),
                    distributed::multihost::Rank{rank});
            }
        } else {
            std::size_t num_entries = 0;
            distributed_context->broadcast(
                tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&num_entries), sizeof(num_entries)),
                distributed::multihost::Rank{rank});
            for (std::size_t i = 0; i < num_entries; i++) {
                tt_metal::AsicID asic_id;
                distributed::MeshCoordinate mesh_coord = distributed::MeshCoordinate(0, 0);
                distributed_context->broadcast(
                    tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&asic_id), sizeof(asic_id)),
                    distributed::multihost::Rank{rank});
                distributed_context->broadcast(
                    tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&(mesh_coord[0])), sizeof(mesh_coord[0])),
                    distributed::multihost::Rank{rank});
                distributed_context->broadcast(
                    tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&(mesh_coord[1])), sizeof(mesh_coord[1])),
                    distributed::multihost::Rank{rank});
                asic_id_to_mesh_coord_map.emplace(asic_id, mesh_coord);
            }
        }
    }

    return asic_id_to_mesh_coord_map;
}

PhysicalSystemDescriptor create_physical_system_descriptor() {
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto& distributed_context = tt::tt_metal::MetalContext::instance().get_distributed_context_ptr();
    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    const auto& rtoptions = tt::tt_metal::MetalContext::instance().rtoptions();
    constexpr bool run_discovery = true;
    const auto& driver = cluster.get_driver();

    return tt::tt_metal::PhysicalSystemDescriptor(driver, distributed_context, &hal, rtoptions, run_discovery);
}

std::vector<BlitzDecodePipelineStage> build_pipeline(
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    const std::unordered_map<tt::tt_metal::AsicID, distributed::MeshCoordinate>& asic_id_to_mesh_coord) {
    std::vector<PhysicalPipelineStageConfig> physical_pipeline_stage_configs = {
        {.tray_id = 3, .entry_node_asic_location = 7, .exit_node_asic_location = 3},
        {.tray_id = 1, .entry_node_asic_location = 3, .exit_node_asic_location = 7},
        {.tray_id = 3, .entry_node_asic_location = 7, .exit_node_asic_location = 4},
        {.tray_id = 4, .entry_node_asic_location = 4, .exit_node_asic_location = 6},
        {.tray_id = 2, .entry_node_asic_location = 6, .exit_node_asic_location = 2},
        {.tray_id = 4, .entry_node_asic_location = 2, .exit_node_asic_location = 5},
        {.tray_id = 3, .entry_node_asic_location = 5, .exit_node_asic_location = 2},
        {.tray_id = 1, .entry_node_asic_location = 2, .exit_node_asic_location = 6},
        {.tray_id = 3, .entry_node_asic_location = 6, .exit_node_asic_location = 4},
        {.tray_id = 4, .entry_node_asic_location = 4, .exit_node_asic_location = 7},
        {.tray_id = 2, .entry_node_asic_location = 7, .exit_node_asic_location = 3},
        {.tray_id = 4, .entry_node_asic_location = 3, .exit_node_asic_location = 7},
        {.tray_id = 2, .entry_node_asic_location = 7, .exit_node_asic_location = 4},
        {.tray_id = 1, .entry_node_asic_location = 4, .exit_node_asic_location = 1},
        {.tray_id = 2, .entry_node_asic_location = 1, .exit_node_asic_location = 4},
        {.tray_id = 1, .entry_node_asic_location = 4, .exit_node_asic_location = 1},

        {.tray_id = 2, .entry_node_asic_location = 1, .exit_node_asic_location = 7},
        {.tray_id = 4, .entry_node_asic_location = 7, .exit_node_asic_location = 3},
        {.tray_id = 2, .entry_node_asic_location = 3, .exit_node_asic_location = 7},
        {.tray_id = 4, .entry_node_asic_location = 7, .exit_node_asic_location = 4},
        {.tray_id = 3, .entry_node_asic_location = 4, .exit_node_asic_location = 6},
        {.tray_id = 1, .entry_node_asic_location = 6, .exit_node_asic_location = 2},
        {.tray_id = 3, .entry_node_asic_location = 2, .exit_node_asic_location = 6},
        {.tray_id = 1, .entry_node_asic_location = 6, .exit_node_asic_location = 1},
        {.tray_id = 2, .entry_node_asic_location = 1, .exit_node_asic_location = 7},
        {.tray_id = 4, .entry_node_asic_location = 7, .exit_node_asic_location = 3},
        {.tray_id = 2, .entry_node_asic_location = 3, .exit_node_asic_location = 7},
        {.tray_id = 4, .entry_node_asic_location = 7, .exit_node_asic_location = 4},
        {.tray_id = 3, .entry_node_asic_location = 4, .exit_node_asic_location = 6},
        {.tray_id = 1, .entry_node_asic_location = 6, .exit_node_asic_location = 2},
        {.tray_id = 3, .entry_node_asic_location = 2, .exit_node_asic_location = 6},
        {.tray_id = 1, .entry_node_asic_location = 6, .exit_node_asic_location = 1},

        {.tray_id = 2, .entry_node_asic_location = 1, .exit_node_asic_location = 4},
        {.tray_id = 1, .entry_node_asic_location = 4, .exit_node_asic_location = 1},
        {.tray_id = 2, .entry_node_asic_location = 1, .exit_node_asic_location = 4},
        {.tray_id = 1, .entry_node_asic_location = 4, .exit_node_asic_location = 6},
        {.tray_id = 3, .entry_node_asic_location = 6, .exit_node_asic_location = 2},
        {.tray_id = 1, .entry_node_asic_location = 2, .exit_node_asic_location = 6},
        {.tray_id = 3, .entry_node_asic_location = 6, .exit_node_asic_location = 4},
        {.tray_id = 4, .entry_node_asic_location = 4, .exit_node_asic_location = 7},
        {.tray_id = 2, .entry_node_asic_location = 7, .exit_node_asic_location = 3},
        {.tray_id = 4, .entry_node_asic_location = 3, .exit_node_asic_location = 5},
        {.tray_id = 3, .entry_node_asic_location = 5, .exit_node_asic_location = 2},
        {.tray_id = 1, .entry_node_asic_location = 2, .exit_node_asic_location = 6},
        {.tray_id = 3, .entry_node_asic_location = 6, .exit_node_asic_location = 4},
        {.tray_id = 4, .entry_node_asic_location = 4, .exit_node_asic_location = 7},
        {.tray_id = 2, .entry_node_asic_location = 7, .exit_node_asic_location = 3},
        {.tray_id = 4, .entry_node_asic_location = 3, .exit_node_asic_location = 1},

        {.tray_id = 3, .entry_node_asic_location = 1, .exit_node_asic_location = 7}};

    const auto num_procs = *(tt::tt_metal::MetalContext::instance().get_distributed_context_ptr()->size());
    std::vector<BlitzDecodePipelineStage> logical_pipeline_stage_configs;
    logical_pipeline_stage_configs.reserve(physical_pipeline_stage_configs.size());

    for (std::size_t stage_index = 0; stage_index < physical_pipeline_stage_configs.size(); stage_index++) {
        const auto& stage_config = physical_pipeline_stage_configs[stage_index];
        auto stage_hostname = physical_system_descriptor.get_hostname_for_rank(stage_index % num_procs);
        auto entry_node_asic_id = physical_system_descriptor.get_asic_id(
            stage_hostname,
            tt::tt_metal::TrayID(stage_config.tray_id),
            tt::tt_metal::ASICLocation(stage_config.entry_node_asic_location));
        auto exit_node_asic_id = physical_system_descriptor.get_asic_id(
            stage_hostname,
            tt::tt_metal::TrayID(stage_config.tray_id),
            tt::tt_metal::ASICLocation(stage_config.exit_node_asic_location));
        logical_pipeline_stage_configs.emplace_back(BlitzDecodePipelineStage{
            .stage_index = stage_index,
            .entry_node_coord = asic_id_to_mesh_coord.at(entry_node_asic_id),
            .exit_node_coord = asic_id_to_mesh_coord.at(exit_node_asic_id)});
    }

    return logical_pipeline_stage_configs;
}

}  // namespace

std::vector<BlitzDecodePipelineStage> generate_blitz_decode_pipeline(const distributed::MeshDevice& mesh_device) {
    auto physical_system_descriptor = create_physical_system_descriptor();
    auto asic_id_to_mesh_coord = get_asic_id_to_mesh_coord_map(mesh_device);
    return build_pipeline(physical_system_descriptor, asic_id_to_mesh_coord);
}

}  // namespace tt::tt_metal::experimental::scaleout
