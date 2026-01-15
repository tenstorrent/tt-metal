// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/experimental/udm/mesh_semaphore.hpp"
#include "tt_metal/experimental/udm/mesh_builder.hpp"
#include "tt_metal/experimental/udm/mesh_program.hpp"
#include "tt_metal/api/tt-metalium/host_api.hpp"
#include "tt_metal/api/tt-metalium/distributed.hpp"
#include <tt_stl/assert.hpp>

namespace tt::tt_metal::experimental::udm {

MeshSemaphoreHandle CreateMeshSemaphore(MeshBuilder& builder, MeshProgram& /* program */, uint32_t initial_value) {
    // Get all gcores from mesh builder
    const auto& all_gcores_vec = builder.get_all_gcores_in_mesh();
    // Get grid IDs and local CoreRangeSets from gcores
    auto grid_to_cores = builder.get_grid_core_range_set_from_gcores(all_gcores_vec);

    // For GlobalSemaphore, we need a single local CoreRangeSet that covers all cores on each device
    // All grids should have the same local core layout, so we can use any grid's CoreRangeSet
    TT_FATAL(!grid_to_cores.empty(), "No grids found in mesh");

    // Get the local CoreRangeSet from the first grid (should be same for all)
    CoreRangeSet local_cores = grid_to_cores.begin()->second;

    // Create a GlobalSemaphore across all devices in the mesh
    auto* mesh_device = builder.mesh_device();
    auto global_semaphore = tt::tt_metal::CreateGlobalSemaphore(mesh_device, local_cores, initial_value);

    // Synchronize mesh device after creating global semaphore
    tt::tt_metal::distributed::Synchronize(mesh_device, std::nullopt, {});

    // Return handle that keeps the GlobalSemaphore alive
    return MeshSemaphoreHandle(std::move(global_semaphore));
}

}  // namespace tt::tt_metal::experimental::udm
