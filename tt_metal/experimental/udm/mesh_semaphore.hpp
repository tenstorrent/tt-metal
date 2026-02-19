// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "tt_metal/experimental/udm/types.hpp"
#include "tt_metal/experimental/udm/mesh_builder.hpp"
#include "tt_metal/api/tt-metalium/global_semaphore.hpp"

namespace tt::tt_metal::experimental::udm {

/**
 * @brief Create a semaphore across all global cores in the mesh
 *
 * @param builder The MeshBuilder containing mesh topology information
 * @param program The MeshProgram to add the semaphore to
 * @param initial_value Initial value of the semaphore
 * @return MeshSemaphoreHandle Map of grid_id to semaphore address for each grid
 */
MeshSemaphoreHandle CreateMeshSemaphore(MeshBuilder& builder, MeshProgram& program, uint32_t initial_value);

}  // namespace tt::tt_metal::experimental::udm
