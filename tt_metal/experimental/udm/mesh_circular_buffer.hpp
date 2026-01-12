// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include "tt_metal/experimental/udm/types.hpp"
#include "tt_metal/experimental/udm/mesh_program.hpp"
#include "tt_metal/api/tt-metalium/circular_buffer_config.hpp"

namespace tt::tt_metal::experimental::udm {

/**
 * @brief Create a circular buffer across all global cores in the mesh
 *
 * @param builder The MeshBuilder containing mesh topology information
 * @param program The MeshProgram to add the circular buffer to
 * @param config Circular buffer configuration
 * @return MeshCBHandle Map of grid_id to circular buffer handle for each grid
 */
MeshCBHandle CreateMeshCircularBuffer(
    MeshBuilder& builder, MeshProgram& program, const tt::tt_metal::CircularBufferConfig& config);

}  // namespace tt::tt_metal::experimental::udm
