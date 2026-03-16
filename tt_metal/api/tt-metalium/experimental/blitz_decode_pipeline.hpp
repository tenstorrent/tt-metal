// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <vector>

#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>

namespace tt::tt_metal::experimental::blitz {

struct BlitzDecodePipelineStage {
    std::size_t stage_index;
    std::size_t mesh_id;
    ::tt::tt_metal::distributed::MeshCoordinate entry_node_coord;
    ::tt::tt_metal::distributed::MeshCoordinate exit_node_coord;
};

std::vector<BlitzDecodePipelineStage> generate_blitz_decode_pipeline(
    const ::tt::tt_metal::distributed::MeshDevice& mesh_device);

}  // namespace tt::tt_metal::experimental::blitz
