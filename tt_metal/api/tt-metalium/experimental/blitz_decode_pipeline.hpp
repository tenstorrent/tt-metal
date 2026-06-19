// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>

namespace tt::tt_metal::experimental::blitz {

struct BlitzDecodePipelineStage {
    std::size_t stage_index;
    ::tt::tt_metal::distributed::MeshCoordinate entry_node_coord;
    ::tt::tt_metal::distributed::MeshCoordinate exit_node_coord;
};

std::vector<BlitzDecodePipelineStage> generate_blitz_decode_pipeline(bool initialize_loopback = true);

}  // namespace tt::tt_metal::experimental::blitz
