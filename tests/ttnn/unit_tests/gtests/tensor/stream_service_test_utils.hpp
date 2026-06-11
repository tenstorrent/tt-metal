// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt_stl/small_vector.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <ttnn/api/ttnn/distributed/distributed_configs.hpp>

namespace ttnn::distributed::test {

// Fully-replicated placements sized to a mesh's dimensionality (identity on a
// 1x1 mesh; full tensor on every device otherwise). Shared by the H2D and D2D
// stream-service gtests, which both distribute their inputs with this mapping.
inline ttsl::SmallVector<tt::tt_metal::distributed::MeshMapperConfig::Placement> replicate_all(
    const tt::tt_metal::distributed::MeshDevice& mesh) {
    return ttsl::SmallVector<tt::tt_metal::distributed::MeshMapperConfig::Placement>(
        mesh.shape().dims(), tt::tt_metal::distributed::MeshMapperConfig::Replicate{});
}

}  // namespace ttnn::distributed::test
