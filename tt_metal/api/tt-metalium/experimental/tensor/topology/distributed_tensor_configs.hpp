// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// NOTE: This file is a copy of TTNN's ttnn/api/ttnn/distributed/distributed_configs.hpp
// at commit 9f3856801448f589170defe41b23c8b9b43e33a2, with modifications to
// use experimental tensor types.

#pragma once

#include <optional>
#include <ostream>
#include <variant>
#include <tt_stl/small_vector.hpp>
#include <tt-metalium/mesh_coord.hpp>

namespace tt::tt_metal::distributed {

struct MeshMapperConfig {
    // Specifies the tensor should be replicated across devices.
    struct Replicate {
        bool operator==(const Replicate&) const = default;
    };

    // Specifies the tensor should be sharded along the specified dimension.
    struct Shard {
        int dim = 0;
        bool operator==(const Shard&) const = default;
    };

    using Placement = std::variant<Replicate, Shard>;
    ttsl::SmallVector<Placement> placements;

    std::optional<MeshShape> mesh_shape_override = std::nullopt;
};

bool operator==(const MeshMapperConfig::Placement& lhs, const MeshMapperConfig::Placement& rhs);
std::ostream& operator<<(std::ostream& os, const MeshMapperConfig::Placement& placement);
std::ostream& operator<<(std::ostream& os, const MeshMapperConfig& config);

struct MeshComposerConfig {
    // Specifies dimension of the tensor to concatenate.
    ttsl::SmallVector<int> dims;

    std::optional<MeshShape> mesh_shape_override = std::nullopt;
};

std::ostream& operator<<(std::ostream& os, const MeshComposerConfig& config);

}  // namespace tt::tt_metal::distributed
