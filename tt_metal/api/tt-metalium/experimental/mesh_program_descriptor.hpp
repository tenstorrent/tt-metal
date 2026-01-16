// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include <vector>
#include <utility>

namespace tt::tt_metal::experimental {

struct MeshProgramDescriptor {
    using MeshPrograms = std::vector<std::pair<distributed::MeshCoordinateRange, ProgramDescriptor>>;
    MeshPrograms mesh_programs;

    // ProgramDescriptor too large for reflection inline storage.
    static constexpr auto attribute_names = std::forward_as_tuple("num_mesh_programs");
    auto attribute_values() const { return std::make_tuple(mesh_programs.size()); }
};

}  // namespace tt::tt_metal::experimental
