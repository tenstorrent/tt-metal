// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include <unordered_map>

namespace tt::tt_metal::experimental {

struct MeshProgramDescriptor {
    std::unordered_map<distributed::MeshCoordinateRange, ProgramDescriptor> mesh_programs;

    static constexpr auto attribute_names = std::forward_as_tuple("num_mesh_programs");
    auto attribute_values() const { return std::forward_as_tuple(mesh_programs.size()); }
};

}  // namespace tt::tt_metal::experimental
