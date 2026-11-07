// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <tuple>
#include <vector>
#include <utility>

namespace tt::tt_metal::experimental {

struct MeshProgramDescriptor {
    using MeshPrograms = std::vector<std::pair<distributed::MeshCoordinateRange, ProgramDescriptor>>;
    MeshPrograms mesh_programs;

    // ProgramDescriptor too large for reflection inline storage.
    // The declared_* attributes report the largest value any device's program declares (std::nullopt if none does).
    static constexpr auto attribute_names = std::forward_as_tuple(
        "num_mesh_programs",
        "declared_dram_bytes_read",
        "declared_dram_bytes_written",
        "declared_flops",
        "declared_math_fidelity");
    auto attribute_values() const {
        std::optional<std::uint64_t> bytes_read;
        std::optional<std::uint64_t> bytes_written;
        std::optional<std::uint64_t> flops;
        std::optional<MathFidelity> math_fidelity;
        for (const auto& [range, program] : mesh_programs) {
            if (!program.performance) {
                continue;
            }
            const auto& performance = *program.performance;
            bytes_read = std::max(bytes_read.value_or(0), performance.dram_bytes_read);
            bytes_written = std::max(bytes_written.value_or(0), performance.dram_bytes_written);
            flops = std::max(flops.value_or(0), performance.flops);
            if (!math_fidelity) {
                math_fidelity = performance.math_fidelity;
            }
        }
        return std::make_tuple(mesh_programs.size(), bytes_read, bytes_written, flops, math_fidelity);
    }
};

}  // namespace tt::tt_metal::experimental
