// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include <vector>
#include <utility>

namespace tt::tt_metal::experimental {

struct MeshProgramDescriptor {
    using MeshPrograms = std::vector<std::pair<distributed::MeshCoordinateRange, ProgramDescriptor>>;
    MeshPrograms mesh_programs;

    // Optional op-internal cross-device resources, created once by the op and parked here so the
    // framework keeps their device-side allocation valid for the cached workload's lifetime
    // (mirrors WorkloadDescriptor::semaphores). The generic_op adapter copies this into the cached
    // shared_variables at program-cache miss. Deliberately EXCLUDED from the program-cache hash, so
    // calls that differ only in semaphore identity still cache-hit. Empty by default.
    std::vector<GlobalSemaphore> semaphores;

    // ProgramDescriptor too large for reflection inline storage. `semaphores` is intentionally
    // omitted from the reflected attributes: the generic_op program-cache key comes from
    // compute_program_hash (mesh_programs only), and keeping the semaphore count out of the
    // reflected attributes guarantees a call that differs only in op-internal semaphores still
    // hits the cache (the whole point of the slot).
    static constexpr auto attribute_names = std::forward_as_tuple("num_mesh_programs");
    auto attribute_values() const { return std::make_tuple(mesh_programs.size()); }
};

}  // namespace tt::tt_metal::experimental
