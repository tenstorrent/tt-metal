// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include <memory>
#include <utility>
#include <vector>

namespace tt::tt_metal {

namespace distributed {
class MeshBuffer;
}  // namespace distributed

/**
 * Declarative description of a mesh-scoped workload.
 *
 * A MeshWorkloadDescriptor pairs the per-coord ProgramDescriptors that make up
 * the workload with the workload-scoped resources those programs reference.
 * The framework realises it into a MeshWorkload on cache miss and keeps the
 * descriptor alive for the cached workload's lifetime so that:
 *   - GlobalSemaphores keep their device-side allocations valid, and
 *   - MeshBuffers keep the device storage their addresses point at valid.
 *
 * Layout choices:
 *   - `programs` is range-keyed rather than coord-keyed so that ops with a
 *     single program replicated across the whole mesh emit just one entry.
 *   - Resources are flat vectors (not named slots) so factories with N
 *     semaphores or N buffers can grow without changing the schema.
 */
struct MeshWorkloadDescriptor {
    // Workload-scoped resources, allocated by the factory during workload
    // build and kept alive for the lifetime of the cached MeshWorkload.
    std::vector<GlobalSemaphore> semaphores;
    std::vector<std::shared_ptr<distributed::MeshBuffer>> buffers;

    // Per-coord program descriptors.  Each entry covers a contiguous
    // MeshCoordinateRange and is materialised into one Program added to the
    // resulting MeshWorkload.
    using PerCoordProgram = std::pair<distributed::MeshCoordinateRange, ProgramDescriptor>;
    std::vector<PerCoordProgram> programs;
};

}  // namespace tt::tt_metal
