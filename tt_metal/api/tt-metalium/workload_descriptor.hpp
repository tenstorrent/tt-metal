// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include <memory>
#include <vector>

namespace tt::tt_metal {

/**
 * Workload-scoped buffer entry held by a WorkloadDescriptor.
 *
 * `owner` is a type-erased shared owner of the underlying device allocation —
 * typically a `std::shared_ptr<ttnn::Tensor>` so the Tensor's destructor (which
 * force-deallocates the MeshBuffer via DeviceStorage::deallocate regardless of
 * shared_ptr<MeshBuffer> ownership) is deferred until the cached workload is
 * evicted.  Holding the raw `std::shared_ptr<distributed::MeshBuffer>` here
 * was insufficient: the source Tensor's destructor would still fire at the
 * end of the factory call and free the underlying device memory.
 *
 * `buffer` is the raw Buffer* used by binding patching on cache hits.  It
 * stays valid as long as `owner` is alive.
 *
 * In Metal 2.0 the explicit Buffer abstraction is going away, so this struct
 * exists as a transitional shim: factories build it once at cache miss and
 * the framework reads `buffer` to find what to patch.
 */
struct WorkloadBuffer {
    std::shared_ptr<void> owner;
    Buffer* buffer = nullptr;
};

/**
 * Declarative description of a mesh-scoped workload.
 *
 * A WorkloadDescriptor pairs the per-coord ProgramDescriptors that make up
 * the workload with the workload-scoped resources those programs reference.
 * The framework realises it into a MeshWorkload on cache miss and keeps the
 * descriptor alive for the cached workload's lifetime so that:
 *   - GlobalSemaphores keep their device-side allocations valid, and
 *   - WorkloadBuffer owners keep the device storage their addresses point at valid.
 *
 * Layout choices:
 *   - `programs` is range-keyed rather than coord-keyed so that ops with a
 *     single program replicated across the whole mesh emit just one entry.
 *   - Resources are flat vectors (not named slots) so factories with N
 *     semaphores or N buffers can grow without changing the schema.
 */
struct WorkloadDescriptor {
    // Workload-scoped resources, allocated by the factory during workload
    // build and kept alive for the lifetime of the cached MeshWorkload.
    std::vector<GlobalSemaphore> semaphores;
    std::vector<WorkloadBuffer> buffers;

    // Per-coord program descriptors.  Each entry covers a contiguous
    // MeshCoordinateRange and is materialised into one Program added to the
    // resulting MeshWorkload.
    struct PerCoordProgram {
        distributed::MeshCoordinateRange range;
        ProgramDescriptor descriptor;
    };
    std::vector<PerCoordProgram> programs;
};

}  // namespace tt::tt_metal
