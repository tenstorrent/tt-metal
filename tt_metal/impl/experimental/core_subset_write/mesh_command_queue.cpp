// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/core_subset_write/mesh_command_queue.hpp>
#include <tt_stl/assert.hpp>

#include "distributed/mesh_command_queue_base.hpp"

namespace tt::tt_metal::experimental::core_subset_write {

void enqueue_write(
    distributed::MeshCommandQueue& cq,
    const std::shared_ptr<distributed::MeshBuffer>& mesh_buffer,
    const tt::tt_metal::DistributedHostBuffer& host_buffer,
    bool blocking,
    const CoreRangeSet& logical_core_filter) {
    auto* base = dynamic_cast<distributed::MeshCommandQueueBase*>(&cq);
    TT_FATAL(base != nullptr, "MeshCommandQueue is not a MeshCommandQueueBase");
    base->enqueue_write_with_core_filter(mesh_buffer, host_buffer, blocking, &logical_core_filter);
}

}  // namespace tt::tt_metal::experimental::core_subset_write
