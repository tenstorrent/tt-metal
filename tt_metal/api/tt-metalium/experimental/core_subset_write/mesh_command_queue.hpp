// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed_host_buffer.hpp>

namespace tt::tt_metal::distributed {
class MeshBuffer;
class MeshCommandQueue;
}  // namespace tt::tt_metal::distributed

namespace tt::tt_metal::experimental::core_subset_write {

// EXPERIMENTAL: may evolve into a method on MeshCommandQueue.
void enqueue_write(
    distributed::MeshCommandQueue& cq,
    distributed::MeshBuffer& mesh_buffer,
    const DistributedHostBuffer& host_buffer,
    bool blocking,
    const CoreRangeSet& logical_core_filter);

}  // namespace tt::tt_metal::experimental::core_subset_write
