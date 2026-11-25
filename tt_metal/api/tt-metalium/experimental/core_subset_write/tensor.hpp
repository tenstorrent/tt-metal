// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/vocabulary/core_coord.hpp>
#include <tt-metalium/memory/tensor/host_tensor.hpp>
#include <tt-metalium/memory/tensor/mesh_tensor.hpp>
#include <tt-metalium/execution/mesh_command_queue.hpp>

namespace tt::tt_metal::experimental::core_subset_write {

// EXPERIMENTAL: may evolve into overloads of tt::tt_metal::enqueue_write_tensor.

void enqueue_write_tensor(
    distributed::MeshCommandQueue& cq,
    const HostTensor& host_tensor,
    const MeshTensor& device_tensor,
    const CoreRangeSet& logical_core_filter);

}  // namespace tt::tt_metal::experimental::core_subset_write
