// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <optional>
#include <vector>

#include <tt-metalium/memory/buffer/buffer.hpp>
#include <tt-metalium/memory/buffer/shard_data_transfer.hpp>
#include <tt-metalium/memory/tensor/mesh_tensor.hpp>
#include <tt-metalium/execution/mesh_command_queue.hpp>

namespace tt::tt_metal {

// ======================================================================================
//                    Byte-based data transfer APIs
// ======================================================================================
//
// Outside the Runtime Tensor graduation surface (HostTensor <-> MeshTensor).
// Explicitly provided for unit mesh (single device) ttnn::Tensor H<->D access.
// Exposed via ttnn/async_runtime.hpp (write_buffer / read_buffer) and have
// external dependencies.

void enqueue_read_tensor(
    distributed::MeshCommandQueue& queue,
    const MeshTensor& device_tensor,
    std::byte* dst,
    const std::optional<BufferRegion>& region = std::nullopt,
    bool blocking = true);

void enqueue_write_tensor(
    distributed::MeshCommandQueue& queue,
    const std::byte* src,
    MeshTensor& device_tensor,
    const std::optional<BufferRegion>& region = std::nullopt);

namespace experimental {

// Experimental: subject to change without notice.

// MeshCommandQueue::enqueue_read_shards / enqueue_write_shards on the buffer of ``device_tensor``, which callers
// outside tt_metal cannot reach. Unlike the single-device APIs above, these take any shards of any mesh.
void enqueue_read_shards(
    distributed::MeshCommandQueue& queue,
    const MeshTensor& device_tensor,
    const std::vector<distributed::ShardDataTransfer>& shard_data_transfers,
    bool blocking = true);

void enqueue_write_shards(
    distributed::MeshCommandQueue& queue,
    MeshTensor& device_tensor,
    const std::vector<distributed::ShardDataTransfer>& shard_data_transfers,
    bool blocking = true);

}  // namespace experimental

}  // namespace tt::tt_metal
