// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/tensor/mesh_tensor.hpp>
#include <tt-metalium/mesh_command_queue.hpp>

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

// One byte range of one shard of a mesh tensor's buffer and the host memory it is copied to / from.
// ``shard`` is the shard's linear index in the mesh (row-major over the mesh shape); ``offset`` and ``size`` are bytes
// of that shard's buffer (for an interleaved buffer, a range of whole pages is one contiguous range); ``host`` points
// at ``size`` bytes of host memory owned by the caller.
struct ShardRegionTransfer {
    uint32_t shard = 0;
    uint64_t offset = 0;
    uint64_t size = 0;
    std::byte* host = nullptr;
};

// Reads / writes many byte ranges of the shards of a mesh tensor in one command-queue call, ordered after every
// command already enqueued on ``queue``. Unlike the single-device APIs above, these work on any mesh. The caller keeps
// the host memory valid (and, for writes, unchanged) until the transfer has completed (``blocking``, or a later
// finish / event on the queue).
void enqueue_read_shard_regions(
    distributed::MeshCommandQueue& queue,
    const MeshTensor& device_tensor,
    const std::vector<ShardRegionTransfer>& transfers,
    bool blocking = true);

void enqueue_write_shard_regions(
    distributed::MeshCommandQueue& queue,
    MeshTensor& device_tensor,
    const std::vector<ShardRegionTransfer>& transfers,
    bool blocking = true);

}  // namespace tt::tt_metal
