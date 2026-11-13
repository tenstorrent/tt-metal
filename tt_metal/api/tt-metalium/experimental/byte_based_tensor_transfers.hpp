// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/memory_pin.hpp>
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
//
// ``shard`` is the shard's linear index in the mesh (row-major over the shape of the queue's mesh device); the shard
// must be on this host. ``offset`` and ``size`` are bytes of that shard's buffer and must both be multiples of the
// buffer's page size (``Buffer::page_size()``, ttnn's ``Tensor.buffer_page_size()``): a transfer moves whole pages,
// page i being bytes [i * page_size, (i + 1) * page_size). ``host`` points at ``size`` bytes of host memory owned by
// the caller; it receives / supplies the pages in page order.
//
// Only interleaved buffers (DRAM or L1) are accepted: there the page index is the buffer's own order. A sharded
// buffer keeps its pages per core in host page order, so a byte range of it names no contiguous device range; such a
// tensor is refused.
struct ShardRegionTransfer {
    uint32_t shard = 0;
    uint64_t offset = 0;
    uint64_t size = 0;
    std::byte* host = nullptr;
};

// Reads / writes many byte ranges of the shards of a mesh tensor in one command-queue call, ordered after every
// command already enqueued on ``queue``. Unlike the single-device APIs above, these work on any mesh.
//
// Every transfer is validated before the first one is enqueued (shard index in range and local, interleaved buffer,
// non-null host pointer, non-zero page-aligned offset / size inside the shard's buffer, checked without overflow), so
// a bad transfer throws with nothing queued: no read is left half-issued on the queue and no shard is written.
//
// Host memory lifetime:
// - write: the bytes are copied into the command queue before the call returns, blocking or not;
// - read, blocking: the bytes have landed when the call returns;
// - read, non-blocking: the queue's reader thread copies into ``host`` later (by the next finish / blocking call /
//   event on the queue). The caller keeps the memory valid until then; ``host_pin`` (optional) is held by the queue
//   until that copy is done and released right after, for memory whose owner can be reference counted.
void enqueue_read_shard_regions(
    distributed::MeshCommandQueue& queue,
    const MeshTensor& device_tensor,
    const std::vector<ShardRegionTransfer>& transfers,
    bool blocking = true,
    MemoryPin host_pin = MemoryPin());

void enqueue_write_shard_regions(
    distributed::MeshCommandQueue& queue,
    MeshTensor& device_tensor,
    const std::vector<ShardRegionTransfer>& transfers,
    bool blocking = true);

}  // namespace tt::tt_metal
