// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/byte_based_tensor_transfers.hpp>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt_stl/assert.hpp>

#include "distributed/mesh_command_queue_base.hpp"
#include "distributed/mesh_device_impl.hpp"
#include "mesh_tensor_impl.hpp"

namespace tt::tt_metal {

void enqueue_read_tensor(
    distributed::MeshCommandQueue& queue,
    const MeshTensor& device_tensor,
    std::byte* dst,
    const std::optional<BufferRegion>& region,
    bool blocking) {
    TT_FATAL(queue.device()->num_devices() == 1, "enqueue_read_tensor only supports single device mesh");
    std::vector<distributed::ShardDataTransfer> shard_data_transfers = {
        distributed::ShardDataTransfer{*distributed::MeshCoordinateRange(queue.device()->shape()).begin()}
            .host_data(dst)
            .region(region)};
    queue.enqueue_read_shards(shard_data_transfers, device_tensor.impl().raw_mesh_buffer(), blocking);
}

void enqueue_write_tensor(
    distributed::MeshCommandQueue& queue,
    const std::byte* src,
    MeshTensor& device_tensor,
    const std::optional<BufferRegion>& region) {
    TT_FATAL(queue.device()->num_devices() == 1, "enqueue_write_tensor only supports single device mesh");
    std::vector<distributed::ShardDataTransfer> shard_data_transfers = {
        distributed::ShardDataTransfer{*distributed::MeshCoordinateRange(queue.device()->shape()).begin()}
            .host_data(const_cast<std::byte*>(src))
            .region(region)};
    queue.enqueue_write_shards(device_tensor.impl().raw_mesh_buffer(), shard_data_transfers, false);
}

namespace {

// The queue's form of ``transfers`` on ``buffer``, after checking every one of them. All the conditions the queue
// would otherwise only meet one transfer at a time, inside its read / write loop, are checked here: a transfer
// refused there leaves the reads before it issued with their descriptors queued and no memcpy request (every later
// read on the queue is then matched one descriptor late, the first of them into this call's host memory), or some
// shards written and others not.
std::vector<distributed::ShardDataTransfer> to_shard_transfers(
    distributed::MeshCommandQueue& queue,
    const distributed::MeshBuffer& buffer,
    const std::vector<ShardRegionTransfer>& transfers) {
    auto* mesh_device = queue.device();
    TT_FATAL(buffer.is_allocated(), "the tensor's buffer is not allocated");
    TT_FATAL(
        buffer.device() == mesh_device,
        "the tensor is on another mesh device than the command queue: shard indices are of the queue's mesh");
    std::vector<distributed::MeshCoordinate> coords;
    for (const auto& c : distributed::MeshCoordinateRange(mesh_device->shape())) {
        coords.push_back(c);
    }
    std::vector<distributed::ShardDataTransfer> out;
    out.reserve(transfers.size());
    for (const auto& t : transfers) {
        TT_FATAL(t.shard < coords.size(), "shard {} out of range: the mesh has {} devices", t.shard, coords.size());
        TT_FATAL(t.host != nullptr && t.size > 0, "empty transfer for shard {}", t.shard);
        const auto& coord = coords[t.shard];
        // The queue skips a shard of another host without an error: the caller's memory would be left as it was.
        TT_FATAL(
            mesh_device->impl().is_local(coord),
            "shard {} ({}) is not on this host: byte-range transfers reach local devices only",
            t.shard,
            coord);
        const Buffer* device_buffer = buffer.get_device_buffer(coord);
        TT_FATAL(
            !is_sharded(device_buffer->buffer_layout()),
            "byte-range transfers need an interleaved buffer: the pages of a sharded buffer are not in device order");
        const uint64_t page_size = device_buffer->page_size();
        const uint64_t buffer_size = device_buffer->size();
        TT_FATAL(
            t.offset % page_size == 0 && t.size % page_size == 0,
            "shard {}: offset {} and size {} must be multiples of the buffer's page size {}",
            t.shard,
            t.offset,
            t.size,
            page_size);
        // Not offset + size <= buffer_size: the sum of two caller-chosen uint64 values can wrap.
        TT_FATAL(
            t.offset <= buffer_size && t.size <= buffer_size - t.offset,
            "shard {}: bytes [{}, {} + {}) are outside the buffer of {} bytes",
            t.shard,
            t.offset,
            t.offset,
            t.size,
            buffer_size);
        out.push_back(distributed::ShardDataTransfer{coord}.host_data(t.host).region(BufferRegion(t.offset, t.size)));
    }
    return out;
}

}  // namespace

void enqueue_read_shard_regions(
    distributed::MeshCommandQueue& queue,
    const MeshTensor& device_tensor,
    const std::vector<ShardRegionTransfer>& transfers,
    bool blocking,
    MemoryPin host_pin) {
    if (transfers.empty()) {
        return;
    }
    const auto& mesh_buffer = device_tensor.impl().raw_mesh_buffer();
    const auto shard_transfers = to_shard_transfers(queue, *mesh_buffer, transfers);
    if (host_pin == nullptr) {
        queue.enqueue_read_shards(shard_transfers, mesh_buffer, blocking);
        return;
    }
    // The pin travels with the queue's memcpy request, as enqueue_read's per-shard pins do: the reader thread drops it
    // once the bytes are in the host memory.
    auto* base = dynamic_cast<distributed::MeshCommandQueueBase*>(&queue);
    TT_FATAL(base != nullptr, "MeshCommandQueue is not a MeshCommandQueueBase");
    std::vector<MemoryPin> pins;
    pins.push_back(std::move(host_pin));
    auto lock = base->lock_api_function_();
    base->enqueue_read_shards_nolock(shard_transfers, mesh_buffer, blocking, std::move(pins));
}

void enqueue_write_shard_regions(
    distributed::MeshCommandQueue& queue,
    MeshTensor& device_tensor,
    const std::vector<ShardRegionTransfer>& transfers,
    bool blocking) {
    if (transfers.empty()) {
        return;
    }
    const auto& mesh_buffer = device_tensor.impl().raw_mesh_buffer();
    queue.enqueue_write_shards(mesh_buffer, to_shard_transfers(queue, *mesh_buffer, transfers), blocking);
}

}  // namespace tt::tt_metal
