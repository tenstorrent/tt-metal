// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/byte_based_tensor_transfers.hpp>

#include <tt-metalium/experimental/per_core_allocation/memory_config.hpp>

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

namespace experimental {

namespace {

// Checks the whole batch before anything is enqueued: a transfer rejected inside the queue would leave the transfers
// before it enqueued without the completion request that ends the read, and a later read would consume their read
// descriptors.
std::vector<distributed::ShardDataTransfer> to_shard_transfers(
    distributed::MeshCommandQueue& queue,
    const MeshTensor& device_tensor,
    const std::vector<ShardRegionTransfer>& transfers) {
    // In an interleaved buffer, byte offset ``i * page_size`` is the start of page i; no such map holds for a sharded
    // buffer, whose pages a byte range would select in shard order.
    TT_FATAL(
        !device_tensor.memory_config().is_sharded(),
        "shard region transfers support interleaved tensors only, got a sharded tensor");
    TT_FATAL(
        !per_core_allocation::is_per_core_allocation(device_tensor.memory_config()),
        "shard region transfers do not support per-core allocated tensors");
    const auto& mesh_buffer = *device_tensor.impl().raw_mesh_buffer();
    const uint64_t page_size = mesh_buffer.page_size();
    const uint64_t buffer_size = mesh_buffer.device_local_size();
    TT_FATAL(page_size > 0, "shard region transfers need a buffer with a nonzero page size");
    std::vector<distributed::MeshCoordinate> coords;
    for (const auto& c : distributed::MeshCoordinateRange(queue.device()->shape())) {
        coords.push_back(c);
    }
    // A run of transfers that continue one another on the same shard, both in the buffer and in host memory, becomes
    // one transfer: callers describe paged data page by page, and a dispatch per page is the slow part.
    std::vector<ShardRegionTransfer> merged;
    merged.reserve(transfers.size());
    for (const auto& t : transfers) {
        TT_FATAL(t.shard < coords.size(), "shard {} out of range: the mesh has {} devices", t.shard, coords.size());
        TT_FATAL(t.host != nullptr && t.size > 0, "empty transfer for shard {}", t.shard);
        TT_FATAL(
            t.offset % page_size == 0 && t.size % page_size == 0,
            "transfer for shard {} (offset {}, size {}) is not a whole number of {}-byte pages",
            t.shard,
            t.offset,
            t.size,
            page_size);
        TT_FATAL(
            t.size <= buffer_size && t.offset <= buffer_size - t.size,
            "transfer for shard {} (offset {}, size {}) is outside the {}-byte buffer",
            t.shard,
            t.offset,
            t.size,
            buffer_size);
        if (!merged.empty()) {
            auto& last = merged.back();
            if (last.shard == t.shard && last.offset + last.size == t.offset && last.host + last.size == t.host) {
                last.size += t.size;
                continue;
            }
        }
        merged.push_back(t);
    }
    std::vector<distributed::ShardDataTransfer> out;
    out.reserve(merged.size());
    for (const auto& t : merged) {
        out.push_back(
            distributed::ShardDataTransfer{coords[t.shard]}.host_data(t.host).region(BufferRegion(t.offset, t.size)));
    }
    return out;
}

}  // namespace

void enqueue_read_shard_regions(
    distributed::MeshCommandQueue& queue,
    const MeshTensor& device_tensor,
    const std::vector<ShardRegionTransfer>& transfers,
    bool blocking) {
    if (transfers.empty()) {
        return;
    }
    queue.enqueue_read_shards(
        to_shard_transfers(queue, device_tensor, transfers), device_tensor.impl().raw_mesh_buffer(), blocking);
}

void enqueue_write_shard_regions(
    distributed::MeshCommandQueue& queue,
    MeshTensor& device_tensor,
    const std::vector<ShardRegionTransfer>& transfers,
    bool blocking) {
    if (transfers.empty()) {
        return;
    }
    queue.enqueue_write_shards(
        device_tensor.impl().raw_mesh_buffer(), to_shard_transfers(queue, device_tensor, transfers), blocking);
}

}  // namespace experimental

}  // namespace tt::tt_metal
