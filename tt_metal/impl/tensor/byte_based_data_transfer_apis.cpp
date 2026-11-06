// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/byte_based_tensor_transfers.hpp>

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

std::vector<distributed::ShardDataTransfer> to_shard_transfers(
    distributed::MeshCommandQueue& queue, const std::vector<ShardRegionTransfer>& transfers) {
    std::vector<distributed::MeshCoordinate> coords;
    for (const auto& c : distributed::MeshCoordinateRange(queue.device()->shape())) {
        coords.push_back(c);
    }
    std::vector<distributed::ShardDataTransfer> out;
    out.reserve(transfers.size());
    for (const auto& t : transfers) {
        TT_FATAL(t.shard < coords.size(), "shard {} out of range: the mesh has {} devices", t.shard, coords.size());
        TT_FATAL(t.host != nullptr && t.size > 0, "empty transfer for shard {}", t.shard);
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
    queue.enqueue_read_shards(to_shard_transfers(queue, transfers), device_tensor.impl().raw_mesh_buffer(), blocking);
}

void enqueue_write_shard_regions(
    distributed::MeshCommandQueue& queue,
    MeshTensor& device_tensor,
    const std::vector<ShardRegionTransfer>& transfers,
    bool blocking) {
    if (transfers.empty()) {
        return;
    }
    queue.enqueue_write_shards(device_tensor.impl().raw_mesh_buffer(), to_shard_transfers(queue, transfers), blocking);
}

}  // namespace tt::tt_metal
