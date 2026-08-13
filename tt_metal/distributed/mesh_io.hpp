// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <memory>
#include <vector>

#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_command_queue.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_event.hpp>
#include <tt-metalium/mesh_trace_id.hpp>

namespace tt::tt_metal::distributed {

class MeshDevice;

template <typename DType>
void WriteShard(
    MeshCommandQueue& mesh_cq,
    const std::shared_ptr<MeshBuffer>& mesh_buffer,
    std::vector<DType>& src,
    const MeshCoordinate& coord,
    bool blocking = false) {
    std::vector<ShardDataTransfer> shard_data_transfers = {ShardDataTransfer{coord}.host_data(src.data())};
    mesh_cq.enqueue_write_shards(mesh_buffer, shard_data_transfers, blocking);
}

template <typename DType>
void ReadShard(
    MeshCommandQueue& mesh_cq,
    std::vector<DType>& dst,
    const std::shared_ptr<MeshBuffer>& mesh_buffer,
    const MeshCoordinate& coord,
    bool blocking = true) {
    // TODO: #26591 - `is_local` Handling should be done under `MeshCommandQueue`.
    auto* mesh_device = mesh_cq.device();
    auto devices = mesh_device->get_view().get_devices(MeshCoordinateRange(coord, coord));
    if (devices.empty()) {
        return;
    }

    auto* shard = mesh_buffer->get_device_buffer(coord);
    dst.resize(shard->page_size() * shard->num_pages() / sizeof(DType));
    std::vector<ShardDataTransfer> shard_data_transfers = {ShardDataTransfer{coord}.host_data(dst.data())};
    mesh_cq.enqueue_read_shards(shard_data_transfers, mesh_buffer, blocking);
}

bool EventQuery(const MeshEvent& event);

MeshTraceId BeginTraceCapture(MeshDevice* device, uint8_t cq_id);

}  // namespace tt::tt_metal::distributed
