// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <memory>
#include <optional>
#include <vector>

#include <tt_stl/span.hpp>
#include <tt-metalium/assert.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_command_queue.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_event.hpp>
#include <tt-metalium/mesh_socket.hpp>
#include <tt-metalium/mesh_trace_id.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/sub_device_types.hpp>

namespace tt {
namespace tt_metal {
class Program;
namespace distributed {
class MeshDevice;
}  // namespace distributed
}  // namespace tt_metal
}  // namespace tt

namespace tt::tt_metal {

class IDevice;

namespace distributed {

// TODO: These APIs are being kept temporarily during the migration.
// EnqueueMeshWorkload has complex compilation logic that needs careful migration.
void EnqueueMeshWorkload(MeshCommandQueue& mesh_cq, MeshWorkload& mesh_workload, bool blocking);

template <typename DType>
void WriteShard(
    MeshCommandQueue& mesh_cq,
    const std::shared_ptr<MeshBuffer>& mesh_buffer,
    std::vector<DType>& src,
    const MeshCoordinate& coord,
    bool blocking = false) {
    std::vector<MeshCommandQueue::ShardDataTransfer> shard_data_transfers = {{
        .shard_coord = coord,
        .host_data = src.data(),
        .region = std::nullopt,
    }};
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
    // Tracking removal of free function APIs in this file in this issue.
    auto mesh_device = mesh_cq.device();
    if (!mesh_device->is_local(coord)) {
        return;
    }

    auto shard = mesh_buffer->get_device_buffer(coord);
    dst.resize(shard->page_size() * shard->num_pages() / sizeof(DType));
    std::vector<MeshCommandQueue::ShardDataTransfer> shard_data_transfers = {{
        .shard_coord = coord,
        .host_data = dst.data(),
        .region = std::nullopt,
    }};
    mesh_cq.enqueue_read_shards(shard_data_transfers, mesh_buffer, blocking);
}

// TODO: WriteShard and ReadShard template functions are kept temporarily.
// They need complex migration as users would need to create ShardDataTransfer vectors.

// TODO: Remove once all consumers have migrated to using MeshCommandQueue/MeshDevice/MeshEvent methods directly.
// Most APIs have been migrated to methods on their respective classes:
// - mesh_cq.enqueue_record_event() instead of EnqueueRecordEvent()
// - mesh_cq.enqueue_wait_for_event() instead of EnqueueWaitForEvent()
// - mesh_cq.finish() instead of Finish()
// - device->begin_mesh_trace() instead of BeginTraceCapture()
// - device->synchronize() instead of Synchronize()
// - event.synchronize() instead of EventSynchronize()
// - event.query() instead of EventQuery()

// Returns true if the distributed environment is initialized and world_size > 1.
bool UsingDistributedEnvironment();

}  // namespace distributed
}  // namespace tt::tt_metal
