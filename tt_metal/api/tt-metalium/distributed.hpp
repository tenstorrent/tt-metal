// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <memory>
#include <optional>
#include <vector>

#include <tt_stl/span.hpp>
#include <tt_stl/assert.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_command_queue.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_event.hpp>
#include <tt-metalium/experimental/sockets/mesh_socket.hpp>
#include <tt-metalium/mesh_trace_id.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/sub_device_types.hpp>

namespace tt::tt_metal {
class Program;
namespace distributed {
class MeshDevice;
}  // namespace distributed
}  // namespace tt::tt_metal

namespace tt::tt_metal {

class IDevice;

namespace distributed {

void EnqueueMeshWorkload(MeshCommandQueue& mesh_cq, MeshWorkload& mesh_workload, bool blocking);

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
    // Tracking removal of free function APIs in this file in this issue.
    auto* mesh_device = mesh_cq.device();
    if (!mesh_device->is_local(coord)) {
        return;
    }

    auto* shard = mesh_buffer->get_device_buffer(coord);
    dst.resize(shard->page_size() * shard->num_pages() / sizeof(DType));
    std::vector<ShardDataTransfer> shard_data_transfers = {ShardDataTransfer{coord}.host_data(dst.data())};
    mesh_cq.enqueue_read_shards(shard_data_transfers, mesh_buffer, blocking);
}

template <typename DType>
void EnqueueWriteMeshBuffer(
    MeshCommandQueue& mesh_cq,
    std::shared_ptr<MeshBuffer>& mesh_buffer,
    const std::vector<DType>& src,
    bool blocking = false) {
    mesh_cq.enqueue_write_mesh_buffer(mesh_buffer, src.data(), blocking);
}

template <typename DType>
void EnqueueReadMeshBuffer(
    MeshCommandQueue& mesh_cq,
    std::vector<DType>& dst,
    std::shared_ptr<MeshBuffer>& mesh_buffer,
    bool blocking = true) {
    // This API supports reading MeshBuffers sharded across devices
    // and a Unit-MeshBuffer with a replicated layout.
    if (mesh_buffer->global_layout() == MeshBufferLayout::SHARDED) {
        dst.resize(mesh_buffer->global_shard_spec().global_size / sizeof(DType));
    } else {
        dst.resize(mesh_buffer->size() / sizeof(DType));
    }
    mesh_cq.enqueue_read_mesh_buffer(dst.data(), mesh_buffer, blocking);
}

// Make the current thread block until the event is recorded by the associated MeshCommandQueue.
void EventSynchronize(const MeshEvent& event);

// Query the status of an event tied to a MeshCommandQueue.
// Returns true if the CQ has completed recording the event, false otherwise.
bool EventQuery(const MeshEvent& event);

MeshTraceId BeginTraceCapture(MeshDevice* device, uint8_t cq_id);

void Synchronize(
    MeshDevice* device, std::optional<uint8_t> cq_id, tt::stl::Span<const SubDeviceId> sub_device_ids = {});

void Finish(MeshCommandQueue& mesh_cq, tt::stl::Span<const SubDeviceId> sub_device_ids = {});

// Returns true if the distributed environment is initialized and world_size > 1.
bool UsingDistributedEnvironment();

}  // namespace distributed
}  // namespace tt::tt_metal
