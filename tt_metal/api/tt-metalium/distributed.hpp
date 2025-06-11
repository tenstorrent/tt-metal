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

MeshWorkload CreateMeshWorkload();

void AddProgramToMeshWorkload(MeshWorkload& mesh_workload, Program&& program, const MeshCoordinateRange& device_range);

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
    auto shard = mesh_buffer->get_device_buffer(coord);
    dst.resize(shard->page_size() * shard->num_pages() / sizeof(DType));
    std::vector<MeshCommandQueue::ShardDataTransfer> shard_data_transfers = {{
        .shard_coord = coord,
        .host_data = dst.data(),
        .region = std::nullopt,
    }};
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
    TT_FATAL(
        mesh_buffer->global_layout() == MeshBufferLayout::SHARDED,
        "Can only read a Sharded MeshBuffer from a MeshDevice.");
    dst.resize(mesh_buffer->global_shard_spec().global_size / sizeof(DType));
    mesh_cq.enqueue_read_mesh_buffer(dst.data(), mesh_buffer, blocking);
}

// Make the specified MeshCommandQueue record an event.
// Host is not notified when this event completes.
// Can be used for CQ to CQ synchronization.
MeshEvent EnqueueRecordEvent(
    MeshCommandQueue& mesh_cq,
    tt::stl::Span<const SubDeviceId> sub_device_ids = {},
    const std::optional<MeshCoordinateRange>& device_range = std::nullopt);

// Make the specified MeshCommandQueue record an event and notify the host when it completes.
// Can be used for CQ to CQ and host to CQ synchronization.
MeshEvent EnqueueRecordEventToHost(
    MeshCommandQueue& mesh_cq,
    tt::stl::Span<const SubDeviceId> sub_device_ids = {},
    const std::optional<MeshCoordinateRange>& device_range = std::nullopt);

// Make the specified MeshCommandQueue wait for the completion of an event.
// This operation is non-blocking on host, however the specified command queue
// will stall until the event is recorded.
void EnqueueWaitForEvent(MeshCommandQueue& mesh_cq, const MeshEvent& event);

// Make the current thread block until the event is recorded by the associated MeshCommandQueue.
void EventSynchronize(const MeshEvent& event);

// Query the status of an event tied to a MeshCommandQueue.
// Returns true if the CQ has completed recording the event, false otherwise.
bool EventQuery(const MeshEvent& event);

MeshTraceId BeginTraceCapture(MeshDevice* device, uint8_t cq_id);

void EndTraceCapture(MeshDevice* device, uint8_t cq_id, const MeshTraceId& trace_id);

void ReplayTrace(MeshDevice* device, uint8_t cq_id, const MeshTraceId& trace_id, bool blocking);

void ReleaseTrace(MeshDevice* device, const MeshTraceId& trace_id);

void Synchronize(
    MeshDevice* device, std::optional<uint8_t> cq_id, tt::stl::Span<const SubDeviceId> sub_device_ids = {});

void Finish(MeshCommandQueue& mesh_cq, tt::stl::Span<const SubDeviceId> sub_device_ids = {});

}  // namespace distributed
}  // namespace tt::tt_metal
