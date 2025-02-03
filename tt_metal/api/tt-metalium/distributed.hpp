// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "mesh_buffer.hpp"
#include "mesh_command_queue.hpp"
#include "mesh_event.hpp"

namespace tt::tt_metal {

inline namespace v0 {

class IDevice;

}  // namespace v0

namespace distributed {

MeshWorkload CreateMeshWorkload();

void AddProgramToMeshWorkload(MeshWorkload& mesh_workload, Program& program, const LogicalDeviceRange& device_range);

void EnqueueMeshWorkload(MeshCommandQueue& mesh_cq, MeshWorkload& mesh_workload, bool blocking);

template <typename DType>
void WriteShard(
    MeshCommandQueue& mesh_cq,
    std::shared_ptr<MeshBuffer>& mesh_buffer,
    std::vector<DType>& src,
    const Coordinate& coord,
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
    std::shared_ptr<MeshBuffer>& mesh_buffer,
    const Coordinate& coord,
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
    std::vector<DType>& src,
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

void EnqueueRecordEvent(
    MeshCommandQueue& mesh_cq,
    const std::shared_ptr<MeshEvent>& event,
    tt::stl::Span<const SubDeviceId> sub_device_ids = {},
    const std::optional<LogicalDeviceRange>& device_range = std::nullopt);

void EnqueueRecordEventToHost(
    MeshCommandQueue& mesh_cq,
    const std::shared_ptr<MeshEvent>& event,
    tt::stl::Span<const SubDeviceId> sub_device_ids = {},
    const std::optional<LogicalDeviceRange>& device_range = std::nullopt);

void EnqueueWaitForEvent(MeshCommandQueue& mesh_cq, const std::shared_ptr<MeshEvent>& event);

void EventSynchronize(const std::shared_ptr<MeshEvent>& event);

void Finish(MeshCommandQueue& mesh_cq, tt::stl::Span<const SubDeviceId> sub_device_ids = {});

}  // namespace distributed
}  // namespace tt::tt_metal
