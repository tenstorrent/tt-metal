// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/distributed/mesh_buffer.hpp"
#include "tt_metal/distributed/mesh_command_queue.hpp"

namespace tt::tt_metal {

inline namespace v0 {

class IDevice;
class Tensor;

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
    mesh_cq.enqueue_write_shard(mesh_buffer, src.data(), coord, blocking);
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
    mesh_cq.enqueue_read_shard(dst.data(), mesh_buffer, coord, blocking);
}

void Finish(MeshCommandQueue& mesh_cq);

}  // namespace distributed
}  // namespace tt::tt_metal
