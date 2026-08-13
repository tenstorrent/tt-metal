// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <vector>

#include <tt_stl/assert.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_command_queue.hpp>
#include <tt-metalium/mesh_event.hpp>

namespace tt::tt_metal::distributed {

// Internal: no MeshCommandQueue equivalent yet (#26591).
bool EventQuery(const MeshEvent& event);

// Internal helpers used by Enqueue*MeshBuffer templates (defined in mesh_io.cpp).
void detail_enqueue_write_mesh_buffer(
    MeshCommandQueue& mesh_cq, const std::shared_ptr<MeshBuffer>& mesh_buffer, const void* src, bool blocking);
void detail_enqueue_read_mesh_buffer(
    MeshCommandQueue& mesh_cq, void* dst, const std::shared_ptr<MeshBuffer>& mesh_buffer, bool blocking);

template <typename DType>
void EnqueueWriteMeshBuffer(
    MeshCommandQueue& mesh_cq,
    std::shared_ptr<MeshBuffer>& mesh_buffer,
    const std::vector<DType>& src,
    bool blocking = false) {
    TT_FATAL(
        src.size() * sizeof(DType) >= mesh_buffer->size(),
        "Source vector is too small for mesh buffer: mesh buffer size={} bytes, source size={} * {} bytes",
        mesh_buffer->size(),
        src.size(),
        sizeof(DType));

    detail_enqueue_write_mesh_buffer(mesh_cq, mesh_buffer, src.data(), blocking);
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
    detail_enqueue_read_mesh_buffer(mesh_cq, dst.data(), mesh_buffer, blocking);
}

}  // namespace tt::tt_metal::distributed
