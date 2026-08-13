// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>

#include <memory>
#include <utility>
#include <variant>

namespace tt::tt_metal::distributed {

class MeshBufferImpl {
public:
    struct OwnedBufferState {
        std::shared_ptr<Buffer> backing_buffer;
    };
    struct ExternallyOwnedState {};
    struct DeallocatedState {};
    using MeshBufferState = std::variant<OwnedBufferState, ExternallyOwnedState, DeallocatedState>;

    MeshBufferImpl(
        const MeshBufferConfig& config,
        const DeviceLocalBufferConfig& device_local_config,
        DeviceAddr device_local_size,
        MeshDevice* mesh_device,
        std::shared_ptr<Buffer> backing_buffer) :
        config_(config),
        device_local_config_(device_local_config),
        mesh_device_(mesh_device->shared_from_this()),
        address_(backing_buffer->address()),
        device_local_size_(device_local_size),
        buffers_(MeshShape(mesh_device->shape())),
        state_(OwnedBufferState{std::move(backing_buffer)}) {}

    MeshBufferImpl(
        const MeshBufferConfig& config,
        const DeviceLocalBufferConfig& device_local_config,
        DeviceAddr address,
        DeviceAddr device_local_size,
        MeshDevice* mesh_device) :
        config_(config),
        device_local_config_(device_local_config),
        mesh_device_(mesh_device->shared_from_this()),
        address_(address),
        device_local_size_(device_local_size),
        buffers_(MeshShape(mesh_device->shape())),
        state_(ExternallyOwnedState{}) {}

    void initialize_device_buffers(MeshBuffer& self);
    bool is_allocated() const;
    void deallocate();
    DeviceAddr size() const;
    const ShardedBufferConfig& global_shard_spec() const;
    uint32_t datum_size_bytes() const;
    Shape2D physical_shard_shape() const;
    std::pair<bool, bool> replicated_dims() const;
    MeshBufferLayout global_layout() const;

    MeshBufferConfig config_;
    DeviceLocalBufferConfig device_local_config_;
    std::weak_ptr<MeshDevice> mesh_device_;
    DeviceAddr address_ = 0;
    DeviceAddr device_local_size_ = 0;
    DistributedMeshContainer<std::shared_ptr<Buffer>> buffers_;
    MeshBufferState state_;
};

}  // namespace tt::tt_metal::distributed
