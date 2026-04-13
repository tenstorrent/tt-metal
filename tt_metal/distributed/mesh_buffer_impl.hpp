// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Internal header — not part of the public API. Do not include from public headers.

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <memory>
#include <variant>

namespace tt::tt_metal::distributed {

struct MeshBuffer::Impl {
    // Owning: backed by an allocation made through `backing_buffer`.
    Impl(
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

    // Non-owning: view over an existing `address`.
    Impl(
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

    // Empty (zero-device) mesh.
    Impl(
        const MeshBufferConfig& config,
        const DeviceLocalBufferConfig& device_local_config,
        DeviceAddr address,
        DeviceAddr device_local_size,
        MeshDevice* mesh_device,
        bool /*empty_tag*/) :
        config_(config),
        device_local_config_(device_local_config),
        mesh_device_(mesh_device->shared_from_this()),
        address_(address),
        device_local_size_(device_local_size),
        buffers_(MeshShape(mesh_device->shape())),
        state_(ExternallyOwnedState{}) {}

    struct OwnedBufferState {
        std::shared_ptr<Buffer> backing_buffer;
    };
    struct ExternallyOwnedState {};
    struct DeallocatedState {};
    using MeshBufferState = std::variant<OwnedBufferState, ExternallyOwnedState, DeallocatedState>;

    void initialize_device_buffers();
    void deallocate();

    MeshBufferConfig config_;
    DeviceLocalBufferConfig device_local_config_;
    std::weak_ptr<MeshDevice> mesh_device_;
    DeviceAddr address_ = 0;
    DeviceAddr device_local_size_ = 0;
    DistributedMeshContainer<std::shared_ptr<Buffer>> buffers_;
    MeshBufferState state_;
};

}  // namespace tt::tt_metal::distributed
