// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <memory>
#include <utility>
#include <variant>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/shape2d.hpp>

namespace tt::tt_metal::distributed {

// Implementation of `MeshBuffer`; owns the per-device buffers and the allocation state.
// Only reachable through `MeshBuffer::impl()`.
class MeshBufferImpl {
public:
    // Creates an owning `MeshBufferImpl`, backed by an allocation made through `backing_buffer`.
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

    // Creates a non-owning `MeshBufferImpl` as "view" over an existing `address`.
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

    ~MeshBufferImpl();

    MeshBufferImpl(const MeshBufferImpl&) = delete;
    MeshBufferImpl& operator=(const MeshBufferImpl&) = delete;
    MeshBufferImpl(MeshBufferImpl&& other) noexcept;
    MeshBufferImpl& operator=(MeshBufferImpl&& other) noexcept;

    // Creates the per-device buffers at `address_`. Must be called exactly once, right after construction.
    void initialize_device_buffers();

    // Returns true if the MeshBuffer is allocated. Note that MeshBuffer is created in the allocated state; either the
    // destructor or the `deallocate` method deallocate the MeshBuffer.
    bool is_allocated() const;

    // Deallocates the MeshBuffer.
    // TODO: Re-consider a need for explicit deallocation methods, as opposed to relying on RAII to clean up the
    // resources.
    void deallocate();

    // Throws an exception if the corresponding MeshDevice is already deallocated
    MeshDevice* device() const;
    DeviceAddr size() const;
    DeviceAddr device_local_size() const { return device_local_size_; }
    DeviceAddr address() const { return address_; }

    MeshBufferLayout global_layout() const;
    const MeshBufferConfig& global_config() const { return config_; }

    const ShardedBufferConfig& global_shard_spec() const;
    const DeviceLocalBufferConfig& device_local_config() const { return device_local_config_; }

    Buffer* get_device_buffer(const MeshCoordinate& device_coord) const;
    Buffer* get_reference_buffer() const;
    Buffer* get_backing_buffer() const;

    uint32_t datum_size_bytes() const;
    Shape2D physical_shard_shape() const;
    std::pair<bool, bool> replicated_dims() const;
    uint32_t page_size() const { return device_local_config_.page_size; }
    uint32_t num_pages() const { return page_size() == 0 ? 0 : device_local_size_ / page_size(); }

    // Direct access to the per-device buffers, for creation paths that populate them selectively.
    DistributedMeshContainer<std::shared_ptr<Buffer>>& buffers() { return buffers_; }
    const DistributedMeshContainer<std::shared_ptr<Buffer>>& buffers() const { return buffers_; }

private:
    MeshBufferConfig config_;
    DeviceLocalBufferConfig device_local_config_;
    std::weak_ptr<MeshDevice> mesh_device_;
    DeviceAddr address_ = 0;
    DeviceAddr device_local_size_ = 0;

    DistributedMeshContainer<std::shared_ptr<Buffer>> buffers_;

    // `MeshBufferState` specifies the state of the MeshBuffer. It can either be:
    // 1. Owned - a single device buffer is responsible for providing the address for the entire mesh buffer.
    // 2. Externally owned - the MeshBuffer was created as a view over an existing address.
    // 3. Deallocated - the MeshBuffer is in the deallocated state.
    struct OwnedBufferState {
        std::shared_ptr<Buffer> backing_buffer;
    };
    struct ExternallyOwnedState {};
    struct DeallocatedState {};
    using MeshBufferState = std::variant<OwnedBufferState, ExternallyOwnedState, DeallocatedState>;
    MeshBufferState state_;
};

}  // namespace tt::tt_metal::distributed
