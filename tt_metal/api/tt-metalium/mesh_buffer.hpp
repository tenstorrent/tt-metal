// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <memory>
#include <optional>
#include <utility>
#include <variant>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_constants.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_device_view.hpp>
#include <tt-metalium/shape2d.hpp>

namespace tt::tt_metal::distributed {

// Specifies how a buffer is laid out across Memory Banks within a single device.
struct DeviceLocalBufferConfig {
    DeviceAddr page_size = 0;

    // Can be DRAM, L1, SYSTEM_MEMORY, L1_SMALL, TRACE.
    BufferType buffer_type = BufferType::DRAM;

    // Can be INTERLEAVED, HEIGHT_SHARDED, WIDTH_SHARDED or BLOCK_SHARDED.
    TensorMemoryLayout buffer_layout = TensorMemoryLayout::INTERLEAVED;

    // Must be set for sharded buffer layouts.
    std::optional<ShardSpecBuffer> shard_parameters;

    // The direction in which memory for this buffer is allocated.
    std::optional<bool> bottom_up;
};

// Specifies MeshBuffer that is replicated across the virtual mesh.
// Write APIs for replicated buffers will write the same data to all devices in the virtual mesh.
struct ReplicatedBufferConfig {
    // Each device will get a buffer of this size.
    DeviceAddr size = 0;
};

// Specifies sharded MeshBuffer.
// Write APIs for sharded buffers will split the data so that each device in the virtual mesh will only get a fraction
// of the data.
struct ShardedBufferConfig {
    // Note: Only 2D sharding and replication is supported by the APIs exposed through this struct.
    // This interface will likely change over time depending on the status of native ND sharding.
    // Global buffer size. Each device will get a fraction of this size.
    DeviceAddr global_size = 0;

    // Global shape of the buffer; at metal-level, we expect the shape to be aligned with the mesh shape.
    Shape2D global_buffer_shape = {0, 0};

    // Shard shape, sent to each device.
    Shape2D shard_shape = {0, 0};

    // Orientation of the shards in a mesh.
    ShardOrientation shard_orientation = ShardOrientation::ROW_MAJOR;

    // Computes the number of bytes per datum in the sharded buffer.
    uint32_t compute_datum_size_bytes() const;

    std::pair<bool, bool> replicated_dims() const;

    Shape2D physical_shard_shape() const;
};

enum class MeshBufferLayout : uint8_t { REPLICATED, SHARDED };
using MeshBufferConfig = std::variant<ReplicatedBufferConfig, ShardedBufferConfig>;

// MeshBuffer allocates a buffer across a mesh of devices according to the specified configuration: either full
// replication, or 2D sharding. The allocation is done in lock-step across all devices in the mesh.
class MeshBuffer {
public:
    static std::shared_ptr<MeshBuffer> create(
        const MeshBufferConfig& mesh_buffer_config,
        const DeviceLocalBufferConfig& device_local_layout,
        MeshDevice* mesh_device,
        std::optional<DeviceAddr> address = std::nullopt);
    ~MeshBuffer();

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
    DeviceAddr address() const { return address_; };

    MeshBufferLayout global_layout() const;
    const MeshBufferConfig& global_config() const { return config_; }

    const ShardedBufferConfig& global_shard_spec() const;
    const DeviceLocalBufferConfig& device_local_config() const { return device_local_config_; }

    std::shared_ptr<Buffer> get_device_buffer(const MeshCoordinate& device_coord) const;

    // TODO: Remove this method, once there is no need to interop MeshBuffer with Buffer.
    std::shared_ptr<Buffer> get_reference_buffer() const;

    uint32_t datum_size_bytes() const;
    Shape2D physical_shard_shape() const;
    std::pair<bool, bool> replicated_dims() const;
    uint32_t page_size() const { return device_local_config_.page_size; }
    uint32_t num_pages() const { return page_size() == 0 ? 0 : device_local_size_ / page_size(); }

private:
    // Creates an owning `MeshBuffer`, backed by an allocation made through `backing_buffer`.
    MeshBuffer(
        const MeshBufferConfig& config,
        const DeviceLocalBufferConfig& device_local_config,
        DeviceAddr device_local_size,
        MeshDevice* mesh_device,
        std::shared_ptr<Buffer> backing_buffer) :
        buffers_(MeshShape(mesh_device->shape()), nullptr),
        config_(config),
        device_local_config_(device_local_config),
        mesh_device_(mesh_device->shared_from_this()),
        address_(backing_buffer->address()),
        device_local_size_(device_local_size),
        state_(OwnedBufferState{std::move(backing_buffer)}) {}

    // Creates a non-owning `MeshBuffer` as "view" over an existing `address`.
    MeshBuffer(
        const MeshBufferConfig& config,
        const DeviceLocalBufferConfig& device_local_config,
        DeviceAddr address,
        DeviceAddr device_local_size,
        MeshDevice* mesh_device) :
        buffers_(MeshShape(mesh_device->shape()), /*fill_value=*/nullptr),
        config_(config),
        device_local_config_(device_local_config),
        mesh_device_(mesh_device->shared_from_this()),
        address_(address),
        device_local_size_(device_local_size),
        state_(ExternallyOwnedState{}) {}

    void initialize_device_buffers();
    MeshBufferConfig config_;
    DeviceLocalBufferConfig device_local_config_;
    std::weak_ptr<MeshDevice> mesh_device_;
    DeviceAddr address_ = 0;
    DeviceAddr device_local_size_ = 0;

    MeshContainer<std::shared_ptr<Buffer>> buffers_;

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

class AnyBuffer {
public:
    AnyBuffer() = default;
    static AnyBuffer create(const tt::tt_metal::ShardedBufferConfig& config);
    static AnyBuffer create(const tt::tt_metal::InterleavedBufferConfig& config);

    Buffer* get_buffer() const;
    bool is_mesh_buffer() const;
    std::shared_ptr<MeshBuffer> get_mesh_buffer() const;

private:
    AnyBuffer(std::shared_ptr<Buffer> buffer);
    AnyBuffer(std::shared_ptr<MeshBuffer> buffer);

    Buffer* buffer_ = nullptr;
    std::variant<std::shared_ptr<Buffer>, std::shared_ptr<distributed::MeshBuffer>> holder_;
};

}  // namespace tt::tt_metal::distributed
