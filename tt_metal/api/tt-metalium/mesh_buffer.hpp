// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <memory>
#include <optional>
#include <utility>
#include <variant>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>

#include <tt-metalium/shape2d.hpp>

namespace tt::tt_metal::distributed {

// Specifies how a buffer is laid out across Memory Banks within a single device.
struct DeviceLocalBufferConfig {
    DeviceAddr page_size = 0;

    // Can be DRAM, L1, SYSTEM_MEMORY, L1_SMALL, TRACE.
    BufferType buffer_type = BufferType::DRAM;

    BufferShardingArgs sharding_args;

    // The direction in which memory for this buffer is allocated.
    std::optional<bool> bottom_up;

    // Optional: Specify the worker sub device this buffer will be allocated on
    std::optional<SubDeviceId> sub_device_id = std::nullopt;
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
        const DeviceLocalBufferConfig& device_local_config,
        MeshDevice* mesh_device,
        std::optional<DeviceAddr> address = std::nullopt);
    ~MeshBuffer();

    // MeshBuffer manages device memory and owns the backing allocation. Copying would create
    // multiple owners of the same device memory, leading to double-free on destruction.
    MeshBuffer(const MeshBuffer&) = delete;
    MeshBuffer& operator=(const MeshBuffer&) = delete;
    MeshBuffer(MeshBuffer&& other) noexcept;
    MeshBuffer& operator=(MeshBuffer&& other) noexcept;

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
    DeviceAddr device_local_size() const;
    DeviceAddr address() const;

    MeshBufferLayout global_layout() const;
    const MeshBufferConfig& global_config() const;

    const ShardedBufferConfig& global_shard_spec() const;
    const DeviceLocalBufferConfig& device_local_config() const;

    Buffer* get_device_buffer(const MeshCoordinate& device_coord) const;

    // TODO: Remove this method, once there is no need to interop MeshBuffer with Buffer.
    // The reference buffer allows "casting" the MeshBuffer to a buffer allocated on a
    // single device. This allows users of this object that only need to query single device
    // attributes to do so without having to keep track of MeshDevice attributes.
    Buffer* get_reference_buffer() const;
    // The backing buffer represents the buffer object keeping the MeshBuffer alive/allocated
    // at its specific address. The backing buffer will not be populated if an address was passed
    // into the creation API.
    Buffer* get_backing_buffer() const;

    uint32_t datum_size_bytes() const;
    Shape2D physical_shard_shape() const;
    std::pair<bool, bool> replicated_dims() const;
    uint32_t page_size() const;
    uint32_t num_pages() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;

    explicit MeshBuffer(std::unique_ptr<Impl> impl);
};

class AnyBuffer {
public:
    AnyBuffer() = default;
    static AnyBuffer create(
        const tt::tt_metal::ShardedBufferConfig& config, std::optional<uint64_t> address = std::nullopt);
    static AnyBuffer create(
        const tt::tt_metal::InterleavedBufferConfig& config, std::optional<uint64_t> address = std::nullopt);

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
