// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "buffer.hpp"
#include "buffer_constants.hpp"
#include "mesh_device.hpp"
#include "mesh_device_view.hpp"
#include "shape2d.hpp"

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
    bool bottom_up = false;
};

// Specifies MeshBuffer that is replicated across the virtual mesh.
struct ReplicatedBufferConfig {
    // Each device will get a buffer of this size.
    DeviceAddr size = 0;
};

// Specifies sharded MeshBuffer.
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

    MeshDevice* device() const { return mesh_device_; }
    DeviceAddr size() const;
    DeviceAddr device_local_size() const { return device_local_size_; }
    DeviceAddr address() const { return address_; };

    MeshBufferLayout global_layout() const;
    const MeshBufferConfig& global_config() const { return config_; }
    // ND Sharding is not supported today. MeshBuffer only supports 2D sharding and
    // replication. Tensor sharding schemes that can be lowered to 2D configurations
    // are thus supported by the MeshCommandQueue.
    const ShardedBufferConfig& global_shard_spec() const;
    const DeviceLocalBufferConfig& device_local_config() const { return device_local_config_; }

    std::shared_ptr<Buffer> get_device_buffer(const Coordinate& device_coord) const;
    uint32_t datum_size_bytes() const;
    Shape2D physical_shard_shape() const;
    std::pair<bool, bool> replicated_dims() const;

private:
    MeshBuffer(
        const MeshBufferConfig& config,
        const DeviceLocalBufferConfig& device_local_config,
        DeviceAddr device_local_size,
        MeshDevice* mesh_device,
        std::shared_ptr<Buffer> backing_buffer) :
        config_(config),
        device_local_config_(device_local_config),
        mesh_device_(mesh_device),
        device_local_size_(device_local_size),
        backing_buffer_(std::move(backing_buffer)) {}

    MeshBuffer(
        const MeshBufferConfig& config,
        const DeviceLocalBufferConfig& device_local_config,
        DeviceAddr address,
        DeviceAddr device_local_size,
        MeshDevice* mesh_device) :
        config_(config),
        device_local_config_(device_local_config),
        mesh_device_(mesh_device),
        address_(address),
        device_local_size_(device_local_size) {}

    void allocate();
    MeshBufferConfig config_;
    DeviceLocalBufferConfig device_local_config_;
    MeshDevice* mesh_device_ = nullptr;
    DeviceAddr address_ = 0;
    DeviceAddr device_local_size_ = 0;

    // TODO: Consider optimizing with SmallVector.
    std::vector<std::vector<std::shared_ptr<Buffer>>> buffers_;
    // Buffer owned by the MeshBuffer. Responsible for interfacing with the
    // single device allocator. This data-structure is not populated if memory
    // for the MeshBuffer is externally owned.
    std::shared_ptr<Buffer> backing_buffer_;
};

}  // namespace tt::tt_metal::distributed
