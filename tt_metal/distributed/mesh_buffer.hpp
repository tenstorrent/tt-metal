// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/impl/buffers/buffer_constants.hpp"
#include "tt_metal/distributed/mesh_device.hpp"
#include "tt_metal/impl/buffers/buffer.hpp"

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
    DeviceAddr buffer_size;
};

// Specifies sharded MeshBuffer.
struct ShardedBufferConfig {
    // Global buffer size. Each device will get a fraction of this size.
    DeviceAddr global_buffer_size;

    // Global shape of the buffer; at metal-level, we expect the shape to be aligned with the mesh shape.
    // TODO: Consider a 2D shape class.
    std::pair<size_t, size_t> global_buffer_shape = {0, 0};

    // Shard shape, sent to each device.
    // TODO: Consider a 2D shape class.
    std::pair<size_t, size_t> shard_shape = {0, 0};

    // Orientation of the shards in a mesh.
    ShardOrientation shard_orientation = ShardOrientation::ROW_MAJOR;

    // Computes the number of bytes per datum in the sharded buffer.
    uint32_t compute_datum_size_bytes() const {
        return global_buffer_size / (global_buffer_shape.first * global_buffer_shape.second);
    }
};

enum class MeshBufferLayout : uint8_t { REPLICATED, SHARDED };
using MeshBufferConfig = std::variant<ReplicatedBufferConfig, ShardedBufferConfig>;

class MeshBuffer {
public:
    static std::shared_ptr<MeshBuffer> create(
        const MeshBufferConfig& mesh_buffer_config,
        const DeviceLocalBufferConfig& device_local_layout,
        MeshDevice* mesh_device);

    MeshDevice* mesh_device() const { return mesh_device_; }
    DeviceAddr device_local_size() const { return device_local_size_; }
    DeviceAddr global_size() const;
    DeviceAddr address() const { return address_; };

    MeshBufferLayout global_layout() const;
    const MeshBufferConfig& global_config() const { return config_; }
    const ShardedBufferConfig& global_shard_spec() const;
    const DeviceLocalBufferConfig& device_local_config() const { return device_local_config_; }

    std::shared_ptr<Buffer> get_device_buffer(uint32_t logical_x, uint32_t logical_y);
    void deallocate();

private:
    MeshBuffer(
        const MeshBufferConfig& config,
        const DeviceLocalBufferConfig& device_local_config,
        DeviceAddr device_local_size,
        MeshDevice* mesh_device) :
        config_(config),
        device_local_config_(device_local_config),
        mesh_device_(mesh_device),
        device_local_size_(device_local_size) {}

    void allocate();

    MeshBufferConfig config_;
    DeviceLocalBufferConfig device_local_config_;
    MeshDevice* mesh_device_ = nullptr;
    DeviceAddr address_ = 0;
    DeviceAddr device_local_size_ = 0;

    // TODO: Conisder optimizing with SmallVector.
    std::vector<std::vector<std::shared_ptr<Buffer>>> buffers_;
};

}  // namespace tt::tt_metal::distributed
