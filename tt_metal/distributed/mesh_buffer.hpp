// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/impl/buffers/buffer_constants.hpp"
#include "tt_metal/tt_stl/overloaded.hpp"
#include "tt_metal/distributed/mesh_device.hpp"
#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/impl/buffers/buffer.hpp"
#include "tt_metal/distributed/mesh_shape.hpp"

namespace tt::tt_metal::distributed {

// Specifies how a buffer is laid out across Memory Banks within a single device.
struct DeviceLocalLayoutConfig {
    DeviceAddr page_size;

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
    std::pair<size_t, size_t> global_buffer_shape = {0, 0};
};

using MeshBufferConfig = std::variant<ReplicatedBufferConfig, ShardedBufferConfig>;

class MeshBuffer {
public:
    static MeshBuffer create(
        const MeshBufferConfig& mesh_buffer_config,
        const DeviceLocalLayoutConfig& device_local_layout,
        BufferType buffer_type,
        MeshDevice* mesh_device);

    MeshDevice* mesh_device() const { return mesh_device_; }
    DeviceAddr device_local_size() const { return device_local_size_; }
    void deallocate() {}

private:
    MeshBuffer(
        const MeshBufferConfig& config,
        const DeviceLocalLayoutConfig& device_local_layout,
        DeviceAddr device_local_size,
        MeshDevice* mesh_device) :
        mesh_device_(mesh_device),
        config_(config),
        device_local_layout_(device_local_layout),
        device_local_size_(device_local_size) {}

    MeshDevice* mesh_device_ = nullptr;
    DeviceAddr device_local_size_;

    MeshBufferConfig config_;
    DeviceLocalLayoutConfig device_local_layout_;
};

}  // namespace tt::tt_metal::distributed
