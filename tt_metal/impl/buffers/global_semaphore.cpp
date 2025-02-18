// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <global_semaphore.hpp>

#include <cstdint>
#include <memory>
#include <vector>

#include <assert.hpp>
#include <core_coord.hpp>
#include <tt_metal.hpp>
#include <tt-metalium/distributed.hpp>
#include <host_api.hpp>
#include <buffer.hpp>
#include <buffer_constants.hpp>
#include <device.hpp>
#include <hal.hpp>

#include "tt_cluster.hpp"

namespace tt::tt_metal {

GlobalSemaphore::GlobalSemaphore(
    IDevice* device, const CoreRangeSet& cores, uint32_t initial_value, BufferType buffer_type) :
    device_(device), cores_(cores) {
    this->setup_buffer(initial_value, buffer_type);
}

GlobalSemaphore::GlobalSemaphore(
    IDevice* device, CoreRangeSet&& cores, uint32_t initial_value, BufferType buffer_type) :
    device_(device), cores_(std::move(cores)) {
    this->setup_buffer(initial_value, buffer_type);
}

void GlobalSemaphore::setup_buffer(uint32_t initial_value, BufferType buffer_type) {
    TT_FATAL(
        buffer_type == BufferType::L1 or buffer_type == BufferType::L1_SMALL,
        "Global semaphore can only be created for L1 buffer types");
    TT_FATAL(device_ != nullptr, "Device cannot be null");
    TT_FATAL(cores_.num_cores() > 0, "CoreRangeSet must have at least one core");
    uint32_t num_cores = cores_.num_cores();
    auto shard_parameters = ShardSpecBuffer(cores_, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {num_cores, 1});
    ShardedBufferConfig sem_shard_config = {
        .device = device_,
        .size = num_cores * sizeof(uint32_t),
        .page_size = sizeof(uint32_t),
        .buffer_type = buffer_type,
        .buffer_layout = TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = std::move(shard_parameters),
    };
    buffer_ = CreateBuffer(sem_shard_config);

    this->reset_semaphore_value(initial_value);
}

IDevice* GlobalSemaphore::device() const { return device_; }

DeviceAddr GlobalSemaphore::address() const { return buffer_->address(); }

void GlobalSemaphore::reset_semaphore_value(uint32_t reset_value) const {
    // Write the initial value to the semaphore to the device
    // Only block for the slow dispatch case
    auto* device = device_;
    device->push_work([device, reset_value, num_cores = cores_.num_cores(), buffer = buffer_] {
        std::vector<uint32_t> host_buffer(num_cores, reset_value);
        if (device->using_slow_dispatch()) {
            detail::WriteToBuffer(*buffer, host_buffer);
            tt::Cluster::instance().l1_barrier(device->id());
        } else {
            // Dynamic resolution of device types is unclean and poor design. This will be cleaned up
            // when MeshBuffer + Buffer and MeshCommandQueue + CommandQueue are unified under the same
            // API
            if (dynamic_cast<distributed::MeshDevice*>(device)) {
                distributed::MeshDevice* mesh_device = dynamic_cast<distributed::MeshDevice*>(device);

                distributed::ReplicatedBufferConfig replicated_buffer_config{.size = buffer->size()};
                distributed::DeviceLocalBufferConfig local_config{
                    .page_size = buffer->page_size(),
                    .buffer_type = buffer->buffer_type(),
                    .buffer_layout = buffer->buffer_layout(),
                    .shard_parameters = buffer->shard_spec(),
                    .bottom_up = buffer->bottom_up()};
                auto mesh_buffer = distributed::MeshBuffer::create(
                    replicated_buffer_config, local_config, mesh_device, buffer->address());
                distributed::EnqueueWriteMeshBuffer(mesh_device->mesh_command_queue(), mesh_buffer, host_buffer);
                // mesh_device->mesh_command_queue().enqueue_write_buffer_broadcast(buffer, host_buffer.data(), false);
            } else {
                EnqueueWriteBuffer(device->command_queue(), buffer, host_buffer, false);
            }
        }
    });
}

}  // namespace tt::tt_metal

namespace std {

std::size_t hash<tt::tt_metal::GlobalSemaphore>::operator()(
    const tt::tt_metal::GlobalSemaphore& global_semaphore) const {
    return tt::stl::hash::hash_objects_with_default_seed(global_semaphore.attribute_values());
}

}  // namespace std
